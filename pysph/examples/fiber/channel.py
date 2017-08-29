"""Shear flow involving a single fiber rotating. (10 mins)
"""
# general imports
import os
import smtplib
import json

# profiling
import cProfile
import pstats

# matplotlib (set up for server use)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# numpy and scipy
import numpy as np
from scipy.integrate import odeint

# mail for notifications
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# PySPH imports
from pysph.base.config import get_config
from pysph.base.nnps import DomainManager, LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline

from pysph.tools.interpolator import Interpolator

from pysph.solver.application import Application
from pysph.solver.utils import load,remove_irrelevant_files
from pysph.solver.solver import Solver

from pysph.sph.integrator import EulerIntegrator, EPECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep, EBGStep
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (SummationDensity, VolumeSummation,
    StateEquation, MomentumEquationPressureGradient, ContinuityEquation,
    MomentumEquationViscosity, MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity,
    VolumeFromMassDensity)
from pysph.sph.fiber.utils import (Damping, HoldPoints, VelocityGradient,
    Contact, ComputeDistance)
from pysph.sph.fiber.beadchain import (Tension, Bending, EBGVelocityReset,
    Friction, ArtificialDamping)


# Jeffrey's equivalent aspect ratio (coarse approximation)
#   H. L. Goldsmith and S. G. Mason
#   CHAPTER 2 - THE MICRORHEOLOGY OF DISPERSIONS A2 - EIRICH, pp. 85–250.
#   Academic Press, 1967.
def get_equivalent_aspect_ratio(aspect_ratio):
    return -0.0017*aspect_ratio**2+0.742*aspect_ratio

# Jeffery's Equation for planar rotation of a rigid (theta=0)
def jeffery_ode(phi, t, ar_equiv, G):
    lbd = (ar_equiv**2-1.0)/(ar_equiv**2+1.0)
    return 0.5*G*(1.0+lbd*np.cos(2.0*phi))

class Channel(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0002, help="Fiber diameter"
        )
        group.add_argument(
            "--ar", action="store", type=int, dest="ar",
            default=11, help="Aspect ratio of fiber"
        )
        group.add_argument(
            "--rho", action="store", type=float, dest="rho0",
            default=1000, help="Rest density"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=1000, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E11, help="Young's modulus"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=4, help="Shear rate"
        )
        group.add_argument(
            "--g", action="store", type=float, dest="g",
            default=0, help="Body force in x-direction"
        )
        group.add_argument(
            "--D", action="store", type=float, dest="D",
            default=10000, help="Damping coefficient for arificial damping"
        )
        group.add_argument(
            "--dim", action="store", type=float, dest="dim",
            default=2, help="Dimension of problem"
        )
        group.add_argument(
            "--mail", action="store", type=str, dest="mail",
            default=None, help="Set notification e-mail adress."
        )
        group.add_argument(
            "--width", action="store", type=int, dest="width",
            default=None, help="Channel width (multiples of fiber diameter)"
        )
        group.add_argument(
            "--holdcenter", action="store_true", dest='holdcenter',
            default=False, help="Holding center particle in place."
        )
        group.add_argument(
            "--vtk", action="store_true", dest='vtk',
            default=False, help="Enable vtk-output during solving."
        )
        group.add_argument(
            "--postonly", action="store_true", dest="postonly",
            default=False, help="Set time to zero and postprocess only."
        )
        group.add_argument(
            "--no-pb", action="store_true", dest="nopb",
            default=False, help="Disable background pressure"
        )
        group.add_argument(
            "--massscale", action="store", type=float, dest="scale_factor",
            default=1E5, help="Factor of mass scaling"
        )
        group.add_argument(
            "--fluidres", action="store", type=float, dest="fluid_res",
            default=1, help="Resolution of fluid particles relative to fiber."
        )


    def consume_user_options(self):
        """Initialization of geometry, properties and time stepping."""

        # Initial spacing of particles is set to the same value as fiber
        # diameter.
        self.dx = self.options.d

        # Smoothing radius is set to the same value as particle spacing. This
        # results for a quintic spline in a radius of influence three times as
        # large as dx
        self.h0 = self.dx

        # The density can be scaled using the mass scaling factor. To account
        # for proper external forces, gravity is scaled just the other way.
        self.options.rho0 = self.options.rho0*self.options.scale_factor
        self.options.g = self.options.g/self.options.scale_factor

        # The fiber length is the aspect ratio times fiber diameter
        self.Lf = self.options.ar*self.dx

        # If a specific width is set, use this as multiple of dx to determine
        # the channel width. Otherwise use the fiber aspect ratio.
        multiples = self.options.width or self.options.ar
        self.Ly = multiples*self.dx + 2*int(0.1*multiples)*self.dx

        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.0*self.Ly + self.dx

        # The position of the fiber's center is set to the center of the
        # channel.
        self.x_fiber = 1/2*self.Lx
        self.y_fiber = 0.5*self.Ly
        self.z_fiber = 0.5*self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled (!) density.
        self.nu = self.options.mu/self.options.rho0

        # For 2 dimensions surface, mass and moments have a different coputation
        # than for 3 dimensions.
        if self.options.dim == 2:
            self.A = self.dx
            self.I = self.dx**3/12
            mass = 3*self.options.rho0*self.dx*self.A
            self.J = 1/12*mass*(self.dx**2 + (3*self.dx)**2)
        else:
            R = self.dx/2
            self.A = np.pi*R**2
            self.I = np.pi*R**4/4.0
            mass = 3*self.options.rho0*self.dx*self.A
            self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.dx)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.Vmax = (self.options.G*self.Ly/2
                    + self.options.g/(2*self.nu)*self.Ly**2/4)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.options.rho0

        # Background pressure in Adami's transport velocity formulation
        if self.options.nopb:
            self.pb = 0.0
        else:
            self.pb = self.p0

        # The time is set to zero, if only postprocessing is required. For a
        # shear flow, it is set to the time for a full period of rotation
        # according to Jeffery's equation. For a Poiseuille flow, it is set to
        # the time to reach steady state for width = 20 and g = 10 to match
        # the FEM result in COMSOL for a single cylinder.
        are = get_equivalent_aspect_ratio(self.options.ar)
        if self.options.postonly:
            self.t = 0
        else:
            if self.options.G > 0.1:
                self.t = 1.0*np.pi*(are+1.0/are)/self.options.G
            else:
                self.t = 1.5E-5*self.options.scale_factor
        print("Simulated time is %g s"%self.t)

        # Time steps
        dt_cfl = 0.4 * self.h0/(self.c0 + self.Vmax)
        dt_viscous = 0.125 * self.h0**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h0/(self.options.g+0.001))
        dt_tension = 0.5*self.dx*np.sqrt(self.options.rho0/self.options.E)
        dt_bending = 0.5*self.dx**2*np.sqrt(self.options.rho0*self.A/(self.options.E*2*self.I))
        print("dt_cfl: %g"%dt_cfl)
        print("dt_viscous: %g"%dt_viscous)
        print("dt_force: %g"%dt_force)
        print("dt_tension: %g"%dt_tension)
        print("dt_bending: %g"%dt_bending)

        # The outer loop time step is set to the minimum of force, cfl and
        # viscous time step. The inner loop operates with a fiber time step
        # computed from tension and bending.
        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.fiber_dt = min(dt_tension, dt_bending)
        print("Time step ratio is %g"%(self.dt/self.fiber_dt))

    def create_scheme(self):
        """There is no scheme used in this application and equaions are set up
        manually."""
        return None

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.options.fluid_res*self.dx
        dx2 = fdx/2

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        _z = np.arange(dx2, self.Ly, fdx)
        fx,fy,fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position. Uncomment proper section for
        # horizontal or vertical alignment respectivley.
        indices = []
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber

            # vertical
            if (fx[i] < xx+self.dx/2 and fx[i] > xx-self.dx/2 and
                fy[i] < yy+self.Lf/2 and fy[i] > yy-self.Lf/2 and
                fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
                indices.append(i)

            #horizontal
            # if (fx[i] < xx+self.Lf/2 and fx[i] > xx-self.Lf/2 and
            #     fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
            #     fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
            #    indices.append(i)

        # Generating fiber particle grid. Uncomment proper section for
        # horizontal or vertical alignment respectivley.

        # vertical fiber
        _fibx = np.array([xx])
        _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/4, self.dx)

        # horizontal fiber
        # _fibx = np.arange(xx-self.Lf/2+self.dx/2, xx+self.Lf/2+self.dx/4, self.dx)
        # _fiby = np.array([yy])

        _fibz = np.array([zz])
        fibx, fiby = np.meshgrid(_fibx, _fiby)
        fibx = fibx.ravel()
        fiby = fiby.ravel()
        fibz = fiby.ravel()

        # Determine the size of dummy region
        ghost_extent = 3*fdx/self.options.fluid_res

        # Create the channel particles at the top
        _y = np.arange(self.Ly+dx2, self.Ly+dx2+ghost_extent, fdx)
        tx,ty,tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        bx,by,bz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the right
        _z = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        _y = np.arange(dx2-ghost_extent, self.Ly+ghost_extent, fdx)
        rx,ry,rz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the left
        _z = np.arange(self.Ly+dx2, self.Ly+dx2+ghost_extent, fdx)
        _y = np.arange(dx2-ghost_extent, self.Ly+ghost_extent, fdx)
        lx,ly,lz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays (and for 3D cas also right and
        # left arrays)
        if self.options.dim ==2:
            cx = np.concatenate((tx, bx))
            cy = np.concatenate((ty, by))
        else:
            cx = np.concatenate((tx, bx, rx, lx))
            cy = np.concatenate((ty, by, ry, ly))
            cz = np.concatenate((tz, bz, rz, lz))

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        if self.options.dim == 2:
            channel = get_particle_array(name='channel', x=cx, y=cy)
            fluid = get_particle_array(name='fluid', x=fx, y=fy)
            fluid.remove_particles(indices)
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby)
        else:
            channel = get_particle_array(name='channel', x=cx, y=cy, z=cz)
            fluid = get_particle_array(name='fluid', x=fx, y=fy, z=fz)
            fluid.remove_particles(indices)
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby, z=fibz)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d, nfiber = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fiber.get_number_of_particles()))

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles()==self.options.ar)

        # Add requisite variables needed for this formulation
        for name in ('V', 'wf','uf','vf','wg','wij','vg','ug', 'phifrac',
                     'awhat', 'avhat','auhat', 'vhat', 'what', 'uhat', 'vmag2',
                     'arho', 'phi0', 'fractag', 'rho0','holdtag', 'eu', 'ev',
                     'ew', 'dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz',
                     'dwdx','dwdy', 'dwdz', 'Fx', 'Fy', 'Fz', 'arho', 'ex',
                     'ey', 'ez'):
            fluid.add_property(name)
            channel.add_property(name)
            fiber.add_property(name)
        for name in ('lprev', 'lnext', 'phi0', 'xcenter',
                     'ycenter', 'rxnext', 'rynext', 'rznext', 'rnext', 'rxprev',
                     'ryprev', 'rzprev', 'rprev', 'fidx'):
            fiber.add_property(name)

        # set the output property arrays
        fluid.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid', 'V'])
        channel.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid', 'V'])
        fiber.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'p',
                        'pid', 'holdtag', 'gid','ug', 'vg', 'wg', 'V', 'Fx',
                        'Fy', 'Fz'])

        # Computation of each particles initial volume.
        volume = fdx**self.options.dim
        fiber_volume = self.dx**self.options.dim

        # Mass is set to get the reference density of rho0.
        fluid.m[:] = volume * self.options.rho0
        channel.m[:] = volume * self.options.rho0
        fiber.m[:] = fiber_volume * self.options.rho0

        # Set initial distances and angles. This represents a straight
        # unstreched fiber in rest state. It fractures, if a segment of length
        # 2 dx is bend more than 11.5°.
        fiber.lprev[:] = self.dx
        fiber.lnext[:] = self.dx
        fiber.phi0[:] = np.pi
        fiber.phifrac[:] = 0.2

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 0
        if self.options.holdcenter:
            idx = int(np.floor(self.options.ar/2))
            fiber.holdtag[idx] = 100

        # assign unique ID (within fiber) to each fiber particle.
        fiber.fidx[:] = range(0,fiber.get_number_of_particles())

        # Set the default density.
        fluid.rho[:] = self.options.rho0
        channel.rho[:] = self.options.rho0
        fiber.rho[:] = self.options.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume
        fiber.V[:] = 1./fiber_volume

        # The smoothing lengths are set accorindg to each particles size.
        fluid.h[:] = self.h0
        channel.h[:] = self.h0
        fiber.h[:] = self.h0

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.y[:]-self.Ly/2)
                    #- 1/2*self.options.g/self.nu*(
                    #    (fluid.y[:]-self.Ly/2)**2-(self.Ly/2)**2))
        fiber.u[:] = self.options.G*(fiber.y[:]-self.Ly/2)
                    #- 1/2*self.options.g/self.nu*(
                    #    (fiber.y[:]-self.Ly/2)**2-(self.Ly/2)**2))

        # Upper and lower walls move uniformly depending on shear rate.
        N = channel.get_number_of_particles()-1
        for i in range(0,N):
            if channel.y[i] > self.Ly/2:
                y = self.Ly/2
            else:
                y = -self.Ly/2
            channel.u[i] = self.options.G*y

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_equations(self):
        """Setting up all equations manually."""

        all = ['fluid', 'channel', 'fiber']
        equations = [
            # The first group computes densities in the fluid phase and corrects
            # the wall's inverse volumes according to their current density. It
            # is applied to all particles including dummies. (real=False)
            Group(
                equations=[
                    EBGVelocityReset(dest='fiber', sources=None),
                    SummationDensity(dest='fluid', sources=all),
                    SummationDensity(dest='fiber', sources=all),
                    VolumeFromMassDensity(dest='channel', sources=all),
                    ComputeDistance(dest='fiber', sources=['fiber']),
                ],
                real=False,
            ),
            # This group mainly updates properties such as velocity gradient,
            # imaginary velocities at wall dummy particles and pressures
            # based on an equation of state. It is applied to all particles
            # including the dummies.
            Group(
                equations=[
                    VelocityGradient(dest='fiber', sources=all),
                    VelocityGradient(dest='fluid', sources=all),
                    SetWallVelocity(dest='channel', sources=['fluid', 'fiber']),
                    SetWallVelocity(dest='fiber', sources=['fluid']),
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                    rho0=self.options.rho0, b=1.0),
                    StateEquation(dest='fiber', sources=None, p0=self.p0,
                                   rho0=self.options.rho0, b=1.0),
                ],
                real=False,
            ),
            # This group updates the pressure of wall particles only. Since
            # these are dummy particles, the flag real is set to false.
            Group(
                equations=[
                    SolidWallPressureBC(dest='channel',
                                        sources=['fluid', 'fiber'],
                                        b=1.0, rho0=self.options.rho0, p0=self.p0),
                ],
                real=False,
            ),
            # This group contains the actual computation of accelerations.
            Group(
                equations=[
                    Friction(dest='fiber', sources=None, J=self.J,
                            A=self.A, mu=self.options.mu, d=self.options.d,
                            ar=self.options.ar),
                    Contact(dest='fiber', sources=['fiber'],
                            E=self.options.E, d=self.options.d,
                            scale=self.options.scale_factor),
                    MomentumEquationPressureGradient(dest='fluid', sources=all,
                                        pb=self.pb, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationPressureGradient(dest='fiber', sources=all,
                                        pb=0.0, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationViscosity(dest='fluid',
                                        sources=['fluid'], nu=self.nu),
                    MomentumEquationViscosity(dest='fiber',
                                        sources=['fluid'], nu=self.nu),
                    SolidWallNoSlipBC(dest='fluid',
                                        sources=['channel','fiber'], nu=self.nu),
                    SolidWallNoSlipBC(dest='fiber',
                                        sources=['channel'], nu=self.nu),
                    MomentumEquationArtificialStress(dest='fluid',
                                        sources=['fluid', 'fiber']),
                    MomentumEquationArtificialStress(dest='fiber',
                                        sources=['fluid', 'fiber']),
                ],
            ),
            # This group resets some varibles: The acceleration and velocity of
            # hold particles is removed and ebg velocities used for the inner
            # loop of bending and tension are set to 0.
            Group(
                equations=[
                    HoldPoints(dest='fiber', sources=None, tag=100),
                ]
            ),
        ]
        return equations

    def _configure(self):
        """The second integrator is a simple Euler-Integrator (accurate
        enough due to very small time steps; very fast) using EBGSteps.
        EBGSteps are basically the same as EulerSteps, exept for the fact
        that they work with an intermediate ebg velocity [eu, ev, ew].
        This velocity does not interfere with the actual velocity, which
        is neseccery to not disturb the real velocity through artificial
        damping in this step. The ebg velocity is initialized for each
        inner loop again and reset in the outer loop."""

        super(Channel, self)._configure()
        # if there are more than 1 particles involved, elastic equations are
        # iterated in an inner loop.
        if self.options.ar > 1:
            # second integrator
            self.fiber_integrator = EulerIntegrator(fiber=EBGStep())
            # The type of spline has no influence here. It must be large enough
            # to contain the next particle though.
            kernel = QuinticSpline(dim=self.options.dim)
            equations = [
                        Group(
                            equations=[
                                ComputeDistance(dest='fiber',sources=['fiber']),
                            ],
                        ),
                        # The first group computes all accelerations based on
                        # tension, bending and damping.
                        Group(
                            equations=[
                                Tension(dest='fiber',
                                        sources=None,
                                        ea=self.options.E*self.A),
                                Bending(dest='fiber',
                                        sources=None,
                                        ei=self.options.E*self.I),
                                Contact(dest='fiber',
                                        sources=['fiber'],
                                        E=self.options.E, d=self.options.d,
                                        scale=self.options.scale_factor),
                                ArtificialDamping(dest='fiber',
                                        sources=None,
                                        d = self.options.D),
                                ],
                            ),
                        # The second group resets accelerations for hold points.
                        Group(
                            equations=[
                                HoldPoints(dest='fiber', sources=None, tag=100),
                            ]
                        ),
                    ]
            # These equations are applied to fiber particles only - that's the
            # reason for computational speed up.
            particles = [p for p in self.particles if p.name == 'fiber']
            # A seperate DomainManager is needed to ensure that particles don't
            # leave the domain.
            domain = DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)
            # A seperate list for the nearest neighbourhood search is benefitial
            # since it is much smaller than the original one.
            nnps = LinkedListNNPS(dim=self.options.dim, particles=particles,
                            radius_scale=kernel.radius_scale, domain=domain,
                            fixed_h=False, cache=False, sort_gids=False)
            # The acceleration evaluator needs to be set up in order to compile
            # it together with the integrator.
            acceleration_eval = AccelerationEval(
                        particle_arrays=particles,
                        equations=equations,
                        kernel=kernel,
                        mode='serial')
            # Compilation of the integrator not using openmp, because the
            # overhead is too large for those few fiber particles.
            comp = SPHCompiler(acceleration_eval, self.fiber_integrator)
            config = get_config()
            config.use_openmp = False
            comp.compile()
            config.use_openmp = True
            acceleration_eval.set_nnps(nnps)

            # Connecting neighbourhood list to integrator.
            self.fiber_integrator.set_nnps(nnps)

    def create_solver(self):
        """Setting up the default integrator for fiber particles"""
        kernel = QuinticSpline(dim=self.options.dim)
        integrator = EPECIntegrator(fluid=TransportVelocityStep(),
                                    fiber=TransportVelocityStep())
        solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
                         tf=self.t, pfreq=int(self.t/(100*self.dt)),
                        vtk=self.options.vtk)
        # solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
        #                  tf=self.t, pfreq=1, vtk=True)
        return solver

    def post_stage(self, current_time, dt, stage):
        """This post stage function gets called after each outer loop and starts
        an inner loop for the fiber iteration."""
        if self.options.ar > 1:
            # 1) predictor
            # 2) post stage 1:
            if stage == 1:
                N = int(np.ceil(self.dt/self.fiber_dt))
                for n in range(0,N):
                    self.fiber_integrator.step(current_time,dt/N)
                    current_time += dt/N
            # 3) Evaluation
            # 4) post stage 2

    def get_meshgrid(self, xx, yy, zz):
        """This function is just a shorthand for the generation of meshgrids."""
        if self.options.dim == 2:
            x, y = np.meshgrid(xx, yy)
            x = x.ravel()
            y = y.ravel()
            z = self.z_fiber*np.ones(np.shape(y))
        else:
            x, y, z = np.meshgrid(xx, yy, zz)
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
        return [x,y,z]

    def _plot_streamlines(self):
        """This function plots streamlines and the pressure field. It
         interpolates the properties from particles using the kernel."""

        # lenght factor m --> mm
        factor = 1000

        # Interpolation grid
        x = np.linspace(0,self.Lx,400)
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)

        # Extract positions of fiber particles from last step to plot them
        # on top of velocities.
        last_output = self.output_files[-1]
        data = load(last_output)
        fiber = data['arrays']['fiber']
        fx = fiber.x
        fy = fiber.y

        # Interpolation (precompiled) of velocities and pressure
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        v = interp.interpolate('v')
        p = interp.interpolate('p')
        vmag = np.sqrt(u**2 + v**2 )

        if self.options.ar == 1:
            upper = 2.5E-5
        else:
            upper = np.max(vmag)

        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.viridis
        levels = np.linspace(0, upper, 30)

        # velocity contour
        vel = plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=upper, vmin=0)
        # streamlines
        stream = plt.streamplot(x*factor,y*factor,u,v, color='k', density=0.5)
        # fiber
        plt.scatter(fx*factor,fy*factor, color='w')

        # set labels
        cbar = plt.colorbar(vel, label='Velocity Magnitude', format='%.2e')
        plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.eps')
        plt.savefig(fig, dpi=300)
        print("Streamplot written to %s."% fig)

        if self.options.ar == 1:
            upper = 100
            lower = -100
        else:
            upper = np.max(p)
            lower = np.min(p)

        # open new plot
        plt.figure()
        #configuring new color map
        cmap = plt.cm.viridis
        levels = np.linspace(lower, upper, 30)

        # pressure contour
        pres = plt.contourf(x*factor,y*factor, p, levels=levels,
                 cmap=cmap,  vmin=lower, vmax=upper)

        # set labels
        cbar = plt.colorbar(pres, label='Pressure')
        plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        # save plot
        p_fig = os.path.join(self.output_dir, 'pressure.eps')
        plt.savefig(p_fig, dpi=300)
        print("Pressure written to %s."% p_fig)

        return[fig, p_fig]

    def _plot_inlet_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the periodic boundary. If
        the fiber has only a single particle, this is interpreted as flow around
        a fiber cylinder and the coresponding FEM solution is plotted as well.
        """
        # length factor m --> mm
        factor = 1000

        # Extract requested output - default is last output.
        output = self.output_files[step_idx]
        data = load(output)

        # Generate meshgrid for interpolation.
        x = np.array([0])
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # solution for undisturbed velocity field.
        u_exact = (self.options.G * (y - self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                            (y-self.Ly/2)**2-(self.Ly/2)**2))

        # FEM solution for disturbed velocity field (ar=1, g=10, G=0, width=20)
        y_fem = np.array([1.20E-04,3.60E-04,6.00E-04,8.40E-04,0.00108,0.00132,
                        0.00156,0.0018,0.00204,0.00228,0.00252,0.00276,0.003,
                        0.00324,0.00348,0.00372,0.00396,0.0042,0.00444,0.00468])

        u_fem = np.array([1.85E-06,5.23E-06,8.17E-06,1.08E-05,1.31E-05,1.50E-05,
                        1.66E-05,1.77E-05,1.85E-05,1.89E-05,1.89E-05,1.85E-05,
                        1.77E-05,1.66E-05,1.50E-05,1.31E-05,1.08E-05,8.18E-06,
                        5.22E-06,1.85E-06])

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u*factor, y*factor , '-k')

        # FEM solution (if applicable)
        if self.options.ar == 1:
            plt.plot(u_fem*factor, y_fem*factor , '--k')

        # undisturbed solution
        plt.plot(u_exact*factor, y*factor, ':k')

        # labels
        plt.title('Velocity at inlet')
        plt.xlabel('Velocity [mm/s]')
        plt.ylabel('Position [mm]')
        if self.options.ar == 1:
            plt.legend(['SPH Simulation', 'FEM', 'No obstacle'])
        else:
            plt.legend(['SPH Simulation', 'No obstacle'])

        # save figure
        fig = os.path.join(self.output_dir, 'inlet_velocity.eps')
        plt.savefig(fig, dpi=300)
        print("Inlet velocity plot written to %s."% fig)

        return (fig)

    def _plot_center_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the center across the
        particle. If the fiber has only a single particle, this is interpreted
        as flow arounda fiber cylinder and the coresponding FEM solution is
        plotted as well.
        """
        # length factor m --> mm
        factor = 1000

        # Extract requested output - default is last output.
        output = self.output_files[step_idx]
        data = load(output)

        # Generate meshgrid for interpolation.
        x = np.array([self.x_fiber])
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # FEM solution for disturbed velocity field (ar=1, g=10, G=0, width=20)
        y_fem = np.array([4.80E-05,1.44E-04,2.40E-04,3.36E-04,4.32E-04,5.28E-04,
                        6.24E-04,7.20E-04,8.16E-04,9.12E-04,1.01E-03,1.10E-03,
                        1.20E-03,1.30E-03,1.39E-03,1.49E-03,1.58E-03,1.68E-03,
                        1.78E-03,1.87E-03,1.97E-03,2.06E-03,2.16E-03,2.26E-03,
                        2.35E-03,2.45E-03,2.54E-03,2.64E-03,2.74E-03,2.83E-03,
                        2.93E-03,3.02E-03,3.12E-03,3.22E-03,3.31E-03,3.41E-03,
                        3.50E-03,3.60E-03,3.70E-03,3.79E-03,3.89E-03,3.98E-03,
                        4.08E-03,4.18E-03,4.27E-03,4.37E-03,4.46E-03,4.56E-03,
                        4.66E-03,4.75E-03])

        u_fem = np.array([1.21E-06,3.50E-06,5.59E-06,7.51E-06,9.26E-06,1.09E-05,
                    1.23E-05,1.36E-05,1.47E-05,1.57E-05,1.66E-05,1.73E-05,
                    1.78E-05,1.83E-05,1.85E-05,1.86E-05,1.84E-05,1.81E-05,
                    1.75E-05,1.66E-05,1.53E-05,1.34E-05,1.05E-05,5.16E-06,0,0,
                    5.13E-06,1.05E-05,1.34E-05,1.53E-05,1.66E-05,1.75E-05,
                    1.81E-05,1.84E-05,1.86E-05,1.85E-05,1.83E-05,1.78E-05,
                    1.73E-05,1.66E-05,1.57E-05,1.47E-05,1.36E-05,1.23E-05,
                    1.09E-05,9.26E-06,7.51E-06,5.59E-06,3.50E-06,1.22E-06])

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u*factor, y*factor , '-k')

        # FEM solution (if applicable)
        plt.plot(u_fem*factor, y_fem*factor , '--k')

        # labels
        plt.title('Velocity at center')
        plt.xlabel('Velocity [mm/s]')
        plt.ylabel('Position [mm]')
        plt.legend(['SPH', 'FEM'])

        # save figure
        fig = os.path.join(self.output_dir, 'center_velocity.eps')
        plt.savefig(fig, dpi=300)
        print("Center velocity plot written to %s."% fig)

        return (fig)

    def _plot_pressure_centerline(self):
        """This function plots the pressure profile along a centerline for a
        single particle."""

        # length factor m --> mm
        factor = 1000

        # Generate meshgrid for interpolation.
        x = np.linspace(0,self.Lx,200)
        y = np.array([self.Ly/2])
        x,y = np.meshgrid(x,y)

        # Set a number of last solutions to average from.
        N = 10

        # averaging the pressure interpolation for N last solutions.
        p = np.zeros((200,))
        for output in self.output_files[-(1+N):-1]:
            data = load(output)
            interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
            interp.update_particle_arrays(list(data['arrays'].values()))
            p += interp.interpolate('p')/N

        # FEM solution for disturbed velocity field (ar=1, g=10, G=0, width=20)
        x_fem = np.array([0.00010,0.00029,0.00048,0.00067,0.00086,0.00106,
                        0.00125,0.00144,0.00163,0.00182,0.00202,0.00221,0.00240,
                        0.00259,0.00278,0.00298,0.00317,0.00336,0.00355,0.00374,
                        0.00394,0.00413,0.00432,0.00451,0.00470,0.00480,0.00490,
                        0.00509,0.00528,0.00547,0.00566,0.00586,0.00605,0.00624,
                        0.00643,0.00662,0.00682,0.00701,0.00720,0.00739,0.00758,
                        0.00778,0.00797,0.00816,0.00835,0.00854,0.00874,0.00893,
                        0.00912,0.00931,0.00950])
        p_fem = np.array([0,1,2,3,3,4,5,5,6,7,7,8,9,9,10,11,12,14,15,17,21,26,
                        36,59,168,np.nan,-169,-58,-35,-25,-20,-17,-15,-13,-11,
                        -10,-9,-9,-8,-7,-6,-6,-5,-4,-4,-3,-2,-2,-1,-1,0])

        # open new plot
        plt.figure()

        # plot SPH solution and FEM solution
        plt.plot(x[0,:]*factor, p, '-k', x_fem*factor, p_fem, '--k')

        # labels
        plt.legend(['SPH Simulation','FEM Result'])
        plt.title('Pressure along center line')
        plt.xlabel('x [mm]')
        plt.ylabel('p [Pa]')

        # save figure
        pcenter_fig = os.path.join(self.output_dir, 'pressure_centerline.eps')
        plt.savefig(pcenter_fig, dpi=300)
        print("Pressure written to %s."% pcenter_fig)

        return pcenter_fig

    def _plot_history(self):
        """This function create all plots employing a iteration over all time
        steps. """

        # length factor m --> mm
        factor = 1000

        # empty list for time
        t = []

        # empty lists for orbit
        x_begin = []
        y_begin = []
        x_end = []
        y_end = []

        # empty list for orientation angle (only applicable for very
        # stiff/almost rigid fiber)
        angle = []
        N = 0

        # empty list for conservation properies
        E_kin = []
        E_p = []
        m = []
        volume = []
        rho = []

        # empty list for reaction forces
        Fx = []
        Fy = []
        Fz = []

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)

            # extracting time
            t.append(data['solver_data']['t'])

            # extrating all arrays.
            fiber = data['arrays']['fiber']
            fluid = data['arrays']['fluid']
            channel = data['arrays']['channel']

            # extrating end-points
            x_begin.append(fiber.x[0])
            y_begin.append(fiber.y[0])
            x_end.append(fiber.x[-1])
            y_end.append(fiber.y[-1])

            # computation of orientation angle
            dxx = fiber.x[0]-fiber.x[-1]
            dyy = fiber.y[0]-fiber.y[-1]
            a = np.arctan(dxx/(dyy+0.01*self.h0)) + N*np.pi
            if len(angle) > 0 and abs(a - angle[-1]) > 1:
                N += 1
                a += np.pi
            angle.append(a)

            # computation of squared velocity and masses from density and volume
            v_fiber = fiber.u**2 + fiber.v**2 + fiber.w**2
            v_fluid = fluid.u**2 + fluid.v**2 + fluid.w**2
            m_fiber = fiber.rho/fiber.V
            m_fluid = fluid.rho/fluid.V
            m_channel = channel.rho/channel.V

            # appending volume, density, mass, pressure and kinetic energy
            volume.append(np.sum(1/fiber.V)+np.sum(1/fluid.V)+np.sum(1/channel.V))
            rho.append(np.sum(fiber.rho)+np.sum(fluid.rho)+np.sum(channel.rho))
            m.append(np.sum(m_fiber)+np.sum(m_fluid)+np.sum(m_channel))
            E_p.append(np.sum(fiber.p/fiber.V)
                        +np.sum(fluid.p/fluid.V)
                        +np.sum(channel.p/channel.V))
            E_kin.append(0.5*np.dot(m_fiber,v_fiber)
                        +0.5*np.dot(m_fluid,v_fluid))

            # extract reaction forces at hold particles
            idx = np.argwhere(fiber.holdtag==100)
            Fx.append(fiber.Fx[idx])
            Fy.append(fiber.Fy[idx])
            Fz.append(fiber.Fz[idx])

        # open new plot
        plt.figure()

        # plot end points
        plt.plot(x_begin, y_begin, '-ok', markersize=3)
        plt.plot(x_end, y_end, '-*k', markersize=3)

        # set equally scaled axis to not distort the orbit
        plt.axis('equal')

        # save plot of orbit
        orbfig = os.path.join(self.output_dir, 'orbitplot.eps')
        plt.savefig(orbfig, dpi=300)
        print("Orbitplot written to %s."% orbfig)

        # Integrate Jeffery's solution
        print("Solving Jeffery's ODE")
        t = np.array(t)
        phi0 = angle[0]
        are = get_equivalent_aspect_ratio(self.options.ar)
        angle_jeffery = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(are,self.options.G))

        # constraint between -pi/2 and pi/2
        #angle_jeffery = (angle_jeffery+np.pi/2.0)%np.pi-np.pi/2.0

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t, angle, 'ok', markersize=3)
        plt.plot(t, angle_jeffery, '-k')

        # labels
        plt.xlabel('t [s]')
        plt.ylabel('Angle [rad]')
        plt.legend(['Simulation', 'Jeffery'])
        plt.title("ar=%g"%self.options.ar)

        # save figure
        angfig = os.path.join(self.output_dir, 'angleplot.eps')
        plt.savefig(angfig, dpi=300)
        print("Angleplot written to %s."% angfig)

        # save angles as *.csv file
        csv_file = os.path.join(self.output_dir, 'angle.csv')
        angle_jeffery = np.reshape(angle_jeffery,(angle_jeffery.size,))
        np.savetxt(csv_file, (t, angle, angle_jeffery), delimiter=',')

        # open new figure
        plt.figure()

        # plot pressure and kinetic energy
        plt.plot(t, E_p, '-k', t, E_kin, ':k')

        # labels
        plt.xlabel('t [s]')
        plt.ylabel('Energy')
        plt.title("Energy")
        plt.legend(['Pressure', 'Kinetic Energy'])

        # save figure
        engfig = os.path.join(self.output_dir, 'energyplot.eps')
        plt.savefig(engfig, dpi=300)
        print("Energyplot written to %s."% engfig)

        # open new plot
        plt.figure()

        # plot relative mass, volume and density
        plt.plot(t, np.array(m)/m[0], '-k',
                 t, np.array(volume)/volume[0], '--k',
                 t, np.array(rho)/rho[0], ':k')

        # labels
        plt.xlabel('t [s]')
        plt.ylabel('Relative value')
        plt.legend(['Mass', 'Volume', 'Density'])

        # save figure
        mfig = os.path.join(self.output_dir, 'massplot.eps')
        plt.savefig(mfig, dpi=300)
        print("Mass plot written to %s."% mfig)

        # hard-coded solutions for total reaction forces and viscous reaction
        # forces from FEM. (ar=1, g=10, G=0, width=20)
        t_fem = np.array([0,50,100,150,200,250,300,350,400,450,500,550,600,650,
                    700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,
                    1350,1400,1450,1500])
        Fv_fem = np.array([0.00003,0.01577,0.02616,0.03370,0.03918,0.04309,
                    0.04586,0.04783,0.04931,0.05036,0.05113,0.05163,0.05198,
                    0.05221,0.05236,0.05247,0.05254,0.05260,0.05263,0.05265,
                    0.05265,0.05265,0.05265,0.05265,0.05265,0.05265,0.05265,
                    0.05265,0.05265,0.05265,0.05265])
        F_fem = np.array([0.00037,0.03139,0.05188,0.06674,0.07754,0.08524,
                    0.09071,0.09460,0.09752,0.09959,0.10109,0.10208,0.10277,
                    0.10324,0.10353,0.10375,0.10388,0.10399,0.10406,0.10409,
                    0.10409,0.10409,0.10409,0.10409,0.10410,0.10410,0.10410,
                    0.10410,0.10410,0.10410,0.10410])

        # applying appropriate scale factors
        t_fem = t_fem/1E5
        t = np.array(t)/self.options.scale_factor*1000

        # Reaction force is plotted only for flow around single cylindrical
        # fiber
        if self.options.ar == 1:
            # open new plot
            plt.figure()

            # plot computed reaction force, total FEM force and viscous FEM
            # force
            plt.plot(t, Fx, '-k', t_fem, F_fem, '--k', t_fem, Fv_fem, ':k')

            # labels
            plt.xlabel('t [ms]')
            plt.ylabel('Force [N/m]')
            plt.title("Reaction Force")
            plt.legend(['SPH Simulation', 'FEM total force', 'FEM viscous force'])

            # save figure
            forcefig = os.path.join(self.output_dir, 'forceplot.eps')
            plt.savefig(forcefig, dpi=300)
            print("Reaction Force plot written to %s."% forcefig)
            return [orbfig, angfig, engfig, forcefig]
        else:
            return [orbfig, angfig, engfig]

    def _send_notification(self, info_fname, attachments=None):
        """Send a notification Mail after succesfull run."""

        gmail_user = "nils.batch.notification"
        gmail_pwd = "batch.notification"

        with open(info_fname, 'r') as f:
            info = json.load(f)
        cpu_time = info.get('cpu_time')

        msg = MIMEMultipart()
        msg['Subject'] = 'Batch results'
        msg['From'] = 'Batch Notification'
        msg['To'] = self.options.mail
        txt = MIMEText(""" Run finished. Parameters were\n
                            Diameter: %g\n
                            Aspect Ratio: %g\n
                            Density: %g\n
                            Absolute Viscosity: %g\n
                            Young's modulus: %g\n
                            Shear Rate: %g\n
                            Damping factor: %g\n
                            CPU Time: %g\n
                        """%(self.options.d, self.options.ar, self.options.rho0,
                            self.options.mu, self.options.E,
                            self.options.G, self.options.D,
                            cpu_time)
                        )
        msg.attach(txt)
        for fname in attachments:
            fp = open(fname, 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pwd)
            server.sendmail(gmail_user, self.options.mail, msg.as_string())
            server.close()
            print ("Successfully sent the mail.")
        except:
            print ("Failed to send mail.")


    def post_process(self, info_fname):
        [streamlines, pressure] = self._plot_streamlines()
        if self.options.ar == 1:
            pressure_centerline = self._plot_pressure_centerline()
            center_velocity = self._plot_center_velocity()
        history = self._plot_history()
        inlet = self._plot_inlet_velocity()
        if self.options.mail:
            self._send_notification(info_fname, [streamlines, history[0], history[1]])

def run_application():
    app = Channel()
    app.run()
    app.post_process(app.info_filename)

if __name__ == '__main__':
    run_application()
    # cProfile.runctx('run_application()', None, locals(), 'stats')
    # p = pstats.Stats('stats')
    # p.sort_stats('tottime').print_stats(10)
    # p.sort_stats('cumtime').print_stats(10)
