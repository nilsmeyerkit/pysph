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
from pysph.sph.ebg.fiber import (Tension, Bending, Vorticity, Friction, Damping,
    HoldPoints, EBGVelocityReset, ArtificialDamping, VelocityGradient)


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
        self.y_fiber = self.Ly/2
        self.z_fiber = self.Ly/2

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
            if not self.options.fluid_res == 1:
                print("Sure to use fluidres != 1 and background pressure?")

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
                self.t = 2.0*np.pi*(are+1.0/are)/self.options.G
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

    # There is no scheme used in this application and equaions are set up
    # manually.
    def create_scheme(self):
        return None

    # The channel has periodix boundary conditions in x-directions.
    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
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
            # if (fx[i] < xx+self.dx/2 and fx[i] > xx-self.dx/2 and
            #     fy[i] < yy+self.Lf/2 and fy[i] > yy-self.Lf/2 and
            #     fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
            #     indices.append(i)

            #horizontal
            if (fx[i] < xx+self.Lf/2 and fx[i] > xx-self.Lf/2 and
                fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
                fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
               indices.append(i)

        # Generating fiber particle grid. Uncomment proper section for
        # horizontal or vertical alignment respectivley.

        # vertical fiber
        # _fibx = np.array([xx])
        # _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/2, self.dx)

        # horizontal fiber
        _fibx = np.arange(xx-self.Lf/2+self.dx/2, xx+self.Lf/2+self.dx/4, self.dx)
        _fiby = np.array([yy])

        _fibz = np.array([zz])
        fibx, fiby = np.meshgrid(_fibx, _fiby)
        fibx = fibx.ravel()
        fiby = fiby.ravel()
        fibz = fiby.ravel()

        # Determine the size of dummy region
        ghost_extent = 5*(self.h0/self.dx)*fdx

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
                     'dwdx','dwdy', 'dwdz', 'Fx', 'Fy', 'Fz', 'arho'):
            fluid.add_property(name)
            channel.add_property(name)
            fiber.add_property(name)
        for name in ('lprev', 'lnext', 'phi0', 'xcenter',
                     'ycenter','rxnext', 'rynext', 'rznext', 'rnext', 'rxprev',
                     'ryprev', 'rzprev', 'rprev'):
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

        # Set the default density.
        fluid.rho[:] = self.options.rho0
        channel.rho[:] = self.options.rho0
        fiber.rho[:] = self.options.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume
        fiber.V[:] = 1./fiber_volume

        # The smoothing lengths are set accorindg to each particles size.
        fluid.h[:] = self.options.fluid_res*self.h0
        channel.h[:] = self.options.fluid_res*self.h0
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
        all = ['fluid', 'channel', 'fiber']
        equations = [
            # The first group computes densities in the fluid phase and corrects
            # the wall's inverse volumes according to their current density. It
            # is applied to all particles including dummies. (real=False)
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=all),
                    #SummationDensity(dest='fiber', sources=all),
                    VolumeFromMassDensity(dest='channel', sources=all),
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
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                    rho0=self.options.rho0, b=1.0),
                    #StateEquation(dest='fiber', sources=None, p0=self.p0,
                    #                rho0=self.options.rho0, b=1.0),
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
                    Friction(dest='fiber', sources=['fiber'], J=self.J,
                                A=self.A, mu=self.options.mu, d=self.options.d,
                                ar=self.options.ar),
                    MomentumEquationPressureGradient(dest='fluid', sources=all,
                                        pb=self.pb, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationPressureGradient(dest='fiber', sources=all,
                                        pb=0.0, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationViscosity(dest='fluid',
                                        sources=['fluid', 'fiber'], nu=self.nu),
                    MomentumEquationViscosity(dest='fiber',
                                       sources=['fluid', 'fiber'], nu=self.nu),
                    SolidWallNoSlipBC(dest='fluid',
                                        sources=['channel',], nu=self.nu),
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
                    EBGVelocityReset(dest='fiber', sources=None),
                ]
            ),
        ]
        return equations

    def _configure(self):
        super(Channel, self)._configure()
        # if there are more than 1 particles involved, elastic equations are
        # iterated in an inner loop.
        if self.options.ar > 1:
            # The second integrator is a simple Euler-Integrator (accurate
            # enough due to very small time steps; very fast) using EBGSteps.
            # EBGSteps are basically the same as EulerSteps, exept for the fact
            # that they work with an intermediate ebg velocity [eu, ev, ew].
            # This velocity does not interfere with the actual velocity, which
            # is neseccery to not disturb the real velocity through artificial
            # damping in this step. The ebg velocity is initialized for each
            # inner loop again and reset in the outer loop.
            self.fiber_integrator = EulerIntegrator(fiber=EBGStep())
            # The type of spline has no influence here. It must be large enough
            # to contain the next particle though.
            kernel = QuinticSpline(dim=self.options.dim)
            equations = [
                        # The first group computes all accelerations based on
                        # tension, bending and damping.
                        Group(
                            equations=[
                                Tension(dest='fiber',
                                        sources=['fiber'],
                                        ea=self.options.E*self.A),
                                Bending(dest='fiber',
                                        sources=['fiber'],
                                        ei=self.options.E*self.I),
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
        # Setting up the default integrator for fiber particles
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
        # This post stage function gets called after each outer loop and starts
        # an inner loop for the fiber iteration.
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
        # This function is just a shorthand for the generation of meshgrids.
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
        # This function plots streamlines and the pressure field. It
        # interpolates the properties from particles using the kernel.

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

        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.viridis
        levels = np.linspace(0, np.max(vmag), 30)

        # velocity contour
        vel = plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=np.max(vmag), vmin=0)
        # streamlines
        stream = plt.streamplot(x*factor,y*factor,u,v, color='k')
        # fiber
        plt.scatter(fx*factor,fy*factor, color='w')

        # set labels
        cbar = plt.colorbar(vel, label='Velocity Magnitude')
        plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.eps')
        plt.savefig(fig, dpi=300)
        print("Streamplot written to %s."% fig)

        # open new plot
        plt.figure()
        #configuring new color map
        cmap = plt.cm.viridis
        levels = np.linspace(-300, 300, 30)

        # pressure contour
        pres = plt.contourf(x*factor,y*factor, p, levels=levels,
                 cmap=cmap, vmax=300, vmin=-300)

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
        output = self.output_files[step_idx]
        data = load(output)

        x = np.array([0])
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        u_exact = (self.options.G * (y - self.Ly/2)
                    - 1/2*self.options.g/self.nu*((y-self.Ly/2)**2-(self.Ly/2)**2))

        plt.figure()
        plt.plot(u, y , '-k')
        plt.plot(u_exact, y, ':k')
        plt.title('Velocity at inlet')
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Position [m]')
        plt.legend(['Simulation', 'Ideal'])
        fig = os.path.join(self.output_dir, 'inlet_velocity.eps')
        plt.savefig(fig, dpi=300)
        print("Inlet velocity plot written to %s."% fig)
        return(fig)

    def _plot_pressure_centerline(self):
        factor = 1000
        x = np.linspace(0,self.Lx,200)
        y = np.array([self.Ly/2])
        x,y = np.meshgrid(x,y)
        N = 10
        p = np.zeros((200,))
        for output in self.output_files[-(1+N):-1]:
            data = load(output)
            interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
            interp.update_particle_arrays(list(data['arrays'].values()))
            p += interp.interpolate('p')/N

        x_fem = np.array([9.60E-05,2.88E-04,4.80E-04,6.72E-04,8.64E-04,0.001056,
                    0.001248,0.00144,0.001632,0.001824,0.002016,0.002208,0.0024,
                    0.002592,0.002784,0.002976,0.003168,0.00336,0.003552,
                    0.003744,0.003936,0.004128,0.00432,0.004512,0.0047,0.0048,
                    0.004909,0.005088,0.00528,0.005472,0.005664,0.005856,
                    0.006048,0.00624,0.006432,0.006624,0.006816,0.007008,
                    0.0072,0.007392,0.007584,0.007776,0.007968,0.00816,
                    0.008352,0.008544,0.008736,0.008928,0.00912,0.009312,
                    0.009504])
        p_fem = np.array([1.58691032,3.218178544,4.371263607,5.586283773,
                    6.816187114,8.082024459,9.393638874,10.7492284,12.13972677,
                    13.57194924,15.05198485,16.76694504,18.4998047,20.25815552,
                    22.45599313,24.90776849,27.84319026,31.28762006,35.51451971,
                    41.18080914,49.93711208,61.37254299,81.9969569,127.4880833,
                    407.5707850,np.nan,-275.5677902,-105.8456749,-61.80262634,
                    -42.04795871,-30.53875332,-24.24458436,-20.20278011,
                    -17.00506646,-14.80382863,-13.43634416,-12.10999473,
                    -11.08772738,-10.21693285,-9.431064134,-8.65649864,
                    -7.839548047,-7.004226958,-6.118488775,-5.212674274,
                    -4.280871816,-3.29811397,-2.290456461,-1.24593802,
                    -0.194508544,1.353145886])
        plt.figure()
        plt.plot(x[0,:]*factor, p, '-k', x_fem*factor, p_fem, '--k')
        plt.legend(['SPH Simulation','FEM Result'])
        plt.xlabel('x [mm]')
        plt.ylabel('p [Pa]')

        pcenter_fig = os.path.join(self.output_dir, 'pressure_centerline.eps')
        plt.savefig(pcenter_fig, dpi=300)
        print("Pressure written to %s."% pcenter_fig)
        return pcenter_fig

    def _plot_history(self):
        factor = 1000
        x_begin = []
        y_begin = []
        x_end = []
        y_end = []
        angle = []
        t = []
        E_kin = []
        E_p = []
        m = []
        volume = []
        rho = []
        Fx = []
        Fy = []
        Fz = []
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)
            fiber = data['arrays']['fiber']
            fluid = data['arrays']['fluid']
            channel = data['arrays']['channel']
            x_begin.append(fiber.x[0])
            y_begin.append(fiber.y[0])
            x_end.append(fiber.x[-1])
            y_end.append(fiber.y[-1])
            dxx = fiber.x[0]-fiber.x[-1]
            dyy = fiber.y[0]-fiber.y[-1]
            angle.append(np.arctan(dxx/(dyy+0.01*self.h0)))
            t.append(data['solver_data']['t'])
            v_fiber = fiber.u**2 + fiber.v**2 + fiber.w**2
            v_fluid = fluid.u**2 + fluid.v**2 + fluid.w**2
            m_fiber = fiber.rho/fiber.V
            m_fluid = fluid.rho/fluid.V
            m_channel = channel.rho/channel.V
            volume.append(np.sum(1/fiber.V)+np.sum(1/fluid.V)+np.sum(1/channel.V))
            rho.append(np.sum(fiber.rho)+np.sum(fluid.rho)+np.sum(channel.rho))
            m.append(np.sum(m_fiber)+np.sum(m_fluid)+np.sum(m_channel))
            E_p.append(np.sum(fiber.p/fiber.V)+np.sum(fluid.p/fluid.V)+np.sum(channel.p/channel.V))
            E_kin.append(0.5*np.dot(m_fiber,v_fiber)
                        +0.5*np.dot(m_fluid,v_fluid))

            idx = np.argwhere(fiber.holdtag==100)
            Fx.append(fiber.Fx[idx][0])
            Fy.append(fiber.Fy[idx][0])
            Fz.append(fiber.Fz[idx][0])

        plt.figure()
        plt.plot(x_begin, y_begin, '-ok', markersize=5)
        plt.plot(x_end, y_end, '-*k', markersize=5)
        plt.axis('equal')
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
        angle_jeffery = (angle_jeffery+np.pi/2.0)%np.pi-np.pi/2.0

        plt.figure()
        plt.plot(t, angle, '*k')
        plt.plot(t, angle_jeffery, '-k')
        plt.xlabel('t [s]')
        plt.ylabel('Angle [rad]')
        plt.legend(['Simulation', 'Jeffery'])
        plt.title("ar=%g"%self.options.ar)
        angfig = os.path.join(self.output_dir, 'angleplot.eps')
        plt.savefig(angfig, dpi=300)
        print("Angleplot written to %s."% angfig)
        csv_file = os.path.join(self.output_dir, 'angle.csv')
        angle_jeffery = np.reshape(angle_jeffery,(angle_jeffery.size,))
        np.savetxt(csv_file, (t, angle, angle_jeffery), delimiter=',')

        plt.figure()
        plt.plot(t, E_p, '-k', t, E_kin, ':k')
        plt.xlabel('t [s]')
        plt.ylabel('Energy')
        plt.title("Energy")
        plt.legend(['Pressure', 'Kinetic Energy'])
        engfig = os.path.join(self.output_dir, 'energyplot.eps')
        plt.savefig(engfig, dpi=300)
        print("Energyplot written to %s."% engfig)

        plt.figure()
        plt.plot(t, np.array(m)/m[0], '-k',
                 t, np.array(volume)/volume[0], '--k',
                 t, np.array(rho)/rho[0], ':k')
        plt.xlabel('t [s]')
        plt.ylabel('Relative value')
        plt.legend(['Mass', 'Volume', 'Density'])
        mfig = os.path.join(self.output_dir, 'massplot.eps')
        plt.savefig(mfig, dpi=300)
        print("Mass plot written to %s."% mfig)

        t_fem = np.array([0,50,100,150,200,250,300,350,400,450,500,550,600,650,
                    700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,
                    1350,1400,1450,1500])
        Fp_fem = np.array([6.95E-04,0.030299595,0.04868658,0.060802176,
                    0.071992602,0.078325888,0.084659173,0.090492935,0.093704197,
                    0.096915459,0.099866885,0.101454168,0.103041451,0.104498144,
                    0.105269236,0.106040328,0.106747500,0.107119093,0.107490685,
                    0.107831369,0.108009777,0.108188185,0.108351729,0.108437236,
                    0.108522743,0.108601121,0.108642069,0.108683017,0.10872055,
                    0.108740151,0.108759753])
        F_fem = np.array([7.48E-04,0.060707871,0.097855135,0.122251195,
                    0.144776438,0.157479887,0.170183337,0.181882322,0.188307871,
                    0.194733419,0.20063835,0.203810039,0.206981728,0.209892324,
                    0.211432182,0.21297204,0.214384218,0.215126076,0.215867934,
                    0.216548076,0.216904208,0.21726034,0.217586799,0.217757474,
                    0.21792815,0.218084595,0.218166326,0.218248057,0.218322972,
                    0.218362095,0.218401219])
        t_fem = t_fem/1E5
        t = np.array(t)/self.options.scale_factor*1000

        if self.options.ar == 1:
            plt.figure()
            plt.plot(t, Fx, '-k', t_fem, F_fem, '--k', t_fem, F_fem-Fp_fem, ':k')
            plt.xlabel('t [ms]')
            plt.ylabel('Force [N/m]')
            plt.title("Reaction Force")
            plt.legend(['SPH Simulation', 'FEM total force', 'FEM viscous force'])
            forcefig = os.path.join(self.output_dir, 'forceplot.eps')
            plt.savefig(forcefig, dpi=300)
            print("Reaction Force plot written to %s."% forcefig)
            return [orbfig, angfig, engfig, forcefig]
        else:
            return [orbfig, angfig, engfig]

    def _send_notification(self, info_fname, attachments=None):
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
        [orbitplot, angleplot, energyplot, force] = self._plot_history()
        inlet = self._plot_inlet_velocity()
        if self.options.mail:
            self._send_notification(info_fname, [streamlines, orbitplot, angleplot])

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
