"""2D shearflow with a single fiber - use --ar to set aspect ratio"""
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
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme
from pysph.sph.integrator import EulerIntegrator, EPECIntegrator
from pysph.sph.integrator_step import EulerStep, EBGStep
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (SummationDensity, VolumeSummation,
    StateEquation, MomentumEquationPressureGradient, ContinuityEquation,
    MomentumEquationViscosity, MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity,
    VolumeFromMassDensity)
from pysph.sph.fiber.utils import (Damping, HoldPoints, VelocityGradient,
    Contact, ComputeDistance, DummyDrag)
from pysph.sph.fiber.beadchain import (Tension, Bending, EBGVelocityReset,
    Friction, ArtificialDamping)


# Jeffrey's equivalent aspect ratio (coarse approximation)
#   H. L. Goldsmith and S. G. Mason
#   CHAPTER 2 - THE MICRORHEOLOGY OF DISPERSIONS A2 - EIRICH, pp. 85–250.
#   Academic Press, 1967.
def get_equivalent_aspect_ratio(aspect_ratio):
    return -0.0017*aspect_ratio**2+0.742*aspect_ratio

# Jeffery's Equation for planar rotation of a rigid (theta=0)
def jeffery_ode(phi, t, ar, G):
    lbd = (ar**2-1.0)/(ar**2+1.0)
    return 0.5*G*(1.0+lbd*np.cos(2.0*phi))

class Channel(Application):
    def create_scheme(self):
        """There is no scheme used in this application and equations are set up
        manually."""
        return None

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
            "--dim", action="store", type=int, dest="dim",
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
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=1.0, help="Number of half rotations."
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
        self.rho0 = self.options.rho0*self.options.scale_factor
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
        self.x_fiber = 0.5*self.Lx
        self.y_fiber = 0.5*self.Ly
        self.z_fiber = 0.5*self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled (!) density.
        self.nu = self.options.mu/self.rho0

        # For 2 dimensions surface, mass and moments have a different coputation
        # than for 3 dimensions.
        if self.options.dim == 2:
            self.A = self.dx
            self.I = self.dx**3/12
            mass = 3*self.rho0*self.dx*self.A
            self.J = 1/12*mass*(self.dx**2 + (3*self.dx)**2)
        else:
            R = self.dx/2
            self.A = np.pi*R**2
            self.I = np.pi*R**4/4.0
            mass = 3*self.rho0*self.dx*self.A
            self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.dx)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.Vmax = (self.options.G*self.Ly/2
                    + self.options.g/(2*self.nu)*self.Ly**2/4)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.rho0

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
        if self.options.postonly:
            self.t = 0
        else:
            if self.options.G > 0.1:
                l = (self.options.ar+1.0/self.options.ar)
                self.t = self.options.rot*np.pi*l/self.options.G
            else:
                self.t = 1.5E-5*self.options.scale_factor
        print("Simulated time is %g s"%self.t)

        dt_tension = 0.5 * self.dx * np.sqrt(self.rho0/self.options.E)
        dt_bending = 0.5 * self.dx**2 * np.sqrt(self.rho0*self.A/(self.options.E*2*self.I))
        self.dt = min(dt_tension, dt_bending)
        print("Time step: %f"%self.dt)


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

        # Generating fiber particle grid. Uncomment proper section for
        # horizontal or vertical alignment respectivley.
        xx = self.x_fiber
        yy = self.y_fiber
        zz = self.z_fiber

        # vertical fiber
        _fibx = np.array([xx])
        _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/4, self.dx)

        # horizontal fiber
        # _fibx = np.arange(xx-self.Lf/2+self.dx/2, xx+self.Lf/2+self.dx/4, self.dx)
        # _fiby = np.array([yy])

        _fibz = np.array([zz])
        fibx,fiby,fibz = self.get_meshgrid(_fibx, _fiby, _fibz)


        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        if self.options.dim == 2:
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby)
        else:
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby, z=fibz)

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles()==self.options.ar)

        # Add requisite variables needed for this formulation
        for name in ('V', 'wf','uf','vf','wg','wij','vg','ug', 'phifrac',
                     'awhat', 'avhat','auhat', 'vhat', 'what', 'uhat', 'vmag2',
                     'arho', 'phi0', 'fractag', 'rho0','holdtag', 'eu', 'ev',
                     'ew', 'dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz',
                     'dwdx','dwdy', 'dwdz', 'Fx', 'Fy', 'Fz', 'arho', 'ex',
                     'ey', 'ez','lprev', 'lnext', 'phi0', 'xcenter', 'ycenter',
                     'rxnext', 'rynext', 'rznext', 'rnext', 'rxprev',
                      'ryprev', 'rzprev', 'rprev', 'fidx'):
            fiber.add_property(name)

        # set the output property arrays
        fiber.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'p',
                        'pid', 'holdtag', 'gid','ug', 'vg', 'wg', 'V', 'Fx',
                        'Fy', 'Fz', 'eu', 'ev', 'ew'])

        # Computation of each particles initial volume.
        volume = fdx**self.options.dim
        fiber_volume = self.dx**self.options.dim

        # Mass is set to get the reference density of rho0.
        fiber.m[:] = fiber_volume * self.rho0

        # Set initial distances and angles. This represents a straight
        # unstreched fiber in rest state. It fractures, if a segment of length
        # 2 dx is bend more than 11.5°.
        fiber.lprev[:] = self.dx
        fiber.lnext[:] = self.dx
        fiber.phi0[:] = np.pi
        fiber.phifrac[:] = 2.0

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 0
        if self.options.holdcenter:
            idx = int(np.floor(self.options.ar/2))
            fiber.holdtag[idx] = 100

        # assign unique ID (within fiber) to each fiber particle.
        fiber.fidx[:] = range(0,fiber.get_number_of_particles())

        # Set the default density.
        fiber.rho[:] = self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        fiber.V[:] = 1./fiber_volume

        # The smoothing lengths are set accorindg to each particles size.
        fiber.h[:] = self.h0

        # Setting the initial velocities for a shear flow.
        fiber.u[:] = self.options.G*(fiber.y[:]-self.Ly/2)
        fiber.dudy[:] = self.options.G

        # Return the particle list.
        return [fiber]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    ComputeDistance(dest='fiber', sources=['fiber'])
                ],
            ),
            Group(
                equations=[
                    DummyDrag(dest='fiber', sources=None, d=self.options.d,
                        mu=self.options.mu, dudy=self.options.G,
                        y0=0.5*self.Ly, scale=self.options.scale_factor),
                    Tension(dest='fiber', sources=None,
                        ea=self.options.E*self.A),
                    Bending(dest='fiber', sources=None,
                        ei=self.options.E*self.I),
                    Friction(dest='fiber', sources=None, J=self.J, A=self.A,
                        mu=self.options.mu, d=self.options.d)
                ],
            ),
        ]
        return equations

    def create_solver(self):
        # Setting up the default integrator for fiber particles
        kernel = QuinticSpline(dim=self.options.dim)
        integrator = EulerIntegrator(fiber=EulerStep())
        solver = Solver(kernel=kernel, dim=self.options.dim,
                        integrator=integrator, dt=self.dt,
                        tf=self.t, N=100, vtk=True)
        return solver

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
        M = 0

        # empty list for roation periods
        T = []
        t0 = 0

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)

            # extracting time
            t.append(data['solver_data']['t'])

            # extrating all arrays.
            fiber = data['arrays']['fiber']

            # extrating end-points
            x_begin.append(fiber.x[0])
            y_begin.append(fiber.y[0])
            x_end.append(fiber.x[-1])
            y_end.append(fiber.y[-1])

            # computation of orientation angle
            dxx = fiber.x[0]-fiber.x[-1]
            dyy = fiber.y[0]-fiber.y[-1]
            a = np.arctan(dxx/(dyy+0.01*self.h0)) + N*np.pi
            if len(angle) > 0 and a - angle[-1] > 3:
                N -= 1
                a -= np.pi
            elif len(angle) > 0 and a-angle[-1] < -3:
                N += 1
                a += np.pi

            # count rotations
            if a-M*np.pi > np.pi:
                T.append(t[-1]-t0)
                t0 = t[-1]
                M += 1
            angle.append(a)

        # evaluate roation statistics
        print(T)
        T_mean = np.mean(T)
        T_std  = np.std(T)
        are = get_equivalent_aspect_ratio(self.options.ar)
        l = (are+1.0/are)
        T_equiv = np.pi*l/self.options.G
        l = (self.options.ar+1.0/self.options.ar)
        T_jef = np.pi*l/self.options.G
        print("Rotational statistics for %d half rotations:"%self.options.rot)
        print("*Mean: %f"%T_mean)
        print("*Standard Deviation: %f"%T_std)
        print("*Jeffery: %f"%T_jef)
        print("*Jeffery(equivalent): %f"%T_equiv)

        # open new plot
        plt.figure()

        # plot end points
        plt.plot(x_begin, y_begin, '-ok', markersize=3)
        plt.plot(x_end, y_end, '-xk', markersize=3)

        # set equally scaled axis to not distort the orbit
        plt.axis('equal')
        plt.title('Orbitplot')

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
                                args=(self.options.ar,self.options.G))
        angle_jeffery_equiv = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(are,self.options.G))

        # constraint between -pi/2 and pi/2
        #angle_jeffery = (angle_jeffery+np.pi/2.0)%np.pi-np.pi/2.0

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t, angle, 'ok', markersize=3)
        plt.plot(t, angle_jeffery_equiv, '--k')
        plt.plot(t, angle_jeffery, '-k')

        # labels
        plt.xlabel('t [s]')
        plt.ylabel('Angle [rad]')
        plt.legend(['Bead Chain', 'Jeffery (equiv.)', 'Jeffery'])
        plt.title("ar=%g"%self.options.ar)

        # save figure
        angfig = os.path.join(self.output_dir, 'angleplot.eps')
        plt.savefig(angfig, dpi=300)
        print("Angleplot written to %s."% angfig)

        # save angles as *.csv file
        csv_file = os.path.join(self.output_dir, 'angle.csv')
        angle_jeffery = np.reshape(angle_jeffery,(angle_jeffery.size,))
        np.savetxt(csv_file, (t, angle, angle_jeffery), delimiter=',')

        return [orbfig, angfig]

    def post_process(self, info_fname):
        history = self._plot_history()

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
