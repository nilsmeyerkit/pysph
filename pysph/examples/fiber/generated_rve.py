"""
################################################################################
Mini RVE
################################################################################
"""
# general imports
import os
import random
import numpy as np
from scipy.integrate import odeint

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain,
                              get_particle_array_beadchain_fiber)

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme


class RVE(Application):
    """Generation of a mini RVE and evaluation of its fiber orientation
    tensor."""
    def create_scheme(self):
        """There is no scheme used in this application and equations are set up
        manually."""
        return BeadChainScheme(['fluid'], ['channel'], ['fibers'], dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.00001, help="Fiber diameter"
        )
        group.add_argument(
            "--ar", action="store", type=int, dest="ar",
            default=25, help="Aspect ratio of fiber"
        )
        group.add_argument(
            "--rho", action="store", type=float, dest="rho0",
            default=1000, help="Rest density"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=250, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=2.5E5, help="Young's modulus"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=1.0, help="Shear rate"
        )
        group.add_argument(
            "--g", action="store", type=float, dest="g",
            default=0, help="Body force in x-direction"
        )
        group.add_argument(
            "--D", action="store", type=float, dest="D",
            default=None, help="Damping coefficient for artificial damping"
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
            "--massscale", action="store", type=float, dest="scale_factor",
            default=None, help="Factor of mass scaling"
        )
        group.add_argument(
            "--folgartucker", action="store_true", dest="folgartucker",
            default=False, help="Decides if Folgar Tucker solution is plotted."
        )
        group.add_argument(
            "--k", action="store", type=float, dest="k",
            default=0.0, help="Friction coefficient between fibers."
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

        # The fiber length is the aspect ratio times fiber diameter
        self.L = 4E-4

        # Computation of a scale factor in a way that dt_cfl exactly matches
        # dt_viscous.
        a = self.h0*0.125*11/0.4
        # nu_needed = a*self.options.G*self.L/2
        nu_needed = (a*self.options.G*self.L/4
                     + np.sqrt(a/8*self.options.g*self.L**2
                               + (a/2)**2/4*self.options.G**2*self.L**2))

        # If there is no other scale scale factor provided, use automatically
        # computed factor.
        auto_scale_factor = self.options.mu/(nu_needed*self.options.rho0)
        self.scale_factor = self.options.scale_factor or auto_scale_factor

        # The density can be scaled using the mass scaling factor. To account
        # for proper external forces, gravity is scaled just the other way.
        self.rho0 = self.options.rho0*self.scale_factor
        self.options.g = self.options.g/self.scale_factor

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled (!) density.
        self.nu = self.options.mu/self.rho0

        # empirical determination for the damping, which is just enough
        self.D = self.options.D or 0.2*self.options.ar

        # mechanical properties
        R = self.dx/2
        self.A = np.pi*R**2
        self.I = np.pi*R**4/4.0
        mass = 3.0*self.rho0*self.dx*self.A
        self.J = 1.0/4.0*mass*R**2 + 1.0/12*mass*(3.0*self.dx)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.Vmax = (self.options.G*self.L/2.0
                     + self.options.g/(2.0*self.nu)*self.L**2/4.0)
        self.c0 = 10.0*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required.
        if self.options.postonly:
            self.t = 0.0
        else:
            self.t = 1
        print("Simulated time is %g s" % self.t)

    def configure_scheme(self):
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=self.A, I=self.I,
            J=self.J, E=self.options.E, D=self.D, gx=self.options.g,
            k=self.options.k)
        # self.scheme.configure(fibers=[])
        self.scheme.configure_solver(tf=self.t, vtk=self.options.vtk, N=100)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.dx
        dx2 = fdx/2

        # Computation of each particles initial volume.
        volume = fdx**3

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume

        # Creating grid points for particles
        _x = np.arange(dx2, self.L, fdx)
        _y = np.arange(dx2, self.L, fdx)
        _z = np.arange(dx2, self.L, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        positions = tuple()
        fractags = tuple()
        fiberindex = (0,)

        first_points = np.loadtxt('first_points.txt', delimiter=',')
        second_points = np.loadtxt('second_points.txt', delimiter=',')

        self.n = len(first_points)

        for p1, p2 in zip(first_points, second_points):
            pos = self.get_fiber_positions(p1, p2, self.options.ar)
            ftags = (len(pos)-1)*[0]+[1]
            N = max(fiberindex)
            findex = range(N + 1, N + len(pos) + 1)
            for i, p in enumerate(pos):
                if p[2] < 0:
                    pos[i] = (p[0], p[1], p[2]+self.L)
                if p[2] > self.L:
                    pos[i] = (p[0], p[1], p[2]-self.L)
                if (abs(pos[i-1][2]-pos[i][2]) > self.L/2.0):
                    ftags[i-1] = 1
            positions = positions + tuple(pos)
            fractags = fractags + tuple(ftags)
            fiberindex = fiberindex + tuple(findex)

        fibx = [position[0] for position in positions]
        fiby = [position[1] for position in positions]
        fibz = [position[2] for position in positions]

        print("Created %d fibers." % self.n)

        # Determine the size of dummy region
        ghost_extent = 3*fdx

        # Create the channel particles at the top
        _z = np.arange(self.L+dx2, self.L+ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _z = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays (and for 3D cas also right and
        # left arrays)
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))
        cz = np.concatenate((tz, bz))

        # Finally create all particle arrays.
        channel = get_particle_array_beadchain(
            name='channel', x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fluid = get_particle_array_beadchain(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        rand_idx = random.sample(range(len(fx)), len(fibx))
        fluid.remove_particles(rand_idx)

        if self.n > 0:
            fibers = get_particle_array_beadchain_fiber(
                name='fibers', x=fibx, y=fiby, z=fibz, m=mass, rho=self.rho0,
                h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi, V=V,
                phifrac=2.0, fidx=fiberindex[:-1], fractag=fractags)

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.z[:]-self.L/2)
        fibers.u[:] = self.options.G*(fibers.z[:]-self.L/2)
        channel.u[:] = self.options.G*(channel.z[:]-self.L/2)

        # Print number of particles.
        print("Generated RVE : nfluid = %d, nwalls = %d, nfiber = %d" % (
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fibers.get_number_of_particles()))

        if self.n > 0:
            return [fluid, channel, fibers]
        else:
            return [fluid, channel]

    def create_domain(self):
        """The channel has periodic boundary conditions."""
        return DomainManager(xmin=0, xmax=self.L, ymin=0, ymax=self.L,
                             periodic_in_x=True, periodic_in_y=True)

    def create_tools(self):
        """Set up fiber integrator."""
        if self.n < 1:
            return []
        else:
            return [FiberIntegrator(self.particles, self.scheme, self.domain,
                                    parallel=True)]

    def get_meshgrid(self, xx, yy, zz):
        """This function is a shorthand for the generation of meshgrids."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def get_fiber_positions(self, p1, p2, N):
        """ Create particles along a straight line.

        Args
        ----
            p1: First point of line

            p2: Second point of line

            N: Number of particles to be created along that line

        Returns
        -------
            list of point positions
        """

        return zip(*[np.linspace(p1[i], p2[i], N) for i in range(len(p1))])

    def post_process(self, info_fname):
        if len(self.output_files) == 0:
            return

        # empty list for time
        t = []

        # empty lists for fiber orientation tensors
        A = []

        # empty list for viscosity
        eta = []

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)

            # extracting time
            t.append(data['solver_data']['t'])

            channel = data['arrays']['channel']
            Fw = np.sqrt(channel.Fwx[:]**2+channel.Fwy[:]**2+channel.Fwz[:]**2)
            surface = self.L**2
            tau = np.sum(Fw)/(2*surface)
            eta.append(tau/self.options.G)

            if self.n > 0:
                # extrating all arrays.
                directions = []
                fiber = data['arrays']['fibers']
                starts = [i*(self.options.ar-1) for i in range(0, self.n)]
                ends = [i*(self.options.ar-1)-1 for i in range(1, self.n + 1)]
                for start, end in zip(starts, ends):
                    px = np.mean(fiber.rxnext[start:end])
                    py = np.mean(fiber.rynext[start:end])
                    pz = np.mean(fiber.rznext[start:end])

                    n = np.array([px, py, pz])
                    p = n/np.linalg.norm(n)
                    directions.append(p)

                N = len(directions)
                a = np.zeros([3, 3])
                for p in directions:
                    for i in range(3):
                        for j in range(3):
                            a[i, j] += 1.0/N*(p[i]*p[j])
                A.append(a.ravel())

        csv_file = os.path.join(self.output_dir, 'N.csv')
        data = np.hstack((np.matrix(t).T, np.vstack(A)))
        np.savetxt(csv_file, data, delimiter=',')


if __name__ == '__main__':
    app = RVE()
    app.run()
    app.post_process(app.info_filename)
