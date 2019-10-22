"""
################################################################################
3D Flow around a single fixed fiber particle.
################################################################################
"""
import os

# numpy and scipy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fluid,
                              get_particle_array_beadchain_solid,
                              get_particle_array_beadchain_fiber)

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files, FloatPBar
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme
from pysph.base.kernels import QuinticSpline, CubicSpline


class SingleParticle(Application):
    def create_scheme(self):
        """The BeadChainScheme is used for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0001, help="Fiber diameter"
        )
        group.add_argument(
            "--rho", action="store", type=float, dest="rho0",
            default=1000.0, help="Rest density"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=1.0, help="Absolute viscosity"
        )
        group.add_argument(
            "--Re", action="store", type=float, dest="Re",
            default=0.5, help="Velocity at cube borders"
        )
        group.add_argument(
            "--size", action="store", type=int, dest="size",
            default=20, help="Cube size (multiples of fiber diameter)"
        )
        group.add_argument(
            "--t", action="store", type=float, dest="t",
            default=0.00005, help="Cube size (multiples of fiber diameter)"
        )
        group.add_argument(
            "--vtk", action="store_true", dest='vtk',
            default=False, help="Enable vtk-output during solving."
        )
        group.add_argument(
            "--postonly", action="store_true", dest="postonly",
            default=False, help="Set time to zero and postprocess only."
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
        self.Lf = self.dx

        # If a specific width is set, use this as multiple of dx to determine
        # the channel width. Otherwise use the fiber aspect ratio.
        self.L = self.options.size*self.dx

        # The density
        self.rho0 = self.options.rho0

        # The position of the fiber's center is set to the center of the cube.
        self.x_fiber = np.array([0.5*self.L])
        self.y_fiber = np.array([0.5*self.L])
        self.z_fiber = np.array([0.5*self.L])

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled density.
        self.nu = self.options.mu/self.rho0

        # applied velocity
        self.v = self.options.Re*self.nu/self.dx

        # damping from empirical guess
        self.D = 1

        # Mass properties of fiber
        R = self.dx/2
        self.A = np.pi*R**2
        self.Ip = np.pi*R**4/4.0
        mass = 3*self.rho0*self.dx*self.A
        self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.dx)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.c0 = 10*self.v
        self.p0 = self.c0**2*self.rho0
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required.
        self.t = self.options.t
        print("Simulated time is %g s" % self.t)

    def configure_scheme(self):
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=self.A,
            Ip=self.Ip, J=self.J, E=1.0, D=self.D)

        kernel = QuinticSpline(dim=3)
        self.scheme.configure_solver(
            tf=self.t, vtk=self.options.vtk, N=20, kernel=kernel)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.dx
        dx2 = fdx/2

        # Creating grid points for particles
        _x = np.arange(dx2, self.L, fdx)
        _y = np.arange(dx2, self.L, fdx)
        _z = np.arange(dx2, self.L, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # add some random noise
        noise = fdx/20
        fx = fx + np.random.normal(0, noise, fx.shape)
        fy = fy + np.random.normal(0, noise, fy.shape)

        # fiber
        fibx, fiby, fibz = self.get_meshgrid(
            self.x_fiber, self.y_fiber, self.z_fiber)

        # Determine the size of dummy region
        ghost_extent = 3*fdx

        # Create the channel particles at the top
        _y = np.arange(self.L+dx2, self.L+dx2+ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the right
        _z = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        _y = np.arange(dx2-ghost_extent, self.L+ghost_extent, fdx)
        rx, ry, rz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the left
        _z = np.arange(self.L+dx2, self.L+dx2+ghost_extent, fdx)
        _y = np.arange(dx2-ghost_extent, self.L+ghost_extent, fdx)
        lx, ly, lz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays
        cx = np.concatenate((tx, bx, rx, lx))
        cy = np.concatenate((ty, by, ry, ly))
        cz = np.concatenate((tz, bz, rz, lz))

        # Computation of each particles initial volume.
        volume = fdx**3

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume

        # Finally create all particle arrays.
        channel = get_particle_array_beadchain_solid(
            name='channel', x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fluid = get_particle_array_beadchain_fluid(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fibx, y=fiby, z=fibz, m=mass, rho=self.rho0,
            h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
            phifrac=2.0, fidx=[0], V=V)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d, nfiber = %d" % (
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fiber.get_number_of_particles()))

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 100

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.v
        channel.u[:] = self.v

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.L, periodic_in_x=True)

    def create_tools(self):
        return [FiberIntegrator(self.particles, self.scheme, self.domain,
                                innerloop=False, updates=False)]

    def get_meshgrid(self, xx, yy, zz):
        """This function is just a shorthand for generation of meshgrids."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def _plot_streamlines(self):
        """This function plots streamlines and the pressure field.

        It interpolates the properties from particles using the kernel.
        """
        from pysph.tools.interpolator import Interpolator
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib import rc
        rc('font', **{'family': 'serif',
                      'serif': ['Computer Modern'],
                      'size': 18})
        rc('text', usetex=True)

        # Interpolation grid
        X = np.linspace(0, self.L, 100)
        Y = np.linspace(0, self.L, 110)
        x, y = np.meshgrid(X, Y)

        # Extract positions of fiber particles from last step to plot them
        # on top of velocities.
        last_output = self.output_files[-1]
        data = load(last_output)
        fiber = data['arrays']['fiber']
        fx = fiber.x
        fy = fiber.y

        # Interpolation (precompiled) of velocities
        interp = Interpolator(list(data['arrays'].values()),
                              x=x, y=y, z=self.L/2.0)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        v = interp.interpolate('v')
        vmag = np.sqrt(u**2 + v**2)/self.v
        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.Reds
        levels = np.linspace(0.0, 1.1, 30)

        # velocity contour (ungly solution against white lines:
        # repeat plots....)
        plt.contourf(x, y, vmag, levels=levels,
                     cmap=cmap, vmin=0.0, vmax=1.1)
        plt.contourf(x, y, vmag, levels=levels,
                     cmap=cmap, vmin=0.0, vmax=1.1)
        vel = plt.contourf(x, y, vmag, levels=levels,
                           cmap=cmap, vmin=0.0, vmax=1.1)
        # streamlines
        y_start = np.linspace(0.0, self.L, 20)
        x_start = np.zeros_like(y_start)
        start_points = np.array(list(zip(x_start, y_start)))
        plt.streamplot(X, Y, u, v,
                       start_points=start_points,
                       color='k',
                       density=100,
                       arrowstyle='-',
                       linewidth=1.0)
        # fiber
        plt.scatter(fx, fy, color='w')

        # set labels
        cbar = plt.colorbar(vel, shrink=0.5)
        cbar.set_label('Velocity $v/v_0$', labelpad=20.0)
        plt.tight_layout()

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Streamplot written to %s." % fig)

    def _plot_history(self):
        """This function create all plots employing a iteration over all time
        steps. """

        # empty list for time
        t = []

        # empty list for reaction forces
        Fx = []
        Fy = []
        Fz = []

        # reference solution
        R = (3/(4*np.pi))**(1.0/3.0)*self.dx
        F = 6.0*np.pi*self.options.mu*R*self.v

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        print("Evaluating Results.")
        bar = FloatPBar(0, len(output_files), show=True)
        for i, fname in enumerate(output_files):
            data = load(fname)
            bar.update(i)

            # extracting time
            t.append(data['solver_data']['t'])

            # extrating all arrays.
            fiber = data['arrays']['fiber']

            # extract reaction forces at hold particles
            idx = np.argwhere(fiber.holdtag == 100)
            Fx.append(fiber.Fx[idx][0]/F)
            Fy.append(fiber.Fy[idx][0]/F)
            Fz.append(fiber.Fz[idx][0]/F)

        bar.finish()

        file = os.path.join(self.output_dir, 'force.csv')

        np.savetxt(file, np.transpose([t, Fx, Fy, Fz]), delimiter=',')

    def post_process(self, info_fname):
        if len(self.output_files) == 0:
            return

        self._plot_history()
        self._plot_streamlines()


if __name__ == '__main__':
        app = SingleParticle()
        app.run()
        app.post_process(app.info_filename)
