"""Example for two-way interaction with fiber.

################################################################################
2D fluid field around fiber - fiber is interpreted to be perpendicular to field
################################################################################
"""
# general imports
import os

# matplotlib (set up for server use)
# matplotlib (set up for server use)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
rc('text', usetex=True)

# numpy and scipy
import numpy as np
from scipy.interpolate import griddata

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain,
                              get_particle_array_beadchain_fiber)

from pysph.tools.interpolator import Interpolator

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files, FloatPBar
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme

from pysph.base.kernels import QuinticSpline, CubicSpline


class Channel(Application):
    """2D fluid field around fiber."""

    def create_scheme(self):
        """The BeadChainScheme is used for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=2)

    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0001, help="Fiber diameter"
        )
        group.add_argument(
            "--rho", action="store", type=float, dest="rho0",
            default=1000, help="Rest density"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=63, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=2.5E9, help="Young's modulus"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=0.0, help="Shear rate"
        )
        group.add_argument(
            "--g", action="store", type=float, dest="g",
            default=10.0, help="Body force in x-direction"
        )
        group.add_argument(
            "--D", action="store", type=float, dest="D",
            default=None, help="Damping coefficient for artificial damping"
        )
        group.add_argument(
            "--width", action="store", type=int, dest="width",
            default=20, help="Channel width (multiples of fiber diameter)"
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
            default=0.5E6, help="Factor of mass scaling"
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

        # If a specific width is set, use this as multiple of dx to determine
        # the channel width. Otherwise use the fiber aspect ratio.
        multiples = self.options.width
        self.Ly = multiples * self.dx + 2 * int(0.1 * multiples) * self.dx

        # The density can be scaled using the mass scaling factor. To account
        # for proper external forces, gravity is scaled just the other way.
        self.scale_factor = self.options.scale_factor
        self.rho0 = self.options.rho0 * self.scale_factor
        self.options.g = self.options.g / self.scale_factor

        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.0 * self.Ly + self.dx

        # The position of the fiber's center is set to the center of the
        # channel.
        self.x_fiber = 0.5 * self.Lx
        self.y_fiber = 0.5 * self.Ly
        self.z_fiber = 0.5 * self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled density.
        self.nu = self.options.mu / self.rho0

        # damping from empirical guess
        self.D = self.options.D or 0.001 * self.scale_factor

        # For 2 dimensions surface, mass and moments have a different
        # coputation than for 3 dimensions.
        self.A = self.dx
        self.Ip = self.dx**3 / 12
        mass = 3 * self.rho0 * self.dx * self.A
        self.J = 1 / 12 * mass * (self.dx**2 + (3 * self.dx)**2)

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.Vmax = (self.options.G * self.Ly / 2 +
                     self.options.g / (2 * self.nu) * self.Ly**2 / 4)
        self.c0 = 10 * self.Vmax
        print("Sound speed is %g s" % self.c0)
        self.p0 = self.c0**2 * self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required. For a
        # shear flow, it is set to the time for a full period of rotation
        # according to Jeffery's equation. For a Poiseuille flow, it is set to
        # the time to reach steady state for width = 20 and g = 10 to match
        # the FEM result in COMSOL for a single cylinder.
        if self.options.postonly:
            self.t = 0
        else:
            self.t = 5E-5 * self.scale_factor
        print("Simulated time is %g s" % self.t)

    def configure_scheme(self):
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu, p0=self.p0, pb=self.pb,
            h0=self.h0, dx=self.dx, A=self.A, Ip=self.Ip, J=self.J,
            E=self.options.E, D=self.D, dim=2, gx=self.options.g,
            viscous_fiber=True)
        # Return the particle list.
        kernel = QuinticSpline(dim=2)
        # kernel = CubicSpline(dim=2)
        self.scheme.configure_solver(
            kernel=kernel, tf=self.t, vtk=self.options.vtk, N=200)
        # self.scheme.configure_solver(
        #   tf=self.t, pfreq=1, vtk=self.options.vtk)

    def create_particles(self):
        """Three particle arrays are created.

        A fluid, representing the polymer matrix, a fiber with additional
        properties and a channel of dummyparticles.
        """
        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.options.fluid_res * self.dx
        dx2 = fdx / 2

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y)

        # add some random noise
        # noise = dx2
        # fx = fx + np.random.normal(0,noise, fx.shape)
        # fy = fy + np.random.normal(0,noise, fy.shape)

        # Remove particles at fiber position. Uncomment proper section for
        # horizontal or vertical alignment respectivley.
        indices = []
        dist = 100000
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber

            if (fx[i] < xx + self.dx / 2 and fx[i] > xx - self.dx / 2 and
                fy[i] < yy + self.dx / 2 and fy[i] > yy - self.dx / 2 and
                    fz[i] < zz + self.dx / 2 and fz[i] > zz - self.dx / 2):
                indices.append(i)
            if ((fx[i] - xx)**2 + (fy[i] - yy)**2 + (fz[i] - zz)**2) < dist:
                min_dist_idx = i

        if len(indices) == 0:
            indices.append(min_dist_idx)

        # fiber
        _fibx = np.array([xx])
        _fiby = np.array([yy])
        fibx, fiby, fibz = self.get_meshgrid(_fibx, _fiby)

        # Determine the size of dummy region
        ghost_extent = 3 * fdx / self.options.fluid_res

        # Create the channel particles at the top
        _y = np.arange(self.Ly + dx2, self.Ly + dx2 + ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2 - ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y)

        # Concatenate the top and bottom arrays (and for 3D cas also right and
        # left arrays)
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))

        # Computation of each particles initial volume.
        volume = fdx**2
        fiber_volume = self.dx**2

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0
        fiber_mass = fiber_volume * self.rho0

        # assign unique ID (within fiber) to each fiber particle.
        fidx = np.array([0])

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1. / volume
        fiber_V = 1. / fiber_volume

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        channel = get_particle_array_beadchain(
            name='channel', x=cx, y=cy, m=mass, rho=self.rho0, h=self.h0, V=V)
        fluid = get_particle_array_beadchain(
            name='fluid', x=fx, y=fy, m=mass, rho=self.rho0, h=self.h0, V=V)
        fluid.remove_particles(indices)
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fibx, y=fiby, m=fiber_mass, rho=self.rho0,
            h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi, phifrac=2.0,
            fidx=fidx, V=fiber_V)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d, nfiber = %d" % (
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fiber.get_number_of_particles()))

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles() == 1)

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 0
        fiber.holdtag[0] = 100

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G * (fluid.y[:] - self.Ly / 2)
        fiber.u[:] = self.options.G * (fiber.y[:] - self.Ly / 2)
        channel.u[:] = self.options.G * (channel.y[:] - self.Ly / 2)

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_tools(self):
        return [FiberIntegrator(self.particles, self.scheme, self.domain,
                                innerloop=False, updates=False)]

    def get_meshgrid(self, xx, yy):
        """This function is just a shorthand for the generation of grids."""
        x, y = np.meshgrid(xx, yy)
        x = x.ravel()
        y = y.ravel()
        z = self.z_fiber * np.ones(np.shape(y))
        return [x, y, z]

    def _plot_streamlines(self):
        """This function plots streamlines and the pressure field.

        It interpolates the properties from particles using the kernel.
        """
        # lenght factor m --> mm
        factor = 1000

        # Interpolation grid
        X = np.linspace(0, self.Lx, 400)
        Y = np.linspace(0, self.Ly, 100)
        x, y = np.meshgrid(X, Y)

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
        vmag = factor * np.sqrt(u**2 + v**2)

        upper = 0.1

        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.viridis
        levels = np.linspace(0, upper, 30)

        # velocity contour (ungly solution against white lines:
        # repeat plots....)
        plt.contourf(x * factor, y * factor, vmag, levels=levels,
                     cmap=cmap, vmax=upper, vmin=0)
        plt.contourf(x * factor, y * factor, vmag, levels=levels,
                     cmap=cmap, vmax=upper, vmin=0)
        vel = plt.contourf(x * factor, y * factor, vmag, levels=levels,
                           cmap=cmap, vmax=upper, vmin=0)
        # streamlines
        y_start = np.linspace(0.0, self.Ly * factor, 20)
        x_start = np.zeros_like(y_start)
        start_points = np.array(list(zip(x_start, y_start)))
        plt.streamplot(X * factor, Y * factor, u, v,
                       start_points=start_points,
                       color='k',
                       density=100,
                       arrowstyle='-',
                       linewidth=1.0)
        # fiber
        plt.scatter(fx * factor, fy * factor, color='w')

        # set labels
        cbar = plt.colorbar(vel,
                            ticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
                            shrink=0.5)
        plt.axis('scaled')
        cbar.set_label('Velocity in mm/s', labelpad=20.0)
        plt.axis((0, factor * self.Lx, 0, factor * self.Ly))
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm ')
        plt.tight_layout()

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Streamplot written to %s." % fig)

        upper = np.max(p)
        lower = np.min(p)

        # open new plot
        plt.figure()
        # configuring new color map
        cmap = plt.cm.viridis
        levels = np.linspace(lower, upper, 30)

        # pressure contour(ugly solution against white lines:
        # repeat plots....)
        plt.contourf(x * factor, y * factor, p, levels=levels,
                     cmap=cmap, vmin=lower, vmax=upper)
        plt.contourf(x * factor, y * factor, p, levels=levels,
                     cmap=cmap, vmin=lower, vmax=upper)
        pres = plt.contourf(x * factor, y * factor, p, levels=levels,
                            cmap=cmap, vmin=lower, vmax=upper)

        # fiber
        plt.scatter(fx * factor, fy * factor, color='w')

        # set labels
        cbar = plt.colorbar(pres, label='Pressure in Pa')
        plt.axis('equal')
        plt.axis((0, factor * self.Lx, 0, factor * self.Ly))
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm')
        plt.tight_layout()

        # save plot
        p_fig = os.path.join(self.output_dir, 'pressure.pdf')
        plt.savefig(p_fig, dpi=300, bbox_inches='tight')
        print("Pressure written to %s." % p_fig)

        return[fig, p_fig]

    def _plot_reference_streamlines(self):
        """This function plots streamlines and the pressure field.

        It interpolates the properties from particles using the kernel.
        """
        # lenght factor m --> mm
        factor = 1000

        # Interpolation grid
        X = np.linspace(0, self.Lx, 400)
        Y = np.linspace(0, self.Ly, 100)
        x, y = np.meshgrid(X, Y)

        fx = self.Lx / 2
        fy = self.Ly / 2

        # Interpolation of velocitie
        data = np.loadtxt('dsm_poiseuille_comsol.txt')
        grid = griddata(data[:, 0:2], data[:, 2:4],
                        (factor * x, factor * y), method='linear')
        u, v = np.transpose(grid)
        u = np.transpose(u)
        v = np.transpose(v)
        vmag = factor * np.sqrt(u**2 + v**2)

        upper = 0.1

        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.viridis
        levels = np.linspace(0, upper, 30)

        # velocity contour (ungly solution against white lines:
        # repeat plots....)
        plt.contourf(x * factor, y * factor, vmag, levels=levels,
                     cmap=cmap, vmax=upper, vmin=0)
        plt.contourf(x * factor, y * factor, vmag, levels=levels,
                     cmap=cmap, vmax=upper, vmin=0)
        vel = plt.contourf(x * factor, y * factor, vmag, levels=levels,
                           cmap=cmap, vmax=upper, vmin=0)
        # streamlines
        y_start = np.linspace(0.0, self.Ly * factor, 20)
        x_start = np.zeros_like(y_start)
        start_points = np.array(list(zip(x_start, y_start)))
        plt.streamplot(X * factor, Y * factor, u, v,
                       start_points=start_points,
                       color='k',
                       density=100,
                       arrowstyle='-',
                       linewidth=1.0)
        # fiber
        plt.scatter(fx * factor, fy * factor, color='w')

        # set labels
        cbar = plt.colorbar(vel,
                            ticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
                            shrink=0.5)
        plt.axis('scaled')
        cbar.set_label('Velocity in mm/s', labelpad=20.0)
        plt.axis((0, factor * self.Lx, 0, factor * self.Ly))
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm ')
        plt.tight_layout()

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot_fem.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("ReferenceStreamplot written to %s." % fig)


    def _plot_inlet_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the periodic boundary.

        If the fiber has only a single particle, this is interpreted as flow
        around a fiber cylinder and the coresponding FEM solution is plotted
        as well.
        """
        # length factor m --> mm
        factor = 1000

        # Extract requested output - default is last output.
        output = self.output_files[step_idx]
        data = load(output)

        # Generate meshgrid for interpolation.
        x = np.array([0])
        y = np.linspace(0, self.Ly, 100)
        x, y = np.meshgrid(x, y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # solution for undisturbed velocity field.
        u_exact = (self.options.G * (y - self.Ly / 2) -
                   0.5 * self.options.g / self.nu * (
                   (y - self.Ly / 2)**2 - (self.Ly / 2)**2))
        u_bulk = (1 / 12 * self.options.g / self.nu * self.Ly**2 *
                  np.ones_like(y))

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u * factor, y * factor, '-k')

        # undisturbed solution
        plt.plot(u_exact * factor, y * factor, ':k')
        # bulk solution
        plt.plot(u_bulk * factor, y * factor, '--k')

        # labels
        plt.xlabel('Velocity $v_1$ in mm/s')
        plt.ylabel('$x_2$ in mm')
        plt.grid()
        plt.legend(['SPH Simulation', 'Pure Poiseuille', 'Poiseuille bulk'])
        plt.tight_layout()

        # save figure
        fig = os.path.join(self.output_dir, 'inlet_velocity.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Inlet velocity plot written to %s." % fig)

        return (fig)

    def _plot_center_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the center.

        If the fiber has only a single particle, this is interpreted
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
        y = np.linspace(0, self.Ly, 100)
        x, y = np.meshgrid(x, y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u * factor, y * factor, '-k')

        # labels
        plt.xlabel('Velocity $v_1$ in mm/s')
        plt.ylabel('$x_2$ in mm')
        plt.grid()
        plt.tight_layout()

        # save figure
        fig = os.path.join(self.output_dir, 'center_velocity.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Center velocity plot written to %s." % fig)

        return (fig)

    def _plot_pressure_centerline(self):
        """This function plots the pressure profile along a centerline for a
        single particle.
        """

        # length factor m --> mm
        factor = 1000

        # Generate meshgrid for interpolation.
        x = np.linspace(0, self.Lx, 200)
        y = np.array([self.Ly / 2])
        x, y = np.meshgrid(x, y)

        # Set a number of last solutions to average from.
        N = 10

        # averaging the pressure interpolation for N last solutions.
        p = np.zeros((200,))
        for output in self.output_files[-(1 + N):-1]:
            data = load(output)
            interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
            interp.update_particle_arrays(list(data['arrays'].values()))
            p += interp.interpolate('p') / N

        # open new plot
        plt.figure()

        # plot SPH solution and FEM solution
        plt.plot(x[0, :] * factor, p, '-k')

        # labels
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('Pressure in Pa')
        plt.grid()
        plt.tight_layout()

        # save figure
        pcenter_fig = os.path.join(self.output_dir, 'pressure_centerline.pdf')
        plt.savefig(pcenter_fig, dpi=300, bbox_inches='tight')
        print("Pressure written to %s." % pcenter_fig)

        return pcenter_fig

    def _plot_history(self):
        """This function create all plots.

        It is employing an iteration over all timesteps.
        """
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

        # empty list for conservation properies
        E_kin = []
        E_p = []
        m = []
        volume = []
        rho = []

        # empty list for reaction forces
        Fx = []
        Fwx = []
        Fy = []
        Fwy = []
        Fz = []
        Fwz = []

        # empty list for roation periods
        T = []
        t0 = 0

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
            fluid = data['arrays']['fluid']
            channel = data['arrays']['channel']

            # extrating end-points
            x_begin.append(fiber.x[0])
            y_begin.append(fiber.y[0])
            x_end.append(fiber.x[-1])
            y_end.append(fiber.y[-1])

            # computation of orientation angle
            dxx = fiber.x[0] - fiber.x[-1]
            dyy = fiber.y[0] - fiber.y[-1]
            a = np.arctan(dxx / (dyy + 0.01 * self.h0)) + N * np.pi
            if len(angle) > 0 and a - angle[-1] > 3:
                N -= 1
                a -= np.pi
            elif len(angle) > 0 and a - angle[-1] < -3:
                N += 1
                a += np.pi

            # count rotations
            if a - M * np.pi > np.pi:
                T.append(t[-1] - t0)
                t0 = t[-1]
                M += 1
            angle.append(a)

            # computation of squared velocity and masses from density and
            # volume
            v_fiber = fiber.u**2 + fiber.v**2 + fiber.w**2
            v_fluid = fluid.u**2 + fluid.v**2 + fluid.w**2
            m_fiber = fiber.rho / fiber.V
            m_fluid = fluid.rho / fluid.V
            m_channel = channel.rho / channel.V

            # appending volume, density, mass, pressure and kinetic energy
            volume.append(np.sum(1 / fiber.V) +
                          np.sum(1 / fluid.V) +
                          np.sum(1 / channel.V))
            rho.append(np.sum(fiber.rho) +
                       np.sum(fluid.rho) +
                       np.sum(channel.rho))
            m.append(np.sum(m_fiber) +
                     np.sum(m_fluid) +
                     np.sum(m_channel))
            E_p.append(np.sum(fiber.p / fiber.V) +
                       np.sum(fluid.p / fluid.V) +
                       np.sum(channel.p / channel.V))
            E_kin.append(0.5 * np.dot(m_fiber, v_fiber) +
                         0.5 * np.dot(m_fluid, v_fluid))

            # extract reaction forces at hold particles
            Fwx.append(fiber.Fwx[0])
            Fwy.append(fiber.Fwy[0])
            Fwz.append(fiber.Fwz[0])
            Fx.append(fiber.Fx[0])
            Fy.append(fiber.Fy[0])
            Fz.append(fiber.Fz[0])

        bar.finish()
        # open new figure
        plt.figure()

        # plot pressure and kinetic energy
        plt.plot(t, E_p, '-k', t, E_kin, ':k')

        # labels
        plt.xlabel('Time $t$ in seconds')
        plt.ylabel('Energy')
        plt.legend(['Pressure', 'Kinetic Energy'])
        plt.grid()
        plt.tight_layout()

        # save figure
        engfig = os.path.join(self.output_dir, 'energyplot.pdf')
        plt.savefig(engfig, dpi=300, bbox_inches='tight')
        print("Energyplot written to %s." % engfig)

        # open new plot
        plt.figure()

        # plot relative mass, volume and density
        plt.plot(t, np.array(m) / m[0], '-k',
                 t, np.array(volume) / volume[0], '--k',
                 t, np.array(rho) / rho[0], ':k')

        # labels
        plt.xlabel('Time $t$ in s')
        plt.ylabel('Relative value')
        plt.legend(['Mass', 'Volume', 'Density'])
        plt.grid()
        plt.tight_layout()

        # save figure
        mfig = os.path.join(self.output_dir, 'massplot.pdf')
        plt.savefig(mfig, dpi=300, bbox_inches='tight')
        print("Mass plot written to %s." % mfig)

        # hard-coded solutions for total reaction forces and viscous reaction
        # forces from FEM. (ar=1, g=10, G=0, width=20)
        t_fem = np.array([0, 5.00E-06, 1.00E-05, 1.50E-05, 2.00E-05, 2.50E-05,
                          3.00E-05, 3.50E-05, 4.00E-05, 4.50E-05, 5.00E-05])
        F_fem = np.array([0, 0.015085, 0.021444, 0.024142, 0.025327, 0.025828,
                          0.026010, 0.026075, 0.026094, 0.026096, 0.026093])

        # Stokes flow force
        u_bulk = 1.0 / 12.0 * self.options.g / self.nu * self.Ly**2
        F_stokes = 3.0 * np.pi * self.options.mu * u_bulk * np.ones_like(F_fem)

        # applying appropriate scale factors
        t = np.array(t) / self.scale_factor

        # open new plot
        plt.figure()

        # plot computed reaction force, total FEM force and viscous FEM
        # force
        plt.plot(t * 1000, Fx, '-k',
                 t_fem * 1000, F_fem, '--k',
                 t_fem * 1000, F_stokes, ':k')

        # labels
        plt.xlabel('Time $t$ in ms')
        plt.ylabel('Force per fiber length in N/m')
        plt.legend(['SPH', 'FEM', 'Stokes'],
                   loc='lower right')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, x2, 0, y2))
        plt.grid()
        plt.tight_layout()

        # save figure
        forcefig = os.path.join(self.output_dir, 'forceplot.pdf')
        plt.savefig(forcefig, dpi=300, bbox_inches='tight')
        try:
            tex_fig = os.path.join(self.output_dir, "forceplot.tex")
            from matplotlib2tikz import save as tikz_save
            tikz_save(tex_fig)
        except ImportError:
            print("Did not write tikz figure.")
        print("Reaction Force plot written to %s." % forcefig)
        return [engfig, forcefig]

    def post_process(self, info_fname):
        if len(self.output_files) == 0:
            return

        self._plot_reference_streamlines()
        [streamlines, pressure] = self._plot_streamlines()
        self._plot_center_velocity()
        self._plot_pressure_centerline()
        self._plot_history()
        self._plot_inlet_velocity()


if __name__ == '__main__':
    app = Channel()
    app.run()
    app.post_process(app.info_filename)
