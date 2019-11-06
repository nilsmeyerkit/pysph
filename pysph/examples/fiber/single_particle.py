"""3D Flow around a single fixed fiber particle."""

import os
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fluid,
                              get_particle_array_beadchain_solid,
                              get_particle_array_beadchain_fiber)

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files

from pysph.sph.scheme import BeadChainScheme
from pysph.base.kernels import CubicSpline


class SingleParticle(Application):
    """3D Flow around a single fixed fiber particle."""

    def create_scheme(self):
        """Use BeadChainScheme is used for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=3)

    def add_user_options(self, group):
        """Add options to aplication."""
        group.add_argument(
            "--dx", action="store", type=float, dest="dx",
            default=0.0001, help="Particle Spacing"
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
            default=0.01, help="Re to determine velocity at cube borders"
        )
        group.add_argument(
            "--size", action="store", type=int, dest="size",
            default=50, help="Cube size (multiples of fiber diameter)"
        )
        group.add_argument(
            "--t", action="store", type=float, dest="t",
            default=0.001, help="Cube size (multiples of fiber diameter)"
        )
        group.add_argument(
            "--vtk", action="store_true", dest='vtk',
            default=False, help="Enable vtk-output during solving."
        )

    def consume_user_options(self):
        """Initialize geometry, properties and time stepping."""
        # Initial spacing of particles
        self.dx = self.options.dx
        self.h0 = self.dx

        # equivalent sphere diameter
        self.d = 2.*(3/(4*np.pi))**(1.0/3.0)*self.dx

        # Cube dimensions
        self.L = self.options.size*self.dx

        # The density
        self.rho0 = self.options.rho0

        # The position of the fiber's center is set to the center of the cube.
        self.x_fiber = np.array([0.5*self.L])
        self.y_fiber = np.array([0.5*self.L])
        self.z_fiber = np.array([0.5*self.L])

        # The kinematic viscosity
        self.nu = self.options.mu/self.rho0

        # applied velocity
        self.v = self.options.Re*self.nu/self.dx

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.c0 = 100*self.v
        self.p0 = self.c0**2*self.rho0
        self.pb = self.p0

    def configure_scheme(self):
        """Set up solver and scheme."""
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=1.0,
            Ip=1.0, J=1.0, E=1.0, D=1.0, d=self.d)

        self.kernel = CubicSpline(dim=3)
        self.scheme.configure_solver(
            tf=self.options.t, vtk=self.options.vtk, N=20, kernel=self.kernel)

    def create_particles(self):
        """Three particle arrays are created.

        A fluid, representing the polymer matrix, a fiber with additional
        properties and a channel of dummy particles.
        """
        # short notation
        fdx = self.dx
        dx2 = fdx/2
        ghost_extent = 3*fdx

        # Creating grid points for particles
        _x = np.arange(dx2, self.L, fdx)
        _y = np.arange(dx2, self.L, fdx)
        _z = np.arange(dx2, self.L, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # add some random noise
        # noise = fdx/20
        # fx = fx + np.random.normal(0, noise, fx.shape)
        # fy = fy + np.random.normal(0, noise, fy.shape)
        # fz = fz + np.random.normal(0, noise, fz.shape)

        # fiber
        fibx, fiby, fibz = self.get_meshgrid(
            self.x_fiber, self.y_fiber, self.z_fiber)

        # remove particles at fiber position
        dist = 100000
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber
            if ((fx[i] - xx)**2 + (fy[i] - yy)**2 + (fz[i] - zz)**2) < dist:
                min_dist_idx = i
                dist = ((fx[i] - xx)**2 + (fy[i] - yy)**2 + (fz[i] - zz)**2)

        # Create the channel particles at the top
        _y = np.arange(self.L+dx2, self.L+dx2+ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the right
        _x = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        _y = np.arange(dx2-ghost_extent, self.L+ghost_extent, fdx)
        rx, ry, rz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the left
        _x = np.arange(self.L+dx2, self.L+dx2+ghost_extent, fdx)
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
        fluid.remove_particles([min_dist_idx])
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fibx, y=fiby, z=fibz, m=mass, rho=self.rho0,
            h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
            phifrac=2.0, fidx=[0], V=V)

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 100

        # Setting the initial velocities
        # fluid.w[:] = self.v
        channel.w[:] = self.v

        # set anayltical solution
        R = self.d/2.
        r = np.sqrt((fluid.x-self.x_fiber)**2
                    + (fluid.y-self.y_fiber)**2
                    + (fluid.z-self.z_fiber)**2)
        theta = np.arccos((fluid.z-self.z_fiber)/r)
        phi = np.arctan2(-(fluid.y-self.y_fiber), -(fluid.x-self.x_fiber))
        ur = self.v*(1. + R**3/(2.*r**3) - (3.*R)/(2.*r))*np.cos(theta)
        ut = -self.v*(1. - R**3/(4.*r**3) - (3.*R)/(4.*r))*np.sin(theta)
        fluid.u[:] = ur*np.sin(theta)*np.cos(phi)+ut*np.cos(theta)*np.cos(phi)
        fluid.v[:] = ur*np.sin(theta)*np.sin(phi)+ut*np.cos(theta)*np.sin(phi)
        fluid.w[:] = ur*np.cos(theta)-ut*np.sin(theta)

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_domain(self):
        """Create periodic boundary conditions in x-direction."""
        return DomainManager(zmin=0, zmax=self.L, periodic_in_z=True)

    def get_meshgrid(self, xx, yy, zz):
        """Generate meshgrids quickly."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def _plots(self):
        """Plot streamlines, pressure and velocity.

        It interpolates the properties from particles using the kernel.
        """
        from pysph.tools.interpolator import Interpolator
        from matplotlib import pyplot as plt

        # Interpolation grid
        X = np.linspace(0, self.L, 100)
        Z = np.linspace(0, self.L, 100)

        # Extract positions of fiber particles from last step to plot them
        # on top of velocities.
        last_output = self.output_files[-1]
        data = load(last_output)
        fiber = data['arrays']['fiber']
        fx = fiber.x
        fy = fiber.y

        # Interpolation (precompiled) of velocities
        x, z = np.meshgrid(X, Z)
        interp = Interpolator(list(data['arrays'].values()),
                              x=x, y=np.array([self.L/2.0]), z=z,
                              kernel=self.kernel)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        w = interp.interpolate('w')
        vmag = np.sqrt(u**2 + w**2)/self.v
        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.Reds
        levels = np.linspace(0.0, 1.0, 10)

        # velocity contour
        plt.contourf(x, z, vmag, levels=levels, cmap=cmap)
        vel = plt.contourf(x, z, vmag, levels=levels, cmap=cmap)
        # streamlines
        x_start = np.linspace(0.0, self.L, 40)
        z_start = np.zeros_like(x_start)
        start_points = np.array(list(zip(x_start, z_start)))
        plt.streamplot(X, Z, u, w,
                       # start_points=start_points,
                       color='k',
                       # density=100,
                       arrowstyle='-',
                       linewidth=1.0)
        # fiber
        plt.scatter(fx, fy, color='w')
        plt.xticks([])
        plt.yticks([])

        # set labels
        cbar = plt.colorbar(vel, shrink=0.5)
        cbar.set_label('Velocity $v/v_0$', labelpad=20.0)
        plt.tight_layout()

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Streamplot written to %s." % fig)

        # Interpolate along line
        x, y, z = np.meshgrid(np.array([self.L/2.0]),
                              np.array([self.L/2.0]),
                              Z)
        interp = Interpolator(list(data['arrays'].values()),
                              x=x, y=y, z=z, kernel=self.kernel)
        interp.update_particle_arrays(list(data['arrays'].values()))
        uz = interp.interpolate('w')
        pz = interp.interpolate('p')
        x, y, z = np.meshgrid(X,
                              np.array([self.L/2.0]),
                              np.array([self.L/2.0]))
        interp = Interpolator(list(data['arrays'].values()),
                              x=x, y=y, z=z, kernel=self.kernel)
        interp.update_particle_arrays(list(data['arrays'].values()))
        ux = interp.interpolate('w')

        # reference solution
        R = self.d/2.
        rz = np.linspace(0, self.L, 500)-self.L/2
        rx = np.linspace(0, self.L, 500)-self.L/2
        ref_uz = 1.0-3.0*R/(2*np.abs(rz))+R**3/(2*np.abs(rz)**3)
        ref_ux = 1.0-3.0*R/(4*np.abs(rx))-R**3/(2*np.abs(rx)**3)
        ref_pz = -3.0/2.0*self.options.mu*self.v*R*rz/(np.abs(rz)**3)
        mask = (np.abs(rz) < R)
        ref_uz[mask] = np.nan
        ref_pz[mask] = np.nan
        mask = (np.abs(rx) < R)
        ref_ux[mask] = np.nan
        pmax = np.nanmax(ref_pz)

        # Velocity plot
        plt.subplot(1, 2, 1)
        plt.plot((rz+self.L/2)/self.dx, ref_uz, Z/self.dx, uz/self.v)
        plt.xlabel('z')
        plt.ylim([0.0, 1.0])
        plt.subplot(1, 2, 2)
        plt.plot((rx+self.L/2)/self.dx, ref_ux, X/self.dx, ux/self.v,)
        plt.xlabel('x')
        plt.ylim([0.0, 1.0])
        # save plot
        fig = os.path.join(self.output_dir, 'velocity.png')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(fig.replace('.png', '.tex'))
        except ImportError:
            print("Did not write tikz figure.")

        # Pressure plot
        plt.figure()
        plt.plot((rz+self.L/2)/self.dx, ref_pz/pmax, Z/self.dx, pz/pmax)
        plt.xlabel('z')
        plt.ylim([-1.0, 1.0])
        # save plot
        fig = os.path.join(self.output_dir, 'pressure.png')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(fig.replace('.png', '.tex'))
        except ImportError:
            print("Did not write tikz figure.")

    def _plot_history(self):
        """Plot forces."""
        # empty list for time
        t = []

        # empty list for reaction forces
        Fx = []
        Fy = []
        Fz = []

        # reference solution
        R = self.d/2.
        F = 6.0*np.pi*self.options.mu*R*self.v

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for i, fname in enumerate(output_files):
            data = load(fname)

            # extracting time
            t.append(data['solver_data']['t'])

            # extrating all arrays.
            fiber = data['arrays']['fiber']

            # extract reaction forces at hold particles
            idx = np.argwhere(fiber.holdtag == 100)
            Fx.append(fiber.Fx[idx][0]/F)
            Fy.append(fiber.Fy[idx][0]/F)
            Fz.append(fiber.Fz[idx][0]/F)

        file = os.path.join(self.output_dir, 'force.csv')

        np.savetxt(file, np.transpose([t, Fx, Fy, Fz]), delimiter=',')

    def post_process(self, info_fname):
        """Build plots and files as results."""
        if len(self.output_files) == 0:
            return

        self._plot_history()
        self._plots()


if __name__ == '__main__':
        app = SingleParticle()
        app.run()
        app.post_process(app.info_filename)
