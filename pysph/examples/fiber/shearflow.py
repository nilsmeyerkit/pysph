"""Example for fiber shearflow.

################################################################################
3D shearflow with a single fiber
################################################################################
"""
# general imports
import os
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fluid,
                              get_particle_array_beadchain_solid,
                              get_particle_array_beadchain_fiber)

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme
from pysph.base.kernels import CubicSpline


def get_cox_aspect_ratio(aspect_ratio):
    u"""Jeffrey's equivalent aspect ratio.

    Approximation from
    Cox et al.
    """
    return 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))


def get_zhang_aspect_ratio(aspect_ratio):
    """Jeffery's equivalent aspect ratio.

    Approximation from
    Zhang et al. 2011
    """
    return (0.000035*aspect_ratio**3 - 0.00467*aspect_ratio**2 +
            0.764*aspect_ratio + 0.404)


def jeffery_ode(phi, t, ar, G):
    """Jeffery's Equation for planar rotation of a rigid."""
    lbd = (ar**2. - 1.)/(ar**2. + 1.)
    return 0.5*G*(1. + lbd*np.cos(2.*phi))


class Channel(Application):
    """Application for the channel flow driven by top an bottom walls."""

    def create_scheme(self):
        """Use BeadChainScheme for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=3)

    def add_user_options(self, group):
        """Add options to aplication."""
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0001, help="Fiber diameter"
        )
        group.add_argument(
            "--ar", action="store", type=int, dest="ar",
            default=11, help="Aspect ratio of fiber"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=63, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E6, help="Young's modulus"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=3.3, help="Shear rate"
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
            "--Re", action="store", type=float, dest="Re",
            default=0.1, help="Desired Particle Reynolds number."
        )
        group.add_argument(
            "--frac", action="store", type=float, dest="phifrac",
            default=5., help="Critical bending angle for fracture."
        )
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=1., help="Number of half rotations."
        )

    def consume_user_options(self):
        """Initialize geometry, properties and time stepping."""
        # Initial spacing of particles is set to the same value as fiber
        # diameter.
        self.dx = self.options.d

        # Smoothing radius is set to the same value as particle spacing. This
        # results for a quintic spline in a radius of influence three times as
        # large as dx
        self.h0 = self.dx

        # The fiber length is the aspect ratio times fiber diameter
        self.Lf = self.options.ar*self.dx

        # Use fiber aspect ratio to determine the channel width.
        self.Ly = self.Lf + 2.*int(0.1*self.options.ar)*self.dx

        # Density from Reynolds number
        self.Vmax = self.options.G*self.Ly/2.
        self.rho0 = (self.options.mu*self.options.Re)/(self.Vmax*self.dx)

        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.*self.Ly + self.dx

        # The position of the fiber's center is set to the center of the
        # channel.
        self.x_fiber = 0.5*self.Lx
        self.y_fiber = 0.5*self.Ly
        self.z_fiber = 0.5*self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled density.
        self.nu = self.options.mu/self.rho0

        # damping from empirical guess
        self.D = 0.01*0.2*self.options.ar

        # mass properties
        R = self.dx/2.
        self.A = np.pi*R**2.
        self.Ip = np.pi*R**4./4.
        mass = 3.*self.rho0*self.dx*self.A
        self.J = 1./4.*mass*R**2. + 1./12.*mass*(3.*self.dx)**2.

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.c0 = 10.*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required. For a
        # shear flow, it is set to the time for a full period of rotation
        # according to Jeffery's equation.
        if self.options.postonly:
            self.t = 0
        else:
            ar = get_zhang_aspect_ratio(self.options.ar)
            lbd = (ar + 1./ar)
            self.t = self.options.rot*np.pi*lbd/self.options.G
        print("Simulated time is %g s" % self.t)

    def configure_scheme(self):
        """Set up solver and scheme."""
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=self.A,
            Ip=self.Ip, J=self.J, E=self.options.E, D=self.D,
            fiber_like_solid=True)
        kernel = CubicSpline(dim=3)
        self.scheme.configure_solver(
            kernel=kernel,
            tf=self.t, vtk=self.options.vtk, N=self.options.rot*200)
        # self.scheme.configure_solver(tf=self.t, pfreq=1,
        #                              vtk = self.options.vtk)

    def create_particles(self):
        """Three particle arrays are created.

        A fluid, representing the polymer matrix, a fiber with additional
        properties and a channel of dummyparticles.
        """
        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.dx
        dx2 = fdx/2.

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        _z = np.arange(dx2, self.Ly, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position. Uncomment proper section for
        # horizontal or vertical alignment respectivley.
        indices = []
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber

            # vertical
            if (fx[i] < xx + self.dx/2. and fx[i] > xx - self.dx/2. and
                fy[i] < yy + self.Lf/2. and fy[i] > yy - self.Lf/2. and
                    fz[i] < zz + self.dx/2. and fz[i] > zz - self.dx/2.):
                indices.append(i)

            # horizontal
            # if (fx[i] < xx+self.Lf/2 and fx[i] > xx-self.Lf/2 and
            #     fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
            #     fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
            #    indices.append(i)

        # Generating fiber particle grid. Uncomment proper section for
        # horizontal or vertical alignment respectivley.

        # vertical fiber
        _fibx = np.array([xx])
        _fiby = np.arange(yy - self.Lf/2. + self.dx/2.,
                          yy + self.Lf/2. + self.dx/4.,
                          self.dx)

        # horizontal fiber
        # _fibx = np.arange(xx-self.Lf/2+self.dx/2,
        #                   xx+self.Lf/2+self.dx/4,
        #                   self.dx)
        # _fiby = np.array([yy])

        _fibz = np.array([zz])
        fibx, fiby, fibz = self.get_meshgrid(_fibx, _fiby, _fibz)

        # Determine the size of dummy region
        ghost_extent = 3.*fdx

        # Create the channel particles at the top
        _y = np.arange(self.Ly + dx2, self.Ly + dx2 + ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2 - ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))
        cz = np.concatenate((tz, bz))

        # Computation of each particles initial volume.
        volume = fdx**3.

        # Mass is set to get the reference density of rho0.
        mass = volume*self.rho0

        # assign unique ID (within fiber) to each fiber particle.
        fidx = range(0, self.options.ar)

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        channel = get_particle_array_beadchain_solid(
            name='channel', x=cx, y=cy, z=cz, m=mass, rho=self.rho0,
            h=self.h0, V=V)
        fluid = get_particle_array_beadchain_fluid(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0,
            h=self.h0, V=V)
        fluid.remove_particles(indices)
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fibx, y=fiby, z=fibz, m=mass,
            rho=self.rho0, h=self.h0, lprev=self.dx, lnext=self.dx,
            phi0=np.pi, phifrac=self.options.phifrac, fidx=fidx, V=V)

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles() == self.options.ar)

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 0
        if self.options.holdcenter:
            idx = int(np.floor(self.options.ar/2))
            fiber.holdtag[idx] = 100

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.y[:] - self.Ly/2.)
        fiber.u[:] = self.options.G*(fiber.y[:] - self.Ly/2.)
        channel.u[:] = self.options.G*np.sign(channel.y[:])*self.Ly/2.

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_domain(self):
        """Create periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, zmin=0, zmax=self.Ly,
                             periodic_in_x=True, periodic_in_z=True)

    def create_tools(self):
        """Add an integrator for the fiber."""
        ud = not self.options.holdcenter
        return [FiberIntegrator(self.particles, self.scheme, self.domain,
                                innerloop=True, updates=ud)]

    def get_meshgrid(self, xx, yy, zz):
        """Generate meshgrids quickly."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def _plots(self):
        """Create plots.

        It is employing a iteration over all time steps.
        """
        # empty list for time
        t = []
        # empty lists for orbit
        x_begin = []
        y_begin = []
        x_end = []
        y_end = []
        # empty list for orientation angle
        angle = []
        N = 0

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for i, fname in enumerate(output_files):
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
            dxx = fiber.x[0] - fiber.x[-1]
            dyy = fiber.y[0] - fiber.y[-1]
            a = np.arctan(dxx / (dyy + 0.01 * self.h0)) + N * np.pi
            if len(angle) > 0 and a - angle[-1] > 3:
                N -= 1
                a -= np.pi
            elif len(angle) > 0 and a - angle[-1] < -3:
                N += 1
                a += np.pi
            angle.append(a)

        # Integrate Jeffery's solution
        print("Solving Jeffery's ODE")
        t = np.array(t)
        phi0 = angle[0]
        ar_zhang = get_zhang_aspect_ratio(self.options.ar)
        ar_cox = get_cox_aspect_ratio(self.options.ar)
        angle_jeffery_zhang = odeint(jeffery_ode, phi0, t, atol=1E-15,
                                     args=(ar_zhang, self.options.G))
        angle_jeffery_cox = odeint(jeffery_ode, phi0, t, atol=1E-15,
                                   args=(ar_cox, self.options.G))

        # constraint between -pi/2 and pi/2
        # angle_jeffery = (angle_jeffery+np.pi/2.)%np.pi-np.pi/2.

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t/self.options.G, angle, '-k')
        plt.plot(t/self.options.G, angle_jeffery_zhang, '--k', color='grey')
        plt.plot(t/self.options.G, angle_jeffery_cox, ':k', color='grey')

        # labels
        plt.xlabel('Strains')
        plt.ylabel('Rotation angle')
        plt.legend(['SPH Simulation', 'Jeffery (Zhang)', 'Jeffery (Cox)',
                    'Jeffery (Goldsm.)'])
        plt.grid()
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, x2, 0, y2))
        ax = plt.gca()
        ax.set_yticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3/2\pi$'])
        plt.tight_layout()

        # save figure
        angfig = os.path.join(self.output_dir, 'angleplot.pdf')
        plt.savefig(angfig, dpi=300, bbox_inches='tight')
        try:
            tex_fig = os.path.join(self.output_dir, "angleplot.tex")
            from matplotlib2tikz import save as tikz_save
            tikz_save(tex_fig)
        except ImportError:
            print("Did not write tikz figure.")
        print("Angleplot written to %s." % angfig)

    def post_process(self, info_fname):
        """Build plots and files as results."""
        if len(self.output_files) == 0:
            return
        self._plots()


if __name__ == '__main__':
    app = Channel()
    app.run()
    app.post_process(app.info_filename)
