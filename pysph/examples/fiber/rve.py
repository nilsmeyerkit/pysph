"""
################################################################################
Mini RVE
################################################################################
"""
# general imports
import os
import random
import itertools
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fluid,
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
        return BeadChainScheme(['fluid'], [], ['fibers'], dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0001, help="Fiber diameter"
        )
        group.add_argument(
            "--ar", action="store", type=int, dest="ar",
            default=10, help="Aspect ratio of fiber"
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
            default=3.3, help="Shear rate"
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
            "--volfrac", action="store", type=float, dest="vol_frac",
            default=0.01, help="Volume fraction of fibers in suspension."
        )
        group.add_argument(
            "--k", action="store", type=float, dest="k",
            default=0.0, help="Friction coefficient between fibers."
        )
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=2.0, help="Number of half rotations."
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
        self.L = self.options.ar*self.dx

        # Computation of a scale factor in a way that dt_cfl exactly matches
        # dt_viscous.
        a = self.h0*0.125*11/0.4
        # nu_needed = a*self.options.G*self.L/2
        nu_needed = (a*self.options.G*self.L/4
                     + np.sqrt(a/8*self.options.g*self.L**2
                               + (a/2)**2/4*self.options.G**2*self.L**2))

        # If there is no other scale scale factor provided, use automatically
        # computed factor.
        if self.options.ar < 35:
            auto_scale_factor = self.options.mu/(nu_needed*self.options.rho0)
        else:
            auto_scale_factor = 0.6*self.options.mu/nu_needed/self.options.rho0
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
        self.Ip = np.pi*R**4/4.0
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
            l = (self.options.ar+1.0/self.options.ar)
            self.t = self.options.rot*np.pi*l/self.options.G
        print("Simulated time is %g s" % self.t)

        fdx = self.dx
        dx2 = fdx/2.0

        _x = np.arange(dx2, self.L, fdx)
        _z = np.arange(dx2, self.L, fdx)

        self.n = int(round(self.options.vol_frac*len(_x)*len(_z)))

    def configure_scheme(self):
        self.scheme.configure(
            rho0=self.rho0,
            c0=self.c0,
            nu=self.nu,
            p0=self.p0,
            pb=self.pb,
            h0=self.h0,
            dx=self.dx,
            A=self.A,
            Ip=self.Ip,
            J=self.J,
            E=self.options.E,
            D=self.D,
            gx=self.options.g,
            k=self.options.k)
        # in case of very low volume fraction
        if self.n < 1:
            self.scheme.configure(fibers=[])
        self.scheme.configure_solver(tf=self.t, vtk=self.options.vtk,
                                     N=self.options.rot*100,
                                     output_only_real=False)

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
        fiber_volume = self.dx**3

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0
        fiber_mass = fiber_volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume
        fiber_V = 1./fiber_volume

        # Creating grid points for particles
        _x = np.arange(dx2, self.L, fdx)
        _y = np.arange(dx2, self.L, fdx)
        _z = np.arange(dx2, self.L, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position.
        indices = []
        fibers = []
        fibx = tuple()
        fiby = tuple()
        fibz = tuple()

        positions = list(itertools.product(_x, _z))
        for yy, zz in random.sample(positions, self.n):
            for i in range(len(fx)):
                xx = 0.5*self.L

                # vertical
                if (fx[i] < xx+self.L/2 and fx[i] > xx-self.L/2 and
                    fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
                        fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
                    indices.append(i)

            # Generating fiber particle grid. Uncomment proper section for
            # horizontal or vertical alignment respectivley.

            # vertical fiber

            _fibx = np.arange(xx-self.L/2+self.dx/2, xx+self.L/2+self.dx/4,
                              self.dx)
            _fiby = np.array([yy])
            _fibz = np.array([zz])
            _fibx, _fiby, _fibz = self.get_meshgrid(_fibx, _fiby, _fibz)
            fibx = fibx + (_fibx,)
            fiby = fiby + (_fiby,)
            fibz = fibz + (_fibz,)

        print("Created %d fibers." % self.n)

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        fluid = get_particle_array_beadchain_fluid(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fluid.remove_particles(indices)
        if self.n > 0:
            fibers = get_particle_array_beadchain_fiber(
                name='fibers', x=np.concatenate(fibx), y=np.concatenate(fiby),
                z=np.concatenate(fibz), m=fiber_mass, rho=self.rho0,
                h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
                phifrac=2.0, fidx=range(self.options.ar*self.n), V=fiber_V)
            # 'Break' fibers in segments
            endpoints = [i*self.options.ar-1 for i in range(1, self.n)]
            fibers.fractag[endpoints] = 1

        # Setting the initial velocities for a shear flow.
        fluid.v[:] = self.options.G*(fluid.x[:]-self.L/2)

        if self.n > 0:
            fibers.v[:] = self.options.G*(fibers.x[:]-self.L/2)
            return [fluid, fibers]
        else:
            return [fluid]

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.L, periodic_in_x=True,
                             ymin=0, ymax=self.L, periodic_in_y=True,
                             zmin=0, zmax=self.L, periodic_in_z=True,
                             gamma_yx=self.options.G,
                             n_layers=1,
                             dt=self.solver.dt)

    def create_tools(self):
        """Set up fiber integrator."""
        if self.n < 1:
            return []
        else:
            return [FiberIntegrator(self.particles, self.scheme,
                                    self.domain,
                                    updates=False,
                                    #  parallel=True
                                    )]

    def get_meshgrid(self, xx, yy, zz):
        """This function is a shorthand for the generation of meshgrids."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def post_process(self, info_fname):
        if len(self.output_files) == 0:
            return

        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt
        t, ke = get_ke_history(self.output_files, 'fluid')
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel('t')
        plt.ylabel('Kinetic energy')
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)

        # empty list for time
        t = []

        # empty lists for fiber orientation tensors
        A = []

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)

            # extracting time
            t.append(data['solver_data']['t'])

            if self.n > 0:
                # extrating all arrays.
                directions = []
                fiber = data['arrays']['fibers']
                startpoints = [i*(self.options.ar-1) for i in range(0, self.n)]
                endpoints = [i*(self.options.ar-1)-1 for i in range(1,
                                                                    self.n+1)]
                for start, end in zip(startpoints, endpoints):
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
    import cProfile
    import pstats
    app = RVE()
    cProfile.runctx('app.run()', None, locals(), 'stats')
    p = pstats.Stats('stats')
    p.sort_stats('tottime').print_stats(20)
    app.post_process(app.info_filename)
