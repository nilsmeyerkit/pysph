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
from pysph.base.kernels import CubicSpline


class RVE(Application):
    """Generation of a mini RVE and evaluation of its fiber orientation
    tensor."""
    def create_scheme(self):
        """There is no scheme used in this application and equations are set up
        manually."""
        return BeadChainScheme(['fluid'], [], ['fibers'], dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--dx", action="store", type=float, dest="dx",
            default=0.0001, help="Particle Spacing"
        )
        group.add_argument(
            "--lf", action="store", type=int, dest="lf",
            default=5, help="Fiber length in multiples of dx"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=1.0, help="Absolute viscosity"
        )
        group.add_argument(
            "--S", action="store", type=float, dest="S",
            default=10, help="Dimensionless fiber stiffness"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=1.0, help="Shear rate"
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
            "--volfrac", action="store", type=float, dest="vol_frac",
            default=0.0014, help="Volume fraction of fibers in suspension."
        )
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=2.0, help="Number of half rotations."
        )
        group.add_argument(
            "--C", action="store", type=float, dest="C",
            default=15.0, help="Cube size as multiples of fiber diameter."
        )
        group.add_argument(
            "--continue", action="store", type=str, dest="continuation",
            default=None, help="Set a file for continuation of run."
        )

    def consume_user_options(self):
        """Initialization of geometry, properties and time stepping."""

        # Initial spacing of particles
        self.dx = self.options.dx
        self.h0 = self.dx

        # The fiber length is the aspect ratio times fiber diameter
        self.L = self.options.lf*self.dx

        # Cube size
        self.C = self.options.C*self.dx

        # Density from Reynolds number
        self.Vmax = self.options.G*self.C/2.
        self.rho0 = (self.options.mu*self.options.Re)/(self.Vmax*self.dx)

        # The kinematic viscosity
        self.nu = self.options.mu/self.rho0

        # empirical determination for the damping, which is just enough
        self.D = 0.002*self.options.lf

        # mass properties
        R = self.dx/(np.sqrt(np.pi))    # Assuming cylindrical shape
        self.d = 2.*R
        self.ar = self.Lf/self.d        # Actual fiber aspect ratio
        print('Aspect ratio is %f' % self.ar)

        self.A = np.pi*R**2.
        self.Ip = np.pi*R**4./4.
        mass = 3.*self.rho0*self.dx*self.A
        self.J = 1./4.*mass*R**2. + 1./12.*mass*(3.*self.dx)**2.

        # stiffness from dimensionless stiffness
        self.E = 4.0/np.pi*(
            self.options.S*self.options.mu*self.options.G*self.ar)

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.c0 = 10.0*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required.
        if self.options.postonly:
            self.t = 0.0
        else:
            lbd = (self.ar+1.0/self.ar)
            self.t = self.options.rot*np.pi*lbd/self.options.G
        print("Simulated time is %g s" % self.t)

        vol_fiber = self.L*self.dx*self.dx
        vol = self.C**3
        self.n = int(round(self.options.vol_frac*vol/vol_fiber))

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
            d=self.d)
        # in case of very low volume fraction
        kernel = CubicSpline(dim=3)
        if self.n < 1:
            self.scheme.configure(fibers=[])
        self.scheme.configure_solver(kernel=kernel,
                                     tf=self.t, vtk=self.options.vtk,
                                     #  pfreq=1,
                                     N=self.options.rot*100,
                                     # output_only_real=False
                                     )

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        if not self.options.continuation:
            # The fluid might be scaled compared to the fiber. fdx is a
            # shorthand for the fluid spacing and dx2 is a shorthand for
            # the half of it.
            fdx = self.dx
            dx2 = fdx/2

            # Computation of each particles initial volume.
            volume = fdx**3

            # Mass is set to get the reference density of rho0.
            mass = volume * self.rho0

            # Initial inverse volume (necessary for transport velocity
            # equations)
            V = 1./volume

            # Creating grid points for particles
            _x = np.arange(dx2, self.C, fdx)
            _y = np.arange(dx2, self.C, fdx)
            _z = np.arange(dx2, self.C, fdx)
            fx, fy, fz = self.get_meshgrid(_x, _y, _z)

            # Remove particles at fiber position.
            indices = []
            fibers = []
            fibx = tuple()
            fiby = tuple()
            fibz = tuple()

            positions = list(itertools.product(_x, _y, _z))
            random.shuffle(positions)
            N = 0
            while N < self.n:
                xx, yy, zz = positions.pop()
                idx_list = []
                for i in range(len(fx)):
                    # periodic extending above
                    if xx+self.L/2 > self.C:
                        if ((fx[i] < (xx+self.L/2-self.C) or
                            fx[i] > xx-self.L/2) and
                            fy[i] < yy+self.dx/2 and
                            fy[i] > yy-self.dx/2 and
                            fz[i] < zz+self.dx/2 and
                                fz[i] > zz-self.dx/2):
                            idx_list.append(i)
                    # periodic extending below
                    elif xx-self.L/2 < 0:
                        if ((fx[i] < xx+self.L/2 or
                            fx[i] > (xx-self.L/2+self.C)) and
                            fy[i] < yy+self.dx/2 and
                            fy[i] > yy-self.dx/2 and
                            fz[i] < zz+self.dx/2 and
                                fz[i] > zz-self.dx/2):
                            idx_list.append(i)
                    # standard case
                    else:
                        if (fx[i] < xx+self.L/2 and
                            fx[i] > xx-self.L/2 and
                            fy[i] < yy+self.dx/2 and
                            fy[i] > yy-self.dx/2 and
                            fz[i] < zz+self.dx/2 and
                                fz[i] > zz-self.dx/2):
                            idx_list.append(i)

                idx_set = set(idx_list)
                if len(idx_set.intersection(set(indices))) == 0:
                    N = N + 1
                    indices = indices + idx_list

                    # Generating fiber particles
                    if self.options.lf % 2 == 1:
                        _fibx = np.linspace(xx-self.options.lf//2*self.dx,
                                            xx+self.options.lf//2*self.dx,
                                            self.options.lf)
                    else:
                        _fibx = np.arange(xx-self.L/2,
                                          xx+self.L/2 - self.dx/4,
                                          self.dx)
                    _fiby = np.array([yy])
                    _fibz = np.array([zz])
                    _fibx, _fiby, _fibz = self.get_meshgrid(_fibx,
                                                            _fiby,
                                                            _fibz)
                    fibx = fibx + (_fibx,)
                    fiby = fiby + (_fiby,)
                    fibz = fibz + (_fibz,)
                else:
                    print("Found a fiber intersection. Trying again...")

            print("Created %d fibers." % N)

            # Finally create all particle arrays. Note that fluid particles are
            # removed in the area, where the fiber is placed.
            fluid = get_particle_array_beadchain_fluid(
                name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0,
                h=self.h0, V=V)
            fluid.remove_particles(indices)

            if self.n > 0:
                fibers = get_particle_array_beadchain_fiber(
                    name='fibers', x=np.concatenate(fibx),
                    y=np.concatenate(fiby),
                    z=np.concatenate(fibz), m=mass, rho=self.rho0,
                    h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
                    phifrac=2.0, fidx=range(self.options.lf*self.n), V=V)
                # 'Break' fibers in segments
                endpoints = [i*self.options.lf-1 for i in range(1, self.n)]
                fibers.fractag[endpoints] = 1

            # Setting the initial velocities for a shear flow.
            fluid.v[:] = self.options.G*(fluid.x[:]-self.C/2)

            if self.n > 0:
                fibers.v[:] = self.options.G*(fibers.x[:]-self.C/2)
                return [fluid, fibers]
            else:
                return [fluid]
        else:
            data = load(self.options.continuation)
            fluid = data['arrays']['fluid']
            fibers = data['arrays']['fibers']
            fibers.phifrac[:] = 2.0
            fibers.phi0[:] = np.pi
            self.solver.t = data['solver_data']['t']
            self.solver.count = data['solver_data']['count']
            return [fluid, fibers]

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.C, periodic_in_x=True,
                             ymin=0, ymax=self.C, periodic_in_y=True,
                             zmin=0, zmax=self.C, periodic_in_z=True,
                             gamma_yx=self.options.G,
                             n_layers=1,
                             dt=self.solver.dt,
                             calls_per_step=2)

    def create_tools(self):
        """Set up fiber integrator."""
        if self.n < 1:
            return []
        else:
            return [FiberIntegrator(self.particles, self.scheme,
                                    self.domain,
                                    updates=True,
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
                startpoints = [i*(self.options.lf-1) for i in range(0, self.n)]
                endpoints = [i*(self.options.lf-1)-1 for i in range(1,
                                                                    self.n+1)]
                for start, end in zip(startpoints, endpoints):
                    px = np.mean(fiber.rxnext[start:end])
                    py = np.mean(fiber.rynext[start:end])
                    pz = np.mean(fiber.rznext[start:end])

                    n = np.array([px, py, pz])
                    norm = np.linalg.norm(n)
                    if norm == 0:
                        p = np.array([1, 0, 0])
                    else:
                        p = n/norm
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
    app.run()
    # cProfile.runctx('app.run()', None, locals(), 'stats')
    # p = pstats.Stats('stats')
    # p.sort_stats('tottime').print_stats(20)
    app.post_process(app.info_filename)
