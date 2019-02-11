"""
################################################################################
Nozzle
################################################################################
"""
# general imports
import numpy as np

# matplotlib (set up for server use)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain,
                              get_particle_array_beadchain_fiber)
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme


class Nozzle(Application):
    """Generation of a Cylinder with a nozzle and a stamp to press fluid through
    the nozzle."""
    def create_scheme(self):
        return BeadChainScheme(['fluid'], ['channel', 'stamp'], ['fibers'],
                               dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--N", action="store", type=float, dest="N",
            default=400, help="Number of particles on circular section plane"
        )
        group.add_argument(
            "--R", action="store", type=float, dest="R",
            default=0.1, help="Cylinder radius"
        )
        group.add_argument(
            "--r", action="store", type=float, dest="r",
            default=0.03, help="Nozzle radius"
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
            default=0.1, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E3, help="Young's modulus"
        )
        group.add_argument(
            "--gx", action="store", type=float, dest="gx",
            default=0, help="Body force in x-direction"
        )
        group.add_argument(
            "--gy", action="store", type=float, dest="gy",
            default=0, help="Body force in y-direction"
        )
        group.add_argument(
            "--gz", action="store", type=float, dest="gz",
            default=0, help="Body force in z-direction"
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
            "--volfrac", action="store", type=float, dest="vol_frac",
            default=0.05, help="Volume fraction of fibers in suspension."
        )
        group.add_argument(
            "--k", action="store", type=float, dest="k",
            default=0.0, help="Friction coefficient between fibers."
        )
        group.add_argument(
            "--speed", action="store", type=float, dest="speed",
            default=0.001, help="Stamp speed."
        )

    def consume_user_options(self):
        """Initialization of properties"""

        # Copy some parameters for easier handling
        self.N = self.options.N
        self.R = self.options.R
        self.rho0 = self.options.rho0
        self.gx = self.options.gx
        self.gy = self.options.gy
        self.gz = self.options.gz

        # Create sunflower seeds and estimate distance
        n = self.generate_parametrized_positions(self.R)
        y, z = self.sunflower_seed(n)
        self.h0 = np.sqrt((y[1] - y[3])**2 + (z[1] - z[3])**2)

        # The fiber length is the aspect ratio times fiber diameter
        self.L = self.options.ar * self.h0

        # The kinematic viscosity is computed from absolute viscosity and
        # density.
        self.nu = self.options.mu / self.rho0

        # empirical determination for the damping, which is just enough
        self.D = self.options.D or 0.01 * self.options.ar

        # mechanical properties
        R = self.h0 / 2
        self.A = np.pi * R**2
        self.Ip = np.pi * R**4 / 4.0
        mass = 3 * self.rho0 * self.h0 * self.A
        self.J = 1 / 4 * mass * R**2 + 1 / 12 * mass * (3 * self.h0)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%.
        # Vmax is 2 times the average speed in the smaller cylinder after the
        # nozzle, which can be computed from the surface ratio.
        self.Vmax = 2 * (self.options.R/self.options.r)**2 * self.options.speed
        self.c0 = 10 * self.Vmax
        self.p0 = self.c0**2 * self.rho0

        # Background pressure is set to zero due to the free surface. This
        # comes with the disadvantage of tensile instabilities.
        self.pb = 0.0

        # The time is set to zero, if only postprocessing is required.
        # Otherwise it is set to the time required for the stamp to fully move
        # forward.
        if self.options.postonly:
            self.t = 0
        else:
            self.t = (self.L + 4 * self.h0) / self.options.speed

        # compute number of fibers
        self.n = int(self.options.vol_frac * self.N)
        print('Creating %d fibers' % self.n)

    def configure_scheme(self):
        self.scheme.configure(
            rho0=self.rho0, c0=self.c0, nu=self.nu, p0=self.p0, pb=self.pb,
            h0=self.h0, dx=self.h0, A=self.A, Ip=self.Ip, J=self.J,
            E=self.options.E, D=self.D, gx=self.gx, gy=self.gy, gz=self.gz,
            k=self.options.k)
        # in case of very low volume fraction
        if self.n < 1:
            self.scheme.configure(fibers=[])
        self.scheme.configure_solver(
            tf=self.t, vtk=self.options.vtk, N=100,
            extra_steppers={'stamp': TransportVelocityStep()})
        # self.scheme.configure_solver(
        #   tf=self.t, pfreq=1, vtk=self.options.vtk,
        #   extra_steppers={'stamp':TransportVelocityStep()})

    def create_particles(self):
        """Four particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties, a channel of dummy
        particles and a moving stamp.
        """
        # Computation of each particle's initial volume.
        volume = np.pi * self.R**2 * self.h0 / self.N

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1. / volume

        # Fluid and fiber particles
        x_pos = np.arange(0, self.L, self.h0)
        fx = np.array([])
        fy = np.array([])
        fz = np.array([])
        fibx = np.array([])
        fiby = np.array([])
        fibz = np.array([])
        fidx = np.array([])
        n = self.generate_parametrized_positions(self.R)
        n_fiber = np.random.choice(n, int(self.options.vol_frac * len(n)),
                                   replace=False)
        n = np.delete(n, n_fiber)
        for j, x in enumerate(x_pos):
            y_2d, z_2d = self.sunflower_seed(n)
            fy = np.append(fy, y_2d.ravel())
            fz = np.append(fz, z_2d.ravel())
            fx = np.append(fx, x * np.ones_like(y_2d.ravel()))
            y_2d, z_2d = self.sunflower_seed(n_fiber)
            fiby = np.append(fiby, y_2d.ravel())
            fibz = np.append(fibz, z_2d.ravel())
            fibx = np.append(fibx, x * np.ones_like(y_2d.ravel()))
            fiber_indices = [i * self.options.ar + j for i in range(0, self.n)]
            fidx = np.append(fidx, np.array(fiber_indices))

        x_pos = np.concatenate((np.arange(-4.0 * self.h0, 0, self.h0),
                                np.arange(self.L, 4.0 * self.L, self.h0)))
        for x in x_pos:
            r = self.radius(x)
            n = self.generate_parametrized_positions(r)
            y_2d, z_2d = self.sunflower_seed(n)
            fy = np.append(fy, y_2d.ravel())
            fz = np.append(fz, z_2d.ravel())
            fx = np.append(fx, x * np.ones_like(y_2d.ravel()))

        # Channel particles
        x_pos = np.arange(-8.0 * self.h0, 4.0 * self.L, self.h0)
        cx = np.array([])
        cy = np.array([])
        cz = np.array([])
        for x in x_pos:
            r_min = self.radius(x)
            r_max = self.radius(x) + 4.0 * self.h0
            n = self.generate_parametrized_positions(r_max, r_min)
            y_2d, z_2d = self.sunflower_seed(n)
            cy = np.append(cy, y_2d.ravel())
            cz = np.append(cz, z_2d.ravel())
            cx = np.append(cx, x * np.ones_like(y_2d.ravel()))

        # Stamp particles
        x_pos = np.arange(-5.0 * self.h0, -9.0 * self.h0, -self.h0)
        sx = np.array([])
        sy = np.array([])
        sz = np.array([])
        for x in x_pos:
            r = self.radius(x)
            n = self.generate_parametrized_positions(r)
            y_2d, z_2d = self.sunflower_seed(n)
            sy = np.append(sy, y_2d.ravel())
            sz = np.append(sz, z_2d.ravel())
            sx = np.append(sx, x * np.ones_like(y_2d.ravel()))

        # Finally create all particle arrays.
        channel = get_particle_array_beadchain(
            name='channel', x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        stamp = get_particle_array_beadchain(
            name='stamp', x=sx, y=sy, z=sz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fluid = get_particle_array_beadchain(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0,
            V=V)
        fibers = get_particle_array_beadchain_fiber(
            name='fibers', x=fibx, y=fiby, z=fibz, m=mass, rho=self.rho0,
            h=self.h0, lprev=self.h0, lnext=self.h0, phi0=np.pi, phifrac=0.1,
            fidx=fidx, V=V)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d" % (
            fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # set stamp velocity
        stamp.u[:] = self.options.speed
        # 'Break' fibers in segments
        fibers.fractag[-self.n:] = 1

        if self.n > 1:
            return [fluid, channel, stamp, fibers]
        else:
            return [fluid, channel, stamp]

    def create_tools(self):
        if self.n < 1:
            return []
        else:
            return [FiberIntegrator(self.particles, self.scheme)]

    def generate_parametrized_positions(self, r_max, r_min=0):
        N_min = int((r_min / self.R)**2 * self.N)
        N_max = int((r_max / self.R)**2 * self.N)
        return(np.arange(N_min, N_max))

    def sunflower_seed(self, n):
        golden_ratio = (1 + np.sqrt(5)) / 2
        r = self.R * np.sqrt(n / self.N)
        theta = 2 * np.pi / golden_ratio**2 * n
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        return (y, z)

    def radius(self, x):
        if x < self.L:
            return self.R
        else:
            return max(self.R * (1 - (x - self.L) / self.L), self.options.r)


if __name__ == '__main__':
    app = Nozzle()
    app.run()
    app.post_process(app.info_filename)
