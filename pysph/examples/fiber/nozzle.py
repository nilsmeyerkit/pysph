"""
################################################################################
Nozzle
################################################################################
"""
# general imports
import os
import random
import itertools
import numpy as np
from scipy.integrate import odeint
from math import sqrt

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
from pysph.solver.utils import load,remove_irrelevant_files
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme


class Nozzle(Application):
    """Generation of a mini RVE and evaluation of its fiber orientation
    tensor."""
    def create_scheme(self):
        """There is no scheme used in this application and equations are set up
        manually."""
        return BeadChainScheme(['fluid'], ['channel', 'stamp'], ['fibers'], dim=3)

    def add_user_options(self, group):
        group.add_argument(
            "--N", action="store", type=float, dest="N",
            default=400, help="Number of particles on circular section plane"
        )
        group.add_argument(
            "--R", action="store", type=float, dest="R",
            default=0.1, help="Nozzle radius"
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
            default=1E9, help="Young's modulus"
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
            default=1E3, help="Factor of mass scaling"
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
        """Initialization of geometry, properties and time stepping."""

        # Initial spacing of particles is set to the same value as fiber
        # diameter.
        self.N = self.options.N
        self.R = self.options.R

        y,z = self.sunflower_seed(self.R)
        self.h0 = np.sqrt((y[0]-y[2])**2+(z[0]-z[2])**2)

        # The fiber length is the aspect ratio times fiber diameter
        self.L = self.options.ar*self.h0

        # If there is no other scale scale factor provided, use automatically
        # computed factor.
        self.scale_factor = self.options.scale_factor

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
        R = self.h0/2
        self.A = np.pi*R**2
        self.I = np.pi*R**4/4.0
        mass = 3*self.rho0*self.h0*self.A
        self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.h0)**2

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.Vmax = abs(self.options.g/(2*self.nu)*self.R**2/4)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required.
        if self.options.postonly:
            self.t = 0
        else:
            self.t = 10
        print("Simulated time is %g s"%self.t)

        # start with no fibers at all
        self.n = 0

    def configure_scheme(self):
        self.scheme.configure(rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.h0, A=self.A, I=self.I,
            J=self.J, E=self.options.E, D=self.D,
            scale_factor=self.scale_factor, gx=self.options.g,
            k=self.options.k, u=self.options.speed)
        # in case of very low volume fraction
        if self.n < 1:
            self.scheme.configure(fibers=[])
        self.scheme.configure_solver(tf=self.t, vtk = self.options.vtk,
            N=100, extra_steppers={'stamp':TransportVelocityStep()})
        #self.scheme.configure_solver(tf=self.t, pfreq=1, vtk = self.options.vtk)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""


        # Computation of each particles initial volume.
        volume = np.pi*self.h0**2*self.h0

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume

        x_pos = np.arange(0, 2.0*self.L, self.h0)

        fx = np.array([])
        fy = np.array([])
        fz = np.array([])
        for x in x_pos:
            r = self.radius(x)
            y_2d, z_2d = self.sunflower_seed(r)
            fy = np.append(fy, y_2d.ravel())
            fz = np.append(fz, z_2d.ravel())
            fx = np.append(fx, x*np.ones_like(y_2d.ravel()))

        x_pos = np.arange(-1.0*self.L, 2.0*self.L, self.h0)

        cx = np.array([])
        cy = np.array([])
        cz = np.array([])
        for x in x_pos:
            r_min = self.radius(x)
            r_max = self.radius(x)+3*self.h0
            y_2d, z_2d = self.sunflower_seed(r_max, r_min)
            cy = np.append(cy, y_2d.ravel())
            cz = np.append(cz, z_2d.ravel())
            cx = np.append(cx, x*np.ones_like(y_2d.ravel()))

        x_pos = np.arange(-self.h0, -4*self.h0, -self.h0)

        sx = np.array([])
        sy = np.array([])
        sz = np.array([])
        for x in x_pos:
            r_max = self.radius(x)
            y_2d, z_2d = self.sunflower_seed(r_max)
            sy = np.append(sy, y_2d.ravel())
            sz = np.append(sz, z_2d.ravel())
            sx = np.append(sx, x*np.ones_like(y_2d.ravel()))

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        channel = get_particle_array_beadchain(name='channel',
                    x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0, V=V)
        stamp = get_particle_array_beadchain(name='stamp',
                    x=sx, y=sy, z=sz, m=mass, rho=self.rho0, h=self.h0, V=V)
        fluid = get_particle_array_beadchain(name='fluid',
                    x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0, V=V)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        if self.n > 1:
            return [fluid, channel, stamp, fibers]
        else:
            return [fluid, channel, stamp]


    # def create_domain(self):
    #     """The channel has periodic boundary conditions in x-direction."""
    #     return DomainManager(xmin=0, xmax=self.L, periodic_in_x=True)

    # def create_tools(self):
    #     if self.n < 1:
    #         return []
    #     else:
    #         return [FiberIntegrator(self.particles, self.scheme, self.domain,
    #                             parallel=True)]

    def sunflower_seed(self, r_max, r_min=0):
        golden_ratio = (1+np.sqrt(5))/2
        N_min = int((r_min/self.R)**2*self.N)
        N_max = int((r_max/self.R)**2*self.N)
        n = np.arange(max(N_min,1),N_max)
        r = self.R*np.sqrt(n/self.N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        return (y,z)

    def radius(self,x):
        if x < self.L:
            return self.R
        else:
            return max(self.R*(1-(x-self.L)/self.L), 0.4*self.R)




if __name__ == '__main__':
    app = Nozzle()
    app.run()
    app.post_process(app.info_filename)
