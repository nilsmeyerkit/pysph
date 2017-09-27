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

# matplotlib (set up for server use)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain,
                              get_particle_array_beadchain_fiber)

from pysph.solver.application import Application
from pysph.solver.utils import load,remove_irrelevant_files
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme


class RVE(Application):
    """Generation of a mini RVE and evaluation of its fiber orientation
    tensor."""
    def create_scheme(self):
        """There is no scheme used in this application and equations are set up
        manually."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber0'], dim=2)

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
            default=1E8, help="Young's modulus"
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
            default=10000, help="Damping coefficient for artificial damping"
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim",
            default=2, help="Dimension of problem"
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
            default=1E5, help="Factor of mass scaling"
        )
        group.add_argument(
            "--volfrac", action="store", type=float, dest="vol_frac",
            default=0.5, help="Volume fraction of fibers in suspension."
        )
        group.add_argument(
            "--folgartucker", action="store_true", dest="folgartucker",
            default=False, help="Decides wether to plot Folgar Tucker solution."
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

        # The density can be scaled using the mass scaling factor. To account
        # for proper external forces, gravity is scaled just the other way.
        self.rho0 = self.options.rho0*self.options.scale_factor
        self.options.g = self.options.g/self.options.scale_factor

        # The fiber length is the aspect ratio times fiber diameter
        self.Lf = self.options.ar*self.dx

        # Determine channel width
        multiples = self.options.ar
        self.Ly = multiples*self.dx + 2*int(0.1*multiples)*self.dx

        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.0*self.Ly + self.dx

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
        self.pb = self.p0

        # The time is set to zero, if only postprocessing is required.
        if self.options.postonly:
            self.t = 0
        else:
            self.t = 10
        print("Simulated time is %g s"%self.t)

        fdx = self.dx
        dx2 = fdx/2

        _x = np.arange(dx2, self.Lx, fdx)
        self.nx = round(self.options.vol_frac*len(_x))
        if self.options.dim == 3:
            _z = np.arange(dx2, self.Ly, fdx)
            self.nz = round(self.options.vol_frac*len(_z))
        else:
            self.nz = 1

    def configure_scheme(self):
        names = ["fiber"+str(i) for i in range(self.nx*self.nz)]
        self.scheme.configure(dim=self.options.dim, fibers=names)
        self.scheme.configure(rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=self.A, I=self.I,
            J=self.J, E=self.options.E, D=self.options.D, dim=self.options.dim,
            scale_factor=self.options.scale_factor, gx=self.options.g,
            k=self.options.k)
        self.scheme.configure_solver(tf=self.t, vtk = self.options.vtk,
            N=100)
        #self.scheme.configure_solver(tf=self.t, pfreq=1, vtk = self.options.vtk)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.dx
        dx2 = fdx/2

        # Computation of each particles initial volume.
        volume = fdx**self.options.dim
        fiber_volume = self.dx**self.options.dim

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0
        fiber_mass = fiber_volume * self.rho0

        # assign unique ID (within fiber) to each fiber particle.
        fidx = range(0,self.options.ar)

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume
        fiber_V = 1./fiber_volume

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        if self.options.dim == 3:
            _z = np.arange(dx2, self.Ly, fdx)
        else:
            _z = np.array([0.5*self.Ly])
        fx,fy,fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position.
        indices = []
        fibers = []
        N = 0
        for xx in random.sample(list(_x), self.nx):
            for zz in random.sample(list(_z), self.nz):
                name = "fiber" + str(N)
                for i in range(len(fx)):
                    yy = 0.5*self.Ly

                    # vertical
                    if (fx[i] < xx+self.dx/2 and fx[i] > xx-self.dx/2 and
                        fy[i] < yy+self.Lf/2 and fy[i] > yy-self.Lf/2 and
                        fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
                        indices.append(i)

                # Generating fiber particle grid. Uncomment proper section for
                # horizontal or vertical alignment respectivley.

                # vertical fiber
                _fibx = np.array([xx])
                _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/4, self.dx)
                _fibz = np.array([zz])
                fibx,fiby,fibz = self.get_meshgrid(_fibx, _fiby, _fibz)

                # Finally create all particle arrays. Note that fluid particles are
                # removed in the area, where the fiber is placed.
                if self.options.dim == 2:
                    fibers.append(get_particle_array_beadchain_fiber(name=name,
                                x=fibx, y=fiby, m=fiber_mass, rho=self.rho0, h=self.h0,
                                lprev=self.dx, lnext=self.dx, phi0=np.pi, phifrac=2.0,
                                fidx=fidx, V=fiber_V))
                else:
                    fibers.append(get_particle_array_beadchain_fiber(name=name,
                                x=fibx, y=fiby, z=fibz, m=fiber_mass, rho=self.rho0,
                                h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
                                phifrac=2.0, fidx=fidx, V=fiber_V))
                N += 1
        print("Created %d fibers."%N)

        # Determine the size of dummy region
        ghost_extent = 3*fdx

        # Create the channel particles at the top
        _y = np.arange(self.Ly+dx2, self.Ly+dx2+ghost_extent, fdx)
        tx,ty,tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        bx,by,bz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the right
        _z = np.arange(-dx2, -dx2-ghost_extent, -fdx)
        _y = np.arange(dx2-ghost_extent, self.Ly+ghost_extent, fdx)
        rx,ry,rz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the left
        _z = np.arange(self.Ly+dx2, self.Ly+dx2+ghost_extent, fdx)
        _y = np.arange(dx2-ghost_extent, self.Ly+ghost_extent, fdx)
        lx,ly,lz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays (and for 3D cas also right and
        # left arrays)
        if self.options.dim ==2:
            cx = np.concatenate((tx, bx))
            cy = np.concatenate((ty, by))
        else:
            cx = np.concatenate((tx, bx, rx, lx))
            cy = np.concatenate((ty, by, ry, ly))
            cz = np.concatenate((tz, bz, rz, lz))


        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        if self.options.dim == 2:
            channel = get_particle_array_beadchain(name='channel',
                        x=cx, y=cy, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid = get_particle_array_beadchain(name='fluid',
                        x=fx, y=fy, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid.remove_particles(indices)
        else:
            channel = get_particle_array_beadchain(name='channel',
                        x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid = get_particle_array_beadchain(name='fluid',
                        x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid.remove_particles(indices)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles()))


        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.y[:]-self.Ly/2)

        # Upper and lower walls move uniformly depending on shear rate.
        N = channel.get_number_of_particles()-1
        for i in range(0,N):
            if channel.y[i] > self.Ly/2:
                y = self.Ly/2
            else:
                y = -self.Ly/2
            channel.u[i] = self.options.G*y

        return [fluid, channel] + fibers

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_tools(self):
        il = self.options.ar > 1
        return [FiberIntegrator(self.particles, self.scheme, self.Lx, il)]

    # def post_stage(self, current_time, dt, stage):
    #     self.nnps.update_domain()

    def get_meshgrid(self, xx, yy, zz):
        """This function is just a shorthand for the generation of meshgrids."""
        if self.options.dim == 2:
            x, y = np.meshgrid(xx, yy)
            x = x.ravel()
            y = y.ravel()
            z = 0.5*self.Ly*np.ones(np.shape(y))
        else:
            x, y, z = np.meshgrid(xx, yy, zz)
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
        return [x,y,z]

    def get_equivalent_aspect_ratio(self, aspect_ratio):
        """Jeffrey's equivalent aspect ratio (coarse approximation)
            H. L. Goldsmith and S. G. Mason
            CHAPTER 2 - THE MICRORHEOLOGY OF DISPERSIONS A2 - EIRICH, pp. 85–250.
            Academic Press, 1967.

        """
        return -0.0017*aspect_ratio**2+0.742*aspect_ratio


    def symm(self, A):
        '''
        This function computes the symmetric part of a fourth order Tensor A
        and returns a symmetric fourth order tensor S.
        '''
        # initial symmetric tensor with zeros
        S = np.zeros((3, 3, 3, 3))

        # Einsteins summation
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # sum of all permutations divided by 4!=24
                        S[i, j, k, l] = 1.0/24.0*(A[i, j, k, l]
                                                  + A[j, i, k, l]
                                                  + A[i, j, l, k]
                                                  + A[j, i, l, k]
                                                  + A[k, l, i, j]
                                                  + A[l, k, i, j]
                                                  + A[k, l, j, i]
                                                  + A[l, k, j, i]
                                                  + A[i, k, j, l]
                                                  + A[k, i, j, l]
                                                  + A[i, k, l, j]
                                                  + A[k, i, l, j]
                                                  + A[j, l, i, k]
                                                  + A[l, j, i, k]
                                                  + A[j, l, k, i]
                                                  + A[l, j, k, i]
                                                  + A[i, l, j, k]
                                                  + A[l, i, j, k]
                                                  + A[i, l, k, j]
                                                  + A[l, i, k, j]
                                                  + A[j, k, i, l]
                                                  + A[k, j, i, l]
                                                  + A[j, k, l, i]
                                                  + A[k, j, l, i])
        return S

    def generate_fourth_order_tensor(self, A):
        """This function utilizes a invariant based optimal fitting closure to
        generate a fourth order tensor from a second order tensor.
        Reference: Chung & Kwon paper about 'Invariant-based optimal fitting
        closure approximation for the numerical prediction of flow-induced fiber
        orientation'

        Input:
        A: Second order orientation tensor
        """

        # build second order orientation tensor in eigensystem representaion
        e1,e2,e3 = np.linalg.eigvals(A)

        # first invariant
        I = e1 + e2 + e3

        # second invariant
        II = e1*e2+e2*e3+e1*e3

        # third invariant
        III = e1*e2*e3

        # coefficients from Chung & Kwon paper
        C1 = np.zeros((1, 21))

        C2 = np.zeros((1, 21))

        C3 = np.array([[0.24940908165786E2,
                        -0.435101153160329E3,
                        0.372389335663877E4,
                        0.703443657916476E4,
                        0.823995187366106E6,
                        -0.133931929894245E6,
                        0.880683515327916E6,
                        -0.991630690741981E7,
                        -0.159392396237307E5,
                        0.800970026849796E7,
                        -0.237010458689252E7,
                        0.379010599355267E8,
                        -0.337010820273821E8,
                        0.322219416256417E5,
                        -0.257258805870567E9,
                        0.214419090344474E7,
                        -0.449275591851490E8,
                        -0.213133920223355E8,
                        0.157076702372204E10,
                        -0.232153488525298E5,
                        -0.395769398304473E10]])

        C4 = np.array([[-0.497217790110754E0,
                        0.234980797511405E2,
                        -0.391044251397838E3,
                        0.153965820593506E3,
                        0.152772950743819E6,
                        -0.213755248785646E4,
                        -0.400138947092812E4,
                        -0.185949305922308E7,
                        0.296004865275814E4,
                        0.247717810054366E7,
                        0.101013983339062E6,
                        0.732341494213578E7,
                        -0.147919027644202E8,
                        -0.104092072189767E5,
                        -0.635149929624336E8,
                        -0.247435106210237E6,
                        -0.902980378929272E7,
                        0.724969796807399E7,
                        0.487093452892595E9,
                        0.138088690964946E5,
                        -0.160162178614234E10]])

        C5 = np.zeros((1, 21))

        C6 = np.array([[0.234146291570999E2,
                        -0.412048043372534E3,
                        0.319553200392089E4,
                        0.573259594331015E4,
                        -0.485212803064813E5,
                        -0.605006113515592E5,
                        -0.477173740017567E5,
                        0.599066486689836E7,
                        -0.110656935176569E5,
                        -0.460543580680696E8,
                        0.203042960322874E7,
                        -0.556606156734835E8,
                        0.567424911007837E9,
                        0.128967058686204E5,
                        -0.152752854956514E10,
                        -0.499321746092534E7,
                        0.132124828143333E9,
                        -0.162359994620983E10,
                        0.792526849882218E10,
                        0.466767581292985E4,
                        -0.128050778279459E11]])

        # build matrix of coefficients by stacking vectors
        C = np.vstack((C1, C2, C3, C4, C5, C6))

        # compute parameters as fith order polynom based on invariants
        beta3 = (C[2, 0]+C[2, 1]*II+C[2, 2]*II**2++C[2, 3]*III+C[2, 4]*III**2
                 + C[2, 5]*II*III+C[2, 6]*II**2*III+C[2, 7]*II*III**2+C[2, 8]*II**3
                 + C[2, 9]*III**3+C[2, 10]*II**3*III+C[2, 11]*II**2*III**2
                 + C[2, 12]*II*III**3+C[2, 13]*II**4+C[2, 14]*III**4
                 + C[2, 15]*II**4*III+C[2, 16]*II**3*III**2+C[2, 17]*II**2*III**3
                 + C[2, 18]*II*III**4+C[2, 19]*II**5+C[2, 20]*III**5)

        beta4 = (C[3, 0]+C[3, 1]*II+C[3, 2]*II**2++C[3, 3]*III+C[3, 4]*III**2
                 + C[3, 5]*II*III+C[3, 6]*II**2*III+C[3, 7]*II*III**2+C[3, 8]*II**3
                 + C[3, 9]*III**3+C[3, 10]*II**3*III+C[3, 11]*II**2*III**2
                 + C[3, 12]*II*III**3+C[3, 13]*II**4+C[3, 14]*III**4
                 + C[3, 15]*II**4*III+C[3, 16]*II**3*III**2+C[3, 17]*II**2*III**3
                 + C[3, 18]*II*III**4+C[3, 19]*II**5+C[3, 20]*III**5)

        beta6 = (C[5, 0]+C[5, 1]*II+C[5, 2]*II**2++C[5, 3]*III+C[5, 4]*III**2
                 + C[5, 5]*II*III+C[5, 6]*II**2*III+C[5, 7]*II*III**2+C[5, 8]*II**3
                 + C[5, 9]*III**3+C[5, 10]*II**3*III+C[5, 11]*II**2*III**2
                 + C[5, 12]*II*III**3+C[5, 13]*II**4+C[5, 14]*III**4
                 + C[5, 15]*II**4*III+C[5, 16]*II**3*III**2+C[5, 17]*II**2*III**3
                 + C[5, 18]*II*III**4+C[5, 19]*II**5+C[5, 20]*III**5)

        beta1 = 3.0/5.0*(-1.0/7.0
                         + 1.0/5.0*beta3*(1.0/7.0+4.0/7.0*II+8.0/3.0*III)
                         - beta4*(1.0/5.0-8.0/15.0*II-14.0/15.0*III)
                         - beta6*(1.0/35.0
                                  - 24.0/105.0*III
                                  - 4.0/35.0*II
                                  + 16.0/15.0*II*III
                                  + 8.0/35.0*II**2))

        beta2 = 6.0/7.0*(1.0
                         - 1.0/5.0*beta3*(1.0+4.0*II)
                         + 7.0/5.0*beta4*(1.0/6.0-II)
                         - beta6*(-1.0/5.0+2.0/3.0*III+4.0/5.0*II-8.0/5.0*II**2))

        beta5 = -4.0/5.0*beta3-7.0/5.0*beta4-6.0/5.0*beta6*(1.0-4.0/3.0*II)

        # second order identy matrix
        delta = np.eye(3)

        # generate fourth order tensor with parameters and tensor algebra
        return (beta1*self.symm(np.einsum('ij,kl->ijkl', delta, delta))
                + beta2*self.symm(np.einsum('ij,kl->ijkl', delta, A))
                + beta3*self.symm(np.einsum('ij,kl->ijkl', A, A))
                + beta4*self.symm(np.einsum('ij,km,ml->ijkl', delta, A, A))
                + beta5*self.symm(np.einsum('ij,km,ml->ijkl', A, A, A))
                + beta6*self.symm(np.einsum('im,mj,kn,nl->ijkl', A, A, A, A)))

    def folgar_tucker_ode(self, A, t, ar, G, Ci=0):
        A = np.reshape(A,(3,3))
        lbd = (ar**2-1.0)/(ar**2+1.0)
        omega = np.array([[0.0, G/2, 0.0],[-G/2, 0.0, 0.0],[0.0, 0.0, 0.0]])
        D = np.array([[0.0, G/2, 0.0],[G/2, 0.0, 0.0],[0.0, 0.0, 0.0]])
        AA = self.generate_fourth_order_tensor(A)
        delta = np.eye(3)
        DADT = (np.einsum('ik,kj->ij', omega, A)
                -np.einsum('ik,kj->ij', A, omega)
                +lbd*(np.einsum('ik,kj->ij', D, A)
                      +np.einsum('ik,kj->ij', A, D)
                      -2*np.einsum('ijkl,kl->ij', AA, D))
                +2*Ci*G*(delta-3*A))
        return DADT.ravel()

    def post_process(self, info_fname):

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

            # extrating all arrays.
            directions = []
            for f in self.scheme.fibers:
                fiber = data['arrays'][f]

                # evaluate fiber orientation at center
                idx = 5
                px = fiber.rxnext[idx]
                py = fiber.rynext[idx]
                pz = fiber.rznext[idx]

                n = np.array([px, py, pz])
                p = n/np.linalg.norm(n, 2)
                directions.append(p)

            N = len(directions)
            a = np.zeros([3,3])
            for p in directions:
                for i in range(3):
                    for j in range(3):
                        a[i, j] += 1.0/N*(p[i]*p[j])
            A.append(a.ravel())

        tt = np.array(t)
        A0 = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
        are = self.get_equivalent_aspect_ratio(self.options.ar)
        Ci = 0.0
        print("Solving Folgar-Tucker equation with Ci=%f"%Ci)
        A_FT = odeint(self.folgar_tucker_ode,A0.ravel(),tt, atol=1E-15,
                                args=(are,self.options.G, Ci))

        # open new plot
        plt.figure()

        # plot Orientation tensor components
        if self.options.folgartucker:
            plt.plot(t, np.vstack(A), tt, A_FT, '-k')
        else:
            plt.plot(t, np.vstack(A))

        plt.legend(['$A_{11}$', '$A_{12}$', '$A_{13}$', '$A_{21}$', '$A_{22}$',
                    '$A_{23}$','$A_{31}$','$A_{32}$','$A_{33}$'])
        plt.title('Orientation Tensor')
        plt.xlabel('t [s]')
        plt.ylabel('Component value')

        # save figure
        fig = os.path.join(self.output_dir, 'orientation_tensor.eps')
        plt.savefig(fig, dpi=300)
        print("Orientation plot written to %s."% fig)

if __name__ == '__main__':
    app = RVE()
    app.run()
    app.post_process(app.info_filename)
