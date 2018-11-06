"""
################################################################################
Flow with fibers in a channel. There are different setups:
    Default:            2D shearflow with a single fiber - use --ar to set
                        aspect ratio
    Nearfield (ar=1):   2D fluid field around fiber (fiber is interpreted to be
                        perpendicular to 2D field.)
                        e.g: pysph run fiber.channel --ar 1 --width 20
                        --massscale 1E8 --G 0 --g 10 --openmp --holdcenter
                        --d 0.0002 --mu 1000
    dim=3 and g>0:      3D Poiseuille flow with moving fiber and obstacle fiber
                        (use smaller artificial damping, e.g. 100!)
                        e.g: pysph run fiber.channel --ar 11 --dim 3 --g 10
                                --G 0 --D 0.1 --vtk --massscale 1E8 --E 1E6
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
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':18})
rc('text', usetex=True)

# numpy and scipy
import numpy as np
from scipy.integrate import odeint

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain,
                              get_particle_array_beadchain_fiber)

from pysph.tools.interpolator import Interpolator

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files, FloatPBar
from pysph.solver.tools import FiberIntegrator

from pysph.sph.scheme import BeadChainScheme


# Jeffrey's equivalent aspect ratio (coarse approximation)
#   H. L. Goldsmith and S. G. Mason
#   CHAPTER 2 - THE MICRORHEOLOGY OF DISPERSIONS A2 - EIRICH, pp. 85–250.
#   Academic Press, 1967.
def get_equivalent_aspect_ratio(aspect_ratio):
    return [1.24*aspect_ratio/np.sqrt(np.log(aspect_ratio)),
            -0.0017*aspect_ratio**2+0.742*aspect_ratio]

# Jeffery's Equation for planar rotation of a rigid (theta=0)
def jeffery_ode(phi, t, ar, G):
    lbd = (ar**2-1.0)/(ar**2+1.0)
    return 0.5*G*(1.0+lbd*np.cos(2.0*phi))

class Channel(Application):
    def create_scheme(self):
        """The BeadChainScheme is used for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=2)

    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0001, help="Fiber diameter"
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
            "--dim", action="store", type=int, dest="dim",
            default=2, help="Dimension of problem"
        )
        group.add_argument(
            "--width", action="store", type=int, dest="width",
            default=None, help="Channel width (multiples of fiber diameter)"
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
            "--no-pb", action="store_true", dest="nopb",
            default=False, help="Disable background pressure"
        )
        group.add_argument(
            "--massscale", action="store", type=float, dest="scale_factor",
            default=None, help="Factor of mass scaling"
        )
        group.add_argument(
            "--fluidres", action="store", type=float, dest="fluid_res",
            default=1, help="Resolution of fluid particles relative to fiber."
        )
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=1.0, help="Number of half rotations."
        )
        group.add_argument(
            "--steps", action="store", type=int, dest="steps",
            default=None, help="Number of steps in inner loop."
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
        self.Lf = self.options.ar*self.dx

        # If a specific width is set, use this as multiple of dx to determine
        # the channel width. Otherwise use the fiber aspect ratio.
        multiples = self.options.width or self.options.ar
        self.Ly = multiples*self.dx + 2*int(0.1*multiples)*self.dx

        # Computation of a scale factor in a way that dt_cfl exactly matches
        # dt_viscous.
        a = self.h0*0.125*11/0.4
        #nu_needed = a*self.options.G*self.Ly/2
        nu_needed = (a*self.options.G*self.Ly/4
                     +np.sqrt(a/8*self.options.g*self.Ly**2
                              +(a/2)**2/4*self.options.G**2*self.Ly**2))

        # If there is no other scale scale factor provided, use automatically
        # computed factor.
        auto_scale_factor = self.options.mu/(nu_needed*self.options.rho0)
        print("Automatic scale factor is %d"%auto_scale_factor)
        self.scale_factor = self.options.scale_factor or auto_scale_factor

        # The density can be scaled using the mass scaling factor. To account
        # for proper external forces, gravity is scaled just the other way.
        self.rho0 = self.options.rho0*self.scale_factor
        self.options.g = self.options.g/self.scale_factor


        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.0*self.Ly + self.dx

        # The position of the fiber's center is set to the center of the
        # channel.
        self.x_fiber = 0.5*self.Lx
        self.y_fiber = 0.5*self.Ly
        self.z_fiber = 0.5*self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled density.
        self.nu = self.options.mu/self.rho0

        # damping from empirical guess
        self.D = self.options.D or 0.001*self.scale_factor

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
        if self.options.nopb:
            self.pb = 0.0
        else:
            self.pb = self.p0

        # The time is set to zero, if only postprocessing is required. For a
        # shear flow, it is set to the time for a full period of rotation
        # according to Jeffery's equation. For a Poiseuille flow, it is set to
        # the time to reach steady state for width = 20 and g = 10 to match
        # the FEM result in COMSOL for a single cylinder.
        if self.options.postonly:
            self.t = 0
        else:
            if self.options.G > 0.1:
                l = (self.options.ar+1.0/self.options.ar)
                self.t = self.options.rot*np.pi*l/self.options.G
            else:
                if self.options.dim == 3:
                    self.t = 0.2E-5*self.scale_factor
                else:
                    self.t = 1.5E-5*self.scale_factor
        print("Simulated time is %g s"%self.t)

    def configure_scheme(self):
        self.scheme.configure(rho0=self.rho0, c0=self.c0, nu=self.nu,
            p0=self.p0, pb=self.pb, h0=self.h0, dx=self.dx, A=self.A, I=self.I,
            J=self.J, E=self.options.E, D=self.D, dim=self.options.dim,
            gx=self.options.g)
        if self.options.dim == 3 and self.options.g > 0:
            self.scheme.configure(fibers=['fiber', 'obstacle'])
        # Return the particle list.
        self.scheme.configure_solver(tf=self.t, vtk = self.options.vtk,
            N=self.options.rot*200)
        #self.scheme.configure_solver(tf=self.t, pfreq=1, vtk = self.options.vtk)

    def create_particles(self):
        """Three particle arrays are created: A fluid, representing the polymer
        matrix, a fiber with additional properties and a channel of dummy
        particles."""

        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.options.fluid_res*self.dx
        dx2 = fdx/2

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        _z = np.arange(dx2, self.Ly, fdx)
        fx,fy,fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position. Uncomment proper section for
        # horizontal or vertical alignment respectivley.
        indices = []
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber

            # vertical
            if (fx[i] < xx+self.dx/2 and fx[i] > xx-self.dx/2 and
                fy[i] < yy+self.Lf/2 and fy[i] > yy-self.Lf/2 and
                fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
                indices.append(i)

            #horizontal
            # if (fx[i] < xx+self.Lf/2 and fx[i] > xx-self.Lf/2 and
            #     fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
            #     fz[i] < zz+self.dx/2 and fz[i] > zz-self.dx/2):
            #    indices.append(i)

            # obstacle
            if self.options.dim == 3 and self.options.g > 0:
                ox = self.x_fiber + 0.1*self.Lx
                oy = self.y_fiber - 0.1*self.Ly
                oz = self.z_fiber
                if (fx[i] < ox+self.dx/2 and fx[i] > ox-self.dx/2 and
                    fy[i] < oy+self.dx/2 and fy[i] > oy-self.dx/2 and
                    fz[i] < oz+self.Lf/2 and fz[i] > oz-self.Lf/2):
                    indices.append(i)

        # Generating fiber particle grid. Uncomment proper section for
        # horizontal or vertical alignment respectivley.

        # vertical fiber
        _fibx = np.array([xx])
        _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/4, self.dx)

        # horizontal fiber
        # _fibx = np.arange(xx-self.Lf/2+self.dx/2, xx+self.Lf/2+self.dx/4, self.dx)
        # _fiby = np.array([yy])

        _fibz = np.array([zz])
        fibx,fiby,fibz = self.get_meshgrid(_fibx, _fiby, _fibz)

        # obstacle fiber
        if self.options.dim == 3 and self.options.g > 0:
            _obsx = np.array([ox])
            _obsy = np.array([oy])
            _obsz = np.arange(oz-self.Lf/2+self.dx/2, oz+self.Lf/2+self.dx/4, self.dx)
            obsx,obsy,obsz = self.get_meshgrid(_obsx, _obsy, _obsz)

        # Determine the size of dummy region
        ghost_extent = 3*fdx/self.options.fluid_res

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


        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        if self.options.dim == 2:
            channel = get_particle_array_beadchain(name='channel',
                        x=cx, y=cy, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid = get_particle_array_beadchain(name='fluid',
                        x=fx, y=fy, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid.remove_particles(indices)
            fiber = get_particle_array_beadchain_fiber(name='fiber',
                        x=fibx, y=fiby, m=fiber_mass, rho=self.rho0, h=self.h0,
                        lprev=self.dx, lnext=self.dx, phi0=np.pi, phifrac=2.0,
                        fidx=fidx, V=fiber_V)
        else:
            channel = get_particle_array_beadchain(name='channel',
                        x=cx, y=cy, z=cz, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid = get_particle_array_beadchain(name='fluid',
                        x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0, V=V)
            fluid.remove_particles(indices)
            fiber = get_particle_array_beadchain_fiber(name='fiber',
                        x=fibx, y=fiby, z=fibz, m=fiber_mass, rho=self.rho0,
                        h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
                        phifrac=2.0, fidx=fidx, V=fiber_V)
            if self.options.dim == 3 and self.options.g > 0:
                obstacle = get_particle_array_beadchain_fiber(name='obstacle',
                            x=obsx, y=obsy, z=obsz, m=fiber_mass, rho=self.rho0,
                            h=self.h0, lprev=self.dx, lnext=self.dx, phi0=np.pi,
                            phifrac=2.0, fidx=fidx, V=fiber_V)

        # Print number of particles.
        print("Shear flow : nfluid = %d, nchannel = %d, nfiber = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fiber.get_number_of_particles()))

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles()==self.options.ar)
        if self.options.dim == 3 and self.options.g > 0:
            assert(obstacle.get_number_of_particles() == self.options.ar)

        # Tag particles to be hold, if requested.
        fiber.holdtag[:] = 0
        if self.options.holdcenter:
            idx = int(np.floor(self.options.ar/2))
            fiber.holdtag[idx] = 100
        if self.options.dim == 3 and self.options.g > 0:
            obstacle.holdtag[0] = 100
            obstacle.holdtag[-1] = 100

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.y[:]-self.Ly/2)
        fiber.u[:] = self.options.G*(fiber.y[:]-self.Ly/2)
        channel.u[:] = self.options.G*(channel.y[:]-self.Ly/2)

        # Return the particle list.
        if self.options.dim == 3 and self.options.g > 0:
            return [fluid, channel, fiber, obstacle]
        else:
            return [fluid, channel, fiber]

    def create_domain(self):
        """The channel has periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_tools(self):
        ud = not self.options.holdcenter
        if self.options.steps:
            print("Using %d inner steps."%self.options.steps)
        return [FiberIntegrator(self.particles, self.scheme, self.domain,
                                innerloop=self.options.ar>1, updates=ud,
                                steps=self.options.steps)]

    def get_meshgrid(self, xx, yy, zz):
        """This function is just a shorthand for the generation of meshgrids."""
        if self.options.dim == 2:
            x, y = np.meshgrid(xx, yy)
            x = x.ravel()
            y = y.ravel()
            z = self.z_fiber*np.ones(np.shape(y))
        else:
            x, y, z = np.meshgrid(xx, yy, zz)
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
        return [x,y,z]

    def _plot_streamlines(self):
        """This function plots streamlines and the pressure field. It
         interpolates the properties from particles using the kernel."""

        # lenght factor m --> mm
        factor = 1000

        # Interpolation grid
        X = np.linspace(0,self.Lx,400)
        Y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(X,Y)

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
        vmag = factor*np.sqrt(u**2 + v**2)

        if self.options.ar == 1:
            upper = 0.025
        else:
            upper = np.max(vmag)

        # open new figure
        plt.figure()
        # configuring color map
        cmap = plt.cm.viridis
        levels = np.linspace(0, upper, 30)

        # velocity contour (ugly solution against white lines:
        # repeat plots....)
        plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=upper, vmin=0)
        plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=upper, vmin=0)
        vel = plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=upper, vmin=0)
        # streamlines
        y_start = np.linspace(0.0, self.Ly*factor, 20)
        x_start = np.zeros_like(y_start)
        start_points = np.array(list(zip(x_start, y_start)))
        stream = plt.streamplot(X*factor,Y*factor,u,v,
                                start_points=start_points,
                                color='k',
                                density=35)
        # fiber
        plt.scatter(fx*factor,fy*factor, color='w')

        # set labels
        if self.options.ar == 1:
            cbar = plt.colorbar(vel,
                        ticks=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                        shrink=0.5)
            plt.axis('scaled')
        else:
            cbar = plt.colorbar(vel)
            plt.axis('equal')
        cbar.set_label('Velocity in mm/s', labelpad=20.0)
        plt.axis((0,factor*self.Lx,0,factor*self.Ly))
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm ')
        plt.tight_layout()

        # save plot
        fig = os.path.join(self.output_dir, 'streamplot.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Streamplot written to %s."% fig)


        upper = np.max(p)
        lower = np.min(p)

        # open new plot
        plt.figure()
        #configuring new color map
        cmap = plt.cm.viridis
        levels = np.linspace(lower, upper, 30)

        # pressure contour(ungly solution against white lines:
        # repeat plots....)
        plt.contourf(x*factor,y*factor, p, levels=levels,
                 cmap=cmap,  vmin=lower, vmax=upper)
        plt.contourf(x*factor,y*factor, p, levels=levels,
                 cmap=cmap,  vmin=lower, vmax=upper)
        pres = plt.contourf(x*factor,y*factor, p, levels=levels,
                 cmap=cmap,  vmin=lower, vmax=upper)

        # fiber
        plt.scatter(fx*factor,fy*factor, color='w')

        # set labels
        cbar = plt.colorbar(pres, label='Pressure in Pa')
        plt.axis('equal')
        plt.axis((0,factor*self.Lx,0,factor*self.Ly))
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm')
        plt.tight_layout()


        # save plot
        p_fig = os.path.join(self.output_dir, 'pressure.pdf')
        plt.savefig(p_fig, dpi=300, bbox_inches='tight')
        print("Pressure written to %s."% p_fig)

        return[fig, p_fig]

    def _plot_inlet_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the periodic boundary. If
        the fiber has only a single particle, this is interpreted as flow around
        a fiber cylinder and the coresponding FEM solution is plotted as well.
        """
        # length factor m --> mm
        factor = 1000

        # Extract requested output - default is last output.
        output = self.output_files[step_idx]
        data = load(output)

        # Generate meshgrid for interpolation.
        x = np.array([0])
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # solution for undisturbed velocity field.
        u_exact = (self.options.G * (y - self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                            (y-self.Ly/2)**2-(self.Ly/2)**2))

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u*factor, y*factor , '-k')

        # undisturbed solution
        plt.plot(u_exact*factor, y*factor, ':k')

        # labels
        plt.xlabel('Velocity $v_1$ in mm/s')
        plt.ylabel('$x_2$ in mm')
        plt.grid()
        plt.legend(['SPH Simulation', 'No obstacle'])
        plt.tight_layout()

        # save figure
        fig = os.path.join(self.output_dir, 'inlet_velocity.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Inlet velocity plot written to %s."% fig)

        return (fig)

    def _plot_center_velocity(self, step_idx=-1):
        """This function plots the velocity profile at the center across the
        particle. If the fiber has only a single particle, this is interpreted
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
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)

        # interpolation of velocity field.
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')

        # open new plot
        plt.figure()

        # SPH solution
        plt.plot(u*factor, y*factor , '-k')

        # labels
        plt.xlabel('Velocity $v_1$ in mm/s')
        plt.ylabel('$x_2$ in mm')
        plt.grid()
        plt.tight_layout()

        # save figure
        fig = os.path.join(self.output_dir, 'center_velocity.pdf')
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Center velocity plot written to %s."% fig)

        return (fig)

    def _plot_pressure_centerline(self):
        """This function plots the pressure profile along a centerline for a
        single particle."""

        # length factor m --> mm
        factor = 1000

        # Generate meshgrid for interpolation.
        x = np.linspace(0,self.Lx,200)
        y = np.array([self.Ly/2])
        x,y = np.meshgrid(x,y)

        # Set a number of last solutions to average from.
        N = 10

        # averaging the pressure interpolation for N last solutions.
        p = np.zeros((200,))
        for output in self.output_files[-(1+N):-1]:
            data = load(output)
            interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
            interp.update_particle_arrays(list(data['arrays'].values()))
            p += interp.interpolate('p')/N

        # FEM solution for disturbed velocity field (ar=1, g=10, G=0, width=20)
        x_fem = np.array([0.00010,0.00029,0.00048,0.00067,0.00086,0.00106,
                        0.00125,0.00144,0.00163,0.00182,0.00202,0.00221,0.00240,
                        0.00259,0.00278,0.00298,0.00317,0.00336,0.00355,0.00374,
                        0.00394,0.00413,0.00432,0.00451,0.00470,0.00480,0.00490,
                        0.00509,0.00528,0.00547,0.00566,0.00586,0.00605,0.00624,
                        0.00643,0.00662,0.00682,0.00701,0.00720,0.00739,0.00758,
                        0.00778,0.00797,0.00816,0.00835,0.00854,0.00874,0.00893,
                        0.00912,0.00931,0.00950])
        p_fem = np.array([0,1,2,3,3,4,5,5,6,7,7,8,9,9,10,11,12,14,15,17,21,26,
                        36,59,168,np.nan,-169,-58,-35,-25,-20,-17,-15,-13,-11,
                        -10,-9,-9,-8,-7,-6,-6,-5,-4,-4,-3,-2,-2,-1,-1,0])

        # open new plot
        plt.figure()

        # plot SPH solution and FEM solution
        plt.plot(x[0,:]*factor, p, '-k', x_fem*factor, p_fem, '--k')

        # labels
        plt.legend(['SPH Simulation','FEM Result'], loc='upper right')
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('Pressure in Pa')
        plt.grid()
        plt.tight_layout()

        # save figure
        pcenter_fig = os.path.join(self.output_dir, 'pressure_centerline.pdf')
        plt.savefig(pcenter_fig, dpi=300, bbox_inches='tight')
        print("Pressure written to %s."% pcenter_fig)

        return pcenter_fig

    def _plot_history(self):
        """This function create all plots employing a iteration over all time
        steps. """

        # length factor m --> mm
        factor = 1000

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
        for i,fname in enumerate(output_files):
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
            dxx = fiber.x[0]-fiber.x[-1]
            dyy = fiber.y[0]-fiber.y[-1]
            a = np.arctan(dxx/(dyy+0.01*self.h0)) + N*np.pi
            if len(angle) > 0 and a - angle[-1] > 3:
                N -= 1
                a -= np.pi
            elif len(angle) > 0 and a-angle[-1] < -3:
                N += 1
                a += np.pi

            # count rotations
            if a-M*np.pi > np.pi:
                T.append(t[-1]-t0)
                t0 = t[-1]
                M += 1
            angle.append(a)

            # computation of squared velocity and masses from density and volume
            v_fiber = fiber.u**2 + fiber.v**2 + fiber.w**2
            v_fluid = fluid.u**2 + fluid.v**2 + fluid.w**2
            m_fiber = fiber.rho/fiber.V
            m_fluid = fluid.rho/fluid.V
            m_channel = channel.rho/channel.V

            # appending volume, density, mass, pressure and kinetic energy
            volume.append(np.sum(1/fiber.V)+np.sum(1/fluid.V)+np.sum(1/channel.V))
            rho.append(np.sum(fiber.rho)+np.sum(fluid.rho)+np.sum(channel.rho))
            m.append(np.sum(m_fiber)+np.sum(m_fluid)+np.sum(m_channel))
            E_p.append(np.sum(fiber.p/fiber.V)
                        +np.sum(fluid.p/fluid.V)
                        +np.sum(channel.p/channel.V))
            E_kin.append(0.5*np.dot(m_fiber,v_fiber)
                        +0.5*np.dot(m_fluid,v_fluid))

            # extract reaction forces at hold particles
            idx = np.argwhere(fiber.holdtag==100)
            if len(idx) > 0 and self.options.ar == 1:
                Fwx.append(fiber.Fwx[idx][0])
                Fwy.append(fiber.Fwy[idx][0])
                Fwz.append(fiber.Fwz[idx][0])
                Fx.append(fiber.Fx[idx][0])
                Fy.append(fiber.Fy[idx][0])
                Fz.append(fiber.Fz[idx][0])

        bar.finish()

        # evaluate roation statistics
        if self.options.G > 0:
            print(T)
            T_mean = np.mean(T)
            T_std  = np.std(T)
            are = get_equivalent_aspect_ratio(self.options.ar)
            l_cox = (are[0]+1.0/are[0])
            l_gmason = (are[1]+1.0/are[1])
            T_cox = np.pi*l_cox/self.options.G
            T_gmason = np.pi*l_gmason/self.options.G
            l = (self.options.ar+1.0/self.options.ar)
            T_jef = np.pi*l/self.options.G
            print("Rotational statistics for %d half rotations:"%self.options.rot)
            print("*Mean: %f"%T_mean)
            print("*Standard Deviation: %f"%T_std)
            print("*Jeffery: %f"%T_jef)
            print("*Jeffery(Cox equivalent): %f"%T_cox)
            print("*Jeffery(Goldsmith/Mason equivalent): %f"%T_gmason)

        # open new plot
        plt.figure()

        # plot end points
        plt.plot(np.array(x_begin)*factor, np.array(y_begin)*factor, '-ok', markersize=3)
        plt.plot(np.array(x_end)*factor, np.array(y_end)*factor, '-xk', markersize=3)

        # set equally scaled axis to not distort the orbit
        plt.axis('equal')
        plt.xlabel('$x_1$ in mm')
        plt.ylabel('$x_2$ in mm')
        plt.grid()
        plt.tight_layout()

        # save plot of orbit
        orbfig = os.path.join(self.output_dir, 'orbitplot.pdf')
        plt.savefig(orbfig, dpi=300, bbox_inches='tight')
        print("Orbitplot written to %s."% orbfig)

        # Integrate Jeffery's solution
        print("Solving Jeffery's ODE")
        t = np.array(t)
        phi0 = angle[0]
        are = get_equivalent_aspect_ratio(self.options.ar)
        angle_jeffery = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(self.options.ar,self.options.G))
        angle_jeffery_cox = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(are[0],self.options.G))
        angle_jeffery_gmason = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(are[1],self.options.G))

        # constraint between -pi/2 and pi/2
        #angle_jeffery = (angle_jeffery+np.pi/2.0)%np.pi-np.pi/2.0

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t, angle, 'ok', markersize=3)
        #plt.plot(t, angle_jeffery_cox, '-.k')
        plt.plot(t, angle_jeffery_gmason, '--k')
        plt.plot(t, angle_jeffery, '-k')

        # labels
        plt.xlabel('Time $t$ in s')
        plt.ylabel('Angle $\phi$')
        plt.legend(['SPH Simulation', 'Jeffery (equiv.)', 'Jeffery'])
        plt.grid()
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,x2,0,y2))
        plt.tight_layout()

        # save figure
        angfig = os.path.join(self.output_dir, 'angleplot.pdf')
        plt.savefig(angfig, dpi=300, bbox_inches='tight')
        print("Angleplot written to %s."% angfig)

        # save angles as *.csv file
        csv_file = os.path.join(self.output_dir, 'angle.csv')
        angle_jeffery = np.reshape(angle_jeffery,(angle_jeffery.size,))
        np.savetxt(csv_file, (t, angle, angle_jeffery), delimiter=',')

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
        print("Energyplot written to %s."% engfig)

        # open new plot
        plt.figure()

        # plot relative mass, volume and density
        plt.plot(t, np.array(m)/m[0], '-k',
                 t, np.array(volume)/volume[0], '--k',
                 t, np.array(rho)/rho[0], ':k')

        # labels
        plt.xlabel('Time $t$ in s')
        plt.ylabel('Relative value')
        plt.legend(['Mass', 'Volume', 'Density'])
        plt.grid()
        plt.tight_layout()

        # save figure
        mfig = os.path.join(self.output_dir, 'massplot.pdf')
        plt.savefig(mfig, dpi=300, bbox_inches='tight')
        print("Mass plot written to %s."% mfig)

        # hard-coded solutions for total reaction forces and viscous reaction
        # forces from FEM. (ar=1, g=10, G=0, width=20)
        t_fem = np.array([0,5.00E-07,1.00E-06,1.50E-06,2.00E-06,2.50E-06,
                          3.00E-06,350E-06,4.00E-06,4.50E-06,5.00E-06,
                          5.50E-06,6.00E-06,6.50E-06,7.00E-06,7.50E-06,
                          8.00E-06,8.50E-06,9.00E-06,9.50E-06,1.00E-05,
                          1.05E-05,1.10E-05,1.15E-05,1.20E-05,1.25E-05,
                          1.30E-05,1.35E-05,1.40E-05,1.45E-05,1.50E-05])
        Fv_fem = np.array([2.62E-06,0.0013741,0.0023354,0.0031959,0.0039842,
                          0.0047131,0.005388,0.0060123,0.0065885,0.0071183,
                          0.0076104,0.0080643,0.0084792,0.0088653,0.0092193,
                          0.0095403,0.0098401,0.010115,0.010364,0.010596,0.010809,
                          0.011002,0.011182,0.011347,0.011496,0.011635,0.011763,
                          0.011878,0.011986,0.012085,0.012174])
        F_fem = np.array([-8.51E-05,0.0027891,0.004684,0.0063805,0.0079349,
                          0.0093724,0.010703,0.011935,0.013071,0.014116,
                          0.015087,0.015982,0.0168,0.017562,0.01826,0.018893,
                          0.019484,0.020026,0.020517,0.020975,0.021395,0.021775,
                          0.02213,0.022456,0.02275,0.023025,0.023277,0.023504,
                          0.023717,0.023912,0.024088])

        # applying appropriate scale factors
        t = np.array(t)/self.scale_factor*1000

        # Reaction force is plotted only for flow around single cylindrical
        # fiber
        if self.options.ar == 1:
            # open new plot
            plt.figure()

            # plot computed reaction force, total FEM force and viscous FEM
            # force
            plt.plot(t, Fx, '-k',
                     #t, Fwx, '.k',
                     t_fem, F_fem, '--k',
                     t_fem, Fv_fem, ':k')

            # labels
            plt.xlabel('Time $t$ in s')
            plt.ylabel('Force per fiber length in N/m')
            plt.legend(['SPH total force',
                        'FEM total force',
                        'FEM viscous force'],
                        loc='lower right')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((0,x2,0,y2))
            plt.grid()
            plt.tight_layout()

            # save figure
            forcefig = os.path.join(self.output_dir, 'forceplot.pdf')
            plt.savefig(forcefig, dpi=300, bbox_inches='tight')
            print("Reaction Force plot written to %s."% forcefig)
            return [orbfig, angfig, engfig, forcefig]
        else:
            return [orbfig, angfig, engfig]


    def post_process(self, info_fname):
        if len(self.output_files) == 0:
            return

        [streamlines, pressure] = self._plot_streamlines()
        if self.options.ar == 1:
            pressure_centerline = self._plot_pressure_centerline()
            center_velocity = self._plot_center_velocity()
        history = self._plot_history()
        inlet = self._plot_inlet_velocity()


if __name__ == '__main__':
        app = Channel()
        app.run()
        app.post_process(app.info_filename)
