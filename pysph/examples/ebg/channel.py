"""Shear flow involving a single fiber rotating. (10 mins)
"""
# general imports
import os
import smtplib
import json

# profiling
import cProfile
import pstats

# matplotlib (set up for server use)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# numpy and scipy
import numpy as np
from scipy.integrate import odeint

# mail for notifications
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# PySPH imports
from pysph.base.config import get_config
from pysph.base.nnps import DomainManager, LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline

from pysph.tools.interpolator import Interpolator

from pysph.solver.application import Application
from pysph.solver.utils import load,remove_irrelevant_files
from pysph.solver.vtk_output import dump_vtk
from pysph.solver.output import output_formats
from pysph.solver.solver import Solver

from pysph.sph.integrator import PECIntegrator, EulerIntegrator, EPECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep, EulerStep, EBGStep
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (SummationDensity,
    StateEquation, MomentumEquationPressureGradient,
    MomentumEquationArtificialViscosity,
    MomentumEquationViscosity, MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity)
from pysph.sph.ebg.fiber import (Tension, Bending, Vorticity, Friction, Damping,
    HoldPoints, EBGVelocityReset, ArtificialDamping, VelocityGradient)

# Changes
#  - CubicSpline
#  - Background Pressure


# Jeffrey's equivalent aspect ratio (coarse approximation)
def get_equivalent_aspect_ratio(aspect_ratio):
    return -0.0017*aspect_ratio**2+0.742*aspect_ratio

# Jeffery's Equation
def jeffery_ode(phi, t, ar_equiv, G):
    lbd = (ar_equiv**2-1.0)/(ar_equiv**2+1.0)
    return 0.5*G*(1.0+lbd*np.cos(2.0*phi))

# Shear flow between two moving plates with shear rate G.
class Channel(Application):
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
            default=1E11, help="Young's modulus"
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
            default=10000, help="Damping coefficient"
        )
        group.add_argument(
            "--dim", action="store", type=float, dest="dim",
            default=2, help="Dimension of problem"
        )
        group.add_argument(
            "--mail", action="store", type=str, dest="mail",
            default=None, help="Set notification e-mail adress."
        )
        group.add_argument(
            "--width", action="store", type=float, dest="width",
            default=None, help="Channel width"
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
            "--postonly", action="store", type=bool, dest="postonly",
            default=False, help="Set time to zero and postprocess only."
        )
        group.add_argument(
            "--massscale", action="store", type=float, dest="scale_factor",
            default=1E5, help="Factor of mass scaling"
        )
        group.add_argument(
            "--fluidres", action="store", type=float, dest="fluid_res",
            default=1, help="Resolution of fluid particles relative to fiber."
        )


    def consume_user_options(self):
        # Numerical setup
        self.dx = self.options.d
        self.h0 = self.options.fluid_res*self.dx

        self.options.g = self.options.g/self.options.scale_factor
        self.options.rho0 = self.options.rho0*self.options.scale_factor

        # fiber length
        self.Lf = self.options.ar*self.dx
        # channel dimensions
        width = self.Lf + 2*int(0.1*self.options.ar)*self.dx
        self.Ly = self.options.width or width
        self.Lx = 2.0*self.Ly + self.dx
        # fiber position
        self.x_fiber = 1/2*self.Lx
        self.y_fiber = self.Ly/2
        self.z_fiber = self.Ly/2

        # fluid properties
        self.nu = self.options.mu/self.options.rho0

        if self.options.dim == 2:
            self.A = self.dx
            self.I = self.dx**3/12
            mass = 3*self.options.rho0*self.dx*self.A
            self.J = 1/12*mass*(self.dx**2 + (3*self.dx)**2)
        else:
            R = self.dx/2
            self.A = np.pi*R**2
            self.I = np.pi*R**4/4.0
            mass = 3*self.options.rho0*self.dx*self.A
            self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.dx)**2

        # computed properties
        self.Vmax = (self.options.G*self.Ly/2
                    + self.options.g/(2*self.nu)*self.Ly**2/4)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.options.rho0
        self.pb = self.p0

        # time
        are = get_equivalent_aspect_ratio(self.options.ar)
        if self.options.postonly:
            self.t = 0
        else:
            if self.options.G > 0.1:
                self.t = 2.0*np.pi*(are+1.0/are)/self.options.G
            else:
                self.t = 1000
        print("Simulated time is %g s"%self.t)

        # Setup time step
        dt_cfl = 0.4 * self.h0/(self.c0 + self.Vmax)
        dt_viscous = 0.125 * self.h0**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h0/(self.options.g+0.001))
        dt_tension = 0.5*self.h0*np.sqrt(self.options.rho0/self.options.E)
        dt_bending = 0.5*self.h0**2*np.sqrt(self.options.rho0*self.A/(self.options.E*2*self.I))
        print("dt_cfl: %g"%dt_cfl)
        print("dt_viscous: %g"%dt_viscous)
        print("dt_force: %g"%dt_force)
        print("dt_tension: %g"%dt_tension)
        print("dt_bending: %g"%dt_bending)

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.fiber_dt = min(dt_tension, dt_bending)
        print("Time step ratio is %g"%(self.dt/self.fiber_dt))

    def create_scheme(self):
        return None

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
        Lx = self.Lx
        Ly = self.Ly
        dx2 = self.options.fluid_res*self.dx/2
        _x = np.arange(dx2, Lx, self.options.fluid_res*self.dx)

        # create the fluid particles
        _y = np.arange(dx2, Ly, self.options.fluid_res*self.dx)
        _z = np.arange(dx2, Ly, self.options.fluid_res*self.dx)
        fx,fy,fz = self.get_meshgrid(_x, _y, _z)

        # remove particles at fiber position
        indices = []
        for i in range(len(fx)):
            # vertical
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber
            # vertical
            # if (fx[i] < xx+0.5*dx2 and fx[i] >= xx-0.5*dx2 and
            #     fy[i] < yy+self.Lf/2+0.5*dx2 and fy[i] >= yy-self.Lf/2-0.5*dx2
            #     and fz[i] <= zz+dx2 and fz[i] >= zz-dx2):
            #     indices.append(i)

            #horizontal
            if (fx[i] < xx+self.Lf/2 and fx[i] > xx-self.Lf/2 and
               fy[i] < yy+self.dx/2 and fy[i] > yy-self.dx/2 and
               fz[i] <= zz+self.dx/2 and fz[i] >= zz-self.dx/2):
               indices.append(i)


        # create fiber
        # fibx = fx[indices]
        # fiby = fy[indices]
        # fibz = fz[indices]

        # vertical fiber
        # _fibx = np.array([xx])
        # _fiby = np.arange(yy-self.Lf/2+self.dx/2, yy+self.Lf/2+self.dx/2, self.dx)

        # horizontal fiber
        _fibx = np.arange(xx-self.Lf/2+self.dx/2, xx+self.Lf/2+self.dx/4, self.dx)
        _fiby = np.array([yy])


        _fibz = np.array([zz])
        fibx, fiby = np.meshgrid(_fibx, _fiby)
        fibx = fibx.ravel()
        fiby = fiby.ravel()
        fibz = fiby.ravel()

        ghost_extent = 5*self.options.fluid_res*self.dx
        # create the channel particles at the top
        _y = np.arange(Ly+dx2, Ly+dx2+ghost_extent, self.options.fluid_res*self.dx)
        tx,ty,tz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -self.options.fluid_res*self.dx)
        bx,by,bz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the right
        _z = np.arange(-dx2, -dx2-ghost_extent, -self.options.fluid_res*self.dx)
        _y = np.arange(dx2-ghost_extent, Ly+ghost_extent, self.options.fluid_res*self.dx)
        rx,ry,rz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the left
        _z = np.arange(Ly+dx2, Ly+dx2+ghost_extent, self.options.fluid_res*self.dx)
        _y = np.arange(dx2-ghost_extent, Ly+ghost_extent, self.options.fluid_res*self.dx)
        lx,ly,lz = self.get_meshgrid(_x, _y, _z)

        # concatenate the top and bottom arrays
        if self.options.dim ==2:
            cx = np.concatenate((tx, bx))
            cy = np.concatenate((ty, by))
        else:
            cx = np.concatenate((tx, bx, rx, lx))
            cy = np.concatenate((ty, by, ry, ly))
            cz = np.concatenate((tz, bz, rz, lz))

        # create the arrays
        if self.options.dim == 2:
            channel = get_particle_array(name='channel', x=cx, y=cy)
            fluid = get_particle_array(name='fluid', x=fx, y=fy)
            fluid.remove_particles(indices)
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby)
        else:
            channel = get_particle_array(name='channel', x=cx, y=cy, z=cz)
            fluid = get_particle_array(name='fluid', x=fx, y=fy, z=fz)
            fluid.remove_particles(indices)
            fiber = get_particle_array(name='fiber', x=fibx, y=fiby, z=fibz)


        print("Shear flow : nfluid = %d, nchannel = %d, nfiber = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles(),
            fiber.get_number_of_particles()))

        assert(fiber.get_number_of_particles()==self.options.ar)

        # add requisite variables needed for this formulation
        for name in ('V', 'wf','uf','vf','wg','wij','vg','ug',
                     'awhat', 'avhat','auhat', 'vhat', 'what', 'uhat', 'vmag2',
                     'arho', 'phi0', 'omegax', 'omegay', 'omegaz',
                     'holdtag', 'eu', 'ev', 'ew', 'testx', 'testy', 'testz',
                     'dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz','dwdx',
                     'dwdy', 'dwdz'):
            fluid.add_property(name)
            channel.add_property(name)
            fiber.add_property(name)
        for name in ('lprev', 'lnext', 'phi0', 'xcenter',
                     'ycenter','rxnext', 'rynext', 'rznext', 'rnext', 'rxprev',
                     'ryprev', 'rzprev', 'rprev'):
            fiber.add_property(name)

        # set the output property arrays
        fluid.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid','omegax', 'omegay', 'omegaz'])
        channel.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid','omegax', 'omegay', 'omegaz'])
        fiber.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'p',
                        'pid', 'holdtag', 'gid','ug', 'vg', 'wg'])

        # volume is set as dx^2
        volume = (self.options.fluid_res*self.dx)**self.options.dim
        fiber_volume = self.dx**self.options.dim

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.options.rho0
        channel.m[:] = volume * self.options.rho0
        fiber.m[:] = fiber_volume * self.options.rho0

        # set initial distances and angles
        fiber.lprev[:] = self.dx
        fiber.lnext[:] = self.dx
        fiber.phi0[:] = np.pi

        # tag particles to be hold
        fiber.holdtag[:] = 0
        if self.options.holdcenter:
            idx = int(np.floor(self.options.ar/2))
            fiber.holdtag[idx] = 100

        # Set the default rho.
        fluid.rho[:] = self.options.rho0
        channel.rho[:] = self.options.rho0
        fiber.rho[:] = self.options.rho0

        # inverse volume (necessary for transport velocity equations)
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume
        fiber.V[:] = 1./fiber_volume

        # smoothing lengths
        fluid.h[:] = self.h0
        channel.h[:] = self.h0
        fiber.h[:] = self.h0/self.options.fluid_res

        # initial velocities
        fluid.u[:] = (self.options.G*(fluid.y[:]-self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                        (fluid.y[:]-self.Ly/2)**2-(self.Ly/2)**2))
        fiber.u[:] = (self.options.G*(fiber.y[:]-self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                        (fiber.y[:]-self.Ly/2)**2-(self.Ly/2)**2))
        channel.u[:] = (self.options.G*(channel.y[:]-self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                        (channel.y[:]-self.Ly/2)**2-(self.Ly/2)**2))

        # return the particle list
        return [fluid, channel, fiber]

    def create_equations(self):
        all = ['fluid', 'channel', 'fiber']
        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=all),
                    SummationDensity(dest='fiber', sources=all),
                ]
            ),
            Group(
                equations=[
                    Vorticity(dest='fiber', sources=all),
                    Vorticity(dest='fluid', sources=all),
                    VelocityGradient(dest='fiber', sources=all),
                    VelocityGradient(dest='fluid', sources=all),
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                    rho0=self.options.rho0, b=1.0),
                    StateEquation(dest='fiber', sources=None, p0=self.p0,
                                    rho0=self.options.rho0, b=1.0),
                    SetWallVelocity(dest='channel', sources=['fluid', 'fiber']),
                ],
            ),
            Group(
                equations=[
                    SolidWallPressureBC(dest='channel',
                                        sources=['fluid', 'fiber'],
                                        b=1.0, rho0=self.options.rho0, p0=self.p0),
                ],
            ),
            Group(
                equations=[
                    Friction(dest='fiber', sources=['fiber'], J=self.J,
                                A=self.A, mu=self.options.mu, d=self.options.d,
                                ar=self.options.ar),
                    MomentumEquationPressureGradient(dest='fluid', sources=all,
                                        pb=self.pb, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationPressureGradient(dest='fiber', sources=all,
                                        pb=0.0, tdamp=0.0,
                                        gx=self.options.g),
                    MomentumEquationViscosity(dest='fluid',
                                        sources=['fluid', 'fiber'], nu=self.nu),
                    MomentumEquationViscosity(dest='fiber',
                                        sources=['fluid', 'fiber'], nu=self.nu),
                    SolidWallNoSlipBC(dest='fluid',
                                        sources=['channel',], nu=self.nu),
                    SolidWallNoSlipBC(dest='fiber',
                                        sources=['channel'], nu=self.nu),
                    MomentumEquationArtificialStress(dest='fluid',
                                        sources=['fluid', 'fiber']),
                    MomentumEquationArtificialStress(dest='fiber',
                                         sources=['fiber', 'fiber']),
                ],
            ),
            Group(
                equations=[
                    HoldPoints(dest='fiber', sources=None, tag=100),
                    EBGVelocityReset(dest='fiber', sources=None),
                ]
            ),
        ]
        return equations

    def _configure(self):
        super(Channel, self)._configure()
        self.fiber_integrator = EulerIntegrator(fiber=EBGStep())
        kernel = QuinticSpline(dim=self.options.dim)
        equations = [
                        Group(
                            equations=[
                                Tension(dest='fiber',
                                        sources=['fiber'],
                                        ea=self.options.E*self.A),
                                Bending(dest='fiber',
                                        sources=['fiber'],
                                        ei=self.options.E*self.I),
                                ArtificialDamping(dest='fiber',
                                        sources=None,
                                        d = self.options.D),
                            ],
                        ),
                        Group(
                            equations=[
                                HoldPoints(dest='fiber', sources=None, tag=100),
                            ]
                        ),
                    ]
        particles = [p for p in self.particles if p.name == 'fiber']
        domain = DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)
        nnps = LinkedListNNPS(dim=self.options.dim, particles=particles,
                        radius_scale=kernel.radius_scale, domain=domain,
                        fixed_h=False, cache=False, sort_gids=False)
        acceleration_eval = AccelerationEval(
                    particle_arrays=particles,
                    equations=equations,
                    kernel=kernel,
                    mode='serial')
        comp = SPHCompiler(acceleration_eval, self.fiber_integrator)
        config = get_config()
        config.use_openmp = False
        comp.compile()
        config.use_openmp = True
        acceleration_eval.set_nnps(nnps)
        self.fiber_integrator.set_nnps(nnps)

    def create_solver(self):
        # Setting up the default integrator for fiber particles
        kernel = QuinticSpline(dim=self.options.dim)
        integrator = EPECIntegrator(fluid=TransportVelocityStep(),
                                    fiber=TransportVelocityStep())
        solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
                         tf=self.t, pfreq=int(self.t/(100*self.dt)),
                        vtk=self.options.vtk)
        # solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
        #                  tf=self.t, pfreq=1, vtk=True)
        return solver

    def post_stage(self, current_time, dt, stage):
        # 1) predictor
        # 2) post stage 1:
        if stage == 1:
            N = int(np.ceil(self.dt/self.fiber_dt))
            for n in range(0,N):
                self.fiber_integrator.step(current_time,dt/N)
                current_time += dt/N
        # 3) Evaluation
        # 4) post stage 2

    def get_meshgrid(self, xx, yy, zz):
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
        factor = 1000
        x = np.linspace(0,self.Lx,400)
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)
        last_output = self.output_files[-1]
        data = load(last_output)
        fiber = data['arrays']['fiber']
        fx = fiber.x
        fy = fiber.y
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        v = interp.interpolate('v')
        vmag = np.sqrt(u**2 + v**2 )

        plt.figure()
        cmap = plt.cm.viridis
        levels = np.linspace(0, np.max(vmag), 30)
        vel = plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=np.max(vmag), vmin=0)
        stream = plt.streamplot(x*factor,y*factor,u,v, color='k')
        cbar = plt.colorbar(vel, label='Velocity Magnitude')
        plt.scatter(fx*factor,fy*factor, color='w')
        #plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        fig = os.path.join(self.output_dir, 'streamplot.eps')
        plt.savefig(fig, dpi=300)
        print("Streamplot written to %s."% fig)
        return(fig)

    def _plot_inlet_velocity(self, step_idx=-1):
        output = self.output_files[step_idx]
        data = load(output)

        x = np.array([0])
        y = np.linspace(0,self.Ly,100)
        x,y = np.meshgrid(x,y)
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        u_exact = (self.options.G * (y - self.Ly/2)
                    - 1/2*self.options.g/self.nu*((y-self.Ly/2)**2-(self.Ly/2)**2))

        plt.figure()
        plt.plot(u, y , '-k')
        plt.plot(u_exact, y, ':k')
        plt.title('Velocity at inlet')
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Position [m]')
        plt.legend(['Simulation', 'Ideal'])
        fig = os.path.join(self.output_dir, 'inlet_velocity.eps')
        plt.savefig(fig, dpi=300)
        print("Inlet velocity plot written to %s."% fig)
        return(fig)


    def _plot_orbit(self):
        factor = 1000
        x_begin = []
        y_begin = []
        x_end = []
        y_end = []
        angle = []
        t = []
        Ekin = []
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)
            fiber = data['arrays']['fiber']
            fluid = data['arrays']['fluid']
            x_begin.append(fiber.x[0])
            y_begin.append(fiber.y[0])
            x_end.append(fiber.x[-1])
            y_end.append(fiber.y[-1])
            dxx = fiber.x[0]-fiber.x[-1]
            dyy = fiber.y[0]-fiber.y[-1]
            angle.append(np.arctan(dxx/(dyy+0.01*self.h0)))
            t.append(data['solver_data']['t'])
            v_fiber = fiber.u**2 + fiber.v**2 + fiber.w**2
            v_fluid = fluid.u**2 + fluid.v**2 + fluid.w**2
            Ekin.append(np.dot(fiber.rho,v_fiber)+np.dot(fluid.rho,v_fluid))
            #print('read file %s'%fname)

        plt.figure()
        plt.plot(x_begin, y_begin, '-ok')
        plt.plot(x_end, y_end, '-*k')
        plt.axis('equal')
        orbfig = os.path.join(self.output_dir, 'orbitplot.eps')
        plt.savefig(orbfig, dpi=300)
        print("Orbitplot written to %s."% orbfig)

        # Integrate Jeffery's solution
        print("Solving Jeffery's ODE")
        t = np.array(t)
        phi0 = angle[0]
        are = get_equivalent_aspect_ratio(self.options.ar)
        angle_jeffery = odeint(jeffery_ode,phi0,t, atol=1E-15,
                                args=(are,self.options.G))

        # constraint between -pi/2 and pi/2
        angle_jeffery = (angle_jeffery+np.pi/2.0)%np.pi-np.pi/2.0

        plt.figure()
        plt.plot(t, angle, '*k')
        plt.plot(t, angle_jeffery, '-k')
        plt.xlabel('t [s]')
        plt.ylabel('Angle [rad]')
        plt.legend(['Simulation', 'Jeffery'])
        plt.title("ar=%g"%self.options.ar)
        angfig = os.path.join(self.output_dir, 'angleplot.eps')
        plt.savefig(angfig, dpi=300)
        print("Angleplot written to %s."% angfig)
        csv_file = os.path.join(self.output_dir, 'angle.csv')
        angle_jeffery = np.reshape(angle_jeffery,(angle_jeffery.size,))
        np.savetxt(csv_file, (t, angle, angle_jeffery), delimiter=',')

        plt.figure()
        E_mean = np.mean(Ekin)
        E_top = 1.01*E_mean*np.ones(np.shape(Ekin))
        E_bot = 0.99*E_mean*np.ones(np.shape(Ekin))
        plt.plot(t[1:], Ekin[1:], '-k')
        plt.plot(t[1:], E_top[1:], ':k')
        plt.plot(t[1:], E_bot[1:], ':k')
        plt.xlabel('t [s]')
        plt.ylabel('Energy ')
        plt.title("Kinetic Energy")
        #x1,x2,y1,y2 = plt.axis()
        #plt.axis((x1,x2,0,y2))
        engfig = os.path.join(self.output_dir, 'energyplot.eps')
        plt.savefig(engfig, dpi=300)
        print("Energyplot written to %s."% engfig)

        return [orbfig, angfig, engfig]

    def _send_notification(self, info_fname, attachments=None):
        gmail_user = "nils.batch.notification"
        gmail_pwd = "batch.notification"

        with open(info_fname, 'r') as f:
            info = json.load(f)
        cpu_time = info.get('cpu_time')

        msg = MIMEMultipart()
        msg['Subject'] = 'Batch results'
        msg['From'] = 'Batch Notification'
        msg['To'] = self.options.mail
        txt = MIMEText(""" Run finished. Parameters were\n
                            Diameter: %g\n
                            Aspect Ratio: %g\n
                            Density: %g\n
                            Absolute Viscosity: %g\n
                            Young's modulus: %g\n
                            Shear Rate: %g\n
                            Damping factor: %g\n
                            CPU Time: %g\n
                        """%(self.options.d, self.options.ar, self.options.rho0,
                            self.options.mu, self.options.E,
                            self.options.G, self.options.D,
                            cpu_time)
                        )
        msg.attach(txt)
        for fname in attachments:
            fp = open(fname, 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pwd)
            server.sendmail(gmail_user, self.options.mail, msg.as_string())
            server.close()
            print ("Successfully sent the mail.")
        except:
            print ("Failed to send mail.")


    def post_process(self, info_fname):
        streamlines = self._plot_streamlines()
        [orbitplot, angleplot, energyplot] = self._plot_orbit()
        inlet = self._plot_inlet_velocity()
        if self.options.mail:
            self._send_notification(info_fname, [streamlines, orbitplot, angleplot])

def run_application():
    app = Channel()
    app.run()
    app.post_process(app.info_filename)

if __name__ == '__main__':
    #run_application()
    cProfile.runctx('run_application()', None, locals(), 'stats')
    p = pstats.Stats('stats')
    p.sort_stats('tottime').print_stats(10)
    p.sort_stats('cumtime').print_stats(10)
