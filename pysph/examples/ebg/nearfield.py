"""Flow with a single hold particle. (10 mins)
"""
import os
import smtplib
import json

# general imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.ndimage.filters import gaussian_filter
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
from pysph.sph.integrator_step import TransportVelocityStep, EulerStep, EBGStep, InletOutletStep
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Group
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet
from pysph.sph.wc.transport_velocity import (SummationDensity,
    StateEquation, MomentumEquationPressureGradient,
    MomentumEquationArtificialViscosity,
    MomentumEquationViscosity, MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity)
from pysph.sph.ebg.fiber import (Tension, Bending, Vorticity, Friction, Damping,
    HoldPoints, EBGVelocityReset, ArtificialDamping, VelocityGradient)

# Shear flow between two moving plates with shear rate G.
class Channel(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--d", action="store", type=float, dest="d",
            default=0.0002, help="Fiber diameter"
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
            default=1000, help="Absolute viscosity"
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E11, help="Young's modulus"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=0, help="Shear rate"
        )
        group.add_argument(
            "--g", action="store", type=float, dest="g",
            default=0, help="Body force in x-direction"
        )
        group.add_argument(
            "--V", action="store", type=float, dest="V",
            default=0, help="Moving wall velocity"
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
            "--pb", action="store_true", dest="pb",
            default=False, help="Apply Adami background pressure"
        )
        group.add_argument(
            "--fluidres", action="store", type=float, dest="fluid_res",
            default=1, help="Relative size of center particle"
        )
        group.add_argument(
            "--postonly", action="store", type=bool, dest="postonly",
            default=False, help="Set time to zero and postprocess only."
        )
        group.add_argument(
            "--massscale", action="store", type=float, dest="scale_factor",
            default=1E5, help="Factor of mass scaling"
        )


    def consume_user_options(self):
        self.factor = 1/self.options.fluid_res
        # Numerical setup
        self.dx = self.options.d/self.factor
        self.h0 = self.dx

        self.options.g = self.options.g/self.options.scale_factor
        self.options.rho0 = self.options.rho0*self.options.scale_factor

        # imaginary fiber length
        self.Lf = self.factor*self.options.ar*self.dx
        # channel dimensions
        width = self.Lf + int(0.2*self.factor*self.options.ar)*self.dx
        self.Ly = self.options.width or width
        self.Lx = 2.0*self.Ly

        # cylinder position
        self.xx = self.Lx/2
        self.yy = self.Ly/2
        self.zz = 0.0

        # fluid properties
        self.nu = self.options.mu/self.options.rho0

        if self.options.dim == 2:
            self.A = self.dx
            self.I = self.dx**3/12
            mass = 3*self.options.rho0*self.dx**2
            self.J = 1/12*mass*(self.dx**2 + (3*self.dx)**2)
        else:
            R = self.dx/2
            self.A = np.pi*R**2
            self.I = np.pi*R**4/4.0
            mass = 3*self.options.rho0*self.dx*self.A
            self.J = 1/4*mass*R**2 + 1/12*mass*(3*self.dx)**2

        # computed properties
        self.Vmax = (self.options.G*self.Ly/2
                    + self.options.g/(2*self.nu)*self.Ly**2/4 + self.options.V)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.options.rho0
        if self.options.pb:
            print("Background pressure activated.")
            self.pb = self.p0
        else:
            self.pb = 0.0

        # time
        self.t = 100.0
        print("Simulated time is %g s"%self.t)

        # Setup time step
        dt_cfl = 0.4 * self.h0/(self.c0 + self.Vmax)
        dt_viscous = 0.125 * self.h0**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h0/(self.options.g+0.001))
        print("dt_cfl: %g"%dt_cfl)
        print("dt_viscous: %g"%dt_viscous)
        print("dt_force: %g"%dt_force)

        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def create_scheme(self):
        return None

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
        Lx = self.Lx
        Ly = self.Ly
        dx2 = self.dx/2
        _x = np.arange(dx2, Lx, self.dx)

        # create the fluid particles
        _y = np.arange(dx2, Ly, self.dx)
        _z = np.arange(dx2, Ly, self.dx)

        fx,fy,fz = self.get_meshgrid(_x, _y, _z)

        # find center particle to remove
        indices = []
        for i in range(len(fx)):
            # vertical
            xx = self.Lx/2
            yy = self.Ly/2
            zz = 0.0
            if (fx[i] < xx+self.factor*dx2 and fx[i] > xx-self.factor*dx2 and
               fy[i] < yy+self.factor*dx2 and fy[i] > yy-self.factor*dx2 and
               fz[i] < zz+self.factor*dx2 and fz[i] > zz-self.factor*dx2):
               indices.append(i)
        fx = np.delete(fx, indices)
        fy = np.delete(fy, indices)
        fz = np.delete(fz, indices)

        # append center particle
        fx = np.append(fx, self.xx)
        fy = np.append(fy, self.yy)
        fz = np.append(fz, self.zz)

        ghost_extent = 5*self.dx
        # create the channel particles at the top
        _y = np.arange(Ly+dx2, Ly+dx2+ghost_extent, self.dx)
        tx,ty,tz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2-ghost_extent, -self.dx)
        bx,by,bz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the right
        _z = np.arange(-dx2, -dx2-ghost_extent, -self.dx)
        _y = np.arange(dx2-ghost_extent, Ly+ghost_extent, self.dx)
        rx,ry,rz = self.get_meshgrid(_x, _y, _z)

        # create the channel particles at the left
        _z = np.arange(Ly+dx2, Ly+dx2+ghost_extent, self.dx)
        _y = np.arange(dx2-ghost_extent, Ly+ghost_extent, self.dx)
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
        else:
            channel = get_particle_array(name='channel', x=cx, y=cy, z=cz)
            fluid = get_particle_array(name='fluid', x=fx, y=fy, z=fz)



        print("nfluid = %d, nchannel = %d"%(
            fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # add requisite variables needed for this formulation
        for name in ('V', 'wf','uf','vf','wg','wij','vg','ug',
                     'awhat', 'avhat','auhat', 'vhat', 'what', 'uhat', 'vmag2',
                     'arho', 'phi0', 'omegax', 'omegay', 'omegaz',
                     'holdtag', 'eu', 'ev', 'ew', 'testx', 'testy', 'testz',
                     'dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz','dwdx',
                     'dwdy', 'dwdz'):
            fluid.add_property(name)
            channel.add_property(name)

        # set the output property arrays
        fluid.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid','omegax', 'omegay', 'omegaz'])
        channel.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm','h', 'p',
                        'pid', 'holdtag', 'gid','omegax', 'omegay', 'omegaz'])

        # volume is set as dx^2
        volume = self.dx**self.options.dim

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.options.rho0
        fluid.m[-1] = volume * self.factor**self.options.dim * self.options.rho0
        channel.m[:] = volume * self.options.rho0

        # tag particles to be hold
        fluid.holdtag[-1] = 100

        # Set the default rho.
        fluid.rho[:] = self.options.rho0
        channel.rho[:] = self.options.rho0

        # inverse volume (necessary for transport velocity equations)
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = self.h0
        fluid.h[-1] = self.factor*self.h0
        channel.h[:] = self.h0

        # initial velocities
        fluid.u[:] = (self.options.G*(fluid.y[:]-self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                        (fluid.y[:]-self.Ly/2)**2-(self.Ly/2)**2)
                    + self.options.V)
        channel.u[:] = (self.options.G*(channel.y[:]-self.Ly/2)
                    - 1/2*self.options.g/self.nu*(
                        (channel.y[:]-self.Ly/2)**2-(self.Ly/2)**2)
                    + self.options.V)

        # return the particle list
        return [fluid, channel]

    def create_equations(self):
        all = ['fluid', 'channel']
        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=all),
                ]
            ),
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=self.p0,
                                    rho0=self.options.rho0, b=1.0),
                    SetWallVelocity(dest='channel', sources=['fluid']),
                ],
            ),
            Group(
                equations=[
                    SolidWallPressureBC(dest='channel',
                                        sources=['fluid'],
                                        b=1.0, rho0=self.options.rho0, p0=self.p0),
                ],
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(dest='fluid', sources=all,
                                        pb=0.0, gx=self.options.g),
                    MomentumEquationViscosity(dest='fluid',
                                        sources=['fluid'], nu=self.nu),
                    SolidWallNoSlipBC(dest='fluid',
                                        sources=['channel'], nu=self.nu),
                    MomentumEquationArtificialStress(dest='fluid',
                                        sources=['fluid']),
                ],
            ),
            Group(
                equations=[
                    HoldPoints(dest='fluid', sources=None, tag=100),
                    HoldPoints(dest='channel', sources=None, tag=100),
                ]
            ),
        ]
        return equations

    def create_solver(self):
        # Setting up the default integrator for fluid particles
        kernel = CubicSpline(dim=self.options.dim)
        integrator = EPECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
                         tf=self.t, pfreq=int(self.t/(100*self.dt)), vtk=True)
        # solver = Solver(kernel=kernel, dim=self.options.dim, integrator=integrator, dt=self.dt,
        #                  tf=self.t, pfreq=1, vtk=True)
        return solver

    def get_meshgrid(self, xx, yy, zz):
        if self.options.dim == 2:
            x, y = np.meshgrid(xx, yy)
            x = x.ravel()
            y = y.ravel()
            z = 0.0*np.ones(np.shape(y))
        else:
            x, y, z = np.meshgrid(xx, yy, zz)
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
        return [x,y,z]

    def _plot_streamlines(self):
        factor = 1000
        x = np.linspace(0,self.Lx,800)
        y = np.linspace(0,self.Ly,200)
        x,y = np.meshgrid(x,y)
        last_output = self.output_files[-1]
        data = load(last_output)
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        u = interp.interpolate('u')
        v = interp.interpolate('v')
        p = interp.interpolate('p')
        vmag = np.sqrt(u**2 + v**2 )


        plt.figure()
        #f, (ax1, ax2) = plt.subplots(2, 1)
        cmap = plt.cm.viridis
        levels = np.linspace(0, np.max(vmag), 30)
        vel = plt.contourf(x*factor,y*factor, vmag, levels=levels,
                 cmap=cmap, vmax=np.max(vmag), vmin=0)
        cbar = plt.colorbar(vel, label='Velocity Magnitude')
        stream = plt.streamplot(x*factor,y*factor,u,v,
                color='k', density=0.51)
        #plt.scatter(self.Lx/2*factor,self.Ly/2*factor, color='k')
        plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        stream_fig = os.path.join(self.output_dir, 'streamplot.png')
        plt.savefig(stream_fig, dpi=300)
        print("Streamplot written to %s."% stream_fig)

        plt.figure()
        cmap = plt.cm.viridis
        levels = np.linspace(-200, 200, 30)
        vel = plt.contourf(x*factor,y*factor, p-p[0], levels=levels,
                 cmap=cmap, vmax=200, vmin=-200)
        cbar = plt.colorbar(vel, label='Pressure')
        plt.axis('equal')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        p_fig = os.path.join(self.output_dir, 'pressure.png')
        plt.savefig(p_fig, dpi=300)
        print("Pressure written to %s."% p_fig)

        x = np.linspace(0,self.Lx,200)
        y = np.array([self.Ly/2])
        x,y = np.meshgrid(x,y)
        last_output = self.output_files[-1]
        data = load(last_output)
        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        interp.update_particle_arrays(list(data['arrays'].values()))
        p = interp.interpolate('p')

        fem = np.loadtxt('/Users/nils/Dropbox/Thesis/Documentation/SPH/Nearfield/poiseuille_pressure_centerline.csv', delimiter=',')
        x_fem = fem[:,0]*factor
        p_fem = fem[:,2]
        plt.figure()
        plt.plot(x[0,:]*factor, p-p[0], '-k', x_fem, p_fem, '.k')
        plt.legend(['SPH Simulation','FEM Result'])
        plt.xlabel('x [mm]')
        plt.ylabel('p [Pa]')

        pcenter_fig = os.path.join(self.output_dir, 'pressure_centerline.png')
        plt.savefig(pcenter_fig, dpi=300)
        print("Pressure written to %s."% pcenter_fig)

        return(stream_fig, p_fig, pcenter_fig)

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
                    - 1/2*self.options.g/self.nu*((y-self.Ly/2)**2-(self.Ly/2)**2)
                    +self.options.V)

        plt.figure()
        plt.plot(u, y , '-k')
        plt.plot(u_exact, y, ':k')
        plt.title('Velocity at inlet')
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Position [m]')
        plt.legend(['Simulation', 'Ideal'])
        fig = os.path.join(self.output_dir, 'inlet_velocity.png')
        plt.savefig(fig, dpi=300)
        print("Inlet velocity plot written to %s."% fig)
        return(fig)


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
        streamlines, pressure, pcenterline = self._plot_streamlines()
        inlet = self._plot_inlet_velocity()
        if self.options.mail:
            self._send_notification(info_fname, [streamlines])


if __name__ == '__main__':
    app = Channel()
    app.run()
    app.post_process(app.info_filename)
