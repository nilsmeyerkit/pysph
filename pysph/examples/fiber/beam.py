"""Tension of a fiber in gravity field (10 seconds).
"""
import os
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# PySPH imports
from pysph.base.utils import get_particle_array_beadchain_fiber
from pysph.base.kernels import QuinticSpline

from pysph.solver.application import Application
from pysph.solver.utils import load
from pysph.solver.solver import Solver

from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep

from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient
from pysph.sph.fiber.utils import Damping, HoldPoints, ComputeDistance
from pysph.sph.fiber.beadchain import Tension, Bending

class Beam(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--D", action="store", type=float, dest="d",
            default=1, help="Factor for damping. 1 is aperiodic limit."
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E8, help="Young's modulus."
        )
        group.add_argument(
            "--N", action="store", type=int, dest="N",
            default=10, help="Number of particles."
        )
        group.add_argument(
            "--gx", action="store", type=float, dest="gx",
            default=0, help="Body force in x-direction."
        )
        group.add_argument(
            "--gy", action="store", type=float, dest="gy",
            default=0, help="Body force in y-direction."
        )
        group.add_argument(
            "--gz", action="store", type=float, dest="gz",
            default=10, help="Body force in z-direction."
        )

    def consume_user_options(self):
        # fiber length
        self.L = 10.0

        # numerical setup
        self.N = self.options.N
        self.dx = self.L/(self.N-1)
        self.h = self.dx

        # fluid properties
        self.rho0 = 1.0
        self.p0 = 1.0

        # fiber properties
        self.A = 1.0
        self.I = self.A/12.0
        self.E = self.options.E

        # Analytical solution for angular eigenfrequencies:
        #       Pi/L np.sqrt(E/rho) (2n-1)/2
        # --> first analytical eigenfrequency:
        self.omega0_tension = np.pi/(2*self.L)*np.sqrt(self.E/self.rho0)
        self.omega0_bending = 3.5156*np.sqrt(
                                self.E*self.I/(self.rho0*self.A*self.L**4))

        # This is valid when gx >> gy and meant to be used for just one case
        if self.options.gx > self.options.gy:
            self.omega0 = self.omega0_tension
        else:
            self.omega0 = self.omega0_bending

        m = self.rho0*self.A*self.dx
        self.D = self.options.d*m*self.omega0
        self.AD = 5*m*self.omega0
        self.gx = self.options.gx
        self.gy = self.options.gy
        self.gz = self.options.gz
        print('Damping: %g, Omega0: %g' % (self.D, self.omega0))

        # setup time step
        dt_force = 0.25 * np.sqrt(
            self.h/(sqrt(self.gx**2+self.gy**2+self.gz**2)))
        dt_tension = 0.5*self.h*np.sqrt(self.rho0/self.E)
        dt_bending = 0.5*self.h**2*np.sqrt(self.rho0*self.A/(self.E*2*self.I))

        self.tf = 4*np.pi/self.omega0

        self.dt = min(dt_force, dt_tension, dt_bending)

    def create_scheme(self):
        return None

    def create_particles(self):
        _x = np.linspace(-self.dx/2, self.L-self.dx/2, self.N)
        _y = np.array([0.0])
        _z = np.array([0.0])
        x, y, z = np.meshgrid(_x, _y, _z)
        fiber_x = x.ravel()
        fiber_y = y.ravel()
        fiber_z = z.ravel()

        # volume is set as dx * A
        volume = self.A * self.dx

        # create array
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fiber_x, y=fiber_y, z=fiber_z, m=volume*self.rho0,
            rho=self.rho0, h=self.h, lprev=self.dx, lnext=self.dx, phi0=np.pi,
            phifrac=2.0, fidx=range(self.N), V=1./volume)


        # tag particles to be hold
        fiber.holdtag[0] = 1
        fiber.holdtag[1] = 2

        # return the particle list
        return [fiber]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    ComputeDistance(dest='fiber', sources=['fiber'])
                ],
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(dest='fiber',
                       sources=['fiber'], pb=0.0, gx=self.gx, gy=self.gy,
                       gz=self.gz),
                    Tension(dest='fiber', sources=None,
                        ea=self.E*self.A),
                    Bending(dest='fiber', sources=None,
                        ei=self.E*self.I),
                    Damping(dest='fiber',
                        sources=None,
                        d = self.D)
                ],
            ),
            Group(
                equations=[
                    HoldPoints(dest='fiber', sources=None, tag=1),
                    HoldPoints(dest='fiber', sources=None, tag=2, x=False,
                               mirror_particle=-1),
                ],
            ),
        ]
        return equations

    def create_solver(self):
        # Setting up the default integrator for fiber particles
        kernel = QuinticSpline(dim=3)
        integrator = EPECIntegrator(fiber=TransportVelocityStep())
        solver = Solver(kernel=kernel, dim=3, integrator=integrator, dt=self.dt,
                         tf=self.tf, N=100,
                         vtk=True)
        return solver

    def _plot_oscillation(self, file):
        t, disp_x, disp_y = np.loadtxt(file, delimiter=',')
        plt.figure()
        plt.plot(t, disp_x, '-k')
        plt.plot(t, disp_y, '--k')
        plt.title("Oscillation (d=%g, w0x=%g, w0y=%g, dt=%g)"%(
            self.D,self.omega0_tension, self.omega0_bending, self.dt))
        plt.xlabel('t'); plt.ylabel('Displacement')
        plt.legend(['x-direction', 'y-direction'])
        plt.grid()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        fig = os.path.join(self.output_dir, "oscillation.pdf")
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Figure written to %s." % fig)

    def _plot_displacement(self, file):
        x0, disp_x, exp_x, disp_y, exp_y = np.loadtxt(file, delimiter=',')
        plt.figure()
        plt.plot(x0, disp_x,'*k')
        plt.plot(x0, exp_x, '-k')
        plt.title("Displacement in x")
        plt.legend(['simulation', 'exact'])
        plt.xlabel('x'); plt.ylabel('u')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        fig = os.path.join(self.output_dir, "displacement_x.pdf")
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Figure written to %s." % fig)

        plt.figure()
        plt.plot(x0, disp_y,'*k')
        plt.plot(x0, exp_y, '-k')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,x2,0,y2))
        plt.title("Displacement in y")
        plt.legend(['simulation', 'exact'])
        plt.xlabel('x'); plt.ylabel('y')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        fig = os.path.join(self.output_dir, "displacement_y.pdf")
        plt.savefig(fig, dpi=300, bbox_inches='tight')
        print("Figure written to %s." % fig)

    def _save_results(self):
        # load times and y values
        x = []
        w = []
        t = []
        for output in self.output_files:
            data = load(output)
            pa = data['arrays']['fiber']
            x.append(pa.x[-1])
            w.append(pa.y[-1])
            t.append(data['solver_data']['t'])
        osc = os.path.join(self.output_dir,"oscillation_%g.csv"%self.options.d)
        np.savetxt(osc, (t,x-x[0], w-w[0]), delimiter=',')

        last_output = self.output_files[-1]
        data = load(last_output)
        pa = data['arrays']['fiber']
        x = pa.x
        y = pa.y

        first_output = self.output_files[0]
        data = load(first_output)
        pa = data['arrays']['fiber']
        x0 = pa.x
        x_exact = self.gx*self.rho0/self.E*(self.L*x0-x0**2/2)
        k = self.rho0*self.gy*self.A/(self.E*self.I)
        y_exact = k/24*(x0**4-4*self.L*x0**3+6*self.L**2*x0**2)

        disp = os.path.join(self.output_dir, "disp_%d.csv"%self.options.N)
        np.savetxt(disp, (x0, x-x0, x_exact, y, y_exact), delimiter=',')
        return [osc, disp]

    def post_process(self, info_fname):
        # dump vtk files after run
        if len(self.output_files) == 0:
            return

        osc, disp = self._save_results()
        self._plot_displacement(disp)
        self._plot_oscillation(osc)


if __name__ == '__main__':
    app = Beam()
    app.run()
    app.post_process(app.info_filename)
