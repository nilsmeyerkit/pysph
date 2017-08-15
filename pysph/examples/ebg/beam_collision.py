"""Tension of a fiber in gravity field (10 seconds).
"""
import os
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.base.config import get_config
from pysph.base.nnps import LinkedListNNPS

from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.solver.vtk_output import dump_vtk
from pysph.solver.output import output_formats
from pysph.solver.solver import Solver

from pysph.sph.integrator import EPECIntegrator, EulerIntegrator
from pysph.sph.integrator_step import TransportVelocityStep, EulerStep, EBGStep
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler

from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient
from pysph.sph.ebg.fiber import (Tension, HoldPoints, Bending, Damping,
    EBGVelocityReset, ArtificialDamping, SimpleContact, Contact)


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

    def consume_user_options(self):
        # fiber length
        self.L = 10.0

        # numerical setup
        self.N = self.options.N
        self.l = self.L/(1-1/(2*self.N))
        self.dx = self.l/self.N     # particle spacing
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
        self.omega0_bending = 3.5156*np.sqrt(self.E*self.I/(self.rho0*self.A*self.L**4))
        if self.options.gx > self.options.gy:
            self.omega0 = self.omega0_tension
        else:
            self.omega0 = self.omega0_bending
        m = self.rho0*self.A*self.dx
        self.D = self.options.d*m*self.omega0
        self.gx = self.options.gx
        self.gy = self.options.gy
        print('Damping: %g, Omega0: %g'%(self.D,self.omega0))

        # setup time step
        dt_force = 0.25 * np.sqrt(self.h/(sqrt(self.gx**2+self.gy**2)))
        dt_tension = 0.5*self.h*np.sqrt(self.rho0/self.E)
        dt_bending = 0.5*self.h**2*np.sqrt(self.rho0*self.A/(self.E*2*self.I))
        print(dt_force)
        print(dt_tension)
        print(dt_bending)

        self.tf = 20

        self.dt = min(dt_force,dt_tension, dt_bending)
        # self.fiber_dt = min(dt_tension, dt_bending)
        # print("Time step ratio is %g"%(self.dt/self.fiber_dt))

    def create_scheme(self):
        return None

    def create_particles(self):
        _x = np.linspace(-self.dx, self.l-self.dx, self.N+1)
        _y = np.array([0.0])
        _z = np.array([0.0])
        x, y, z = np.meshgrid(_x, _y, _z)
        fiber1_x = x.ravel()
        fiber1_y = y.ravel()
        fiber1_z = z.ravel()

        _x = np.array([0.75*self.l])
        _y = np.array([-2*self.dx])
        _z = np.linspace(-0.25*self.l, 0.75*self.l, self.N+1)
        # _x = np.linspace(-self.dx, self.l-self.dx, self.N+1)
        # _y = np.array([-2*self.dx])
        # _z = np.array([0.0])
        x, y, z = np.meshgrid(_x, _y, _z)
        fiber2_x = x.ravel()
        fiber2_y = y.ravel()
        fiber2_z = z.ravel()

        # create array
        fiber1 = get_particle_array(name='fiber1', x=fiber1_x, y=fiber1_y, z=fiber1_z)
        fiber2 = get_particle_array(name='fiber2', x=fiber2_x, y=fiber2_y, z=fiber2_z)

        # add requisite variables needed for this formulation
        for name in ('V', 'wij', 'vmag2', 'lprev', 'lnext', 'arho',
                     'uf', 'vf', 'wf', 'ug', 'vg', 'wg', 'phi0', 'phifrac',
                     'auhat', 'avhat','awhat', 'uhat', 'vhat', 'what', 'fractag',
                     'rxnext', 'rynext', 'rznext', 'rnext', 'rxprev', 'ryprev',
                     'rzprev', 'rprev', 'eu', 'ev', 'ew', 'holdtag', 'Fx',
                     'Fy', 'Fz', 'omegax', 'omegay', 'omegaz'):
            fiber1.add_property(name)
            fiber2.add_property(name)

        # set the output property arrays
        fiber1.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'fractag',
                              'h', 'p', 'pid', 'tag', 'gid', 'lprev'])
        fiber2.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'fractag',
                              'h', 'p', 'pid', 'tag', 'gid', 'lprev'])

        # set initial distances
        fiber1.lprev[:] = self.dx
        fiber1.lnext[:] = self.dx
        fiber1.phi0[:] = np.pi
        fiber1.phifrac[:] = 0.2
        fiber2.lprev[:] = self.dx
        fiber2.lnext[:] = self.dx
        fiber2.phi0[:] = np.pi
        fiber2.phifrac[:] = 0.2

        # tag particles to be hold
        fiber1.holdtag[:] = 0
        fiber1.holdtag[0] = 2
        fiber1.holdtag[1] = 1
        fiber1.holdtag[2] = 2

        # test fracture
        #fiber2.fractag[int(self.N/2)] = 1

        # volume is set as dx * A
        volume = self.A * self.dx

        # mass is set to get the reference density of rho0
        fiber1.m[:] = volume * self.rho0
        fiber2.m[:] = volume * self.rho0

        # Set the default rho.
        fiber1.rho[:] = self.rho0
        fiber2.rho[:] = self.rho0

        # inverse volume (necessary for transport velocity equations)
        fiber1.V[:] = 1./volume
        fiber2.V[:] = 1./volume

        # smoothing lengths
        fiber1.h[:] = self.h
        fiber2.h[:] = self.h

        # return the particle list
        return [fiber1, fiber2]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    MomentumEquationPressureGradient(dest='fiber1',
                       sources=['fiber1', 'fiber2'], pb=0.0, gx=self.gx, gy=self.gy),
                    MomentumEquationPressureGradient(dest='fiber2',
                       sources=['fiber1', 'fiber2'], pb=0.0, gx=self.gx, gy=self.gy),
                    Tension(dest='fiber1',
                        sources=['fiber1'],
                        ea=self.E*self.A),
                    Tension(dest='fiber2',
                        sources=['fiber2'],
                        ea=self.E*self.A),
                    Bending(dest='fiber1',
                        sources=['fiber1'],
                        ei=self.E*self.I),
                    Bending(dest='fiber2',
                        sources=['fiber2'],
                        ei=self.E*self.I),
                    Contact(dest='fiber1',
                       sources=['fiber1', 'fiber2'],
                       E = self.E, d=self.dx, k=1),
                    Contact(dest='fiber2',
                        sources=['fiber1', 'fiber2'],
                        E = self.E, d=self.dx, k=1),
                    Damping(dest='fiber1',
                        sources=None,
                        d = self.D),
                    Damping(dest='fiber2',
                        sources=None,
                        d = self.D)
                ],
            ),
            Group(
                equations=[
                    HoldPoints(dest='fiber1', sources=None, tag=2, x=False),
                    HoldPoints(dest='fiber1', sources=None, tag=1, y=False),
                ],
            ),
        ]
        return equations

    # def _configure(self):
    #     super(Beam, self)._configure()
    #     self.fiber_integrator = EulerIntegrator(fiber=EulerStep())
    #     kernel = QuinticSpline(dim=3)
    #     equations = [
    #                     Group(
    #                         equations=[
    #                             Tension(dest='fiber',
    #                                     sources=['fiber'],
    #                                     ea=self.E*self.A),
    #                             Bending(dest='fiber',
    #                                     sources=['fiber'],
    #                                     ei=self.E*self.I),
    #                             Damping(dest='fiber',
    #                                     sources=None,
    #                                     d = self.D)
    #                         ],
    #                     ),
    #                     Group(
    #                         equations=[
    #                             HoldPoints(dest='fiber', sources=None, tag=2,
    #                                 x=False),
    #                             HoldPoints(dest='fiber', sources=None, tag=1,
    #                                 y=False),
    #                         ]
    #                     ),
    #                 ]
    #     particles = [p for p in self.particles if p.name == 'fiber']
    #     nnps = LinkedListNNPS(dim=3, particles=particles,
    #                     radius_scale=kernel.radius_scale,
    #                     fixed_h=False, cache=False, sort_gids=False)
    #     acceleration_eval = AccelerationEval(
    #                 particle_arrays=particles,
    #                 equations=equations,
    #                 kernel=kernel,
    #                 mode='serial')
    #     comp = SPHCompiler(acceleration_eval, self.fiber_integrator)
    #     config = get_config()
    #     config.use_openmp = False
    #     comp.compile()
    #     config.use_openmp = True
    #     acceleration_eval.set_nnps(nnps)
    #     self.fiber_integrator.set_nnps(nnps)

    def create_solver(self):
        # Setting up the default integrator for fiber particles
        kernel = QuinticSpline(dim=3)
        integrator = EPECIntegrator(fiber1=TransportVelocityStep(),
            fiber2=TransportVelocityStep())
        solver = Solver(kernel=kernel, dim=3, integrator=integrator, dt=self.dt,
                         tf=self.tf, pfreq=int(np.ceil(self.tf/(1000*self.dt))),
                         vtk=True)
        return solver

    # def post_stage(self, current_time, dt, stage):
    #     # 1) predictor
    #     # 2) post stage 1:
    #     if stage == 1:
    #         N = int(np.ceil(self.dt/self.fiber_dt))
    #         for n in range(0,N):
    #             self.fiber_integrator.step(current_time,dt/N)
    #             current_time += dt/N
    #     # 3) Evaluation
    #     # 4) post stage 2

    def _plot_oscillation(self, file):
        t, disp_x, disp_y = np.loadtxt(file, delimiter=',')
        plt.figure()
        plt.plot(t, disp_x, '-k')
        plt.plot(t, disp_y, '--k')
        plt.title("Oscillation (d=%g, w0x=%g, w0y=%g, dt=%g)"%(
            self.D,self.omega0_tension, self.omega0_bending, self.dt))
        plt.xlabel('t'); plt.ylabel('Displacement')
        plt.legend(['x-direction', 'y-direction'])
        fig = os.path.join(self.output_dir, "oscillation.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s." % fig)

    def _plot_displacement(self, file):
        x0, disp_x, exp_x, disp_y, exp_y = np.loadtxt(file, delimiter=',')
        plt.figure()
        plt.plot(x0, disp_x,'*k')
        plt.plot(x0, exp_x, '-k')
        plt.title("Displacement in x")
        plt.legend(['simulation', 'exact'])
        plt.xlabel('x'); plt.ylabel('u')
        fig = os.path.join(self.output_dir, "displacement_x.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s." % fig)

        plt.figure()
        plt.plot(x0, disp_y,'*k')
        plt.plot(x0, exp_y, '-k')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,x2,0,y2))
        plt.title("Displacement in y")
        plt.legend(['simulation', 'exact'])
        plt.xlabel('x'); plt.ylabel('y')
        fig = os.path.join(self.output_dir, "displacement_y.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s." % fig)

    def _save_results(self):
        # load times and y values
        x = []
        w = []
        t = []
        for output in self.output_files:
            data = load(output)
            pa = data['arrays']['fiber1']
            x.append(pa.x[-1])
            w.append(pa.y[-1])
            t.append(data['solver_data']['t'])
        osc = os.path.join(self.output_dir,"oscillation_%g.csv"%self.options.d)
        np.savetxt(osc, (t,x-x[0], w-w[0]), delimiter=',')

        last_output = self.output_files[-1]
        data = load(last_output)
        pa = data['arrays']['fiber1']
        x = pa.x
        y = pa.y

        first_output = self.output_files[0]
        data = load(first_output)
        pa = data['arrays']['fiber1']
        x0 = pa.x
        x_exact = self.gx*self.rho0/self.E*(self.L*x0-x0**2/2)
        k = self.rho0*self.gy*self.A/(self.E*self.I)
        y_exact = k/24*(x0**4-4*self.L*x0**3+6*self.L**2*x0**2)

        disp = os.path.join(self.output_dir, "disp_%d.csv"%self.options.N)
        np.savetxt(disp, (x0, x-x0, x_exact, y, y_exact), delimiter=',')
        return [osc, disp]

    def post_process(self, info_fname):
        # dump vtk files after run
        osc, disp = self._save_results()
        self._plot_displacement(disp)
        self._plot_oscillation(osc)


if __name__ == '__main__':
    app = Beam()
    app.run()
    app.post_process(app.info_filename)
