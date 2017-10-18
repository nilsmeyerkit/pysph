
class Tool(object):
    """A tool is typically an object that can be used to perform a
    specific task on the solver's pre_step/post_step or post_stage callbacks.
    This can be used for a variety of things.  For example, one could save a
    plot, print debug statistics or perform remeshing etc.

    To create a new tool, simply subclass this class and overload any of its
    desired methods.
    """

    def pre_step(self, solver):
        """If overloaded, this is called automatically before each integrator
        step.  The method is passed the solver instance.
        """
        pass

    def post_stage(self, current_time, dt, stage):
        """If overloaded, this is called automatically after each integrator
        stage, i.e. if the integrator is a two stage integrator it will be
        called after the first and second stages.

        The method is passed (current_time, dt, stage).  See the the
        `Integrator.one_timestep` methods for examples of how this is called.
        """
        pass

    def post_step(self, solver):
        """If overloaded, this is called automatically after each integrator
        step.  The method is passed the solver instance.
        """
        pass



class SimpleRemesher(Tool):
    """A simple tool to periodically remesh a given array of particles onto an
    initial set of points.
    """
    def __init__(self, app, array_name, props, freq=100, xi=None, yi=None,
                 zi=None, kernel=None):
        """Constructor.

        Parameters
        ----------

        app : pysph.solver.application.Application
            The application instance.
        array_name: str
            Name of the particle array that needs to be remeshed.
        props : list(str)
            List of properties to interpolate.
        freq : int
            Frequency of remeshing operation.
        xi, yi, zi : ndarray
            Positions to remesh the properties onto.  If not specified they
            are taken from the particle arrays at the time of construction.
        kernel: any kernel from pysph.base.kernels

        """
        from pysph.solver.utils import get_array_by_name
        self.app = app
        self.particles = app.particles
        self.array = get_array_by_name(self.particles, array_name)
        self.props = props
        if xi is None:
            xi = self.array.x
        if yi is None:
            yi = self.array.y
        if zi is None:
            zi = self.array.z
        self.xi, self.yi, self.zi = xi.copy(), yi.copy(), zi.copy()
        self.freq = freq
        from pysph.tools.interpolator import Interpolator
        if kernel is None:
            kernel = app.solver.kernel
        self.interp = Interpolator(
            self.particles, x=self.xi, y=self.yi, z=self.zi,
            kernel=kernel,
            domain_manager=app.create_domain()
        )

    def post_step(self, solver):
        if solver.count%self.freq == 0 and solver.count > 0:
            self.interp.nnps.update()
            data = dict(x=self.xi, y=self.yi, z=self.zi)
            for prop in self.props:
                data[prop] = self.interp.interpolate(prop)
            self.array.set(**data)

class FiberIntegrator(Tool):
    def __init__(self, all_particles, scheme, domain, innerloop=True, updates=True,
            parallel=False):
        """The second integrator is a simple Euler-Integrator (accurate
        enough due to very small time steps; very fast) using EBGSteps.
        EBGSteps are basically the same as EulerSteps, exept for the fact
        that they work with an intermediate ebg velocity [eu, ev, ew].
        This velocity does not interfere with the actual velocity, which
        is neseccery to not disturb the real velocity through artificial
        damping in this step. The ebg velocity is initialized for each
        inner loop again and reset in the outer loop."""
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.integrator_step import EBGStep
        from pysph.base.config import get_config
        from pysph.sph.integrator import EulerIntegrator
        from pysph.sph.scheme import BeadChainScheme
        from pysph.sph.equation import Group
        from pysph.sph.fiber.utils import (HoldPoints, Contact, ComputeDistance)
        from pysph.sph.fiber.beadchain import (Tension, Bending,
            ArtificialDamping)
        from pysph.base.nnps import DomainManager, LinkedListNNPS
        from pysph.sph.acceleration_eval import AccelerationEval
        from pysph.sph.sph_compiler import SPHCompiler

        if not isinstance(scheme, BeadChainScheme):
            raise TypeError("Scheme must be BeadChainScheme")

        self.innerloop = innerloop
        self.dt = scheme.dt
        self.fiber_dt = scheme.fiber_dt
        self.domain_updates = updates

        # if there are more than 1 particles involved, elastic equations are
        # iterated in an inner loop.
        if self.innerloop:
            # second integrator
            #self.fiber_integrator = EulerIntegrator(fiber=EBGStep())
            steppers = {}
            for f in scheme.fibers:
                steppers[f] = EBGStep()
            self.fiber_integrator = EulerIntegrator(**steppers)
            # The type of spline has no influence here. It must be large enough
            # to contain the next particle though.
            kernel = QuinticSpline(dim=scheme.dim)
            equations = []
            g1 = []
            for fiber in scheme.fibers:
                g1.append(ComputeDistance(dest=fiber,sources=[fiber]))
            equations.append(Group(equations=g1))

            g2 = []
            for fiber in scheme.fibers:
                g2.append(Tension(dest=fiber, sources=None, ea=scheme.E*scheme.A))
                g2.append(Bending(dest=fiber, sources=None, ei=scheme.E*scheme.I))
                g2.append(Contact(dest=fiber, sources=scheme.fibers, E=scheme.E,
                            d=scheme.dx, k=scheme.k, lim=scheme.lim))#,scale=scheme.scale_factor))
                g2.append(ArtificialDamping(dest=fiber, sources=None, d=scheme.D))
            equations.append(Group(equations=g2))

            g3 = []
            for fiber in scheme.fibers:
                g3.append(HoldPoints(dest=fiber, sources=None, tag=100))
            equations.append(Group(equations=g3))

            # These equations are applied to fiber particles only - that's the
            # reason for computational speed up.
            particles = [p for p in all_particles if p.name in scheme.fibers]
            # A seperate DomainManager is needed to ensure that particles don't
            # leave the domain.
            xmin = domain.manager.xmin
            ymin = domain.manager.ymin
            zmin = domain.manager.zmin
            xmax = domain.manager.xmax
            ymax = domain.manager.ymax
            zmax = domain.manager.zmax
            periodic_in_x = domain.manager.periodic_in_x
            periodic_in_y = domain.manager.periodic_in_y
            periodic_in_z = domain.manager.periodic_in_z
            self.domain = DomainManager(xmin=xmin, xmax=xmax, ymin=ymin,
                                        ymax=ymax, zmin=zmin, zmax=zmax,
                                        periodic_in_x=periodic_in_x,
                                        periodic_in_y=periodic_in_y,
                                        periodic_in_z=periodic_in_z)
            # A seperate list for the nearest neighbourhood search is benefitial
            # since it is much smaller than the original one.
            nnps = LinkedListNNPS(dim=scheme.dim, particles=particles,
                            radius_scale=kernel.radius_scale, domain=self.domain,
                            fixed_h=False, cache=False, sort_gids=False)
            # The acceleration evaluator needs to be set up in order to compile
            # it together with the integrator.
            if parallel:
                self.acceleration_eval = AccelerationEval(
                            particle_arrays=particles,
                            equations=equations,
                            kernel=kernel)
            else:
                self.acceleration_eval = AccelerationEval(
                            particle_arrays=particles,
                            equations=equations,
                            kernel=kernel,
                            mode='serial')
            # Compilation of the integrator not using openmp, because the
            # overhead is too large for those few fiber particles.
            comp = SPHCompiler(self.acceleration_eval, self.fiber_integrator)
            if parallel:
                comp.compile()
            else:
                config = get_config()
                config.use_openmp = False
                comp.compile()
                config.use_openmp = True
            self.acceleration_eval.set_nnps(nnps)

            # Connecting neighbourhood list to integrator.
            self.fiber_integrator.set_nnps(nnps)

    def post_stage(self, current_time, dt, stage):
        """This post stage function gets called after each outer loop and starts
        an inner loop for the fiber iteration."""
        from math import ceil
        if self.innerloop:
            # 1) predictor
            # 2) post stage 1:
            if stage == 1:
                N = int(ceil(self.dt/self.fiber_dt))
                for n in range(0,N):
                    self.fiber_integrator.step(current_time,dt/N)
                    current_time += dt/N
                    if self.domain_updates:
                        self.domain.update()
            # 3) Evaluation
            # 4) post stage 2
