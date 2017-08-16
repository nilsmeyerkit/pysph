"""
Equations for Fibers
##############################

"""

from pysph.sph.equation import Equation
from math import sqrt, acos, atan, sin, pi, floor

class HoldPoints(Equation):
    r"""**Holds flagged points **

    Points tagged with 'holdtag' == tag are excluded from accelaration. This
    little trick allows testing of fibers with fixed BCs.
    """

    def __init__(self, dest, sources, tag, x=True, y=True, z=True):
        r"""
        Parameters
        ----------
        tags : int
            tag of fixed particle defined as property 'holdtag'
        x : boolean
            True, if x-position should not be changed
        y : boolean
            True, if y-position should not be changed
        z : boolean
            True, if z-position should not be changed
        """
        self.tag = tag
        self.x = x
        self.y = y
        self.z = z
        super(HoldPoints, self).__init__(dest, sources)

    def loop(self, d_idx, d_holdtag, d_au, d_av, d_aw, d_auhat, d_avhat,
             d_awhat, d_u, d_v, d_w, d_Fx, d_Fy, d_Fz, d_m):
        if d_holdtag[d_idx] == self.tag :
            if self.x:
                d_Fx[d_idx] =  d_m[d_idx] * d_au[d_idx]
                d_au[d_idx] = 0
                d_auhat[d_idx] = 0
                d_u[d_idx] = 0
            if self.y:
                d_Fy[d_idx] =  d_m[d_idx] * d_av[d_idx]
                d_av[d_idx] = 0
                d_avhat[d_idx] = 0
                d_v[d_idx] = 0
            if self.z:
                d_Fz[d_idx] =  d_m[d_idx] * d_aw[d_idx]
                d_aw[d_idx] = 0
                d_awhat[d_idx] = 0
                d_w[d_idx] = 0

class EBGVelocityReset(Equation):
    r"""** Resets EBG velocities **"""

    def loop(self, d_idx, d_eu, d_ev, d_ew, d_ex, d_ey, d_ez, d_u, d_v, d_w, dt):
        d_eu[d_idx] = 0
        d_ev[d_idx] = 0
        d_ew[d_idx] = 0

        d_u[d_idx] += d_ex[d_idx]/dt
        d_v[d_idx] += d_ey[d_idx]/dt
        d_w[d_idx] += d_ez[d_idx]/dt

        d_ex[d_idx] = 0
        d_ey[d_idx] = 0
        d_ez[d_idx] = 0


class Vorticity(Equation):
    r"""** Computes vorticity of velocity field**

    According to Monaghan 1992 (2.12).
    """

    def initialize(self, d_idx, d_omegax, d_omegay, d_omegaz):
        d_omegax[d_idx] = 0.0
        d_omegay[d_idx] = 0.0
        d_omegaz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_omegax, d_omegay, d_omegaz,
            DWIJ, VIJ):
        v = s_m[s_idx]/d_rho[d_idx]
        d_omegax[d_idx] += v*(VIJ[1]*DWIJ[2]-VIJ[2]*DWIJ[1])
        d_omegay[d_idx] += v*(VIJ[2]*DWIJ[0]-VIJ[0]*DWIJ[2])
        d_omegaz[d_idx] += v*(VIJ[0]*DWIJ[1]-VIJ[1]*DWIJ[0])

class VelocityGradient(Equation):
    r"""** Computes 2nd order tensor representing the velocity gradient**
    """

    def initialize(self, d_idx, d_dudx, d_dudy, d_dudz, d_dvdx, d_dvdy, d_dvdz,
                    d_dwdx, d_dwdy, d_dwdz):
        d_dudx[d_idx] = 0.0
        d_dudy[d_idx] = 0.0
        d_dudz[d_idx] = 0.0

        d_dvdx[d_idx] = 0.0
        d_dvdy[d_idx] = 0.0
        d_dvdz[d_idx] = 0.0

        d_dwdx[d_idx] = 0.0
        d_dwdy[d_idx] = 0.0
        d_dwdz[d_idx] = 0.0


    def loop(self, d_idx, s_idx, d_rho, s_m, d_dudx, d_dudy, d_dudz, d_dvdx,
            d_dvdy, d_dvdz,d_dwdx, d_dwdy, d_dwdz, DWIJ, VIJ):
        v = s_m[s_idx]/d_rho[d_idx]
        d_dudx[d_idx] -= v*VIJ[0]*DWIJ[0]
        d_dudy[d_idx] -= v*VIJ[0]*DWIJ[1]
        d_dudz[d_idx] -= v*VIJ[0]*DWIJ[2]

        d_dvdx[d_idx] -= v*VIJ[1]*DWIJ[0]
        d_dvdy[d_idx] -= v*VIJ[1]*DWIJ[1]
        d_dvdz[d_idx] -= v*VIJ[1]*DWIJ[2]

        d_dwdx[d_idx] -= v*VIJ[2]*DWIJ[0]
        d_dwdy[d_idx] -= v*VIJ[2]*DWIJ[1]
        d_dwdz[d_idx] -= v*VIJ[2]*DWIJ[2]


class Tension(Equation):
    r"""**Linear elastic fiber tension**

    Particle acceleration based on fiber tension is computed. The source
    must be chosen to be the same as the destination particles
    """

    def __init__(self, dest, sources, ea):
        r"""
        Parameters
        ----------
        ea : float
            rod stiffness (elastic modulus x section area)
        """
        self.ea = ea
        super(Tension, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_lprev, d_lnext, s_fractag, d_fractag,
             d_au, d_av, d_aw, XIJ, RIJ):
        # interaction with previous particle
        if d_idx == s_idx+1 and s_fractag[s_idx] == 0:
            t = self.ea*(RIJ/d_lprev[d_idx]-1)
            d_au[d_idx] -= (t*XIJ[0]/RIJ)/d_m[d_idx]
            d_av[d_idx] -= (t*XIJ[1]/RIJ)/d_m[d_idx]
            d_aw[d_idx] -= (t*XIJ[2]/RIJ)/d_m[d_idx]

        # interaction with next particle
        if d_idx == s_idx-1 and d_fractag[d_idx] == 0:
            t = self.ea*(RIJ/d_lnext[d_idx]-1)
            d_au[d_idx] -= (t*XIJ[0]/RIJ)/d_m[d_idx]
            d_av[d_idx] -= (t*XIJ[1]/RIJ)/d_m[d_idx]
            d_aw[d_idx] -= (t*XIJ[2]/RIJ)/d_m[d_idx]

class ArtificialDamping(Equation):
    r"""**Damp EBG particle motion**

    EBG Particles are damped based on EBG velocity. Use this in combination with
    EBGStep and EBGVelocityReset.
    """

    def __init__(self, dest, sources, d):
        r"""
        Parameters
        ----------
        d : float
            damping coefficient
        """
        self.d = d
        super(ArtificialDamping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_eu, d_ev, d_ew, d_au, d_av, d_aw):
        d_au[d_idx] -= 2*self.d*d_eu[d_idx]/d_m[d_idx]
        d_av[d_idx] -= 2*self.d*d_ev[d_idx]/d_m[d_idx]
        d_aw[d_idx] -= 2*self.d*d_ew[d_idx]/d_m[d_idx]

class Damping(Equation):
    r"""**Damp particle motion**

    Particles are damped. Difference to ArtificialDamping: This damps real
    particle velocities and therefore affects not only the fiber iteration.
    """

    def __init__(self, dest, sources, d):
        r"""
        Parameters
        ----------
        d : float
            damping coefficient
        """
        self.d = d
        super(Damping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_au, d_av, d_aw):
        d_au[d_idx] -= 2*self.d*d_u[d_idx]/d_m[d_idx]
        d_av[d_idx] -= 2*self.d*d_v[d_idx]/d_m[d_idx]
        d_aw[d_idx] -= 2*self.d*d_w[d_idx]/d_m[d_idx]

class Bending(Equation):
    r"""**Linear elastic fiber bending**

    Particle acceleration based on fiber bending is computed. The source
    particles must be chosen to be the same as the destination particles
    """

    def __init__(self, dest, sources, ei):
        r"""
        Parameters
        ----------
        ei : float
            bending stiffness (elastic modulus x 2nd order moment)
        """
        self.ei = ei
        super(Bending, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev, s_fractag,
                d_fractag, XIJ, RIJ):
        '''The loop saves vectors to previous and next particle only.'''
        if d_idx == s_idx+1:
            if s_fractag[s_idx] == 0:
                d_rxprev[d_idx] = XIJ[0]
                d_ryprev[d_idx] = XIJ[1]
                d_rzprev[d_idx] = XIJ[2]
                d_rprev[d_idx] = RIJ
            else:
                d_rprev[d_idx] = 0.0
        if d_idx == s_idx-1:
            if d_fractag[d_idx] == 0:
                d_rxnext[d_idx] = XIJ[0]
                d_rynext[d_idx] = XIJ[1]
                d_rznext[d_idx] = XIJ[2]
                d_rnext[d_idx] = RIJ
            else:
                d_rnext[d_idx] = 0.0

    def post_loop(self, d_idx, d_m, d_phi0, d_fractag, d_phifrac,
                d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev,
                d_au, d_av, d_aw):
        if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:
            # vector to previous particle
            xab = d_rxprev[d_idx]
            yab = d_ryprev[d_idx]
            zab = d_rzprev[d_idx]
            rab = d_rprev[d_idx]
            # vector to next particle
            xbc = d_rxnext[d_idx]
            ybc = d_rynext[d_idx]
            zbc = d_rznext[d_idx]
            rbc = d_rnext[d_idx]

            # normed dot product between vectors
            # (limited to catch round off errors)
            dot_prod_norm = (xab*xbc+yab*ybc+zab*zbc)/(rab*rbc)
            dot_prod_norm = max(-1, dot_prod_norm)
            dot_prod_norm = min(1, dot_prod_norm)
            # angle between vectors
            phi = acos(dot_prod_norm)
            # direction of angle from cross product
            norm = rab*rbc*sin(phi)
            nx = (yab*zbc-zab*ybc)/norm
            ny = (zab*xbc-xab*zbc)/norm
            nz = (xab*ybc-yab*xbc)/norm

            # momentum
            Mx = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*nx
            My = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*ny
            Mz = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*nz

            if abs(phi-d_phi0[d_idx]) > d_phifrac[d_idx]:
                d_fractag[d_idx] = 1
                #d_fractag[d_idx+1] = 1

            # forces on neighbouring particles
            Fabx = (My*zab-Mz*yab)/(rab**2)
            Faby = (Mz*xab-Mx*zab)/(rab**2)
            Fabz = (Mx*yab-My*xab)/(rab**2)
            Fbcx = (My*zbc-Mz*ybc)/(rbc**2)
            Fbcy = (Mz*xbc-Mx*zbc)/(rbc**2)
            Fbcz = (Mx*ybc-My*xbc)/(rbc**2)

            d_au[d_idx] += (Fabx-Fbcx)/d_m[d_idx]
            d_av[d_idx] += (Faby-Fbcy)/d_m[d_idx]
            d_aw[d_idx] += (Fabz-Fbcz)/d_m[d_idx]
            d_au[d_idx+1] += Fbcx/d_m[d_idx+1]
            d_av[d_idx+1] += Fbcy/d_m[d_idx+1]
            d_aw[d_idx+1] += Fbcz/d_m[d_idx+1]
            d_au[d_idx-1] -= Fabx/d_m[d_idx-1]
            d_av[d_idx-1] -= Faby/d_m[d_idx-1]
            d_aw[d_idx-1] -= Fabz/d_m[d_idx-1]


class Friction(Equation):
    r"""**Fiber bending due to friction on fictive surfaces**

    Since the fiber represented by a beadchain of particles has no thickness, a
    term has do compensate fritction due to shear on the particles surface.
    The source particles must be chosen to be the same as the destination
    particles.
    """

    def __init__(self, dest, sources, J, A, mu, d, ar):
        r"""
        Parameters
        ----------
        J : float
            moment of inertia
        A : float
            shell surface area (2D: 2*dx and 3D: dx*pi*d/2)
        mu : float
            absolute viscosity
        d : float
            fiber diameter
        ar : float
            fiber aspect ratio
        """
        self.J = J
        self.A = A
        self.mu = mu
        self.d = d
        self.ar = ar
        super(Friction, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
       d_au[d_idx] = 0.0
       d_av[d_idx] = 0.0
       d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev, d_fractag, s_fractag,
                XIJ, RIJ):
        '''The loop saves vectors to previous and next particle only.'''
        if d_idx == s_idx+1:
            if s_fractag[s_idx] == 0:
                d_rxprev[d_idx] = XIJ[0]
                d_ryprev[d_idx] = XIJ[1]
                d_rzprev[d_idx] = XIJ[2]
                d_rprev[d_idx] = RIJ
            else:
                d_rprev[d_idx] = 0.0
        if d_idx == s_idx-1:
            if d_fractag[d_idx] == 0:
                d_rxnext[d_idx] = XIJ[0]
                d_rynext[d_idx] = XIJ[1]
                d_rznext[d_idx] = XIJ[2]
                d_rnext[d_idx] = RIJ
            else:
                d_rnext[d_idx] = 0.0

    def post_loop(self, d_idx, d_m, d_rho, d_rxnext, d_rynext, d_rznext,
                d_rnext, d_rxprev, d_ryprev, d_rzprev, d_rprev, d_fractag,
                d_au, d_av, d_aw, d_dudx, d_dudy, d_dudz, d_dvdx,
                d_dvdy, d_dvdz, d_dwdx, d_dwdy, d_dwdz):
        if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:

            #mu = self.nu*d_rho[d_idx]

            dx = d_rxnext[d_idx]-d_rxprev[d_idx]
            dy = d_rynext[d_idx]-d_ryprev[d_idx]
            dz = d_rznext[d_idx]-d_rzprev[d_idx]
            r = sqrt(dx**2+dy**2+dz**2)
            s1 = dx/r
            s2 = dy/r
            s3 = dz/r

            # ensuring that [sx sy sz] is not parallel to [1 0 0]
            if abs(s2) > 1E-14 or abs(s3) > 1E-14:
                fac = (2*self.A * self.d/2 * self.mu)/(s2**2+s3**2)

                Mx = fac*((s1*s2**2*s3+s1*s3**3)*d_dvdx[d_idx]
                    +(-s1**2*s2*s3+s2*s3)*d_dvdy[d_idx]
                    +(-s1**2*s3**2-s2**2)*d_dvdz[d_idx]
                    +(-s1*s2**3-s1*s2*s3**2)*d_dwdx[d_idx]
                    +(s1**2*s2**2+s3**2)*d_dwdy[d_idx]
                    +(s1**2*s2*s3-s2*s3)*d_dwdz[d_idx])
                My = fac*((-s1*s2**2*s3-s1*s3**3)*d_dudx[d_idx]
                    +(s1**2*s2*s3-s2*s3)*d_dudy[d_idx]
                    +(s1**2*s3**2+s2**2)*d_dudz[d_idx]
                    +(-s2**4-2*s2**2*s3**2-s3**4)*d_dwdx[d_idx]
                    +(s1*s2**3+s1*s2*s3**2)*d_dwdy[d_idx]
                    +(s1*s2**2*s3+s1*s3**3)*d_dwdz[d_idx])
                Mz = fac*((s1*s2**3+s1*s2*s3**2)*d_dudx[d_idx]
                    +(-s1**2*s2**2-s3**2)*d_dudy[d_idx]
                    +(-s1**2*s2*s3+s2*s3)*d_dudz[d_idx]
                    +(s2**4+2*s2**2*s3**2+s3**4)*d_dvdx[d_idx]
                    +(-s1*s2**3-s1*s2*s3**2)*d_dvdy[d_idx]
                    +(-s1*s2**2*s3-s1*s3**3)*d_dvdz[d_idx])
            else:
                fac = (2*self.A * self.d/2 * self.mu)/(s1**2+s3**2)

                Mx = fac*((-s1*s2**2*s3+s1*s3)*d_dvdx[d_idx]
                    +(s1**2*s2*s3+s2*s3**3)*d_dvdy[d_idx]
                    +(-s2**2*s3**2-s1**2)*d_dvdz[d_idx]
                    +(-s1**3*s2-s1*s2*s3**2)*d_dwdx[d_idx]
                    +(s1**4+2*s1**2*s3**2+s3**4)*d_dwdy[d_idx]
                    +(-s1**2*s2*s3-s2*s3**3)*d_dwdz[d_idx])
                My = fac*((s1*s2**2*s3-s1*s3)*d_dudx[d_idx]
                    +(-s1**2*s2*s3-s2*s3**3)*d_dudy[d_idx]
                    +(s2**2*s3**2+s1**2)*d_dudz[d_idx]
                    +(-s1**2*s2**2-s3**2)*d_dwdx[d_idx]
                    +(s1**3*s2+s1*s2*s3**2)*d_dwdy[d_idx]
                    +(-s1*s2**2*s3+s1*s3)*d_dwdz[d_idx])
                Mz = fac*((s1**3*s2+s1*s2*s3**2)*d_dudx[d_idx]
                    +(-s1**4-2*s1**2*s3**2-s3**4)*d_dudy[d_idx]
                    +(s1**2*s2*s3+s2*s3**3)*d_dudz[d_idx]
                    +(s1**2*s2**2+s3**2)*d_dvdx[d_idx]
                    +(-s1**3*s2-s1*s2*s3**2)*d_dvdy[d_idx]
                    +(s1*s2**2*s3-s1*s3)*d_dvdz[d_idx])

            d_au[d_idx+1] -= (My*d_rznext[d_idx]-Mz*d_rynext[d_idx])/(2*self.J)
            d_av[d_idx+1] -= (Mz*d_rxnext[d_idx]-Mx*d_rznext[d_idx])/(2*self.J)
            d_aw[d_idx+1] -= (Mx*d_rynext[d_idx]-My*d_rxnext[d_idx])/(2*self.J)

            d_au[d_idx-1] -= (My*d_rzprev[d_idx]-Mz*d_ryprev[d_idx])/(2*self.J)
            d_av[d_idx-1] -= (Mz*d_rxprev[d_idx]-Mx*d_rzprev[d_idx])/(2*self.J)
            d_aw[d_idx-1] -= (Mx*d_ryprev[d_idx]-My*d_rxprev[d_idx])/(2*self.J)
        # else:
        #     # end points
        #     if d_rnext[d_idx] < 1E-14:
        #         nx = -d_rxprev[d_idx]/d_rprev[d_idx]
        #         ny = -d_ryprev[d_idx]/d_rprev[d_idx]
        #         nz = -d_rzprev[d_idx]/d_rprev[d_idx]
        #     else:
        #         nx = -d_rxnext[d_idx]/d_rnext[d_idx]
        #         ny = -d_rynext[d_idx]/d_rnext[d_idx]
        #         nz = -d_rznext[d_idx]/d_rnext[d_idx]
        #
        #     fac = -self.A*self.mu/d_m[d_idx]*self.ar/(self.ar-1)
        #
        #     d_au[d_idx] += fac * (d_dudx[d_idx]*nx
        #                             +d_dudy[d_idx]*ny
        #                             +d_dudz[d_idx]*nz)
        #     d_av[d_idx] += fac * (d_dvdx[d_idx]*nx
        #                             +d_dvdy[d_idx]*ny
        #                             +d_dvdz[d_idx]*nz)
        #     d_aw[d_idx] += fac * (d_dwdx[d_idx]*nx
        #                             +d_dwdy[d_idx]*ny
        #                             +d_dwdz[d_idx]*nz)

class SimpleContact(Equation):
    """This class computes simple fiber repulsion to stop penetration. It
    computes the force between two spheres as Hertz pressure."""
    def __init__(self, dest, sources, E, d, pois=0.3):
        r"""
        Parameters
        ----------
        E : float
            Young's modulus
        d : float
            fiber diameter
        pois : flost
            poisson number
        """
        self.E = E
        self.d = d
        self.pois = pois
        super(SimpleContact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
       d_au[d_idx] = 0.0
       d_av[d_idx] = 0.0
       d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_au, d_av, d_aw, XIJ, RIJ):
        if not s_idx == d_idx and RIJ < self.d:
            E_star = 1/(2*((1-self.pois**2)/self.E))
            # effective radius for two spheres of same size
            R = self.d/4
            F = 4/3 * E_star * sqrt(R) * abs(self.d-RIJ)**1.5
            d_au[d_idx] += XIJ[0]/RIJ * F/d_m[d_idx]
            d_av[d_idx] += XIJ[1]/RIJ * F/d_m[d_idx]
            d_aw[d_idx] += XIJ[2]/RIJ * F/d_m[d_idx]

class Contact(Equation):
    """This class computes fiber repulsion to stop penetration. It
    computes the force between two spheres based on Hertz pressure between two
    cylinders. This Equation requires a computation of ditances by the Bending
    equation."""
    def __init__(self, dest, sources, E, d, pois=0.3, k=0, scale=1):
        r"""
        Parameters
        ----------
        E : float
            Young's modulus
        d : float
            fiber diameter
        pois : float
            poisson number
        k : float
            friction coefficient between fibers
        scale : float
            scale factor countering mass scaling
        """
        self.E = E
        self.d = d
        self.pois = pois
        self.k = k
        self.scale = scale
        super(Contact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_Fx, d_Fy, d_Fz):
       d_au[d_idx] = 0.0
       d_av[d_idx] = 0.0
       d_aw[d_idx] = 0.0
       d_Fx[d_idx] = 0.0
       d_Fy[d_idx] = 0.0
       d_Fz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_au, d_av, d_aw, d_rxnext, d_rynext,
        d_rznext, d_rnext, d_rxprev, d_ryprev, d_rzprev, d_rprev, s_rxnext,
        s_rynext, s_rznext, s_rnext, s_rxprev, s_ryprev, s_rzprev, s_rprev,
        d_Fx, d_Fy, d_Fz, d_fractag, XIJ, VIJ, RIJ):

        # not contact, if
        # - particle is too far away to cause contact (sqrt(6)*R is max. dist.)
        # - source particle is destination particle
        # - particle is neighbour
        if (RIJ > 1E-14
            and RIJ < sqrt(6)/2*self.d
            and abs(RIJ-d_rprev[d_idx]) > 1E-14
            and abs(RIJ-d_rnext[d_idx]) > 1E-14):

            # elastic factor from Hertz' pressure in contact
            E_star = 1/(2*((1-self.pois**2)/self.E))

            # case for two fiber ends
            if ((d_rnext[d_idx] < 1E-14 or d_rprev[d_idx] < 1E-14) and
                (s_rnext[s_idx] < 1E-14 or s_rprev[s_idx] < 1E-14)):

                F = self.scale*2*max(self.d-RIJ,0)*self.d*E_star
                V = sqrt(VIJ[0]**2 + VIJ[1]**2 + VIJ[2]**2)

                d_Fx[d_idx] += (XIJ[0]/RIJ * F - self.k*VIJ[0]/V)/d_m[d_idx]
                d_Fy[d_idx] += (XIJ[1]/RIJ * F - self.k*VIJ[1]/V)/d_m[d_idx]
                d_Fz[d_idx] += (XIJ[2]/RIJ * F - self.k*VIJ[2]/V)/d_m[d_idx]

                d_au[d_idx] += (XIJ[0]/RIJ * F - self.k*VIJ[0]/V)/d_m[d_idx]
                d_av[d_idx] += (XIJ[1]/RIJ * F - self.k*VIJ[1]/V)/d_m[d_idx]
                d_aw[d_idx] += (XIJ[2]/RIJ * F - self.k*VIJ[2]/V)/d_m[d_idx]

            # case for fiber end in destination fiber
            elif d_rnext[d_idx] < 1E-14 or d_rprev[d_idx] < 1E-14:
                # direction of source fiber
                sx = s_rxnext[s_idx]-s_rxprev[s_idx]
                sy = s_rynext[s_idx]-s_ryprev[s_idx]
                sz = s_rznext[s_idx]-s_rzprev[s_idx]
                sr = sqrt(sx**2+sy**2+sz**2)

                # relative velocity for friction term
                v_rel = sx*VIJ[0]+sy*VIJ[1]+sz*VIJ[2]
                v_rel_x = v_rel * sx/sr
                v_rel_y = v_rel * sy/sr
                v_rel_z = v_rel * sz/sr
                v_rel = sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
                v_rel = v_rel if v_rel > 1E-14 else 1

                # distance computation
                dot_prod = (sx*XIJ[0] + sy*XIJ[1] + sz*XIJ[2])/sr
                tx = XIJ[0]-dot_prod*sx/sr
                ty = XIJ[1]-dot_prod*sy/sr
                tz = XIJ[2]-dot_prod*sz/sr
                tr = sqrt(tx**2 + ty**2 + tz**2)

                F = self.scale*2*max(self.d-tr,0)*self.d*E_star

                d_Fx[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                d_Fy[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                d_Fz[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]

                d_au[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                d_av[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                d_aw[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]

            elif s_rnext[s_idx] < 1E-14 or s_rprev[s_idx] < 1E-14:
                # direction of destination fiber
                dx = d_rxnext[d_idx]-d_rxprev[d_idx]
                dy = d_rynext[d_idx]-d_ryprev[d_idx]
                dz = d_rznext[d_idx]-d_rzprev[d_idx]
                dr = sqrt(dx**2+dy**2+dz**2)

                # relative velocity for friction term
                v_rel = dx*VIJ[0]+dy*VIJ[1]+dz*VIJ[2]
                v_rel_x = v_rel * dx/dr
                v_rel_y = v_rel * dy/dr
                v_rel_z = v_rel * dz/dr
                v_rel = sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
                v_rel =  v_rel if v_rel > 1E-14 else 1

                # distance computation
                dot_prod = (dx*XIJ[0] + dy*XIJ[1] + dz*XIJ[2])/dr
                tx = XIJ[0]-dot_prod*dx/dr
                ty = XIJ[1]-dot_prod*dy/dr
                tz = XIJ[2]-dot_prod*dz/dr
                tr = sqrt(tx**2 + ty**2 + tz**2)

                F = self.scale*2*max(self.d-tr,0)*self.d*E_star

                d_Fx[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                d_Fy[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                d_Fz[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]

                d_au[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                d_av[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                d_aw[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]
            else:
                # direction of destination fiber
                dx = d_rxnext[d_idx]-d_rxprev[d_idx]
                dy = d_rynext[d_idx]-d_ryprev[d_idx]
                dz = d_rznext[d_idx]-d_rzprev[d_idx]
                dr = sqrt(dx**2+dy**2+dz**2)
                dx = dx/dr; dy = dy/dr; dz = dz/dr

                # direction of source fiber
                sx = s_rxnext[s_idx]-s_rxprev[s_idx]
                sy = s_rynext[s_idx]-s_ryprev[s_idx]
                sz = s_rznext[s_idx]-s_rzprev[s_idx]
                sr = sqrt(sx**2+sy**2+sz**2)
                sx = sx/sr; sy = sy/sr; sz = sz/sr

                # normal direction at contact
                nx = dy * sz - dz * sy
                ny = dz * sx - dx * sz
                nz = dx * sy - dy * sx
                nr = sqrt(nx**2+ny**2+nz**2)

                # 3 vectors not in plane
                if abs(nx*XIJ[0]+ny*XIJ[1]+nz*XIJ[2]) > 1E-14:
                    nx = -nx/nr
                    ny = -ny/nr
                    nz = -nz/nr

                    # relative velocity in each fiber direction
                    v_rel_d = dx*VIJ[0]+dy*VIJ[1]+dz*VIJ[2]
                    v_rel_s = sx*VIJ[0]+sy*VIJ[1]+sz*VIJ[2]
                    v_rel_x = v_rel_d * dx/dr + v_rel_s * sx/sr
                    v_rel_y = v_rel_d * dy/dr + v_rel_s * sy/sr
                    v_rel_z = v_rel_d * dz/dr + v_rel_s * sz/sr
                    v_rel = sqrt(v_rel_x**2+v_rel_y**2+v_rel_z**2)
                    v_rel = v_rel if v_rel > 1E-14 else 1

                    y = nx*XIJ[0]+ny*XIJ[1]+nz*XIJ[2]

                    F = self.scale*2*max(self.d-y,0)*self.d*E_star

                    d_Fx[d_idx] += (F*nx - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                    d_Fy[d_idx] += (F*ny - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                    d_Fz[d_idx] += (F*nz - self.k*F*v_rel_z/v_rel)/d_m[d_idx]

                    d_au[d_idx] += (F*nx - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                    d_av[d_idx] += (F*ny - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                    d_aw[d_idx] += (F*nz - self.k*F*v_rel_z/v_rel)/d_m[d_idx]
                else:
                    # direction of destination fiber
                    dx = d_rxnext[d_idx]-d_rxprev[d_idx]
                    dy = d_rynext[d_idx]-d_ryprev[d_idx]
                    dz = d_rznext[d_idx]-d_rzprev[d_idx]
                    dr = sqrt(dx**2+dy**2+dz**2)

                    # relative velocity for friction term
                    v_rel = dx*VIJ[0]+dy*VIJ[1]+dz*VIJ[2]
                    v_rel_x = v_rel * dx/dr
                    v_rel_y = v_rel * dy/dr
                    v_rel_z = v_rel * dz/dr
                    v_rel = sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
                    v_rel =  v_rel if v_rel > 1E-14 else 1

                    # distance computation
                    dot_prod = (dx*XIJ[0] + dy*XIJ[1] + dz*XIJ[2])/dr
                    tx = XIJ[0]-dot_prod*dx/dr
                    ty = XIJ[1]-dot_prod*dy/dr
                    tz = XIJ[2]-dot_prod*dz/dr
                    tr = sqrt(tx**2 + ty**2 + tz**2)

                    F = self.scale*2*max(self.d-tr,0)*self.d*E_star

                    d_Fx[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                    d_Fy[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                    d_Fz[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]

                    d_au[d_idx] += (F*tx/tr - self.k*F*v_rel_x/v_rel)/d_m[d_idx]
                    d_av[d_idx] += (F*ty/tr - self.k*F*v_rel_y/v_rel)/d_m[d_idx]
                    d_aw[d_idx] += (F*tz/tr - self.k*F*v_rel_z/v_rel)/d_m[d_idx]
