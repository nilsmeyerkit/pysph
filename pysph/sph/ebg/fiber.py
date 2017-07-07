"""
Element Bending Group
##############################

"""

from pysph.sph.equation import Equation
from math import sqrt, acos, atan, sin, pi, floor

class HoldPoints(Equation):
    r"""**Holds Flagged Points **

    Points tagged with 'hold' are excluded from accelaration. This little trick
    allows testing of Element Bending Groups with fixed BCs.
    """

    def __init__(self, dest, sources, tag, x=True, y=True, z=True):
        r"""
        Parameters
        ----------
        tags : int
            tag of fixed particle
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
             d_awhat):
        if d_holdtag[d_idx] == self.tag :
            if self.x :
                d_au[d_idx] = 0
                d_auhat[d_idx] = 0
            if self.y :
                d_av[d_idx] = 0
                d_avhat[d_idx] = 0
            if self.z :
                d_aw[d_idx] = 0
                d_awhat[d_idx] = 0

class EBGVelocityReset(Equation):
    '''Resets EBG velocities.'''
    def loop(self, d_idx, d_eu, d_ev, d_ew):
        d_eu[d_idx] = 0
        d_ev[d_idx] = 0
        d_ew[d_idx] = 0

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

    def loop(self, d_idx, s_idx, d_m, d_lprev, d_lnext,
             d_au, d_av, d_aw, XIJ, RIJ):
        # interaction with previous particle
        if d_idx == s_idx+1:
            t = self.ea*(RIJ/d_lprev[d_idx]-1)
            d_au[d_idx] -= (t*XIJ[0]/RIJ)/d_m[d_idx]
            d_av[d_idx] -= (t*XIJ[1]/RIJ)/d_m[d_idx]
            d_aw[d_idx] -= (t*XIJ[2]/RIJ)/d_m[d_idx]

        # interaction with next particle
        if d_idx == s_idx-1:
            t = self.ea*(RIJ/d_lnext[d_idx]-1)
            d_au[d_idx] -= (t*XIJ[0]/RIJ)/d_m[d_idx]
            d_av[d_idx] -= (t*XIJ[1]/RIJ)/d_m[d_idx]
            d_aw[d_idx] -= (t*XIJ[2]/RIJ)/d_m[d_idx]

class Damping(Equation):
    r"""**Damp particle motion**

    EBG Particles are damped.
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

    def loop(self, d_idx, d_m, d_eu, d_ev, d_ew, d_au, d_av, d_aw):
        d_au[d_idx] -= 2*self.d*d_eu[d_idx]/d_m[d_idx]
        d_av[d_idx] -= 2*self.d*d_ev[d_idx]/d_m[d_idx]
        d_aw[d_idx] -= 2*self.d*d_ew[d_idx]/d_m[d_idx]


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
                d_rxprev, d_ryprev, d_rzprev, d_rprev, XIJ, RIJ):
        '''The loop saves vectors to previous and next particle only.'''
        if d_idx == s_idx+1:
            d_rxprev[d_idx] = XIJ[0]
            d_ryprev[d_idx] = XIJ[1]
            d_rzprev[d_idx] = XIJ[2]
            d_rprev[d_idx] = RIJ
        if d_idx == s_idx-1:
            d_rxnext[d_idx] = XIJ[0]
            d_rynext[d_idx] = XIJ[1]
            d_rznext[d_idx] = XIJ[2]
            d_rnext[d_idx] = RIJ

    def post_loop(self, d_idx, d_tag, d_m, d_phi0,
                d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev,
                d_au, d_av, d_aw, d_omegax, d_omegay, d_omegaz):
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
    r"""**Fiber bending based on friction**#

    .... The source
    particles must be chosen to be the same as the destination particles
    """

    def __init__(self, dest, sources, J, A, nu, d, ar):
        r"""
        Parameters
        ----------
        J : float
            moment of inertia
        A : float
            shell surface area (2D: 2*dx and 3D: dx*pi*d/2)
        mu : float
            viscosity
        """
        self.J = J
        self.A = A
        self.nu = nu
        self.d = d
        self.ar = ar
        super(Friction, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_testx, d_testy, d_testz):
       d_au[d_idx] = 0.0
       d_av[d_idx] = 0.0
       d_aw[d_idx] = 0.0
       d_testx[d_idx] = 0.0
       d_testy[d_idx] = 0.0
       d_testz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev, d_x, d_y, d_z, XIJ, RIJ):
        '''The loop saves vectors to previous and next particle only.'''
        if d_idx == s_idx+1:
            d_rxprev[d_idx] = XIJ[0]
            d_ryprev[d_idx] = XIJ[1]
            d_rzprev[d_idx] = XIJ[2]
            d_rprev[d_idx] = RIJ
        if d_idx == s_idx-1:
            d_rxnext[d_idx] = XIJ[0]
            d_rynext[d_idx] = XIJ[1]
            d_rznext[d_idx] = XIJ[2]
            d_rnext[d_idx] = RIJ

    def post_loop(self, d_idx, d_m, d_rho, d_x, d_y, d_z,
                d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev,
                d_au, d_av, d_aw, d_omegax, d_omegay, d_omegaz,
                d_testx, d_testy, d_testz, d_dudx, d_dudy, d_dudz, d_dvdx,
                d_dvdy, d_dvdz, d_dwdx, d_dwdy, d_dwdz):
        if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:
            mu = self.nu*d_rho[d_idx]

            dx = d_rxnext[d_idx]-d_rxprev[d_idx]
            dy = d_rynext[d_idx]-d_ryprev[d_idx]
            dz = d_rznext[d_idx]-d_rzprev[d_idx]
            r = sqrt(dx**2+dy**2+dz**2)
            s1 = dx/r
            s2 = dy/r
            s3 = dz/r

            # ensuring that [sx sy sz] is not parallel to [1 0 0]
            if abs(s2) > 1E-14 or abs(s3) > 1E-14:
                fac = (self.A * self.d/2 * mu)/(s2**2+s3**2)

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
                fac = (self.A * self.d/2 * mu)/(s1**2+s3**2)

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
            # just for debugging
            d_testx[d_idx+1] -= (My*d_rznext[d_idx]-Mz*d_rynext[d_idx])/(2*self.J)
            d_testy[d_idx+1] -= (Mz*d_rxnext[d_idx]-Mx*d_rznext[d_idx])/(2*self.J)
            d_testz[d_idx+1] -= (Mx*d_rynext[d_idx]-My*d_rxnext[d_idx])/(2*self.J)

            d_testx[d_idx-1] -= (My*d_rzprev[d_idx]-Mz*d_ryprev[d_idx])/(2*self.J)
            d_testy[d_idx-1] -= (Mz*d_rxprev[d_idx]-Mx*d_rzprev[d_idx])/(2*self.J)
            d_testz[d_idx-1] -= (Mx*d_ryprev[d_idx]-My*d_rxprev[d_idx])/(2*self.J)


    # def post_loop(self, d_idx, d_m, d_rho, d_x, d_y, d_z,
    #             d_rxnext, d_rynext, d_rznext, d_rnext,
    #             d_rxprev, d_ryprev, d_rzprev, d_rprev,
    #             d_au, d_av, d_aw, d_omegax, d_omegay, d_omegaz,
    #             d_testx, d_testy, d_testz, d_dudx, d_dudy, d_dudz, d_dvdx,
    #             d_dvdy, d_dvdz, d_dwdx, d_dwdy, d_dwdz):
    #     if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:
    #         mu = self.nu*d_rho[d_idx]
    #
    #         dx = d_rxnext[d_idx]-d_rxprev[d_idx]
    #         dy = d_rynext[d_idx]-d_ryprev[d_idx]
    #         dz = d_rznext[d_idx]-d_rzprev[d_idx]
    #         r = sqrt(dx**2+dy**2+dz**2)
    #         epsilon = 0.001*r
    #
    #         phiz = atan(dy/(dx+epsilon))
    #         phiy = atan(dz/(dx+epsilon))
    #         phix = atan(dz/(dy+epsilon))
    #
    #         nx = sin(phiz) * sin(phiy)
    #         ny = cos(phiz) * cos(phix)
    #         nz = cos(phiy) * sin(phix)
    #
    #         tx = mu * (d_dudx[d_idx]*nx + d_dudy[d_idx]*ny + d_dudz[d_idx]*nz)
    #         ty = mu * (d_dvdx[d_idx]*nx + d_dvdy[d_idx]*ny + d_dudz[d_idx]*nz)
    #         tz = mu * (d_dwdx[d_idx]*nx + d_dwdy[d_idx]*ny + d_dwdz[d_idx]*nz)
    #
    #         Mx = self.A * self.d/2 * (ty*nz - tz*ny)
    #         My = self.A * self.d/2 * (tz*nx - tx*nz)
    #         Mz = self.A * self.d/2 * (tx*ny - ty*nx)
    #
    #         d_au[d_idx+1] += (My*d_rznext[d_idx]-Mz*d_rynext[d_idx])/(2*self.J)
    #         d_av[d_idx+1] += (Mz*d_rxnext[d_idx]-Mx*d_rznext[d_idx])/(2*self.J)
    #         d_aw[d_idx+1] += (Mx*d_rynext[d_idx]-My*d_rxnext[d_idx])/(2*self.J)
    #
    #         d_au[d_idx-1] += (My*d_rzprev[d_idx]-Mz*d_ryprev[d_idx])/(2*self.J)
    #         d_av[d_idx-1] += (Mz*d_rxprev[d_idx]-Mx*d_rzprev[d_idx])/(2*self.J)
    #         d_aw[d_idx-1] += (Mx*d_ryprev[d_idx]-My*d_rxprev[d_idx])/(2*self.J)
    #
    #         # just for debugging
    #         d_testx[d_idx+1] += (My*d_rznext[d_idx]-Mz*d_rynext[d_idx])/(2*self.J)
    #         d_testy[d_idx+1] += (Mz*d_rxnext[d_idx]-Mx*d_rznext[d_idx])/(2*self.J)
    #         d_testz[d_idx+1] += (Mx*d_rynext[d_idx]-My*d_rxnext[d_idx])/(2*self.J)
    #
    #         d_testx[d_idx-1] += (My*d_rzprev[d_idx]-Mz*d_ryprev[d_idx])/(2*self.J)
    #         d_testy[d_idx-1] += (Mz*d_rxprev[d_idx]-Mx*d_rzprev[d_idx])/(2*self.J)
    #         d_testz[d_idx-1] += (Mx*d_ryprev[d_idx]-My*d_rxprev[d_idx])/(2*self.J)



# class Friction(Equation):
#     r"""**Fiber bending based on friction**
#
#     Rigid rotation based on friction. Does not work across periodic boundaries!
#     """
#
#     def __init__(self, dest, sources, J, k, n=2):
#         r"""
#         Parameters
#         ----------
#         J : float
#             moment of inertia
#         k : float
#             friction coefficient for torque from vorticity
#         n : int
#             rigid motion rod length
#         """
#         self.J = J
#         self.k = k
#         self.N = n-1
#         self.hold_tag_count = 1
#         super(Friction, self).__init__(dest, sources)
#
#     def initialize(self, d_idx, d_au, d_av, d_aw):
#         d_au[d_idx] = 0.0
#         d_av[d_idx] = 0.0
#         d_aw[d_idx] = 0.0
#
#     def loop(self, d_idx, d_x, d_y, d_z, d_omegax, d_omegay, d_omegaz,
#                 d_au, d_av, d_aw, d_m, d_tag, d_testx, d_testy, d_testz,
#                 d_holdtag):
#         if ((d_idx+self.hold_tag_count)%(self.N+1) == 0 and d_idx > 0):
#             xx = 0.0; yy = 0.0; zz = 0.0
#             Mx = 0.0; My = 0.0; Mz = 0.0
#
#             # temporary workaround for angles
#             #ddx = d_x[d_idx-self.N] - d_x[d_idx+1]
#             #ddy = d_y[d_idx-self.N] - d_y[d_idx+1]
#             #phi = atan(ddx/(ddy+0.001*ddx))
#
#             n = self.N+1
#             for idx in range(d_idx-self.N, d_idx+1):
#                 xx += 1.0/n * d_x[idx]
#                 yy += 1.0/n * d_y[idx]
#                 zz += 1.0/n * d_z[idx]
#                 # ox += 1.0/n * d_omegax[idx]
#                 # oy += 1.0/n * d_omegay[idx]
#                 # oz += 1.0/n * d_omegaz[idx]
#                 Mx += 1.0/n*self.k*d_omegax[idx]
#                 My += 1.0/n*self.k*d_omegay[idx]
#                 Mz += 1.0/n*self.k*d_omegaz[idx]
#
#
#             for idx in range(d_idx-self.N, d_idx+1):
#                 xab = d_x[idx]-xx
#                 yab = d_y[idx]-yy
#                 zab = d_z[idx]-zz
#
#                 d_au[idx] += (My*zab-Mz*yab)/self.J
#                 d_av[idx] += (Mz*xab-Mx*zab)/self.J
#                 d_aw[idx] += (Mx*yab-My*xab)/self.J
#
#                 # just for Debugging:
#                 if d_holdtag[idx] == 0:
#                     d_testx[idx] = (My*zab-Mz*yab)/self.J
#                     d_testy[idx] = (Mz*xab-Mx*zab)/self.J
#                     d_testz[idx] = (Mx*yab-My*xab)/self.J
#
#         # handle holds
#         if not d_holdtag[d_idx] == 0:
#             self.hold_tag_count +=1
