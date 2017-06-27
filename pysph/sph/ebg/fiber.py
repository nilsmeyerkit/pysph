"""
Element Bending Group
##############################

"""

from pysph.sph.equation import Equation
from math import sqrt, acos, sin, pi, floor

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

#class Friction(Equation):
#    r"""**Fiber bending based on friction**#
#
#    Particle acceleration based on fiber bending is computed. The source
#    particles must be chosen to be the same as the destination particles
#    """
#
#    def __init__(self, dest, sources, k):
#        r"""
#        Parameters
#        ----------
#        k : float
#            friction coefficient for torque from vorticity
#        """
#        self.k = k
#        super(Friction, self).__init__(dest, sources)
#
#    def initialize(self, d_idx, d_au, d_av, d_aw):
#        d_au[d_idx] = 0.0
#        d_av[d_idx] = 0.0
#        d_aw[d_idx] = 0.0
#
#    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
#                d_rxprev, d_ryprev, d_rzprev, d_rprev, XIJ, RIJ):
#        '''The loop saves vectors to previous and next particle only.'''
#        if d_idx == s_idx+1:
#            d_rxprev[d_idx] = XIJ[0]
#            d_ryprev[d_idx] = XIJ[1]
#            d_rzprev[d_idx] = XIJ[2]
#            d_rprev[d_idx] = RIJ
#        if d_idx == s_idx-1:
#            d_rxnext[d_idx] = XIJ[0]
#            d_rynext[d_idx] = XIJ[1]
#            d_rznext[d_idx] = XIJ[2]
#            d_rnext[d_idx] = RIJ
#
#    def post_loop(self, d_idx, d_m,
#                d_rxnext, d_rynext, d_rznext, d_rnext,
#                d_rxprev, d_ryprev, d_rzprev, d_rprev,
#                d_au, d_av, d_aw, d_omegax, d_omegay, d_omegaz):
#        if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:
#            # vector to previous particle
#            xab = d_rxprev[d_idx]
#            yab = d_ryprev[d_idx]
#            zab = d_rzprev[d_idx]
#            rab = d_rprev[d_idx]
#            # vector to next particle
#            xbc = d_rxnext[d_idx]
#            ybc = d_rynext[d_idx]
#            zbc = d_rznext[d_idx]
#            rbc = d_rnext[d_idx]
#
#            # Averaged vorticity
#            #ox = (d_omegax[d_idx-1]+d_omegax[d_idx]+d_omegax[d_idx+1])/3
#            #oy = (d_omegay[d_idx-1]+d_omegay[d_idx]+d_omegay[d_idx+1])/3
#            #oz = (d_omegaz[d_idx-1]+d_omegaz[d_idx]+d_omegaz[d_idx+1])/3
#            ox = d_omegax[d_idx]
#            oy = d_omegay[d_idx]
#            oz = d_omegaz[d_idx]
#
#            kabx = self.k*(oy*zab-oz*yab)/(rab**2)
#            kaby = self.k*(oz*xab-ox*zab)/(rab**2)
#            kabz = self.k*(ox*yab-oy*xab)/(rab**2)
#            kbcx = -self.k*(oy*zbc-oz*ybc)/(rbc**2)
#            kbcy = -self.k*(oz*xbc-ox*zbc)/(rbc**2)
#            kbcz = -self.k*(ox*ybc-oy*xbc)/(rbc**2)
#
#            d_au[d_idx+1] += kbcx/d_m[d_idx+1]
#            d_av[d_idx+1] += kbcy/d_m[d_idx+1]
#            d_aw[d_idx+1] += kbcz/d_m[d_idx+1]
#            d_au[d_idx-1] -= kabx/d_m[d_idx-1]
#            d_av[d_idx-1] -= kaby/d_m[d_idx-1]
#            d_aw[d_idx-1] -= kabz/d_m[d_idx-1]

class Friction(Equation):
    r"""**Fiber bending based on friction**

    Rigid rotation based on friction. Does not work across periodic boundaries!
    """

    def __init__(self, dest, sources, k, n):
        r"""
        Parameters
        ----------
        k : float
            friction coefficient for torque from vorticity
        n : int
            rigid motion rod length
        """
        self.k = k
        self.N = n-1
        super(Friction, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_x, d_y, d_z, d_omegax, d_omegay, d_omegaz,
                d_au, d_av, d_aw, d_m, d_tag):

        if ((d_idx+1)%(self.N+1) == 0 and d_idx > 0):
            xx = 0.0; yy = 0.0; zz = 0.0
            ox = 0.0; oy = 0.0; oz = 0.0
            # scale k
            rx = d_x[d_idx]-d_x[d_idx-self.N]
            ry = d_y[d_idx]-d_y[d_idx-self.N]
            rz = d_z[d_idx]-d_z[d_idx-self.N]
            k = self.k/sqrt(rx**2+ry**2+rz**2)
            #k = self.k

            n = self.N+1
            for idx in range(d_idx-self.N, d_idx+1):
                xx += 1.0/n * d_x[idx]
                yy += 1.0/n * d_y[idx]
                zz += 1.0/n * d_z[idx]
                ox += 1.0/n * d_omegax[idx]
                oy += 1.0/n * d_omegay[idx]
                oz += 1.0/n * d_omegaz[idx]

            for idx in range(d_idx-self.N, d_idx+1):
                xab = d_x[idx]-xx
                yab = d_y[idx]-yy
                zab = d_z[idx]-zz

                d_au[idx] += k*(oy*zab-oz*yab)
                d_av[idx] += k*(oz*xab-ox*zab)
                d_aw[idx] += k*(ox*yab-oy*xab)
