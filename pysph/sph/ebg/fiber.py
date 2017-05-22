"""
Element Bending Group
##############################

"""

from pysph.sph.equation import Equation
from math import sqrt, acos, sin, pi

class HoldPoints(Equation):
    r"""**Holds Flagged Points **

    Points tagged with 'tag' are excluded from accelaration. This little trick allows
    testing of EBGs with fixed BCs. 
    """

    def __init__(self, dest, sources, tags, x=True, y=True, z=True):
        r"""
        Parameters
        ----------
        tags : list(float)
            tag of fixed particle
        x : boolean
            True, if x-position should not be changed
        y : boolean
            True, if y-position should not be changed
        z : boolean
            True, if z-position should not be changed
        """
        self.tags = tags
        self.x = x
        self.y = y
        self.z = z
        super(HoldPoints, self).__init__(dest, sources)

    def loop(self, d_idx, d_tag, d_au, d_av, d_aw):
        if d_tag[d_idx] in self.tags :
            if self.x : d_au[d_idx] = 0 
            if self.y : d_av[d_idx] = 0
            if self.z : d_aw[d_idx] = 0

class Tension(Equation):
    r"""**Linear elastic fiber tension**

    Particle acceleration based on fiber tension is computed. First particle in fiber 
    must be tagged with 100 and last one must be tagged with -100.
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
    
    def loop(self, d_idx, d_tag, d_m, d_lprev, d_lnext,
                d_x, d_y, d_z, d_au, d_av, d_aw):
        # interaction with previous particle
        if not d_tag[d_idx] == 100:
            Dx = d_x[d_idx]-d_x[d_idx-1]
            Dy = d_y[d_idx]-d_y[d_idx-1]
            Dz = d_z[d_idx]-d_z[d_idx-1]
            l = sqrt(Dx**2+Dy**2+Dz**2)
            t = self.ea*(l/d_lprev[d_idx]-1)
            d_au[d_idx] -= (t*Dx/l)/d_m[d_idx]
            d_av[d_idx] -= (t*Dy/l)/d_m[d_idx]
            d_aw[d_idx] -= (t*Dz/l)/d_m[d_idx]

        # interaction with next particle
        if not d_tag[d_idx] == -100:
            Dx = d_x[d_idx]-d_x[d_idx+1]
            Dy = d_y[d_idx]-d_y[d_idx+1]
            Dz = d_z[d_idx]-d_z[d_idx+1]
            l = sqrt(Dx**2+Dy**2+Dz**2)
            t = self.ea*(l/d_lprev[d_idx]-1)
            d_au[d_idx] -= (t*Dx/l)/d_m[d_idx]
            d_av[d_idx] -= (t*Dy/l)/d_m[d_idx]
            d_aw[d_idx] -= (t*Dz/l)/d_m[d_idx]

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

    Particle acceleration based on fiber bending is computed. First particle in fiber 
    must be tagged with 100 and last one must be tagged with -100.

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

    def loop(self, d_idx, d_tag, d_m, d_phi0,
                d_x, d_y, d_z, d_au, d_av, d_aw):
        # interaction with previous particle
        if not d_tag[d_idx] == 100 and not d_tag[d_idx] == -100:
            # vector to previous particle
            xab = d_x[d_idx-1]-d_x[d_idx]
            yab = d_y[d_idx-1]-d_y[d_idx]
            zab = d_z[d_idx-1]-d_z[d_idx]
            # vector to next particle
            xbc = d_x[d_idx+1]-d_x[d_idx]
            ybc = d_y[d_idx+1]-d_y[d_idx]
            zbc = d_z[d_idx+1]-d_z[d_idx]

            # norms
            rab = sqrt(xab**2+yab**2+zab**2)
            rbc = sqrt(xbc**2+ybc**2+zbc**2)
            # normed dot product between vectors (limited to catch round off errors)
            dot_prod = (xab*xbc+yab*ybc+zab*zbc)/(rab*rbc)
            dot_prod = max(-1, dot_prod)
            dot_prod = min(1, dot_prod)
            # angle between vectors
            phi = acos(dot_prod)
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
            Fabx = -(My*zab-Mz*yab)/(rab**2)
            Faby = -(Mz*xab-Mx*zab)/(rab**2)
            Fabz = -(Mx*yab-My*xab)/(rab**2)
            Fbcx = (My*zbc-Mz*ybc)/(rbc**2)
            Fbcy = (Mz*xbc-Mx*zbc)/(rbc**2)
            Fbcz = (Mx*ybc-My*xbc)/(rbc**2)

            d_au[d_idx] += (Fabx+Fbcx)/d_m[d_idx]
            d_av[d_idx] += (Faby+Fbcy)/d_m[d_idx]
            d_aw[d_idx] += (Fabz+Fbcz)/d_m[d_idx]
            d_au[d_idx+1] -= Fbcx/d_m[d_idx+1]
            d_av[d_idx+1] -= Fbcy/d_m[d_idx+1]
            d_aw[d_idx+1] -= Fbcz/d_m[d_idx+1]
            d_au[d_idx-1] -= Fabx/d_m[d_idx-1]
            d_av[d_idx-1] -= Faby/d_m[d_idx-1]
            d_aw[d_idx-1] -= Fabz/d_m[d_idx-1]