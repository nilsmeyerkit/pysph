"""
Utitlity equations for fibers
##############################

"""

from pysph.sph.equation import Equation
from math import sqrt, acos, atan, sin, pi, floor

class ComputeDistance(Equation):
    r"""** Compute Distances to neighbours**
    The loop saves vectors to previous and next particle only."""

    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
                d_rxprev, d_ryprev, d_rzprev, d_rprev, s_fractag, d_fidx,
                s_fidx, d_fractag, s_rxnext, s_rynext, s_rznext, s_rnext,
                s_rxprev, s_ryprev, s_rzprev, s_rprev, XIJ, RIJ):
        if d_fidx[d_idx] == s_fidx[s_idx]+1:
            if s_fractag[s_idx] == 0:
                d_rxprev[d_idx] = -XIJ[0]
                d_ryprev[d_idx] = -XIJ[1]
                d_rzprev[d_idx] = -XIJ[2]
                d_rprev[d_idx] = RIJ
            else:
                d_rxprev[d_idx] = 0.0
                d_ryprev[d_idx] = 0.0
                d_rzprev[d_idx] = 0.0
                d_rprev[d_idx] = 0.0
                s_rnext[s_idx] = 0.0
                s_rxnext[s_idx] = 0.0
                s_rynext[s_idx] = 0.0
                s_rznext[s_idx] = 0.0
        if d_fidx[d_idx] == s_fidx[s_idx]-1:
            if d_fractag[d_idx] == 0:
                d_rxnext[d_idx] = -XIJ[0]
                d_rynext[d_idx] = -XIJ[1]
                d_rznext[d_idx] = -XIJ[2]
                d_rnext[d_idx] = RIJ
            else:
                s_rxprev[s_idx] = 0.0
                s_ryprev[s_idx] = 0.0
                s_rzprev[s_idx] = 0.0
                s_rprev[s_idx] = 0.0
                d_rxnext[d_idx] = 0.0
                d_rynext[d_idx] = 0.0
                d_rznext[d_idx] = 0.0
                d_rnext[d_idx] = 0.0

class HoldPoints(Equation):
    r"""**Holds flagged points **

    Points tagged with 'holdtag' == tag are excluded from accelaration. This
    little trick allows testing of fibers with fixed BCs.
    """

    def __init__(self, dest, sources, tag, x=True, y=True, z=True,
                 mirror_particle=0):
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
        mirror_particle : int
            idx shift to a particle, which displacement should be mirrored to
            origin
        """
        self.tag = tag
        self.x = x
        self.y = y
        self.z = z
        self.mirror = mirror_particle
        super(HoldPoints, self).__init__(dest, sources)

    def loop(self, d_idx, d_holdtag, d_au, d_av, d_aw, d_auhat, d_avhat,
             d_awhat, d_u, d_v, d_w, d_x, d_y, d_z, d_Fx, d_Fy, d_Fz, d_m):
        if d_holdtag[d_idx] == self.tag :
            if self.x:
                d_Fx[d_idx] =  d_m[d_idx] * d_au[d_idx]
                d_au[d_idx] = 0
                d_auhat[d_idx] = 0
                d_u[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_x[d_idx+self.mirror] = -d_x[d_idx]
                d_u[d_idx+self.mirror] = -d_u[d_idx]

            if self.y:
                d_Fy[d_idx] =  d_m[d_idx] * d_av[d_idx]
                d_av[d_idx] = 0
                d_avhat[d_idx] = 0
                d_v[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_y[d_idx+self.mirror] = -d_y[d_idx]
                d_v[d_idx+self.mirror] = -d_v[d_idx]

            if self.z:
                d_Fz[d_idx] =  d_m[d_idx] * d_aw[d_idx]
                d_aw[d_idx] = 0
                d_awhat[d_idx] = 0
                d_w[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_z[d_idx+self.mirror] = -d_z[d_idx]
                d_w[d_idx+self.mirror] = -d_w[d_idx]

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

    def loop(self, d_idx, s_idx, d_fidx, s_fidx, d_m, d_au, d_av, d_aw,
            XIJ, RIJ):
        if not s_fidx[s_idx] == d_fidx[d_idx] and RIJ < self.d:
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
    def __init__(self, dest, sources, E, d, dim, pois=0.3, k=0.0, lim=0.5,
                 eta0=0.0):
        r"""
        Parameters
        ----------
        E : float
            Young's modulus
        d : float
            fiber diameter
        dim : int
            dimensionaltiy of the problem
        pois : float
            poisson number
        k : float
            friction coefficient between fibers
        eta0 : float
            viscosity of suspension fluid
        """
        self.E = E
        self.d = d
        self.pois = pois
        self.k = k
        self.dim = dim
        self.lim = lim
        self.eta0 = eta0
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0
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
        d_Fx, d_Fy, d_Fz, d_fractag, d_tag, s_tag, XIJ, VIJ, RIJ):

        # not contact, if
        # - particle is too far away to cause contact (sqrt(6)*R is max. dist.)
        # - source particle is destination particle
        # - particle is neighbour
        if (RIJ > 1E-14
            and RIJ < 1.5*self.d
            and abs(RIJ-d_rprev[d_idx]) > 1E-14
                and abs(RIJ-d_rnext[d_idx]) > 1E-14):

            # case for two fiber ends
            if ((d_rnext[d_idx] < 1E-14 or d_rprev[d_idx] < 1E-14) and
                    (s_rnext[s_idx] < 1E-14 or s_rprev[s_idx] < 1E-14)):

                d = self.d - RIJ

                self.compute_force(
                    XIJ[0]/RIJ, XIJ[1]/RIJ, XIJ[2]/RIJ, VIJ[0], VIJ[1], VIJ[2],
                    d)

            # case for fiber end in destination fiber
            elif d_rnext[d_idx] < 1E-14 or d_rprev[d_idx] < 1E-14:

                # direction of source fiber
                sx = s_rxprev[s_idx] - s_rxnext[s_idx]
                sy = s_ryprev[s_idx] - s_rynext[s_idx]
                sz = s_rzprev[s_idx] - s_rznext[s_idx]
                sr = sqrt(sx**2 + sy**2 + sz**2)

                # relative velocity for friction term
                v_rel = sx*VIJ[0] + sy*VIJ[1] + sz*VIJ[2]
                v_rel_x = v_rel * sx/sr
                v_rel_y = v_rel * sy/sr
                v_rel_z = v_rel * sz/sr

                # distance computation
                dot_prod = (sx*XIJ[0] + sy*XIJ[1] + sz*XIJ[2])/sr
                tx = XIJ[0] - dot_prod*sx/sr
                ty = XIJ[1] - dot_prod*sy/sr
                tz = XIJ[2] - dot_prod*sz/sr
                tr = sqrt(tx**2 + ty**2 + tz**2)

                d = self.d - RIJ

                self.compute_force(
                    tx/tr, ty/tr, tz/tr, v_rel_x, v_rel_y, v_rel_z, d)

            # case for fiber end in source fiber
            elif s_rnext[s_idx] < 1E-14 or s_rprev[s_idx] < 1E-14:
                # direction of destination fiber
                dx = d_rxprev[d_idx]-d_rxnext[d_idx]
                dy = d_ryprev[d_idx]-d_rynext[d_idx]
                dz = d_rzprev[d_idx]-d_rznext[d_idx]
                dr = sqrt(dx**2 + dy**2 + dz**2)

                # relative velocity for friction term
                v_rel = dx*VIJ[0] + dy*VIJ[1] + dz*VIJ[2]
                v_rel_x = v_rel * dx/dr
                v_rel_y = v_rel * dy/dr
                v_rel_z = v_rel * dz/dr

                # distance computation
                dot_prod = (dx*XIJ[0] + dy*XIJ[1] + dz*XIJ[2])/dr
                tx = XIJ[0] - dot_prod*dx/dr
                ty = XIJ[1] - dot_prod*dy/dr
                tz = XIJ[2] - dot_prod*dz/dr
                tr = sqrt(tx**2 + ty**2 + tz**2)

                d = self.d-RIJ
                self.compute_force(
                    tx/tr, ty/tr, tz/tr, v_rel_x, v_rel_y, v_rel_z, d)
            else:
                # direction of destination fiber
                dx = d_rxprev[d_idx] - d_rxnext[d_idx]
                dy = d_ryprev[d_idx] - d_rynext[d_idx]
                dz = d_rzprev[d_idx] - d_rznext[d_idx]
                dr = sqrt(dx**2 + dy**2 + dz**2)

                dx = dx/dr
                dy = dy/dr
                dz = dz/dr

                # direction of source fiber
                sx = s_rxprev[s_idx] - s_rxnext[s_idx]
                sy = s_ryprev[s_idx] - s_rynext[s_idx]
                sz = s_rzprev[s_idx] - s_rznext[s_idx]
                sr = sqrt(sx**2 + sy**2 + sz**2)

                sx = sx/sr
                sy = sy/sr
                sz = sz/sr

                # normal direction at contact
                nx = dy*sz - dz*sy
                ny = dz*sx - dx*sz
                nz = dx*sy - dy*sx
                nr = sqrt(nx**2 + ny**2 + nz**2)

                # 3 vectors not in plane
                if abs(nx*XIJ[0] + ny*XIJ[1] + nz*XIJ[2]) > 1E-14 and nr > 1E14:
                    nx = -nx/nr
                    ny = -ny/nr
                    nz = -nz/nr

                    # relative velocity in each fiber direction
                    v_rel_d = dx*VIJ[0] + dy*VIJ[1] + dz*VIJ[2]
                    v_rel_s = sx*VIJ[0] + sy*VIJ[1] + sz*VIJ[2]
                    v_rel_x = v_rel_d * dx/dr + v_rel_s * sx/sr
                    v_rel_y = v_rel_d * dy/dr + v_rel_s * sy/sr
                    v_rel_z = v_rel_d * dz/dr + v_rel_s * sz/sr

                    y = nx*XIJ[0] + ny*XIJ[1] + nz*XIJ[2]

                    d = self.d - RIJ

                    self.compute_force(
                        nx, ny, nz, v_rel_x, v_rel_y, v_rel_z, d)
                else:
                    # direction of destination fiber
                    dx = d_rxnext[d_idx] - d_rxprev[d_idx]
                    dy = d_rynext[d_idx] - d_ryprev[d_idx]
                    dz = d_rznext[d_idx] - d_rzprev[d_idx]
                    dr = sqrt(dx**2+dy**2+dz**2)

                    # relative velocity for friction term
                    v_rel = dx*VIJ[0] + dy*VIJ[1] + dz*VIJ[2]
                    v_rel_x = v_rel * dx/dr
                    v_rel_y = v_rel * dy/dr
                    v_rel_z = v_rel * dz/dr

                    # distance computation
                    dot_prod = (dx*XIJ[0] + dy*XIJ[1] + dz*XIJ[2])/dr
                    tx = XIJ[0] - dot_prod*dx/dr
                    ty = XIJ[1] - dot_prod*dy/dr
                    tz = XIJ[2] - dot_prod*dz/dr
                    tr = sqrt(tx**2 + ty**2 + tz**2)

                    d = self.d-RIJ

                    self.compute_force(
                        tx/tr, ty/tr, tz/tr, v_rel_x, v_rel_y, v_rel_z, d)

            d_Fx[d_idx] += self.Fx
            d_Fy[d_idx] += self.Fy
            d_Fz[d_idx] += self.Fz

            d_au[d_idx] += self.Fx/d_m[d_idx]
            d_av[d_idx] += self.Fy/d_m[d_idx]
            d_aw[d_idx] += self.Fz/d_m[d_idx]


    def compute_force(self, dirx=0.0, diry=0.0, dirz=0.0, vx=0.0, vy=0.0,
                      vz=0.0, d=0.0):
        '''This force can be either a contact force (d>0) or a lubrication
            force (d<0).
        '''
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0

        if d > 0:
            # elastic factor from Hertz' pressure in contact
            E_star = 1/(2*((1-self.pois**2)/self.E))

            d = min(self.lim*self.d, d)

            if self.dim == 3:
                F = 4/3 * d**1.5 * sqrt(self.d/2) * E_star
            else:
                # workaround for 2D contact (2 reactangular surfaces)
                F = self.E*d

            v_rel = sqrt(vx**2 + vy**2 + vz**2)
            v_rel = v_rel if v_rel > 1E-14 else 1

            self.Fx = (F*dirx - self.k*F*vx/v_rel)
            self.Fy = (F*diry - self.k*F*vy/v_rel)
            self.Fz = (F*dirz - self.k*F*vz/v_rel)
