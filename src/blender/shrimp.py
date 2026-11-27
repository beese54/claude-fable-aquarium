"""Red cherry shrimp (Neocaridina davidi), built and posed by code.

The shrimp faces +X, Z is up, and the origin is the point on the ground under
the middle of the body: the feet stand on z = 0. One mesh holds every part on
material slots. The shell (carapace, abdomen, telson), eyes, rostrum, antennal
scales and tail fan are rigid; every appendage is a thin tapering tube whose
centreline is solved again on each pose (two-bone IK for the legs, angle
sweeps for swimmerets and antennae), so a pose is one numpy pass that rewrites
the vertex array. The anatomy is drawn in millimetres along a curved spine and
the finished mesh is scaled so rostrum tip to tail spans LENGTH_MM.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from bl_common import MM, new_material
from fish import _math, _mix, _smooth

LENGTH_MM = 22.0      # adult female, rostrum tip to tail fan
CYCLE = 0.8           # s; one step of every walking leg. All motion repeats with it.
DUTY = 0.65           # fraction of a step that a foot is planted
STRIDE_MM = 1.6       # drawing units, before the LENGTH_MM scale
LIFT_MM = 0.7

# Arc length along the spine (drawing mm): carapace, abdominal segments 1-6, telson.
_SEGS = np.array([7.5, 1.7, 2.0, 2.1, 1.6, 1.4, 2.0, 2.3])
_JOINTS = np.concatenate(([0.0], np.cumsum(_SEGS)))
S0, S1 = -0.55, float(_JOINTS[-1])     # front of the head, tip of the telson
SPINE_Z = 4.30                         # spine height over the legs (s = 4)

# Spine pitch (degrees, positive rises going backward): the back rises to a
# hump at abdominal segment 3, then the abdomen arches down and the tail fan
# flattens out again just clear of the ground.
_PITCH = np.array([
    (-0.55, 8.0), (4.0, 9.0), (7.5, 11.0), (9.5, 11.0), (10.8, 0.0), (11.8, -16.0),
    (13.0, -36.0), (14.5, -50.0), (16.0, -52.0), (17.6, -40.0), (18.4, -22.0), (19.2, -8.0),
    (20.6, 2.0),
])

# station, top, bottom, half-width (mm from the spine). The body is laterally
# compressed: much deeper than it is wide.
_PROFILE = np.array([
    (-0.55, 0.30, -0.45, 0.35),    # front between the eyes; the back slopes down into the rostrum
    (-0.35, 0.62, -0.88, 0.60),
    (0.00, 0.92, -1.22, 0.85),
    (0.60, 1.32, -1.60, 1.08),
    (1.50, 1.78, -1.98, 1.32),
    (3.00, 2.25, -2.28, 1.50),
    (4.80, 2.48, -2.38, 1.56),
    (6.50, 2.42, -2.32, 1.50),
    (7.50, 2.28, -2.25, 1.42),
    (7.56, 2.18, -2.20, 1.36),     # carapace margin overhangs abdominal segment 1
    (9.20, 2.22, -2.45, 1.36),     # deep female pleura form the brood chamber
    (11.2, 2.05, -2.20, 1.24),
    (13.3, 1.50, -1.50, 1.00),
    (14.9, 1.08, -1.00, 0.80),
    (16.3, 0.85, -0.75, 0.64),
    (18.0, 0.62, -0.52, 0.52),     # narrow end of segment 6 before the fan
    (18.3, 0.36, -0.26, 0.55),     # telson: flat and pointed
    (19.8, 0.16, -0.09, 0.28),
    (20.6, 0.05, -0.03, 0.06),
])

_PALETTES = {
    #          base               back (dorsal)       stripe              speckle             leg base            leg tip
    "red":    ((0.42, 0.010, 0.007), (0.16, 0.003, 0.002), (0.62, 0.16, 0.11), (0.70, 0.22, 0.16), (0.50, 0.06, 0.04), (0.62, 0.30, 0.26)),
    "orange": ((0.55, 0.12, 0.01), (0.25, 0.04, 0.004), (0.75, 0.40, 0.15), (0.80, 0.45, 0.20), (0.60, 0.20, 0.05), (0.70, 0.50, 0.35)),
    "yellow": ((0.60, 0.40, 0.02), (0.32, 0.18, 0.01), (0.80, 0.70, 0.30), (0.85, 0.75, 0.40), (0.65, 0.48, 0.10), (0.75, 0.68, 0.45)),
    "blue":   ((0.02, 0.06, 0.30), (0.005, 0.015, 0.10), (0.20, 0.30, 0.55), (0.25, 0.35, 0.60), (0.05, 0.10, 0.35), (0.35, 0.42, 0.55)),
}

_Y = np.array((0.0, 1.0, 0.0))
_Z = np.array((0.0, 0.0, 1.0))


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _profile(s):
    s = np.asarray(s, dtype=np.float64)
    top, bot, hw = (np.interp(s, _PROFILE[:, 0], _PROFILE[:, k]) for k in (1, 2, 3))
    # Each abdominal segment widens toward its rear edge, which overlaps the
    # next one; the pleura hang lowest mid-segment, giving the scalloped belly.
    k = np.clip(np.searchsorted(_JOINTS, s, side="right") - 1, 0, len(_SEGS) - 1)
    u = np.clip((s - _JOINTS[k]) / _SEGS[k], 0.0, 1.0)
    flare = np.where((k >= 1) & (k <= 6), 0.965 + 0.035 * u ** 1.5, 1.0)
    pleura = np.where((k >= 1) & (k <= 5), 0.88 + 0.18 * np.sin(np.pi * u), 1.0)
    return top * flare, bot * flare * pleura, hw * flare


class _Anatomy:
    """The curved spine and the attachment points hung off it (drawing mm)."""

    def __init__(self, x0=0.0):
        s = np.linspace(S0, S1, 900)
        th = np.radians(np.interp(s, _PITCH[:, 0], _PITCH[:, 1]))
        ds = np.diff(s)
        tx, tz = -np.cos(th), np.sin(th)
        x = np.concatenate(([0.0], np.cumsum(0.5 * (tx[1:] + tx[:-1]) * ds)))
        z = np.concatenate(([0.0], np.cumsum(0.5 * (tz[1:] + tz[:-1]) * ds)))
        z += SPINE_Z - np.interp(4.0, s, z)
        self.s, self.x, self.z, self.th = s, x + x0, z, th

    def frame(self, s):
        """Spine point, tangent (pointing tailward) and dorsal normal at s."""
        x = np.interp(s, self.s, self.x)
        z = np.interp(s, self.s, self.z)
        th = np.interp(s, self.s, self.th)
        return (np.array((x, 0.0, z)), np.array((-math.cos(th), 0.0, math.sin(th))),
                np.array((math.sin(th), 0.0, math.cos(th))))

    def at(self, s, y, zl):
        """A point y to the side of the spine and zl along its dorsal normal."""
        p, _, n = self.frame(s)
        return p + y * _Y + zl * n

    def bottom(self, s):
        return float(_profile(s)[1])

    def top(self, s):
        return float(_profile(s)[0])


# --- geometry helpers ---------------------------------------------------------------

def _tube(c, ra, rb, ref, sides, mat, rb_dn=None, phase=0.0):
    """A tapering tube along centreline c. ra is the radius across ref, rb the
    radius toward ref (rb_dn on the far side), so flat blades are tubes too."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    ra = np.broadcast_to(np.asarray(ra, dtype=np.float64), (n,))
    rb = np.broadcast_to(np.asarray(rb, dtype=np.float64), (n,))
    rd = rb if rb_dn is None else np.broadcast_to(np.asarray(rb_dn, dtype=np.float64), (n,))
    return dict(c=c, ra=ra, rb=rb, rd=rd, ref=np.asarray(ref, dtype=np.float64),
                sides=sides, mat=mat, phase=phase)


def _tube_verts(tb):
    c, ref = tb["c"], tb["ref"]
    T = np.gradient(c, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # (N1, N2, T) right-handed, so rings wound by increasing angle face outward
    N1 = np.cross(ref, T)
    ln = np.linalg.norm(N1, axis=1)
    bad = ln < 0.15
    if bad.any():
        alt = np.array((1.0, 0.0, 0.0)) if abs(ref[0]) < 0.7 else _Z
        N1[bad] = np.cross(alt, T[bad])
        ln = np.linalg.norm(N1, axis=1)
    N1 /= ln[:, None]
    N2 = np.cross(T, N1)
    a = tb["phase"] + 2 * math.pi * np.arange(tb["sides"]) / tb["sides"]
    ca, sa = np.cos(a)[None, :], np.sin(a)[None, :]
    rz = np.where(sa >= 0, tb["rb"][:, None], tb["rd"][:, None])
    v = (c[:, None, :] + (tb["ra"][:, None] * ca)[..., None] * N1[:, None, :]
         + (rz * sa)[..., None] * N2[:, None, :])
    return v.reshape(-1, 3)


def _polyline(pts, counts):
    """Points along a polyline: counts[i] stations per segment, joints shared."""
    out = [np.asarray(pts[0], dtype=np.float64)]
    for (p, q), k in zip(zip(pts[:-1], pts[1:]), counts):
        p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
        for i in range(1, k + 1):
            out.append(p + (q - p) * i / k)
    return np.array(out)


def _curve(start, length, n, psi, beta, side):
    """A flagellum: integrate heading psi (out from +X toward the animal's
    side) and pitch beta, both arrays over n stations."""
    d = np.stack((np.cos(beta) * np.cos(psi), side * np.cos(beta) * np.sin(psi), np.sin(beta)), -1)
    step = length / (n - 1)
    seg = 0.5 * (d[1:] + d[:-1]) * step
    return np.asarray(start) + np.concatenate((np.zeros((1, 3)), np.cumsum(seg, axis=0)))


def _ik(h, p, a, b, pole):
    """Knee of a two-bone chain from hip h to end p, bent toward pole."""
    d = p - h
    dist = np.linalg.norm(d)
    e = d / dist
    L = min(max(dist, abs(a - b) + 1e-3), a + b - 1e-3)
    along = (a * a - b * b + L * L) / (2 * L)
    hgt = math.sqrt(max(a * a - along * along, 0.0))
    nrm = pole - e * np.dot(pole, e)
    return h + e * along + _unit(nrm) * hgt


def _paddle(u, wmax):
    # rounded oar blade: narrow stalk, widest past the middle, round tip
    return wmax * np.sin(np.pi * np.clip(u * 0.92 + 0.08, 0.0, 1.0)) ** 0.55


# --- parts --------------------------------------------------------------------------

BODY, LIMB, EYE, FIN = range(4)


def _body_rings(an, M=16):
    ss = np.unique(np.concatenate((
        np.linspace(S0, S1, 78),
        np.concatenate([(j - 0.05, j + 0.05) for j in _JOINTS[1:-1]]))))
    top, bot, hw = _profile(ss)
    th = 2 * np.pi * np.arange(M) / M
    c, s = np.cos(th), np.sin(th)
    yk = np.sign(s) * np.abs(s) ** 0.8
    # rounded back, boxier belly where the pleura and carapace sides hang down
    zk = np.sign(c) * np.abs(c) ** np.where(c >= 0, 0.9, 0.62)
    verts, uvs = [], []
    for k, sk in enumerate(ss):
        p, _, n = an.frame(sk)
        mid, half = (top[k] + bot[k]) / 2, (top[k] - bot[k]) / 2
        ring = p[None, :] + (hw[k] * yk)[:, None] * _Y[None, :] + (mid + half * zk)[:, None] * n[None, :]
        verts.append(ring)
        uvs.extend((float((sk - S0) / (S1 - S0)), float(0.5 + 0.5 * cj)) for cj in c)
    return np.concatenate(verts), len(ss), M, uvs


def _static_tubes(an):
    tubes = []
    # rostrum: a laterally flat blade with a serrated upper edge
    n = 32
    u = np.linspace(0, 1, n)
    # it leaves the carapace as a crest and runs forward at eye level
    base = an.at(2.2, 0.0, an.top(2.2) - 0.45)
    tip = np.array((an.frame(S0)[0][0] + 2.8, 0.0, an.at(-0.25, 0.0, 0.95)[2]))
    c = base[None, :] + (tip - base)[None, :] * u[:, None]
    c[:, 2] += 0.15 * u ** 3
    teeth = np.where((u > 0.12) & (u < 0.72), 0.11 * (1 - np.mod(u * 26, 1.0)) ** 1.5, 0.0)
    hi = 0.36 * (1 - u) ** 0.8 + teeth + 0.01
    lo = 0.30 * (1 - u) ** 0.9 + np.where(u > 0.7, 0.05 * (1 - np.mod(u * 20, 1.0)), 0.0) + 0.01
    tubes.append(_tube(c, 0.13 * (1 - u) + 0.015, hi, _Z, 6, FIN, rb_dn=lo, phase=0.0))

    for side in (1, -1):
        # eye stalk and eye
        eb = an.at(0.35, side * 0.35, 0.55)
        ec = an.at(-0.25, side * 1.2, 0.85) + np.array((0.15, 0.0, 0.0))
        tubes.append(_tube(_polyline([eb, ec], [5]), np.linspace(0.24, 0.30, 6), np.linspace(0.24, 0.30, 6), _Z, 6, BODY))
        ax = _unit(ec - eb)
        r = 0.44
        xs = np.linspace(-r, r, 9)
        rr = np.sqrt(np.maximum(r * r - xs * xs, 0.0004))
        # the compound eye is a little taller than it is long
        tubes.append(_tube(ec[None, :] + xs[:, None] * ax[None, :], rr * 1.0, rr * 1.18, _Z, 10, EYE))

        # antennal scale (scaphocerite): flat blade beside the rostrum
        b0 = an.at(0.0, side * 0.80, -0.30)
        u = np.linspace(0, 1, 10)
        c = b0[None, :] + _unit((0.97, side * 0.20, 0.06))[None, :] * (2.3 * u)[:, None]
        tubes.append(_tube(c, _paddle(u, 0.34), 0.05 * (1 - u) + 0.02, _Z, 6, FIN))

        # tail fan: outer and inner uropod, spread in the plane of the telson
        p, _, N = an.frame(_JOINTS[7] - 0.2)
        T = an.frame(0.5 * (_JOINTS[7] + S1))[1]     # the fan lies along the telson, not segment 6
        root = p + side * 0.30 * _Y - 0.05 * N
        # the blades overlap each other and the telson, so the fan reads as one
        for ang, length, w, roll in ((21.0, 3.3, 0.70, 0.70), (9.0, 2.9, 0.56, 0.35)):
            g = math.radians(ang)
            d = _unit(math.cos(g) * T + side * math.sin(g) * _Y - 0.10 * N)
            u = np.linspace(0, 1, 9)
            c = root[None, :] + d[None, :] * (length * u)[:, None]
            # outer blades roll down at their outer edge: seen from behind the
            # fan is a shallow inverted V, so it still shows from the side
            ref = _unit(math.cos(roll) * N + side * math.sin(roll) * _Y)
            tubes.append(_tube(c, _paddle(u, w), 0.05 * (1 - u) + 0.02, ref, 6, FIN))
    return tubes


_WALK_LEGS = (
    # s on spine, foot (dx from hip, y out), bones: merus, carpus+propodus, dactyl; wave offset
    (4.5, 2.6, 3.5, 3.3, 3.0, 0.70, 2 / 3),
    (5.4, 0.3, 3.8, 3.5, 3.2, 0.75, 1 / 3),
    (6.3, -2.8, 3.6, 3.6, 3.3, 0.75, 0.0),
)
_CHELAE = (
    # s on spine, merus, carpus+chela, ground reach (dx ahead of the mouth, y out), phase
    (2.5, 1.3, 1.6, 1.3, 0.8, 0.0),
    (3.4, 1.7, 2.1, 2.0, 1.2, 0.25),
)


def _frac(x):
    return x - math.floor(x)


def _appendages(an, t, walk):
    tubes = []
    w = 2 * math.pi * t / CYCLE        # one turn per cycle: every phase below is a multiple
    mouth_s = 1.0
    for side in (1, -1):
        alt = 0.5 if side < 0 else 0.0

        # walking legs 3-5: planted feet slide back while the body moves on, then
        # lift and swing forward; the wave runs tail to head, sides alternate
        for s, dx, fy, a, b, c_len, off in _WALK_LEGS:
            hip = an.at(s, side * 0.75, an.bottom(s) + 0.30)
            ph = _frac(t / CYCLE - off + alt)
            if ph < DUTY:
                q = ph / DUTY
                fx, fz = STRIDE_MM * walk * (0.5 - q), 0.0
            else:
                q = (ph - DUTY) / (1 - DUTY)
                fx = STRIDE_MM * walk * (-0.5 + (0.5 - 0.5 * math.cos(math.pi * q)))
                fz = LIFT_MM * walk * math.sin(math.pi * q)
            foot = np.array((hip[0] + dx + fx, side * fy, fz + 0.06))
            ankle = foot + c_len * _unit((0.0, side * 0.30, 0.95))
            pole = np.array((0.25 * dx, side * 1.0, 0.35))
            knee = _ik(hip, ankle, a, b, pole)
            c = _polyline([hip, knee, ankle, foot], [3, 3, 2])
            r = np.array((0.15, 0.14, 0.13, 0.125, 0.11, 0.10, 0.09, 0.075, 0.05))
            tubes.append(_tube(c, r, r, (1.0, 0.0, 0.0), 4, LIMB, phase=math.pi / 4))

        # feeding legs 1-2: tiny claws pick at the ground and carry food to the mouth
        mouth = an.at(mouth_s, side * 0.25, an.bottom(mouth_s) - 0.05)
        for s, a, b, reach, ry, ph0 in _CHELAE:
            hip = an.at(s, side * 0.50, an.bottom(s) + 0.30)
            p = _frac(2 * t / CYCLE + ph0 + alt)
            pick = (0.5 - 0.5 * math.cos(2 * math.pi * p)) * (1.0 - 0.5 * walk)
            ground = np.array((mouth[0] + reach, side * ry, 0.12))
            tip = mouth + (ground - mouth) * pick + side * 0.15 * math.sin(2 * math.pi * p) * _Y
            tip[2] = max(tip[2], 0.10)
            elbow = _ik(hip, tip, a, b, np.array((-0.3, side * 1.0, 0.2)))
            c = np.vstack((_polyline([hip, elbow], [3]),
                           elbow[None, :] + (tip - elbow)[None, :] * np.array((0.25, 0.5, 0.65, 0.8, 0.92, 1.0))[:, None]))
            r = np.array((0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.10, 0.11, 0.07, 0.03))
            tubes.append(_tube(c, r, r, (1.0, 0.0, 0.0), 4, LIMB, phase=math.pi / 4))

        # third maxillipeds: leg-like mouthparts held forward under the head
        mb = an.at(1.3, side * 0.35, an.bottom(1.3) + 0.30)
        nod = 0.12 * math.sin(2 * w + (math.pi if side < 0 else 0.0))
        p1 = mb + np.array((0.6, side * 0.30, -0.55))
        p2 = p1 + np.array((1.0, side * 0.15, -0.25 + nod))
        p3 = p2 + np.array((0.8, -side * 0.05, -0.05 + nod))
        c = _polyline([mb, p1, p2, p3], [2, 3, 3])
        r = np.linspace(0.11, 0.035, len(c))
        tubes.append(_tube(c, r, r, (1.0, 0.0, 0.0), 4, LIMB, phase=math.pi / 4))

        # swimmerets: five pairs under the abdomen fanning in a metachronal wave
        # (tail end leads), two flat rami on a short stalk
        for k in range(1, 6):
            s = 0.5 * (_JOINTS[k] + _JOINTS[k + 1])
            p, T, N = an.frame(s)
            base = p + side * 0.40 * _Y + 0.80 * an.bottom(s) * N
            beat = 4 * w + 0.9 * k
            alpha = 0.30 + 0.38 * math.sin(beat)
            scale = (1.0, 1.05, 1.0, 0.9, 0.75)[k - 1]

            def ray(ang, splay):
                return _unit(-math.cos(ang) * N - math.sin(ang) * T + side * splay * _Y)

            d = ray(alpha, 0.15)
            stalk = base + d * 0.55 * scale
            tubes.append(_tube(_polyline([base, stalk], [2]), np.array((0.09, 0.08, 0.07)),
                               np.array((0.11, 0.10, 0.09)), _Y, 4, LIMB, phase=math.pi / 4))
            lag = alpha + 0.3 * math.cos(beat)       # rami trail the stalk on each stroke
            u = np.linspace(0, 1, 6)
            # two rami open in a V along the body; the blades are turned half
            # toward the side so the fringe still shows in profile
            blade = _unit(side * _Y + 0.9 * T)
            for spread, splay, ln in ((0.30, 0.10, 1.30), (-0.15, 0.25, 1.15)):
                dr = ray(lag + spread, splay)
                c = stalk[None, :] + dr[None, :] * (ln * scale * u)[:, None]
                tubes.append(_tube(c, 0.035 * (1 - u) + 0.02, _paddle(u, 0.17), blade, 4, LIMB))

        # long antennae: a short stalk, then a whip as long as the body sweeping
        # out and back; a slow wave travels along it
        sw = 2.1 if side < 0 else 0.0
        a0 = an.at(0.2, side * 0.60, -0.60)
        a1 = a0 + 1.3 * _unit((0.93, side * 0.30, 0.20))
        n = 40
        u = np.linspace(0, 1, n)
        # it turns early (first few mm) and then trails back past the body
        turn = (1 - np.exp(-u / 0.22)) / (1 - math.exp(-1 / 0.22))
        psi = (math.radians(20) + math.radians(140) * turn
               + 0.22 * u * np.sin(w - 3.0 * u + sw) + 0.08 * u * np.sin(2 * w + 1.3 + sw))
        # rising first, so the whips arch over the back, then sinking behind the tail
        beta = (math.radians(30) * (1 - u) ** 2 - math.radians(16) * u ** 2
                + 0.10 * u * np.sin(w + 1.0 + sw))
        flag = _curve(a1, 19.0, n, psi, beta, side)
        c = np.vstack((_polyline([a0, a1], [3])[:-1], flag))
        c[:, 2] = np.maximum(c[:, 2], 0.20)          # a whip that reaches the ground lies on it
        # no thinner than 0.15 mm, or the whip vanishes at 3 px per mm
        r = np.concatenate((np.linspace(0.16, 0.12, 3), np.linspace(0.11, 0.08, n)))
        tubes.append(_tube(c, r, r, _Z, 3, LIMB))

        # antennules: a stalk forking into a longer outer and a shorter inner whip, flicking
        b0 = an.at(-0.35, side * 0.30, 0.20)
        b1 = b0 + 1.7 * _unit((0.95, side * 0.18, 0.25))
        c = _polyline([b0, b1], [4])
        r = np.linspace(0.13, 0.09, len(c))
        tubes.append(_tube(c, r, r, _Z, 4, LIMB, phase=math.pi / 4))
        flick = 0.15 * math.sin(2 * w + (1.7 if side < 0 else 0.0))
        for (p0, p1, b_0, b_1, ln, nn, k) in ((18, 40, 25, 5, 3.6, 12, 0.0), (5, 12, 0, -10, 2.4, 9, 1.1)):
            u = np.linspace(0, 1, nn)
            psi = np.radians(p0 + (p1 - p0) * u)
            beta = np.radians(b_0 + (b_1 - b_0) * u) + flick * u * math.cos(k)
            c = _curve(b1, ln, nn, psi, beta, side)
            r = np.linspace(0.09, 0.08, nn)
            tubes.append(_tube(c, r, r, _Z, 3, LIMB))
    return tubes


# --- mesh ---------------------------------------------------------------------------

class _Model:
    """Topology and rigid vertices, shared by every shrimp (they differ only by pose)."""

    def __init__(self):
        # measure the shell once to centre it on the origin and set the scale
        x = self._rigid(_Anatomy())[:, 0]
        self.anatomy = _Anatomy(-(x.min() + x.max()) / 2)
        self.scale = LENGTH_MM / float(x.max() - x.min())

        b_verts, K, M, b_uvs = _body_rings(self.anatomy)
        verts, faces, uvs, mats = [], [], [], []

        def grid(off, nu, nv, mat):
            for i in range(nu - 1):
                for j in range(nv):
                    a = off + i * nv + j
                    b = off + i * nv + (j + 1) % nv
                    faces.append((a, b, b + nv, a + nv))
                    mats.append(mat)

        verts.append(b_verts)
        uvs.extend(b_uvs)
        grid(0, K, M, BODY)
        # close the head and the telson tip with fans
        n0 = K * M
        an = self.anatomy
        verts.append(np.array([an.frame(S0)[0], an.frame(S1)[0]]))
        uvs.extend([(0.0, 0.5), (1.0, 0.5)])
        for j in range(M):
            jn = (j + 1) % M
            faces.append((n0, jn, j))
            faces.append((n0 + 1, (K - 1) * M + j, (K - 1) * M + jn))
            mats.extend((BODY, BODY))
        off = n0 + 2

        def add_tubes(tubes):
            nonlocal off
            for tb in tubes:
                n, sd = len(tb["c"]), tb["sides"]
                verts.append(_tube_verts(tb))
                uvs.extend((i / (n - 1), j / sd) for i in range(n) for j in range(sd))
                grid(off, n, sd, tb["mat"])
                off += n * sd

        add_tubes(_static_tubes(an))
        self.n_rigid = off
        add_tubes(_appendages(an, 0.0, 1.0))
        self.verts = np.concatenate(verts)
        self.faces, self.uvs, self.mats = faces, uvs, mats

    @staticmethod
    def _rigid(an):
        b_verts = _body_rings(an)[0]
        return np.concatenate([b_verts] + [_tube_verts(tb) for tb in _static_tubes(an)])

    def posed(self, t, walk):
        """Vertex positions in metres at unit size."""
        co = np.empty_like(self.verts)
        co[:self.n_rigid] = self.verts[:self.n_rigid]
        co[self.n_rigid:] = np.concatenate([_tube_verts(tb) for tb in _appendages(self.anatomy, t, walk)])
        return co * (self.scale * MM)


# --- materials ---------------------------------------------------------------------

def shrimp_materials(colour="red"):
    base, back, stripe, speck, leg, leg_tip = _PALETTES[colour]

    # shell -------------------------------------------------------------------
    body, nt, out = new_material(f"ShrimpBody_{colour}")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    s, v = sep.outputs["X"], sep.outputs["Y"]           # s: head to tail; v: 1 on the back, 0 on the belly
    back_m = _smooth(nt, v, 0.45, 0.92)
    stripe_m = _math(nt, "MULTIPLY", _smooth(nt, v, 0.972, 0.992), _smooth(nt, s, 0.06, 0.12))
    stripe_m = _math(nt, "MULTIPLY", stripe_m, _smooth(nt, s, 0.70, 0.82, 0.6, 0.0))
    tc = N.new("ShaderNodeTexCoord")
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3000.0     # per metre: sub-millimetre flecks
    noise.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    speck_m = _smooth(nt, noise.outputs["Fac"], 0.64, 0.74, 0.0, 0.3)
    # chromatophores are patchy: dense red blotches over clearer, paler shell
    blot = N.new("ShaderNodeTexNoise")
    blot.inputs["Scale"].default_value = 700.0
    blot.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], blot.inputs["Vector"])
    clear_m = _smooth(nt, blot.outputs["Fac"], 0.42, 0.62, 0.0, 1.0)
    belly_m = _smooth(nt, v, 0.08, 0.30, 1.0, 0.0)       # the underside is thinner and clearer
    col = _mix(nt, _math(nt, "MAXIMUM", _math(nt, "MULTIPLY", clear_m, 0.15),
                         _math(nt, "MULTIPLY", belly_m, 0.35)), base, stripe)
    col = _mix(nt, back_m, col, back)
    col = _mix(nt, stripe_m, col, stripe)
    col = _mix(nt, speck_m, col, speck)
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    # light scatters through the thin shell; the pigmented back is more opaque
    L.new(_math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", back_m, 0.45)), bsdf.inputs["Subsurface Weight"])
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.18, 0.12)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0012
    L.new(_math(nt, "MULTIPLY", _math(nt, "MAXIMUM", clear_m, belly_m), 0.25), bsdf.inputs["Transmission Weight"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    bsdf.inputs["Coat Weight"].default_value = 0.35    # wet cuticle: small sharp highlights
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    # a faintly uneven cuticle breaks the long tube-like highlight into glints
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.00003
    L.new(blot.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Coat Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # legs, swimmerets, antennae: paler and clearer toward the tips ----------------
    limb, nt, out = new_material(f"ShrimpLimb_{colour}")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    u = _smooth(nt, sep.outputs["X"], 0.05, 0.9)
    # red bands at the joints over clear cuticle, the way cherry shrimp legs look
    band = _math(nt, "SINE", _math(nt, "MULTIPLY", sep.outputs["X"], 2 * math.pi * 4.5))
    clear = _math(nt, "MAXIMUM", u, _smooth(nt, band, 0.2, 0.8, 0.0, 0.7))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(_mix(nt, clear, leg, leg_tip), bsdf.inputs["Base Color"])
    L.new(_math(nt, "MULTIPLY_ADD", clear, 0.4), bsdf.inputs["Transmission Weight"])
    N[-1].inputs[2].default_value = 0.2
    L.new(_math(nt, "MULTIPLY_ADD", u, -0.35), bsdf.inputs["Alpha"])
    N[-1].inputs[2].default_value = 1.0
    bsdf.inputs["Subsurface Weight"].default_value = 0.5
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.15)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0003
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Coat Weight"].default_value = 0.5
    bsdf.inputs["Coat Roughness"].default_value = 0.08
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # eye -----------------------------------------------------------------------
    eye, nt, out = new_material("ShrimpEye")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.010, 0.006, 0.005, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.35   # a strong ring highlight reads as a cartoon pupil
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    # rostrum, antennal scales, tail fan: thin translucent blades ------------------
    fin, nt, out = new_material(f"ShrimpFin_{colour}")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    u = _smooth(nt, sep.outputs["X"], 0.2, 1.0)        # red at the root, clear at the rim
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(_mix(nt, u, base, leg_tip), bsdf.inputs["Base Color"])
    L.new(_math(nt, "MULTIPLY_ADD", u, 0.35), bsdf.inputs["Transmission Weight"])
    N[-1].inputs[2].default_value = 0.3
    L.new(_math(nt, "MULTIPLY_ADD", u, -0.3), bsdf.inputs["Alpha"])
    N[-1].inputs[2].default_value = 1.0
    bsdf.inputs["Subsurface Weight"].default_value = 0.6
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.15)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0005
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Coat Weight"].default_value = 0.5
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return [body, limb, eye, fin]


# --- posing ---------------------------------------------------------------------------

def _alive(mats):
    """Cached materials die when the scene is reset in the same session."""
    try:
        return bool(mats) and all(m.name in bpy.data.materials for m in mats)
    except ReferenceError:
        return False


class Shrimp:
    """One shrimp in the scene. pose() is a pure function of its arguments."""

    _model = None
    _materials = {}
    # how fast the body must travel (m/s at size 1) for planted feet not to slip
    WALK_SPEED = None

    def __init__(self, name, collection=None, size=1.0, colour="red"):
        if Shrimp._model is None:
            Shrimp._model = _Model()
            Shrimp.WALK_SPEED = STRIDE_MM * Shrimp._model.scale * MM / (DUTY * CYCLE)
        if not _alive(Shrimp._materials.get(colour)):
            Shrimp._materials[colour] = shrimp_materials(colour)
        m = Shrimp._model
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in m.posed(0.0, 1.0)], [], m.faces)
        uvl = me.uv_layers.new(name="UVMap")
        uvs = np.array(m.uvs, dtype=np.float32)
        vi = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", vi)
        uvl.data.foreach_set("uv", uvs[vi].ravel())
        me.polygons.foreach_set("material_index", m.mats)
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        for mat in Shrimp._materials[colour]:
            me.materials.append(mat)
        me.update()
        self.obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(self.obj)
        self.size = size
        self.obj.cycles.is_caustics_receiver = True

    def pose(self, position, yaw=0.0, t=0.0, up=(0, 0, 1), walk=1.0):
        """position: metres, the ground point under the middle of the body.
        yaw: heading in radians about up (0 faces +X when up is +Z). t: seconds;
        the pose repeats every CYCLE. up: surface normal. walk: 0 standing and
        grazing .. 1 walking."""
        walk = min(max(float(walk), 0.0), 1.0)
        co = Shrimp._model.posed(float(t), walk)
        me = self.obj.data
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.update()
        # the shortest rotation from +Z onto up keeps yaw meaning the same heading
        # as the surface tilts away from level
        tilt = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(up).normalized())
        rot = tilt.to_matrix().to_4x4() @ Matrix.Rotation(yaw, 4, "Z")
        self.obj.matrix_world = Matrix.Translation(Vector(position)) @ rot @ Matrix.Scale(self.size, 4)
