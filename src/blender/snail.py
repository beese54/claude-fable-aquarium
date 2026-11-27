"""Zebra nerite snail (Neritina natalensis), built and posed by code.

The snail faces +X with the middle of its sole at the origin; Z is the surface
normal. The shell is a true seashell sweep: a growing D-shaped section carried
along a logarithmic spiral. A nerite expands so fast per turn that the last
whorl swallows the rest, leaving a low dome with a small spire at the back.
The body is fitted to the shell rather than the other way round: the foot's
outline is the shell opening, so the shell sits on it the way the real animal
carries it. One mesh holds shell, body and eyes on separate material slots, so
a pose is a single pass over the vertex array: the tentacles are rebuilt along
bent centrelines, the shell rocks about its opening and a ripple runs on the foot.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from bl_common import MM, new_material
from fish import _math, _mix, _smooth

SHELL_MM = 14.0                # apex to front of the lip, the size aquarists quote

# --- shell: logarithmic spiral sweep ---------------------------------------------
# Raup's parameters. W, the growth of the spiral per turn, is what separates a
# nerite (tens) from a pond snail (about 2): the last whorl dwarfs the others.
_W = 15.0
_TURNS = 1.5                   # older whorls are far too small to matter
_KA, _KB = 0.93, 1.2           # section semi-axes over spiral radius (radial, axial)
# The inner side of the section reaches past the axis (negative flattening),
# so each whorl wraps over the last: no open coil, the spire stays a low knob.
_FLAT = -0.3
_T = 0.3                       # slide along the axis per unit radius: a low spire
_PSI = math.radians(45.0)      # section leans so the opening faces the ground
# The bare sweep is lower and wider than the species (a log spiral lying on its
# opening cannot be much more than half as high as it is long); an affine
# squeeze keeps the spiral intact and brings it to about 14 x 11.5 x 9.7 mm.
_STRETCH = (1.0, 0.82, 1.45)
_LIFT = 0.75                   # mm from sole to the lip
_SHELL_NTH, _SHELL_NPH = 64, 36


def _rot(axis, ang):
    return np.array(Matrix.Rotation(ang, 3, Vector(axis)))


def _shell_points():
    """(nth, nph, 3) points in mm and their (turns, around) UVs: the opening
    faces down at height _LIFT and is centred on the origin, the spire is at -X."""
    nth, nph = _SHELL_NTH, _SHELL_NPH
    g = math.log(_W) / (2 * math.pi)
    s = np.linspace(0.0, 1.0, nth)
    th = -_TURNS * 2 * math.pi * (1.0 - s)
    r = np.exp(g * th)
    R = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    Tg = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], 1)
    Z = np.array([0.0, 0.0, 1.0])
    A = math.cos(_PSI) * Z[None, :] + math.sin(_PSI) * Tg
    C = r[:, None] * R - (_T * r)[:, None] * Z[None, :]
    ph = 2 * math.pi * np.arange(nph) / nph
    cph, sph = np.cos(ph), np.sin(ph)
    rad = np.where(cph < 0, cph * (1.0 - _FLAT), cph)     # D-shaped: inner side differs
    P = (C[:, None, :]
         + (_KA * r)[:, None, None] * rad[None, :, None] * R[:, None, :]
         + (_KB * r)[:, None, None] * sph[None, :, None] * A[:, None, :])
    uv = np.stack(np.meshgrid(_TURNS * s, ph / (2 * math.pi), indexing="ij"), -1)

    # turn the opening (normal of the last section) to face straight down
    n = np.array([0.0, math.cos(_PSI), -math.sin(_PSI)])
    M = np.array(Vector(n).rotation_difference(Vector((0, 0, -1))).to_matrix())
    P = P @ M.T
    apex = M @ C[0]
    # the spire goes to the back: apex to the middle of the shell in plan is
    # the snail's long axis
    xy = P.reshape(-1, 3)[:, :2]
    ax = (xy.min(0) + xy.max(0)) / 2 - apex[:2]
    P = P @ _rot((0, 0, 1), -math.atan2(ax[1], ax[0])).T
    P = P * np.array(_STRETCH)
    flat = P.reshape(-1, 3)
    P = P * (SHELL_MM / (flat[:, 0].max() - flat[:, 0].min()))
    c = P[-1].mean(0)
    P = P - np.array([c[0], c[1], P[..., 2].min() - _LIFT])
    # The lip is rolled inward a little: a real shell has thickness, and from
    # below (through the glass) this rim is the striped edge around the foot.
    lip = P[-1]
    inward = -lip[:, :2] / np.linalg.norm(lip[:, :2], axis=1, keepdims=True)
    roll = [lip + np.c_[inward * 0.30, np.zeros(len(lip))],
            lip + np.c_[inward * 0.42, np.full(len(lip), 0.30)]]
    P = np.concatenate([P, np.stack(roll)], 0)
    uv = np.concatenate([uv, uv[-1:] + np.array([0.004, 0.0]), uv[-1:] + np.array([0.008, 0.0])], 0)
    # mesh winding that makes the normals point out of the shell
    return P[:, ::-1], uv[:, ::-1]


# --- mesh builder ----------------------------------------------------------------

class _Builder:
    def __init__(self):
        self.verts, self.faces, self.uvs, self.mats = [], [], [], []

    def add_verts(self, pts, uvs):
        off = len(self.verts)
        self.verts.extend(tuple(float(c) for c in p) for p in pts)
        self.uvs.extend(tuple(u) for u in uvs)
        return off

    def grid(self, off, nu, nv, mat):
        """Rings of nv vertices, closed around, joined ring to ring."""
        for i in range(nu - 1):
            for j in range(nv):
                a = off + i * nv + j
                b = off + i * nv + (j + 1) % nv
                self.faces.append((a, b, b + nv, a + nv))
                self.mats.append(mat)

    def fan(self, ring_off, nv, centre, mat, flip=False):
        for j in range(nv):
            a, b = ring_off + j, ring_off + (j + 1) % nv
            self.faces.append((a, centre, b) if flip else (a, b, centre))
            self.mats.append(mat)


def _ellipsoid(b, centre, radii, mat, nu=9, nv=14):
    cx, cy, cz = centre
    pts, uvs = [(cx, cy, cz - radii[2])], [(0.0, 0.0)]
    for i in range(1, nu):
        lat = -math.pi / 2 + math.pi * i / nu
        for j in range(nv):
            lon = 2 * math.pi * j / nv
            pts.append((cx + radii[0] * math.cos(lat) * math.cos(lon),
                        cy + radii[1] * math.cos(lat) * math.sin(lon),
                        cz + radii[2] * math.sin(lat)))
            uvs.append((j / nv, i / nu))
    pts.append((cx, cy, cz + radii[2]))
    uvs.append((0.0, 1.0))
    off = b.add_verts(pts, uvs)
    b.fan(off + 1, nv, off, mat, flip=True)
    b.grid(off + 1, nu - 1, nv, mat)
    b.fan(off + 1 + (nu - 2) * nv, nv, off + len(pts) - 1, mat)


# Body profile as (fraction of the opening's outline, height mm). The first
# rows are exactly z = 0: the sole is a flat plate on the surface, a little
# inside the opening so that from below the lip rings it. It rolls up into a
# rim, then the mantle wall rises to just inside the lip, so there is never a
# gap between foot and shell, and domes over inside the shell.
_BODY_PROFILE = [(0.30, 0.0), (0.58, 0.0), (0.78, 0.0), (0.84, 0.04), (0.875, 0.18),
                 (0.885, 0.40), (0.895, 0.62), (0.93, _LIFT + 0.12), (0.84, _LIFT + 0.60),
                 (0.62, _LIFT + 1.25), (0.32, _LIFT + 1.65)]
_BODY_TOP = _LIFT + 1.75
_NANG = 40


def _opening_outline(lip):
    """Radius of the shell opening (centred, in plan) at _NANG even angles,
    smoothed: the foot follows it, so the shell edge rings the sole."""
    ang = np.arctan2(lip[:, 1], lip[:, 0])
    rad = np.hypot(lip[:, 0], lip[:, 1])
    o = np.argsort(ang)
    alpha = 2 * math.pi * np.arange(_NANG) / _NANG
    r = np.interp(np.where(alpha > math.pi, alpha - 2 * math.pi, alpha), ang[o], rad[o], period=2 * math.pi)
    for _ in range(3):
        r = (np.roll(r, 1) + 2 * r + np.roll(r, -1)) / 4
    # the sole reaches forward under the head, as it does when crawling
    r = r + 1.2 * np.clip(np.cos(alpha), 0.0, None) ** 4
    return alpha, r


# --- head and tentacles ------------------------------------------------------------

_TENT_LEN = 5.0
_TENT_R0 = 0.26
_TENT_RINGS, _TENT_SIDES = 12, 8


class _Head:
    """Where the head parts sit, placed from the front of the opening."""

    def __init__(self, x_front):
        self.front = x_front
        self.neck = np.array([x_front - 2.0, 0.0, 1.0])          # pivot of the head sway
        self.head = (np.array([x_front - 0.2, 0.0, 1.05]), (2.0, 2.0, 0.95))
        self.snout = (np.array([x_front + 1.3, 0.0, 0.62]), (1.05, 1.35, 0.50))
        self.bases = {}
        for side in (1, -1):
            base = np.array([x_front + 0.55, side * 0.95, 1.55])
            yaw, pitch = side * math.radians(32.0), math.radians(22.0)
            d = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
            self.bases[side] = (base, d)


def _tentacle(base, d0, d_yaw=0.0, d_pitch=0.0, curl=0.0, length=1.0):
    """Tentacle surface as (rings * sides + 1, 3). Bend angles grow along the
    tentacle so it curves like a muscle-driven feeler, not a stiff rod."""
    L = _TENT_LEN * length
    s = np.linspace(-0.06, 1.0, _TENT_RINGS)        # the first ring is buried in the head
    up = np.array([0.0, 0.0, 1.0])
    side_ax = np.cross(up, d0)
    side_ax /= np.linalg.norm(side_ax)
    pts = []
    p = base + d0 * s[0] * L
    for i, si in enumerate(s):
        k = max(si, 0.0)
        d = _rot(up, d_yaw * k) @ (_rot(side_ax, -(d_pitch * k + curl * k * k)) @ d0)
        if i:
            p = p + d * (si - s[i - 1]) * L
        n1 = np.cross(d, up)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(n1, d)
        # a short swelling at the base, then a long taper to a fine tip
        rad = _TENT_R0 * (1.0 - 0.78 * k ** 0.8) * (1.0 + 0.3 * max(0.0, 1 - k / 0.15))
        for j in range(_TENT_SIDES):
            a = 2 * math.pi * j / _TENT_SIDES
            pts.append(p + rad * (math.cos(a) * n1 + math.sin(a) * n2))
    pts.append(p + d * 0.07)
    return np.array(pts)


def _smoothstep(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _tentacle_motion(t, side):
    """Bend, curl and length of one tentacle at time t: slow incommensurate
    sines for the waving, and a periodic pulse for a brief partial retraction."""
    tau = 2 * math.pi
    ph = 0.0 if side > 0 else 1.7
    d_yaw = 0.20 * math.sin(tau * t / 2.9 + ph) + 0.08 * math.sin(tau * t / 6.7 + 2.1 * ph)
    d_pitch = 0.20 * math.sin(tau * t / 3.7 + 0.6 + ph) + 0.07 * math.sin(tau * t / 1.9 + ph)
    x = (t / (8.3 if side > 0 else 11.1) + (0.45 if side > 0 else 0.4)) % 1.0
    pulse = _smoothstep(0.0, 0.06, x) * (1.0 - _smoothstep(0.12, 0.30, x))
    return d_yaw, d_pitch + 0.5 * pulse, 0.35 + 0.6 * pulse, 1.0 - 0.35 * pulse


# --- whole mesh --------------------------------------------------------------------

def build_nerite_mesh(name="Nerite"):
    b = _Builder()
    parts = {}

    # shell (slot 0)
    P, uv = _shell_points()
    nth, nph = P.shape[:2]
    off = b.add_verts(P.reshape(-1, 3), uv.reshape(-1, 2))
    b.grid(off, nth, nph, 0)
    parts["shell"] = (off, len(b.verts))

    # foot and mantle (slot 1), following the opening
    alpha, rad = _opening_outline(P[-1, :, :2])
    ca, sa = np.cos(alpha), np.sin(alpha)
    pts, uvs = [], []
    for k, (f, z) in enumerate(_BODY_PROFILE):
        for j in range(_NANG):
            pts.append((f * rad[j] * ca[j], f * rad[j] * sa[j], z))
            uvs.append((j / _NANG, k / len(_BODY_PROFILE)))
    pts += [(0.0, 0.0, 0.0), (0.0, 0.0, _BODY_TOP)]
    uvs += [(0.0, 0.0), (0.0, 1.0)]
    off = b.add_verts(pts, uvs)
    nring = len(_BODY_PROFILE)
    b.fan(off, _NANG, off + nring * _NANG, 1, flip=True)
    b.grid(off, nring, _NANG, 1)
    b.fan(off + (nring - 1) * _NANG, _NANG, off + nring * _NANG + 1, 1)
    parts["foot"] = (off, len(b.verts))

    # head: a rounded mass under the front lip, and a short broad snout
    head = _Head(float(P[-1, :, 0].max()))
    h0 = len(b.verts)
    _ellipsoid(b, head.head[0], head.head[1], 1, nu=8, nv=14)
    _ellipsoid(b, head.snout[0], head.snout[1], 1, nu=7, nv=12)
    parts["head"] = (h0, len(b.verts))

    # eyes (slot 2): small black beads at the outer base of each tentacle
    e0 = len(b.verts)
    for side in (1, -1):
        base, d = head.bases[side]
        _ellipsoid(b, base + np.array([0.0, side * 0.36, -0.08]) - d * 0.05, (0.2, 0.2, 0.2), 2, nu=5, nv=8)
    parts["eyes"] = (e0, len(b.verts))

    # tentacles (slot 1)
    for side in (1, -1):
        t0 = len(b.verts)
        uvs = [(i / _TENT_RINGS, j / _TENT_SIDES) for i in range(_TENT_RINGS) for j in range(_TENT_SIDES)]
        b.add_verts(_tentacle(*head.bases[side]), uvs + [(1.0, 0.0)])
        b.grid(t0, _TENT_RINGS, _TENT_SIDES, 1)
        b.fan(t0 + (_TENT_RINGS - 1) * _TENT_SIDES, _TENT_SIDES, t0 + _TENT_RINGS * _TENT_SIDES, 1)
        parts["tent%+d" % side] = (t0, len(b.verts))

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c * MM for c in v) for v in b.verts], [], b.faces)
    uvl = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        uvl.data[loop.index].uv = b.uvs[loop.vertex_index]
    me.polygons.foreach_set("material_index", b.mats)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    return me, parts, head


# --- materials -----------------------------------------------------------------------

def nerite_materials():
    # shell -------------------------------------------------------------------
    shell, nt, out = new_material("NeriteShell")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    turns, around = sep.outputs["X"], sep.outputs["Y"]

    # Stripes come from pigment cells along the lip that stay on as the shell
    # grows, so each runs with the sweep from the spire to the lip (constant
    # position around the section) and wanders sideways as the rhythm drifts.
    # Noise space wraps around the section (cos, sin) so the pattern has no seam.
    ang = _math(nt, "MULTIPLY", around, 2 * math.pi)
    comb = N.new("ShaderNodeCombineXYZ")
    L.new(_math(nt, "MULTIPLY", turns, 2.0), comb.inputs[0])
    L.new(_math(nt, "MULTIPLY", _math(nt, "COSINE", ang), 0.5), comb.inputs[1])
    L.new(_math(nt, "MULTIPLY", _math(nt, "SINE", ang), 0.5), comb.inputs[2])

    def noise(scale, detail=3.0):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        L.new(comb.outputs[0], n.inputs["Vector"])
        return _math(nt, "SUBTRACT", n.outputs["Fac"], 0.5)

    zig = _math(nt, "PINGPONG", _math(nt, "MULTIPLY", turns, 13.0), 0.5)
    jag = _math(nt, "ADD", _math(nt, "MULTIPLY", zig, 0.8), _math(nt, "MULTIPLY", noise(2.5), 1.4))
    jag = _math(nt, "ADD", jag, _math(nt, "MULTIPLY", noise(14.0, 2.0), 0.35))
    phase = _math(nt, "ADD", _math(nt, "MULTIPLY", around, 22.0), jag)
    wave = _math(nt, "SINE", _math(nt, "MULTIPLY", phase, 2 * math.pi))
    thr = _math(nt, "MULTIPLY_ADD", noise(4.0), 1.6)          # stripes swell and thin
    N[-1].inputs[2].default_value = 0.05
    stripe = _smooth(nt, _math(nt, "SUBTRACT", wave, thr), -0.10, 0.10)

    gold = _mix(nt, _math(nt, "ADD", noise(3.0), 0.5), (0.20, 0.17, 0.025), (0.38, 0.25, 0.022))
    col = _mix(nt, stripe, gold, (0.008, 0.007, 0.004))
    # the oldest whorls are worn chalky, as in most aquarium nerites
    col = _mix(nt, _smooth(nt, turns, 0.75, 0.35), col, (0.30, 0.29, 0.26))
    geo = N.new("ShaderNodeNewGeometry")
    col = _mix(nt, geo.outputs["Backfacing"], col, (0.62, 0.66, 0.68))    # porcelain inside

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # body --------------------------------------------------------------------
    body, nt, out = new_material("NeriteBody")
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    mott = N.new("ShaderNodeTexNoise")
    mott.inputs["Scale"].default_value = 4000.0     # flecks a few tenths of a mm
    mott.inputs["Detail"].default_value = 4.0
    L.new(tc.outputs["Object"], mott.inputs["Vector"])
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    sole = _smooth(nt, sep.outputs["Z"], 0.00012, 0.00002)        # sole: paler, sparse flecks
    spots = _smooth(nt, mott.outputs["Fac"], 0.52, 0.62)
    spots = _math(nt, "MULTIPLY", spots, _math(nt, "SUBTRACT", 0.85, _math(nt, "MULTIPLY", sole, 0.45)))
    base = _mix(nt, sole, (0.22, 0.22, 0.21), (0.46, 0.46, 0.43))
    col = _mix(nt, spots, base, (0.045, 0.045, 0.042))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Subsurface Weight"].default_value = 0.3
    bsdf.inputs["Subsurface Scale"].default_value = 0.0006
    bsdf.inputs["Coat Weight"].default_value = 0.35       # wet mucus sheen
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # eye ---------------------------------------------------------------------
    eye, nt, out = new_material("NeriteEye")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.004, 0.004, 0.005, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.06
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return [shell, body, eye]


# --- posing ------------------------------------------------------------------------

class Snail:
    """One snail on a surface. pose() is a pure function of its arguments."""

    _materials = {}

    def __init__(self, name, collection=None, size=1.0, kind="nerite"):
        if kind != "nerite":
            raise ValueError(f"unknown snail kind {kind!r}")
        if kind not in Snail._materials:
            Snail._materials[kind] = nerite_materials()
        me, self.parts, self.head = build_nerite_mesh(name)
        for m in Snail._materials[kind]:
            me.materials.append(m)
        self.obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(self.obj)
        self.size = size
        n = len(me.vertices)
        rest = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", rest)
        self.rest = rest.reshape(n, 3).astype(np.float64) / MM          # mm
        self.obj.cycles.is_caustics_receiver = True

    def pose(self, position, yaw=0.0, t=0.0, up=(0, 0, 1)):
        """position: metres, the surface point under the middle of the foot.
        yaw: heading in radians about up; 0 faces +X (+Z if up is along X).
        t: seconds; drives tentacles, shell rock and foot ripple.
        up: the surface normal the sole is pressed against."""
        co = self.rest.copy()
        tau = 2 * math.pi

        # the shell rocks a fraction of a degree about its opening as the foot
        # pulls; at most ~0.1 mm at the lip, so it never cuts into the sole
        a, b = self.parts["shell"]
        R = _rot((1, 0, 0), 0.012 * math.sin(tau * t / 3.3)) @ _rot((0, 1, 0), 0.008 * math.sin(tau * t / 4.7 + 1.0))
        piv = np.array([0.0, 0.0, _LIFT + 0.4])
        co[a:b] = (co[a:b] - piv) @ R.T + piv

        # pedal ripple: a low wave running forward along the foot. Sole rows
        # (z = 0) only slide in-plane so the sole stays on the surface.
        a, b = self.parts["foot"]
        f = co[a:b]
        w = np.sin(tau * (f[:, 0] / 4.0 - t / 1.6))
        f[:, 2] += 0.06 * w * np.clip(f[:, 2] / 0.6, 0.0, 1.0) * (f[:, 2] < _LIFT)
        f[:, 1] *= 1.0 + 0.012 * w * (f[:, 2] < _LIFT)

        # the head sways slowly, carrying eyes and tentacles with it
        h = self.head
        hr = _rot((0, 0, 1), 0.07 * math.sin(tau * t / 6.1 + 0.4)) @ _rot((0, 1, 0), 0.03 * math.sin(tau * t / 4.1))
        for key in ("head", "eyes"):
            a, b = self.parts[key]
            co[a:b] = (co[a:b] - h.neck) @ hr.T + h.neck
        for side in (1, -1):
            a, b = self.parts["tent%+d" % side]
            base, d0 = h.bases[side]
            co[a:b] = _tentacle(h.neck + hr @ (base - h.neck), hr @ d0, *_tentacle_motion(t, side))

        self.obj.data.vertices.foreach_set("co", (co * MM).astype(np.float32).ravel())
        self.obj.data.update()

        u = Vector(up).normalized()
        ref = Vector((1, 0, 0)) if abs(u.x) < 0.95 else Vector((0, 0, 1))
        fwd = Matrix.Rotation(yaw, 3, u) @ (ref - u * ref.dot(u)).normalized()
        rot = Matrix((fwd, u.cross(fwd), u)).transposed().to_4x4()
        self.obj.matrix_world = Matrix.Translation(Vector(position)) @ rot @ Matrix.Scale(self.size, 4)
