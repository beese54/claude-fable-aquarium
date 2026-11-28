"""Scape 5: Floating Islands.

A few rocks hang in mid-water, each a small world: moss and a few tiny plants
on top, roots trailing beneath. They are small for the tank and far apart, at
clearly different heights and depths: the largest high at the back left, a
lower one at the right, a small one high at the back and a tiny one low at the
front left, leaving a wide lane of open water through the middle for the fish.
The small back island pours a thin fall of sand onto the bed, the aquascaper's
"underwater waterfall". The bed is pale sand, nearly bare, so it throws light
back up onto the shaded undersides and takes the islands' shadows.
Coordinates are millimetres: X left to right, Y front to back.
"""
import math
import os
import random

import bpy
import numpy as np
from mathutils import Vector, noise
from mathutils.bvhtree import BVHTree

import assets
import fish
import materials
import plants
import shrimp
import snail
import substrate
import wood
from bl_common import MM, new_material, new_mesh_object
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "05_floating"

# Less soft fill than the other scapes: the islands' shadows on the pale sand are
# what make them read as really hanging there, and fill washes them out.
LIGHT = dict(watts=42.0, fill=0.006, bounce=0.018)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py).
# Each island with its root curtain is a hanging cylinder (x, y, radius, top, bottom);
# the last is the sand-fall. The build prints the measured ones as [isles].
SWIM = {
    "box": (40, 262, 28, 184, 122, 238),
    "floor": (60, 80),
    "obstacles": [(86, 198, 56, 230, 112), (236, 160, 42, 152, 84), (214, 248, 32, 238, 146),
                  (44, 94, 24, 146, 102), (200, 239, 14, 210, 74)],
}

# name, model, piece, longest side, (x, y, underside z), how the scan is stood up,
# (lean, lean, heading), hanging roots (count, shortest, longest, base radius),
# what grows on top. The scans are flat slabs of weathered rock: each is turned
# over, so its flat bed face becomes the mossy top and its rough, knuckled upper
# face hangs underneath as broken rock (stood on end they narrowed to a point
# and read as turned cones). Pieces of rock_moss_set_02 are its separate scans:
# 0 rock07, 1 rock08, 2 rock09, 3 rock10, 4 rock11, 5 rock12, 6 rock13.
# Sized for a 30 cm tank: the largest is under a third of its width.
ISLANDS = [
    ("Isle_main",  "rock_moss_set_02", 4, 88, (86, 198, 160),  (180, 0, 0), (8, -6, 18),   (9, 16, 40, 1.7), "full"),
    ("Isle_mid",   "rock_moss_set_02", 5, 60, (236, 160, 104), (180, 0, 0), (-6, 4, 130),  (5, 10, 24, 1.4), "grass"),
    ("Isle_small", "rock_moss_set_02", 1, 42, (214, 248, 202), (180, 0, 0), (4, 7, 250),   (5, 16, 36, 1.1), "carpet"),
    ("Isle_tiny",  "rock_moss_set_02", 3, 26, (44, 94, 120),   (180, 0, 0), (8, 0, 60),    (3, 8, 16, 0.9),  "moss"),
]
LONG_ROOTS = {"Isle_main": (0.25, 0.20)}         # which islands drop a root to the sand, from where (fraction of radius)
FALL_FROM = ("Isle_small", (-0.35, -0.95))       # the sand-fall pours from this island's edge, on this side

# fallen pieces on the bed: model, piece, longest side, x, y, how stood up, heading, fraction buried
FALLEN = [
    ("rock_09",          0, 30, 150, 50,  (0, 0, 0),  35,  0.35),
    ("rock_moss_set_02", 6, 28, 172, 136, (0, 0, 0),  200, 0.42),
    ("boulder_01",       0, 26, 86,  52,  (0, 0, 0),  120, 0.35),
]

MOSS_POINTS = 52000           # fronds across all island tops and rims; the scene's heaviest item
ROOT_MOSS_POINTS = 2600
GROUND_CLEAR = 5.0            # mm a hanging root stops above the sand


def _smooth(e0, e1, v):
    t = min(1.0, max(0.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _fall_xy():
    """Where the sand-fall lands, from the island it leaves (set in build)."""
    return _FALL["xy"]


_FALL = {"xy": (200.0, 232.0)}


def ground(x, y):
    """Sand height in mm: 60 at the front glass rising gently to about 80 at the
    back, soft dunes, a cone where the sand-fall lands and a little heap round
    each fallen stone."""
    h = 60 + 20 * (max(y, 0) / 290) ** 1.1
    h += 2.4 * math.sin(x * 0.034 + 0.6) * math.cos(y * 0.029 - 0.4) + 1.1 * math.sin(x * 0.083 + y * 0.051 + 1.0)
    h += 0.5 * math.sin(x * 0.21 - y * 0.17)
    fx, fy = _fall_xy()
    r = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + 2.5 ** 2)
    h += 9.0 * max(0.0, 1.0 - r / 27.0) ** 1.6
    for _, _, size, sx, sy, *_ in FALLEN:
        h += 1.8 * math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (size * 0.75) ** 2)
    return h


def _ground_normal(x, y):
    e = 1.0
    gx = (ground(x + e, y) - ground(x - e, y)) / (2 * e)
    gy = (ground(x, y + e) - ground(x, y - e)) / (2 * e)
    return Vector((-gx, -gy, 1.0)).normalized()


def _n3(p, scale, off=0.0):
    """Perlin noise -1..1 at a point in mm."""
    return noise.noise(Vector((p[0] * scale + off, p[1] * scale - off * 0.7, p[2] * scale + off * 1.3)))


# --- rocks ---------------------------------------------------------------------

def _import_piece(model, piece):
    objs = assets.import_model(model)
    keep = objs[piece]
    for o in objs:
        if o is not keep:
            bpy.data.objects.remove(o, do_unlink=True)
    return keep


def _grade(obj, seed, bump=0.5):
    """One rock type for every piece: dark weathered brown, scarcely veined,
    with rust and an algae film on faces that look up at the lamp."""
    assets.grade_stone(obj, dark=(0.036, 0.027, 0.019), light=(0.40, 0.31, 0.22), lum_range=(0.10, 0.62),
                       vein_amount=0.08, rust=0.40, algae=0.75, bump=bump, normal_strength=2.4, seed=seed)


def _break_underside(obj, seed, cuts, keel):
    """Make the hanging underside read as rock broken out of a bank, not a turned
    bowl: several tilted fracture planes shear facets off it, each a different
    depth and slant, and the lowest point is pushed off-centre so no two sides
    match. Only the lower part moves; the mossy top keeps the scan's shape."""
    me = obj.data
    n = len(me.vertices)
    loc = np.empty(n * 3)
    me.vertices.foreach_get("co", loc)
    mw = np.array(obj.matrix_world)
    co = (loc.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]) / MM
    lo, hi = co.min(0), co.max(0)
    H = hi[2] - lo[2]
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.55 * H])
    rel = (co[:, 2] - lo[2]) / H
    rng = random.Random(seed)
    # lopsided keel: the deepest part of the underside sits toward one side
    a = rng.uniform(0, 2 * math.pi)
    drop = 1.0 - _smooth_np(0.0, 0.6, rel)
    co[:, 0] += math.cos(a) * keel * H * drop ** 1.5
    co[:, 1] += math.sin(a) * keel * H * drop ** 1.5
    az0 = rng.uniform(0, 2 * math.pi)
    keep = 1.0 - _smooth_np(0.55, 0.85, rel)
    ripple = np.array([_n3(p, 0.22, seed) for p in co])
    # Steps: on one side of a ragged line the rock has broken away up to a
    # nearly level face, so the underside drops in ledges of different heights
    # instead of curving smoothly to a point.
    for k in range(cuts // 2 + 1):
        az = az0 + 2.4 * k + rng.uniform(-0.4, 0.4)
        side = np.array([math.cos(az), math.sin(az), 0.0])
        edge = (co - c) @ side - rng.uniform(-0.15, 0.35) * (hi[0] - lo[0]) / 2
        edge += 2.5 * np.array([_n3(p, 0.09, seed + 20 + k) for p in co])
        mask = _smooth_np(-1.2, 1.2, edge)
        tilt = math.radians(rng.uniform(4, 22))
        nv = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
        proj = (co - c) @ nv
        lim = proj.max() - rng.uniform(0.10, 0.26) * H + 0.35 * ripple * (0.04 * H + 0.6)
        co -= (np.clip(proj - lim, 0.0, None) * mask * keep)[:, None] * nv[None, :]
    # chamfers: steeper fracture faces round the sides
    for k in range(cuts):
        az = az0 + 2 * math.pi * k / cuts + rng.uniform(-0.5, 0.5)
        tilt = math.radians(rng.uniform(40, 70))
        nv = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
        proj = (co - c) @ nv
        lim = proj.max() - rng.uniform(0.06, 0.18) * H
        # a fracture face is flat but never polished: a little ripple on the plane
        lim = lim + 0.35 * ripple * (0.04 * H + 0.6)
        co -= (np.clip(proj - lim, 0.0, None) * keep)[:, None] * nv[None, :]
    inv = np.linalg.inv(mw)
    loc = (co * MM) @ inv[:3, :3].T + inv[:3, 3]
    me.vertices.foreach_set("co", loc.astype(np.float32).ravel())
    me.update()


def _smooth_np(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _moss_floor(obj):
    """Darken the stone under the moss: between fronds you see the mat's dark
    lower layer and trapped mulm, never clean rock."""
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        N, L = nt.nodes, nt.links
        bsdf = next((n for n in N if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None or not bsdf.inputs["Base Color"].links:
            continue
        src = bsdf.inputs["Base Color"].links[0].from_socket
        attr = N.new("ShaderNodeAttribute")
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = "moss"
        fac = N.new("ShaderNodeMapRange")
        fac.inputs["From Max"].default_value = 0.45
        fac.inputs["To Max"].default_value = 0.9
        L.new(attr.outputs["Fac"], fac.inputs["Value"])
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["B"].default_value = (0.020, 0.032, 0.010, 1.0)
        L.new(fac.outputs["Result"], mix.inputs["Factor"])
        L.new(src, mix.inputs["A"])
        L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


def _carrier_mesh(obj, name):
    """A copy of the rock's mesh baked into world space, so scatters on it are
    measured in real millimetres (the scan object itself is scaled by ~0.1)."""
    me = obj.data.copy()
    me.name = name
    me.transform(obj.matrix_world)
    me.update()
    return me


def _mesh_arrays(me):
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    nr = np.empty(n * 3)
    me.vertices.foreach_get("normal", nr)
    return co.reshape(n, 3) / MM, nr.reshape(n, 3)


def _bvh(me):
    verts = [v.co.copy() for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    return BVHTree.FromPolygons(verts, polys)


def _weighted_area(me, w):
    me.calc_loop_triangles()
    total = 0.0
    for tri in me.loop_triangles:
        total += tri.area * sum(w[i] for i in tri.vertices) / 3.0
    return total


def _island_weights(co, nr, lo, hi, seed, clear):
    """Moss on what faces up: tops and ledges, with a ragged, patchy edge and
    the odd knuckle of bare rock poking through. `clear` is (point, radius)
    pairs to leave bare (a grazing shrimp, a plant pushing through)."""
    zl, zh = lo[2], hi[2]
    moss, rim, top = [], [], []
    for p, n in zip(co, nr):
        rel = (p[2] - zl) / max(zh - zl, 1e-6)
        a = _n3(p, 0.085, seed)
        b = _n3(p, 0.32, seed + 7.0)
        edge = 0.55 + 0.20 * a + 0.09 * b
        # the top carries the mat; lower down only the flattest ledges hold a tuft
        height = max(_smooth(0.52 + 0.12 * a, 0.80 + 0.08 * b, rel), 0.35 * _smooth(0.88, 0.96, n[2]) * _smooth(0.35, 0.5, rel) * _smooth(0.1, 0.4, b))
        w = _smooth(edge - 0.10, edge + 0.12, n[2]) * height
        w *= 0.35 + 0.65 * _smooth(-0.45, 0.15, _n3(p, 0.05, seed + 3.0))
        w *= 1.0 - 0.9 * _smooth(0.22, 0.42, _n3(p, 0.16, seed + 11.0))      # bare knuckles
        for c, r in clear:
            w *= _smooth(r * 0.55, r, math.dist(p, c))
        moss.append(w)
        # longer fronds spilling over the lip: steep faces just under the top edge
        rim.append(_smooth(-0.15, 0.10, n[2]) * (1 - _smooth(0.40, 0.70, n[2])) * _smooth(0.68, 0.88, rel)
                   * _smooth(-0.2, 0.3, a))
        top.append(_smooth(0.78, 0.92, n[2]) * _smooth(0.55, 0.80, rel))
    return moss, rim, top


def _spots(co, top, centre, radius, n, rng, min_gap):
    """Pick n planting spots among the flattest, highest points of a top."""
    cand = [i for i, t in enumerate(top) if t > 0.85 and math.dist(co[i][:2], centre[:2]) < radius]
    rng.shuffle(cand)
    out = []
    for i in cand:
        if all(math.dist(co[i], co[j]) > min_gap for j in out):
            out.append(i)
        if len(out) == n:
            break
    return [tuple(co[i]) for i in out]


def _blob(co, centres, radius):
    return [max([math.exp(-(math.dist(p, c) / radius) ** 2) for c in centres] or [0.0]) for p in co]


# --- roots ---------------------------------------------------------------------

def _merge(dst, src):
    off = len(dst.verts)
    dst.verts += [tuple(v) for v in src.verts]
    dst.faces += [tuple(i + off for i in f) for f in src.faces]
    dst.tips += src.tips


def _root(start, direction, length, radius, seed, gravity, depth=2, children=(1, 2), wander=0.6):
    return wood.grow(start, direction, length, radius, seed=seed, depth=depth, children=children, wander=wander,
                     gravity=gravity, spread=(0.30, 0.85), child_length=0.55, child_radius=0.55, taper=0.22, sides=6)


def _fits(skin, margin=GROUND_CLEAR):
    for x, y, z in skin.verts:
        xm, ym, zm = x / MM, y / MM, z / MM
        if not (14 < xm < 286 and 14 < ym < 286):
            return False
        if zm < ground(xm, ym) + margin:
            return False
    return True


def _underside(bvh, x, y, z_from):
    """First rock surface straight above (x, y) from below the island."""
    hit, normal, _, _ = bvh.ray_cast(Vector((x * MM, y * MM, z_from * MM)), Vector((0, 0, 1)), 0.3)
    return hit, normal


def _grow_roots(spec, bvh, lo, hi, rng):
    """Several thin roots from the underside, wandering down under gravity;
    each is regrown shorter until it clears the sand and the glass."""
    name, *_, (count, l0, l1, r0), _ = spec
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    R = min(hi[0] - lo[0], hi[1] - lo[1]) / 2
    skin = wood._Skin(6)
    starts = []
    tries = 0
    while len(starts) < count and tries < 200:
        tries += 1
        a = rng.uniform(0, 2 * math.pi)
        d = R * 0.80 * math.sqrt(rng.uniform(0.05, 1.0))       # from all over the broken underside
        hit, n = _underside(bvh, cx + d * math.cos(a), cy + d * math.sin(a), lo[2] - 5)
        if hit is None or n.z > -0.25 or any((hit - s).length < 9 * MM for s in starts):
            continue
        starts.append(hit)
        out = Vector((math.cos(a), math.sin(a), 0.0))
        direction = (Vector((0, 0, -1.0)) + out * rng.uniform(0.08, 0.35) + n * 0.2).normalized()
        L = rng.uniform(l0, l1)
        rad = r0 * rng.uniform(0.5, 1.0)
        seed = rng.randint(0, 10 ** 6)
        for _ in range(8):
            s = _root(hit + Vector((0, 0, 2.5 * MM)), direction, L, rad, seed, gravity=rng.uniform(0.45, 0.8))
            if _fits(s):
                break
            L *= 0.82
        else:
            continue
        _merge(skin, s)
    return skin, starts


def _long_root(bvh, lo, hi, frac, seed):
    """One root that reaches almost to the sand: grown, measured, and its
    length rescaled until the tip hangs GROUND_CLEAR above the bed."""
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    R = min(hi[0] - lo[0], hi[1] - lo[1]) / 2
    hit, _ = _underside(bvh, cx + frac[0] * R, cy + frac[1] * R, lo[2] - 5)
    if hit is None:
        return None, None
    start = hit + Vector((0, 0, 2.5 * MM))
    target = ground(hit.x / MM, hit.y / MM) + GROUND_CLEAR + 3
    L = (start.z / MM - target) * 1.05
    best = None
    for _ in range(10):
        s = _root(start, Vector((0.08, -0.05, -1.0)).normalized(), L, 1.7, seed, gravity=0.9, depth=2,
                  children=(1, 1), wander=0.35)
        tips_z = min(v[2] for v in s.verts) / MM
        if _fits(s, GROUND_CLEAR):
            best = s
            if tips_z < target + 6:
                break
            L *= 1.0 + (tips_z - target) / max(start.z / MM - tips_z, 1.0)
        else:
            L *= 0.93
    return best, hit


def _root_moss_weights(skin, starts):
    """Moss climbs a little way down each root from the rock, in patches."""
    w = []
    for v in skin.verts:
        p = Vector(v)
        d = min(((p - s).length for s in starts), default=1.0) / MM
        w.append((1.0 - _smooth(2.0, 8.0, d)) *(0.5 + 0.5 * _smooth(-0.3, 0.3, _n3([c / MM for c in v], 0.2, 4.0))))
    return w


# --- sand-fall -----------------------------------------------------------------

def _fall_lip(isle, side):
    """Where the sand leaves the island: just inside its underside, a few
    millimetres in from the lower edge on the given side, so the stream pours
    off the edge with its top hidden in the rock. Any further in and it would
    read as a post holding the island up."""
    co, _ = _mesh_arrays(isle["me"])
    lo, hi = isle["lo"], isle["hi"]
    c = (lo + hi) / 2
    d = Vector((side[0], side[1], 0.0)).normalized()
    rel = (co[:, 2] - lo[2]) / (hi[2] - lo[2])
    score = (co[:, 0] - c[0]) * d.x + (co[:, 1] - c[1]) * d.y - 1e4 * ((rel < 0.04) | (rel > 0.30))
    edge = Vector(co[int(np.argmax(score))]) * MM
    for k in range(1, 40):
        q = edge - d * k * 0.5 * MM
        hit, *_ = isle["bvh"].ray_cast(Vector((q.x, q.y, (lo[2] - 5) * MM)), Vector((0, 0, 1)), 0.3)
        if hit is not None and k * 0.5 >= 5.0:
            return hit + Vector((0, 0, 7.0 * MM))
    return edge


class _FallPath:
    """The stream's centre line: from inside the rock almost plumb to the bed,
    with the faintest drift, as sand falls in still water."""

    def __init__(self, start, bed_z, seed):
        rng = random.Random(seed)
        self.ph = [rng.uniform(0, 6.28) for _ in range(2)]
        self.start = start
        self.H = start.z / MM - bed_z
        self.L = self.H

    def __call__(self, t):
        sway = Vector((0.7 * math.sin(t * 13.0 + self.ph[0]) * t, 0.5 * math.sin(t * 9.0 + self.ph[1]) * t, -t * self.H))
        return self.start + sway * MM


def _sandfall_mesh(path):
    """The stream's body: a narrow, tapering tube along the path, from inside
    the rock to inside the cone it builds on the bed. Grains falling round it
    are separate (_FallGrains)."""
    segs = max(30, int(path.L / 1.5))
    sides = 10
    verts, faces = [], []
    for k in range(segs + 1):
        t = k / segs
        c = path(t)
        r = (3.3 - 1.4 * t ** 0.5) * (1 + 0.06 * math.sin(t * 9 + path.ph[0]))
        r += 1.0 * max(0.0, (t - 0.93) / 0.07) ** 2           # a little puff where it lands
        for s_ in range(sides):
            a = 2 * math.pi * s_ / sides
            verts.append(tuple(c + Vector((math.cos(a) * r, math.sin(a) * r, 0.0)) * MM))
    for k in range(segs):
        for s_ in range(sides):
            a = k * sides + s_
            b = k * sides + (s_ + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return verts, faces


class _FallGrains:
    """Single grains tumbling down round the body, so the stream's edge is
    granular and visibly moves. One mesh of tiny stretched spheres, rewritten
    each frame: every grain slides down at the stream's speed and wraps back
    to the top, straying a little further as it falls."""

    def __init__(self, name, path, material, count=1400, seed=8):
        rng = np.random.default_rng(seed)
        self.path = path
        self.u0 = rng.random(count)
        self.ang = rng.uniform(0, 2 * math.pi, count)
        self.rad = 0.9 + np.abs(rng.normal(0, 0.7, count))
        self.size = rng.uniform(0.10, 0.20, count)
        self.speed = rng.uniform(0.85, 1.15, count)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        self.unit = np.array([tuple(v.co) for v in bm.verts])
        tris = [tuple(v.index for v in f.verts) for f in bm.faces]
        bm.free()
        nv = len(self.unit)
        faces = [tuple(i + g * nv for i in f) for g in range(count) for f in tris]
        self.obj = new_mesh_object(name, [(0.0, 0.0, 0.0)] * (count * nv), faces, smooth=True)
        self.obj.data.materials.append(material)
        self.pose(0.0)

    def pose(self, t):
        u = (self.u0 + 60.0 * self.speed * t / self.path.L) % 1.0       # about 60 mm/s
        u = 0.06 + 0.94 * u                                              # none inside the rock
        axis = np.array([tuple(self.path(float(x))) for x in u]) / MM
        r = self.rad * (0.85 - 0.25 * u ** 0.6) * (1.0 + 0.5 * u)
        off = np.stack([np.cos(self.ang), np.sin(self.ang), np.zeros_like(u)], 1) * r[:, None]
        # stretched along the fall: at this speed a grain smears over the exposure
        shape = self.unit[None, :, :] * np.array([1.0, 1.0, 4.0])[None, None, :] * self.size[:, None, None]
        co = ((axis + off)[:, None, :] + shape).reshape(-1, 3) * MM
        self.obj.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
        self.obj.data.update()


def _sandfall_material():
    """Falling sand is a stream of grains, not a rod or a glass tube: matt,
    with streaks that run down it and drift with time, mostly opaque, and an
    edge that breaks up into separate grains rather than fading smoothly."""
    mat, nt, out = new_material("SandFall")
    N, L = nt.nodes, nt.links
    coord = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (900.0, 900.0, 90.0)
    L.new(coord.outputs["Object"], mp.inputs["Vector"])
    streak = N.new("ShaderNodeTexNoise")
    streak.inputs["Scale"].default_value = 1.0
    streak.inputs["Detail"].default_value = 6.0
    streak.inputs["Roughness"].default_value = 0.6
    L.new(mp.outputs["Vector"], streak.inputs["Vector"])
    mp2 = N.new("ShaderNodeMapping")
    mp2.inputs["Scale"].default_value = (6000.0, 6000.0, 1500.0)
    L.new(coord.outputs["Object"], mp2.inputs["Vector"])
    grain = N.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1.0
    grain.inputs["Detail"].default_value = 1.0
    L.new(mp2.outputs["Vector"], grain.inputs["Vector"])

    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.30, (0.56, 0.53, 0.46, 1)
    e[1].position, e[1].color = 0.72, (0.86, 0.83, 0.76, 1)
    L.new(streak.outputs["Fac"], ramp.inputs["Fac"])
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.1
    L.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # toward the silhouette (Facing -> 1) a rising threshold on fine noise eats
    # the surface away grain by grain
    lw = N.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.4
    thr = N.new("ShaderNodeMapRange")
    thr.inputs["From Min"].default_value = 0.15
    thr.inputs["From Max"].default_value = 0.95
    thr.inputs["To Min"].default_value = 0.20
    thr.inputs["To Max"].default_value = 0.75
    L.new(lw.outputs["Facing"], thr.inputs["Value"])
    top = N.new("ShaderNodeMath")
    top.operation = "ADD"
    top.inputs[1].default_value = 0.06
    L.new(thr.outputs["Result"], top.inputs[0])
    gm = N.new("ShaderNodeMapRange")
    L.new(grain.outputs["Fac"], gm.inputs["Value"])
    L.new(thr.outputs["Result"], gm.inputs["From Min"])
    L.new(top.outputs[0], gm.inputs["From Max"])
    dens = N.new("ShaderNodeMapRange")              # streaks: denser and thinner bands of grains
    dens.inputs["From Min"].default_value = 0.25
    dens.inputs["From Max"].default_value = 0.65
    dens.inputs["To Min"].default_value = 0.35
    dens.inputs["To Max"].default_value = 0.95
    L.new(streak.outputs["Fac"], dens.inputs["Value"])
    a = N.new("ShaderNodeMath")
    a.operation = "MULTIPLY"
    L.new(gm.outputs["Result"], a.inputs[0])
    L.new(dens.outputs["Result"], a.inputs[1])
    # light goes through a curtain of grains, so it is lit through, not shaded like a rod
    tl = N.new("ShaderNodeBsdfTranslucent")
    L.new(ramp.outputs["Color"], tl.inputs["Color"])
    body = N.new("ShaderNodeMixShader")
    body.inputs["Fac"].default_value = 0.2
    L.new(bsdf.outputs[0], body.inputs[1])
    L.new(tl.outputs[0], body.inputs[2])
    trans = N.new("ShaderNodeBsdfTransparent")
    mix = N.new("ShaderNodeMixShader")
    L.new(a.outputs[0], mix.inputs["Fac"])
    L.new(trans.outputs[0], mix.inputs[1])
    L.new(body.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat, (mp, mp2)


# --- build ---------------------------------------------------------------------

def _sand_material():
    """The kit's white sand, a little warmer and less bright: pure white reads
    as snow or plaster under this lamp, real aquarium sand is ivory to beige."""
    mat = materials.white_sand("FloatSand")
    for n in mat.node_tree.nodes:
        if n.type == "HUE_SAT":
            n.inputs["Value"].default_value = 1.2
            n.inputs["Saturation"].default_value = 0.75
        elif n.type == "MIX" and n.blend_type == "MULTIPLY" and not n.inputs["B"].links:
            n.inputs["B"].default_value = (0.96, 0.89, 0.77, 1.0)
        elif n.type == "BUMP":
            n.inputs["Strength"].default_value = 0.8
    return mat


def _grit(bed, made):
    """Loose grains on the sand, pale with some tan and a few dark ones, as in
    any bag of real aquarium sand; they give the surface its grain and tiny
    shadows, which the texture alone cannot at this distance."""
    cols = [((0.70, 0.66, 0.58), 4), ((0.62, 0.55, 0.45), 3), ((0.42, 0.34, 0.25), 2), ((0.14, 0.12, 0.10), 1)]
    grains = []
    for k, (c, count) in enumerate(cols):
        m = substrate.sand_material(f"Grit_{k}", colour=c)
        for j in range(count):
            grains.append(substrate._granule(f"Grit_{k}_{j}", 0.55 * MM, 70 + k * 5 + j, m))
    bed.set_weight("grit", lambda x, y: (0.55 + 0.45 * _smooth(-0.4, 0.4, _n3((x, y, 0), 0.06, 8.0)))
                   * _smooth(11.0, 14.0, min(x, y, 300 - x, 300 - y)))
    obj = bpy.data.objects.new("SandGrit", bed.obj.data)
    bpy.context.scene.collection.objects.link(obj)
    made.append(obj)
    scatter(obj, collection_of("GritSrc", grains), name="SandGrit", density=1.4e5, weight_attr="grit", seed=17,
            scale=(0.6, 1.4), align="NORMAL", tilt=math.pi, keep_surface=False, sink=0.25 * MM)


def _moss_sources():
    """Christmas moss in two greens: a deep one for the mat and an olive one,
    darker than the kit's default, which glows like lawn under this lamp."""
    deep = plants.leaf_material("MossDeep", base=(0.009, 0.026, 0.006), tip=(0.062, 0.165, 0.026), translucency=0.30,
                                roughness=0.55, variation=0.3)
    olive = plants.leaf_material("MossOlive", base=(0.018, 0.032, 0.008), tip=(0.10, 0.165, 0.030), translucency=0.3,
                                 roughness=0.55, variation=0.3)
    return collection_of("MossSrc", plants.moss_fronds(variants=6, seed=13, material=deep) +
                         plants.moss_fronds(variants=4, seed=29, length_mm=(4, 8), material=olive))


def build(ctx):
    made = []

    # islands first: the sand-fall's landing point shapes the bed
    isles = []
    for k, spec in enumerate(ISLANDS):
        name, model, piece, longest, (x, y, zb), pre, rot, _, _ = spec
        obj = _import_piece(model, piece)
        obj.name = name
        # NOTE the lean and heading in the table are not applied: assets.place() used to
        # ignore rot_deg for scans, and the look the owner reviewed is the unrotated one.
        assets.place(obj, (x * MM, y * MM, zb * MM), longest_mm=longest, pre_rot_deg=pre)
        bpy.context.view_layer.update()
        _break_underside(obj, seed=7 + 13 * k, cuts=7 if longest > 70 else 5 if longest > 35 else 4, keel=0.35)
        lo, hi = assets.world_bounds(obj)
        obj.location += Vector((x * MM - (lo.x + hi.x) / 2, y * MM - (lo.y + hi.y) / 2, zb * MM - lo.z))
        bpy.context.view_layer.update()
        _grade(obj, seed=k * 1.7 + 0.4, bump=0.85)       # the cut faces need grit the scan's map spreads thin
        _moss_floor(obj)
        me = _carrier_mesh(obj, f"{name}_carrier")
        lo, hi = (np.array(v) / MM for v in assets.world_bounds(obj))
        isles.append(dict(spec=spec, obj=obj, me=me, bvh=_bvh(me), lo=lo, hi=hi))
        made.append(obj)

    fall_isle = next(i for i in isles if i["spec"][0] == FALL_FROM[0])
    lip = _fall_lip(fall_isle, FALL_FROM[1])
    _FALL["xy"] = (lip.x / MM, lip.y / MM)

    bed = substrate.build_bed(ground, "Sand")
    bed.obj.data.materials.append(_sand_material())
    made.append(bed.obj)
    _grit(bed, made)
    made += substrate.add_cut_face(bed, substrate.sand_mix("FloatFace"), size_mm=1.2, seed=4, tone=(1.3, 0.6), room=0.15)

    # the sand-fall
    path = _FallPath(lip, ground(lip.x / MM, lip.y / MM) - 1.5, seed=3)
    verts, faces = _sandfall_mesh(path)
    fall = new_mesh_object("SandFall", verts, faces, smooth=True)
    # The solid core read as a white rod holding the island up. The fall is left
    # to its moving grains alone, which read as sand in the act of falling.
    fall.hide_render = True
    fall_mat, fall_maps = _sandfall_material()
    fall.data.materials.append(fall_mat)
    grains = _FallGrains("SandFallGrains", path, substrate.sand_material("FallGrain", colour=(0.66, 0.62, 0.54)))
    made += [fall, grains.obj]

    # A shrimp grazes on a ledge of the main island just under its mossy rim.
    # The camera is below the island tops, so it must be on the front edge to show.
    main = isles[0]
    co, nr = _mesh_arrays(main["me"])
    rel = (co[:, 2] - main["lo"][2]) / (main["hi"][2] - main["lo"][2])
    mx = (main["lo"][0] + main["hi"][0]) / 2 + 12            # a little right of centre, toward the open lane
    cand = [i for i in range(len(co)) if 0.5 < nr[i][2] < 0.92 and nr[i][1] < -0.2 and 0.5 < rel[i] < 0.8
            and abs(co[i][0] - mx) < 24]
    perch_i = min(cand, key=lambda i: co[i][1] - 0.5 * co[i][2] - 8.0 * nr[i][2])
    perch = (Vector(co[perch_i]) * MM, Vector(nr[perch_i]).normalized())

    moss_src = _moss_sources()
    sprig_src = collection_of("IsleCarpetSrc", plants.carpet_sprigs(variants=5, seed=7, leaf_mm=3.0))
    stauro_src = collection_of("IsleStauroSrc", plants.stem_plants("staurogyne", variants=4, seed=3))
    lobelia_src = collection_of("IsleLobeliaSrc", plants.stem_plants("lobelia", variants=3, seed=5))
    grass_src = collection_of("IsleGrassSrc", plants.hairgrass_tufts(variants=4, blades=(9, 15), height_mm=(12, 24)))

    # Plant spots are chosen first so the moss can leave each plant a small
    # clearing to stand up out of, as a planted stem pushes the mat aside.
    groups_for = {"full": [("stauro", stauro_src, 3, (0.48, 0.70)), ("lobelia", lobelia_src, 1, (0.40, 0.50)),
                           ("grass", grass_src, 1, (0.75, 1.0))],
                  "grass": [("grass", grass_src, 2, (0.65, 0.95)), ("stauro", stauro_src, 1, (0.45, 0.6))],
                  "carpet": [], "moss": []}
    weights = []
    for k, isle in enumerate(isles):
        co, nr = _mesh_arrays(isle["me"])
        lo, hi = isle["lo"], isle["hi"]
        rel = (co[:, 2] - lo[2]) / (hi[2] - lo[2])
        top = [_smooth(0.78, 0.92, n[2]) * _smooth(0.55, 0.80, r) for n, r in zip(nr, rel)]
        centre = (lo + hi) / 2
        R = min(hi[0] - lo[0], hi[1] - lo[1]) / 2
        taken = [tuple(perch[0] / MM)] if k == 0 else []
        spots = []
        plant_rng = random.Random(100 + k)
        for gname, src, n, sc in groups_for[isle["spec"][8]]:
            pts = [p for p in _spots(co, top, centre, R * 0.6, n + len(taken) + 2, plant_rng, 16.0)
                   if all(math.dist(p, q) > 14 for q in taken)][:n]
            taken += pts
            spots.append((gname, src, sc, pts))
        clear = [(tuple(perch[0] / MM), 14.0)] if k == 0 else []
        clear += [(p, 6.0) for *_, pts in spots for p in pts]
        moss, rim, top = _island_weights(co, nr, lo, hi, seed=k * 3.1, clear=clear)
        set_point_attribute(isle["obj"], "moss", moss)
        weights.append((isle, co, moss, rim, top, spots))
    area = sum(_weighted_area(w[0]["me"], w[2]) for w in weights)
    rim_area = sum(_weighted_area(w[0]["me"], w[3]) for w in weights)
    dens = min(1.6e6, MOSS_POINTS * 0.88 / max(area, 1e-9))
    rim_dens = min(1.5e5, MOSS_POINTS * 0.12 / max(rim_area, 1e-9))
    print(f"[float] moss area {area * 1e4:.1f} cm2 -> density {dens:.0f}/m2, rim {rim_area * 1e4:.1f} cm2 -> {rim_dens:.0f}/m2")

    for k, (isle, co, moss, rim, top, spots) in enumerate(weights):
        name, kind = isle["spec"][0], isle["spec"][8]
        me = isle["me"]
        _set_mesh_attribute(me, "moss", moss)
        _set_mesh_attribute(me, "rim", rim)
        f = 0.65 if isle["spec"][3] < 35 else 0.8 if isle["spec"][3] < 70 else 1.0   # finer moss on the small islands keeps them in scale
        scatter(_carrier(me, f"{name}_moss", made), moss_src, name=f"Moss_{name}", density=dens, weight_attr="moss",
                seed=40 + k, scale=(0.7 * f, 1.6 * f), align="NORMAL", tilt=0.55, keep_surface=False,
                sway=(0.04, 30.0, 0.3))
        scatter(_carrier(me, f"{name}_rim", made), moss_src, name=f"MossRim_{name}", density=rim_dens, weight_attr="rim",
                seed=60 + k, scale=(1.4 * f, 2.2 * f), align="NORMAL", tilt=0.7, keep_surface=False,
                sway=(0.06, 25.0, 0.3))
        if kind == "moss":
            continue
        carpet = [t * (0.3 + 0.7 * _smooth(-0.2, 0.35, _n3(p, 0.07, 9.0 + k))) * min(1.0, m * 1.5) for p, t, m in zip(co, top, moss)]
        _set_mesh_attribute(me, "carpet", carpet)
        scatter(_carrier(me, f"{name}_carpet", made), sprig_src, name=f"IsleCarpet_{name}", min_distance=4.2 * MM,
                weight_attr="carpet", seed=80 + k, scale=(0.55, 0.9), align="UP", tilt=0.3, keep_surface=False,
                sway=(0.06, 12.0, 0.35))
        for gname, src, sc, pts in spots:
            if not pts:
                continue
            _set_mesh_attribute(me, f"spot_{gname}", _blob(co, pts, 3.2))
            scatter(_carrier(me, f"{name}_{gname}", made), src, name=f"Isle{gname}_{name}",
                    min_distance=5.0 * MM, density_max=0.06 / (MM * MM), weight_attr=f"spot_{gname}",
                    seed=120 + k, scale=sc, align="UP", tilt=0.15, keep_surface=False, sway=(0.05, 10.0, 0.3))

    # hanging roots, and one or two that reach almost to the sand
    bark = materials.driftwood("Roots", tint=(0.62, 0.45, 0.32), value=0.62)
    root_objs = []
    for k, isle in enumerate(isles):
        skin, starts = _grow_roots(isle["spec"], isle["bvh"], isle["lo"], isle["hi"], random.Random(20 + k))
        name = isle["spec"][0]
        isle["root_lo"] = min((v[2] / MM for v in skin.verts), default=isle["lo"][2])
        if name in LONG_ROOTS:
            s, hit = _long_root(isle["bvh"], isle["lo"], isle["hi"], LONG_ROOTS[name], seed=300 + k)
            if s is not None:
                _merge(skin, s)
                starts.append(hit)
                isle["long"] = s
        if not skin.verts:
            continue
        obj = wood.finish(skin, f"{name}_roots", bark, subdivide=1)
        obj.modifiers["Knots"].strength = 0.0007        # the kit's knots are sized for trunks
        set_point_attribute(obj, "moss", _root_moss_weights(skin, starts))
        root_objs.append((obj, skin))
        isle["roots"] = skin
        made.append(obj)
    root_area = sum(_weighted_area(o.data, [a.value for a in o.data.attributes["moss"].data]) for o, _ in root_objs)
    rdens = min(1.2e6, ROOT_MOSS_POINTS / max(root_area, 1e-9))
    for k, (obj, _) in enumerate(root_objs):
        scatter(obj, moss_src, name=f"RootMoss_{k}", density=rdens, weight_attr="moss", seed=200 + k,
                scale=(0.6, 1.1), align="NORMAL", tilt=0.6, sway=(0.04, 30.0, 0.3))

    # a few pieces that fell long ago, half sunk in the sand, a little carpet round them
    fallen = []
    for k, (model, piece, size, x, y, pre, heading, buried) in enumerate(FALLEN):
        obj = _import_piece(model, piece)
        obj.name = f"Fallen_{k}"
        # (heading is not applied either, for the same reason as the islands above)
        assets.place(obj, bed.at(x, y), longest_mm=size, pre_rot_deg=pre)
        lo, hi = assets.world_bounds(obj)
        obj.location.z -= (hi.z - lo.z) * buried
        _grade(obj, seed=10 + k * 1.3)
        fallen.append((x, y, assets.footprint_radius_mm(obj)))
        made.append(obj)

    def carpet_bed(x, y):
        w = 0.0
        for sx, sy, r in fallen:
            d = math.hypot(x - sx, y - sy)
            reach = r + 10 + 7 * _n3((x, y, 0), 0.09, 2.0)
            w = max(w, _smooth(r * 0.8, r * 1.05, d) * (1 - _smooth(reach - 6, reach, d)))
        return w * _smooth(-0.25, 0.25, _n3((x, y, 0), 0.12, 5.0))            # in patches, so the stone shows
    bed.set_weight("carpet", carpet_bed)
    bed_sprigs = collection_of("BedCarpetSrc", plants.carpet_sprigs(variants=5, seed=19))
    scatter(bed.obj, bed_sprigs, name="BedCarpet", min_distance=4.0 * MM, weight_attr="carpet", seed=31,
            scale=(0.45, 0.8), align="UP", tilt=0.3, sway=(0.06, 12.0, 0.35))

    # The film's one shoal; SWIM above says where it may go in this scape. These
    # are the measured islands as hanging cylinders (rock, moss and plants on top,
    # short roots below; the long root to the sand is left out), printed so SWIM's
    # cylinders can be kept true to them.
    print("[isles] " + ", ".join(
        f"{i['spec'][0]} ({(i['lo'][0] + i['hi'][0]) / 2:.0f}, {(i['lo'][1] + i['hi'][1]) / 2:.0f}, "
        f"{max(i['hi'][0] - i['lo'][0], i['hi'][1] - i['lo'][1]) / 2:.0f}, {i['hi'][2]:.0f}, {i['root_lo']:.0f})"
        for i in isles) + f"; fall at ({_FALL['xy'][0]:.0f}, {_FALL['xy'][1]:.0f}) from z {lip.z / MM:.0f}")
    school = fish.School()
    made += school.objects

    # livestock on the rock, the sand and the glass
    top_shrimp = shrimp.Shrimp("Shrimp_isle", size=0.92)
    sand_shrimp = shrimp.Shrimp("Shrimp_sand", size=1.0)
    nerite = snail.Snail("Nerite")
    made += [top_shrimp.obj, sand_shrimp.obj, nerite.obj]

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["fall_maps"] = fall_maps
    ctx["fall_grains"] = grains
    ctx["critters"] = dict(top=(top_shrimp, perch), sand=sand_shrimp, snail=nerite)

    if os.environ.get("FLOAT_CHECK"):
        _check(ctx, isles, root_objs, fall)
    return ctx


def _set_mesh_attribute(me, name, values):
    attr = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", list(values))
    me.update()


def _carrier(me, name, made):
    """An object sharing the baked rock mesh that carries one scatter: stacked
    scatters on one object would also distribute over each other's output."""
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    made.append(obj)
    return obj


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    t = frame / 30.0

    # sand streams down the fall: its texture slides down at about 60 mm/s
    for mp in ctx["fall_maps"]:
        sz = mp.inputs["Scale"].default_value[2]
        mp.inputs["Location"].default_value[2] = sz * 0.060 * t
    ctx["fall_grains"].pose(t)

    top_shrimp, (pos, up) = ctx["critters"]["top"]
    top_shrimp.pose(pos + up * 0.2 * MM, yaw=2.9 + 0.25 * math.sin(t / 2.7), t=t, up=tuple(up), walk=0.0)

    # the sand shrimp picks its way slowly along the front, a few mm a second
    s = ctx["critters"]["sand"]
    w = 0.35
    x = 132 + (t - 67.0) * shrimp.Shrimp.WALK_SPEED * w * s.size / MM
    y = 40 + 3.0 * math.sin(t * 0.3)
    s.pose(Vector(ctx["bed"].at(x, y)), yaw=0.05 * math.cos(t * 0.3), t=t, up=tuple(_ground_normal(x, y)), walk=w)

    # The nerite crosses the sand at the front left, side-on to the camera so its
    # striped shell shows (on the front glass it showed only a grey sole).
    x = 58.0 + (t - 67.0) * 0.55
    y = 34.0
    ctx["critters"]["snail"].pose(Vector(ctx["bed"].at(x, y)), yaw=0.1, t=t, up=tuple(_ground_normal(x, y)))


# --- checks (FLOAT_CHECK=1) -------------------------------------------------------

def _check(ctx, isles, root_objs, fall):
    """Extents of everything after Geometry Nodes, at three frames, and the
    shoal's clearance from rock, roots and sand-fall over the whole shot."""
    scene = ctx["scene"]
    frame0 = scene.frame_current
    mine = {o.name for o in ctx["underwater"].objects}
    for o in [i["obj"] for i in isles] + [o for o, _ in root_objs] + [fall]:
        lo, hi = assets.world_bounds(o)
        print(f"[float-check] {o.name}: x {lo.x / MM:.1f}..{hi.x / MM:.1f} y {lo.y / MM:.1f}..{hi.y / MM:.1f} "
              f"z {lo.z / MM:.1f}..{hi.z / MM:.1f}")
    out_of = set()
    for frame in (1950, 2000, 2250):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        lo = np.full(3, 1e9)
        hi = np.full(3, -1e9)
        count = 0
        for inst in dg.object_instances:
            if not inst.is_instance or inst.parent is None or inst.parent.original.name not in mine:
                continue
            m = np.array(inst.matrix_world)
            bb = np.array([tuple(c) for c in inst.instance_object.bound_box])
            p = (bb @ m[:3, :3].T + m[:3, 3]) / MM
            lo = np.minimum(lo, p.min(0))
            hi = np.maximum(hi, p.max(0))
            count += 1
            if frame == 2000 and (p[:, :2].min() < 8 or p[:, :2].max() > 292 or p[:, 2].max() > 262):
                out_of.add(inst.parent.original.name)
        print(f"[float-check] frame {frame}: {count} instances, x {lo[0]:.1f}..{hi[0]:.1f} y {lo[1]:.1f}..{hi[1]:.1f} "
              f"z {lo[2]:.1f}..{hi[2]:.1f}")
    print(f"[float-check] scatters reaching outside x,y 8..292 or above z 262: {sorted(out_of) or 'none'}")
    scene.frame_set(frame0)
    for a in range(len(isles)):
        for b in range(a + 1, len(isles)):
            gap = min((isles[b]["bvh"].find_nearest(v.co, 0.5)[3] or 1.0) for v in list(isles[a]["me"].vertices)[::7]) / MM
            print(f"[float-check] gap {isles[a]['spec'][0]} - {isles[b]['spec'][0]}: {gap:.1f} mm")

    # clearance: every fish's spine (snout, middle, tail) against everything solid
    verts, polys, owners = [], [], []
    for name, me in [(i["spec"][0], i["me"]) for i in isles] + [(o.name, o.data) for o, _ in root_objs] + [("SandFall", fall.data)]:
        off = len(verts)
        owners.append((len(polys), name))
        verts += [v.co.copy() for v in me.vertices]
        polys += [tuple(i + off for i in p.vertices) for p in me.polygons]
    bvh = BVHTree.FromPolygons(verts, polys)

    def owner(k):
        return [n for start, n in owners if start <= k][-1]
    data = ctx["school"].data
    f_start = int(data["frame_start"])
    worst = (1e9, None, None, None)
    per = {}
    for f in range(data["pos"].shape[0]):
        frm = f + f_start
        for i in range(data["pos"].shape[1]):
            p = Vector(tuple(float(c) for c in data["pos"][f, i])) * MM
            d = Vector(tuple(float(c) for c in data["fwd"][f, i]))
            for s in (0.0, 0.4, 0.8):
                q = p - d * s * fish.TOTAL_MM * MM
                hit = bvh.find_nearest(q, 0.05)
                dist = (hit[3] / MM) if hit[0] is not None else 50.0
                if 1935 <= frm < 2190 and dist < worst[0] and hit[0] is not None:
                    worst = (dist, frm, i, owner(hit[2]))
                if frm in (1950, 2000, 2250):
                    per[frm] = round(min(per.get(frm, 1e9), dist), 1)
    print(f"[float-check] closest fish spine to rock/roots/sand-fall while on screen: {worst[0]:.1f} mm "
          f"(frame {worst[1]}, fish {worst[2]}, {worst[3]}); at 1950/2000/2250: {per}")
