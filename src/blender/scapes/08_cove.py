"""Scape 8: Cove.

A sheltered cove. Golden-brown rock is stacked into two uneven walls, a tall
bulky one at the back left that sends a low arm forward along the glass, and a
lower, slimmer one at the back right. They meet right of centre where a long
stone has come down off the left wall and lodged on a stub of the right one,
leaving a window under it that looks through to the lit sand behind. Pale sand
spills forward from under it as a low rippled fan, a small beach, and
the bowl of open water over it is left to the shoal. Moss has taken the joints
and a few shaded ledges, small Bucephalandra sit in the cracks, and broken
rock lies where the walls meet the sand.

The stones are not posed by hand. Base stones are bedded in the sand; every
stone above them is dropped until it touches what is already there, and the
leaning ones are then let tip about that first contact until a second (and a
third) stops them, as a real stone would come to rest. So the fallen stone
truly lies on its two supports. Coordinates are millimetres: X left to right, Y front to
back.
"""
import math
import os
import random

import bpy
import numpy as np
from mathutils import Matrix, Vector, noise
from mathutils.bvhtree import BVHTree

import assets
import fish
import materials
import plants
import shrimp
import substrate
from bl_common import MM, new_mesh_object
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "08_cove"

# A little less soft fill than the default: the pale beach throws plenty of
# light back up, and the hollow under the fallen stone has to stay in shadow.
LIGHT = dict(watts=43.0, fill=0.006, bounce=0.02)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py):
# the bowl of the cove, in front of the arch and over the beach. The cylinders
# cover the rock that stands near it; the build prints the measured stones as [cove].
SWIM = {
    "box": (48, 256, 28, 168, 112, 238),
    "floor": (62, 78),
    "obstacles": [(72, 240, 44, 172), (32, 203, 28, 132), (126, 227, 24, 130), (188, 224, 46, 148),
                  (250, 240, 34, 146), (254, 198, 28, 112), (50, 154, 32, 105), (242, 150, 28, 90)],
}

ARCH = (176.0, 222.0)          # the window's centre on plan

# name, model, piece, how the scan is stood up, (tilt, tilt, heading), x, y, how it is set:
#   ("bed", height above the sand, fraction buried)   bedded in the sand
#   ("drop", longest side, bite)                       dropped onto what is below, then pressed in `bite` mm
#   ("lean", longest side, bite[, most roll])          dropped, then let tip until it lodges
# Order matters: a stone only rests on those listed before it.
# Sized for a 30 cm cube: the highest point is under half the water depth and the
# back wall leaves the right-hand corner open.
STONES = [
    # left wall: the bulk of the cove, and its arm coming forward along the glass
    ("L_back",   "moon_rock_05",     0, (0, 0, 0),  (0, -4, 30),   62,  250, ("bed", 62, 0.24)),
    ("L_side",   "moon_rock_06",     0, (90, 0, 0), (0, 5, 80),    34,  198, ("bed", 50, 0.26)),
    ("L_fill",   "rock_moss_set_02", 4, (0, 0, 0),  (0, 0, 200),   104, 258, ("bed", 40, 0.28)),
    ("L_arm1",   "rock_moss_set_02", 4, (0, 0, 0),  (0, 4, 60),    44,  150, ("bed", 32, 0.30)),
    ("L_arm2",   "rock_moss_set_02", 1, (0, 0, 0),  (0, -5, 15),   62,  112, ("bed", 19, 0.34)),
    ("L_toe",    "rock_moss_set_02", 3, (0, 0, 0),  (0, 3, 250),   88,  86,  ("bed", 9, 0.40)),
    # the arch: a tall stub on the left, a low one on the right, and the beam fallen across them
    ("A_left",   "rock_moss_set_02", 6, (0, 90, 0), (0, 3, 20),    124, 226, ("bed", 55, 0.24)),
    ("A_right",  "moon_rock_02",     0, (0, 0, 0),  (0, -4, 15),   232, 222, ("bed", 24, 0.30)),
    ("A_lintel", "moon_rock_01",     0, (0, 0, 0),  (0, 15, 8),    184, 222, ("lean", 106, 2.5, 6.0)),
    # right wall: lower and slimmer, stopping short of the corner
    ("R_main",   "moon_rock_07",     0, (0, 90, 0), (0, 4, 200),   252, 246, ("bed", 54, 0.26)),
    ("R_side",   "rock_moss_set_02", 6, (0, 0, 0),  (0, -5, 80),   252, 200, ("bed", 32, 0.30)),
    ("R_arm",    "rock_moss_set_02", 5, (0, 0, 0),  (0, 3, 210),   244, 150, ("bed", 18, 0.36)),
    ("R_toe",    "moon_rock_04",     0, (0, 0, 0),  (0, 0, 130),   224, 118, ("bed", 8, 0.42)),
    # what has come to rest on them
    ("L_top",    "moon_rock_03",     0, (0, 0, 0),  (8, -6, 150),  72,  240, ("lean", 58, 3.0)),
    ("L_slab",   "moon_rock_04",     0, (0, 0, 0),  (62, 0, 10),   84,  204, ("lean", 62, 2.5)),
    ("A_wedge",  "rock_moss_set_02", 2, (0, 0, 0),  (0, 20, 60),   114, 238, ("lean", 46, 2.5)),
    ("R_cap",    "rock_moss_set_02", 0, (0, 0, 0),  (0, 8, 160),   250, 230, ("lean", 46, 3.0)),
    ("A_prop",   "rock_09",          0, (0, 0, 0),  (58, 0, -25),  228, 196, ("lean", 54, 2.5)),
    # strays on the beach and behind the window
    ("S_1",      "moon_rock_06",     0, (0, 0, 0),  (0, 0, 40),    122, 158, ("bed", 8, 0.40)),
    ("S_2",      "rock_moss_set_02", 2, (0, 0, 0),  (0, 0, 300),   200, 92,  ("bed", 5, 0.45)),
    ("S_3",      "moon_rock_07",     0, (0, 0, 0),  (0, 0, 75),    154, 190, ("bed", 9, 0.40)),
    ("S_back",   "moon_rock_03",     0, (0, 0, 0),  (0, 0, 20),    170, 276, ("bed", 11, 0.35)),
]

# The stone: warm golden brown, like sandstone that has yellowed in the tank.
# Per scan, the brightness range of its photographed colour that is spread from
# dark to light, chosen so the different scans come out as one kind of rock.
STONE_DARK = (0.055, 0.030, 0.013)
STONE_LIGHT = (0.60, 0.40, 0.19)
LUM = {"rock_09": (0.03, 0.32), "rock_moss_set_02": (0.06, 0.44)}
LUM_DEFAULT = (0.18, 0.72)

MOSS_POINTS = 24000            # fronds across all the rock; the scene's heaviest item
BUCE = 8                       # Bucephalandra clumps in the cracks
SHRIMP_ROCK = "L_arm2"         # the stone one shrimp grazes on


def _smooth(e0, e1, v):
    t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _np_smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _n2(x, y, scale, off=0.0):
    """Perlin noise -1..1 at a point in mm."""
    return noise.noise(Vector((x * scale + off, y * scale - off * 0.7, off * 1.3)))


def _wobble(co, f, seed):
    """Cheap smooth noise -1..1 over an array of points (mm)."""
    x, y, z = co[:, 0] * f, co[:, 1] * f, co[:, 2] * f
    return (np.sin(x + seed + 1.7 * np.sin(y * 0.83 + z * 0.6)) * np.cos(y * 1.1 + 1.3 * np.sin(z * 0.9 + x * 0.57 + seed))
            + 0.5 * np.sin(z * 1.9 + x * 1.3 - seed) * np.sin(y * 2.3 + seed * 0.7)) / 1.5


# --- the sand ----------------------------------------------------------------------

def _fan(x, y):
    """0..1: the tongue of clean pale sand that spreads from the arch mouth to the front glass."""
    t = min(1.0, max(0.0, (ARCH[1] - 18 - y) / 170.0))        # 0 at the arch mouth .. 1 at the front glass
    cx = ARCH[0] - 26.0 * t                                    # it drifts left on its way forward
    half = 24.0 + 46.0 * t
    return math.exp(-((x - cx) / half) ** 2) * _smooth(0.0, 0.18, t)


def ground(x, y):
    """Sand height in mm. It climbs gently to the back, is banked against the
    foot of each wall (more on the left), is scoured into a shallow hollow under
    the arch, and lies in front of it as a low rippled fan: the beach."""
    yy = min(max(y, 0.0), 300.0) / 290.0
    h = 62.0 + 13.0 * yy ** 1.15
    h += 12.0 * math.exp(-(((x - 56) / 60) ** 2 + ((y - 228) / 74) ** 2))
    h += 7.0 * math.exp(-(((x - 250) / 42) ** 2 + ((y - 226) / 58) ** 2))
    h -= 4.5 * math.exp(-(((x - ARCH[0]) / 30) ** 2 + ((y - ARCH[1]) / 26) ** 2))
    fan = _fan(x, y)
    h += 4.0 * fan * (1.0 - 0.45 * min(1.0, max(0.0, (ARCH[1] - 18 - y) / 170.0)))
    # ripples across the fan, bent and broken so no two crests match
    ph = y * 0.42 + 2.6 * math.sin(x * 0.045 + 0.7) + 1.4 * math.sin(x * 0.11 + y * 0.03)
    h += 0.75 * fan * math.sin(ph) * (0.6 + 0.4 * math.sin(x * 0.07 + y * 0.05 + 1.0))
    # soft swells everywhere
    h += 1.5 * math.sin(x * 0.036 + 0.6 + 0.8 * math.sin(y * 0.027)) * math.cos(y * 0.031 - 0.4)
    h += 0.6 * math.sin(x * 0.085 + y * 0.052 + 1.0)
    return h


def _ground_normal(x, y):
    e = 1.0
    gx = (ground(x + e, y) - ground(x - e, y)) / (2 * e)
    gy = (ground(x, y + e) - ground(x, y - e)) / (2 * e)
    return Vector((-gx, -gy, 1.0)).normalized()


class _Field:
    """ground() sampled on a grid, for the many thousand lookups the stones need."""

    def __init__(self, step=2.0):
        self.step = step
        self.n = int(300 / step) + 1
        self.h = np.array([[ground(i * step, j * step) for i in range(self.n)] for j in range(self.n)])

    def __call__(self, x, y):
        fx = np.clip(np.asarray(x) / self.step, 0, self.n - 1.001)
        fy = np.clip(np.asarray(y) / self.step, 0, self.n - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        h = self.h
        return h[j, i] * (1 - u) * (1 - v) + h[j, i + 1] * u * (1 - v) + h[j + 1, i] * (1 - u) * v + h[j + 1, i + 1] * u * v


def _sand_material():
    """The kit's white sand, warmer and less bright (pure white reads as snow
    under this lamp), a little darker where the painted `foot` attribute says
    debris has gathered against the rock, and faintly layered so the cut face
    at the glass shows how it settled."""
    mat = materials.white_sand("CoveSand")
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    for n in nt.nodes:
        if n.type == "HUE_SAT":
            n.inputs["Value"].default_value = 1.20
            n.inputs["Saturation"].default_value = 0.78
        elif n.type == "MIX" and n.blend_type == "MULTIPLY" and not n.inputs["B"].links:
            n.inputs["B"].default_value = (0.97, 0.90, 0.77, 1.0)
        elif n.type == "BUMP":
            n.inputs["Strength"].default_value = 0.8
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    sand = bsdf.inputs["Base Color"].links[0].from_socket
    coord = N.new("ShaderNodeTexCoord")
    lay_map = N.new("ShaderNodeMapping")
    lay_map.inputs["Scale"].default_value = (16.0, 16.0, 420.0)
    L.new(coord.outputs["Object"], lay_map.inputs["Vector"])
    lay = N.new("ShaderNodeTexNoise")
    lay.inputs["Scale"].default_value = 1.0
    lay.inputs["Detail"].default_value = 5.0
    lay.inputs["Roughness"].default_value = 0.6
    L.new(lay_map.outputs["Vector"], lay.inputs["Vector"])
    lay_v = N.new("ShaderNodeMapRange")
    lay_v.inputs["From Min"].default_value = 0.30
    lay_v.inputs["From Max"].default_value = 0.70
    lay_v.inputs["To Min"].default_value = 0.70
    lay_v.inputs["To Max"].default_value = 1.06
    L.new(lay.outputs["Fac"], lay_v.inputs["Value"])
    layered = N.new("ShaderNodeMix")
    layered.data_type = "RGBA"
    layered.blend_type = "MULTIPLY"
    layered.inputs["Factor"].default_value = 1.0
    L.new(sand, layered.inputs["A"])
    L.new(lay_v.outputs["Result"], layered.inputs["B"])
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "foot"
    brk = N.new("ShaderNodeTexNoise")
    brk.inputs["Scale"].default_value = 140.0
    brk.inputs["Detail"].default_value = 5.0
    L.new(coord.outputs["Object"], brk.inputs["Vector"])
    amt = N.new("ShaderNodeMath")
    amt.operation = "MULTIPLY"
    L.new(attr.outputs["Fac"], amt.inputs[0])
    L.new(brk.outputs["Fac"], amt.inputs[1])
    dirty = N.new("ShaderNodeMix")
    dirty.data_type = "RGBA"
    dirty.blend_type = "MULTIPLY"
    dirty.inputs["B"].default_value = (0.50, 0.42, 0.30, 1.0)
    L.new(amt.outputs[0], dirty.inputs["Factor"])
    L.new(layered.outputs["Result"], dirty.inputs["A"])
    pale = N.new("ShaderNodeAttribute")
    pale.attribute_type = "GEOMETRY"
    pale.attribute_name = "pale"
    off = N.new("ShaderNodeMapRange")
    off.inputs["To Min"].default_value = 1.0
    off.inputs["To Max"].default_value = 0.0
    L.new(pale.outputs["Fac"], off.inputs["Value"])
    dull = N.new("ShaderNodeMix")
    dull.data_type = "RGBA"
    dull.blend_type = "MULTIPLY"
    dull.inputs["B"].default_value = (0.74, 0.66, 0.53, 1.0)
    L.new(off.outputs["Result"], dull.inputs["Factor"])
    L.new(dirty.outputs["Result"], dull.inputs["A"])
    L.new(dull.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def _grit(bed, made):
    """Loose grains on the sand and down its cut face at the glass: pale, with
    some tan and a few dark ones, as in any bag of real sand. They give the
    surface grain and tiny shadows that a texture alone cannot."""
    cols = [((0.70, 0.65, 0.56), 4), ((0.62, 0.53, 0.41), 3), ((0.44, 0.34, 0.23), 2), ((0.15, 0.12, 0.10), 1)]
    grains = []
    for k, (c, count) in enumerate(cols):
        m = substrate.sand_material(f"CoveGrit_{k}", colour=c)
        for j in range(count):
            grains.append(substrate._granule(f"CoveGrit_{k}_{j}", 0.6 * MM, 70 + k * 5 + j, m))
    obj = bpy.data.objects.new("SandGrit", bed.obj.data)
    bpy.context.scene.collection.objects.link(obj)
    made.append(obj)
    scatter(obj, collection_of("CoveGritSrc", grains), name="SandGrit", density=1.7e5, weight_attr="grit", seed=17,
            scale=(0.55, 1.6), align="NORMAL", tilt=math.pi, keep_surface=False, sink=0.25 * MM)


# --- stone -------------------------------------------------------------------------

def _import_piece(model, piece=0):
    objs = assets.import_model(model)
    keep = objs[piece]
    for o in objs:
        if o is not keep:
            bpy.data.objects.remove(o, do_unlink=True)
    # glTF objects arrive with quaternion rotation, in which assets.place's tilt
    # and heading (set as Euler angles) are silently ignored
    keep.rotation_mode = "XYZ"
    return keep


def _grade(obj, model, seed, tone=1.0):
    """One kind of stone for every scan: golden brown with darker brown in the
    hollows, hardly veined, rust-stained, only a trace of algae film."""
    assets.grade_stone(obj, dark=STONE_DARK, light=tuple(c * tone for c in STONE_LIGHT),
                       lum_range=LUM.get(model, LUM_DEFAULT), vein_amount=0.04, rust=0.5, algae=0.16, bump=0.5,
                       normal_strength=2.4, seed=seed)
    # grade_stone's procedural layers are sized for a scan about 10 cm across
    lo, hi = Vector(obj.bound_box[0]), Vector(obj.bound_box[6])
    m = max(hi - lo) / 0.10
    for slot in obj.material_slots:
        for n in slot.material.node_tree.nodes:
            if n.type in {"TEX_NOISE", "TEX_WAVE"}:
                n.inputs["Scale"].default_value /= m


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
        fac.inputs["From Min"].default_value = 0.10
        fac.inputs["From Max"].default_value = 0.55
        fac.inputs["To Max"].default_value = 0.85
        L.new(attr.outputs["Fac"], fac.inputs["Value"])
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["B"].default_value = (0.022, 0.030, 0.010, 1.0)
        L.new(fac.outputs["Result"], mix.inputs["Factor"])
        L.new(src, mix.inputs["A"])
        L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


class _Rock:
    """A placed stone with its mesh in world millimetres, for contact tests."""

    def __init__(self, name, model, obj):
        self.name, self.model, self.obj = name, model, obj

    def bake(self):
        bpy.context.view_layer.update()
        me = self.obj.data
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        nr = np.empty(n * 3)
        me.vertices.foreach_get("normal", nr)
        mw = np.array(self.obj.matrix_world)
        self.co = (co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]) / MM
        nw = nr.reshape(n, 3) @ mw[:3, :3].T
        self.nr = nw / np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
        me.calc_loop_triangles()
        t = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        self.tris = t.reshape(-1, 3)
        self.bvh = BVHTree.FromPolygons(self.co.tolist(), self.tris.tolist())
        self.lo, self.hi = self.co.min(0), self.co.max(0)
        return self


def _world_points(obj, limit=2600):
    bpy.context.view_layer.update()
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    mw = np.array(obj.matrix_world)
    co = (co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]) / MM
    return co[::max(1, n // limit)].copy()


def _penetrating(pts, statics, field, bury, tol=0.4):
    """Which points are inside a placed stone, or more than `bury` mm under the sand."""
    bad = pts[:, 2] < field(pts[:, 0], pts[:, 1]) - bury - tol
    for s in statics:
        m = ~bad & np.all((pts > s.lo) & (pts < s.hi), axis=1)
        for i in np.nonzero(m)[0]:
            p = Vector(pts[i].tolist())
            loc, nrm, _, dist = s.bvh.find_nearest(p, 12.0)
            if loc is not None and dist > tol and (p - loc).dot(nrm) < -0.8 * dist:
                bad[i] = True
    return bad


def _rotate(pts, pivot, axis, ang):
    v = pts - pivot
    c, s = math.cos(ang), math.sin(ang)
    return pivot + v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1 - c)


def _rot_matrix(pivot, axis, ang):
    p = Vector((pivot * MM).tolist())
    return Matrix.Translation(p) @ Matrix.Rotation(ang, 4, Vector(axis.tolist())) @ Matrix.Translation(-p)


def _tip(pts, pivot, axis, held, statics, field, bury, max_deg, step_deg=0.5):
    """Turn the points about the line through `pivot` along `axis` until some
    part of the stone away from where it is already held touches something.
    Returns the angle turned and where it touched (None if it never did)."""
    free = np.ones(len(pts), dtype=bool)
    for q in held:
        free &= np.linalg.norm(pts - q, axis=1) > 14.0
    loose = pts[free]
    for k in range(1, int(max_deg / step_deg) + 1):
        ang = math.radians(k * step_deg)
        moved = _rotate(loose, pivot, axis, ang)
        bad = _penetrating(moved, statics, field, bury)
        if bad.any():
            return ang, moved[bad].mean(0)
    return math.radians(max_deg), None


def _settle(obj, statics, field, lean, bite, bury=5.0, max_roll=12.0):
    """Let a stone come to rest: straight down until it first touches a placed
    stone (or beds `bury` mm into the sand); a leaning stone then tips about
    that contact, its centre of mass falling, until a second contact stops it,
    and rolls about the line through the two until a third does. Finally it is
    pressed in `bite` mm, so no hairline of light shows under it."""
    pts = _world_points(obj)
    lift = 330.0 - pts[:, 2].min()
    pts[:, 2] += lift
    clear = pts[:, 2] - field(pts[:, 0], pts[:, 1]) + bury
    down = Vector((0.0, 0.0, -1.0))
    for s in statics:
        m = (pts[:, 0] > s.lo[0]) & (pts[:, 0] < s.hi[0]) & (pts[:, 1] > s.lo[1]) & (pts[:, 1] < s.hi[1])
        for i in np.nonzero(m)[0]:
            hit = s.bvh.ray_cast(Vector(pts[i].tolist()), down, 400.0)
            if hit[0] is not None and hit[3] < clear[i]:
                clear[i] = hit[3]
    fall = float(clear.min())
    first = pts[clear <= fall + 0.8].mean(0) - np.array([0.0, 0.0, fall])
    pts[:, 2] -= fall
    M = Matrix.Translation((0.0, 0.0, (lift - fall) * MM))
    note = f"first touch ({first[0]:.0f}, {first[1]:.0f}, {first[2]:.0f})"
    if lean:
        com = pts.mean(0)
        h = np.array([com[0] - first[0], com[1] - first[1], 0.0])
        if np.linalg.norm(h) > 2.0:
            axis = np.array([-h[1], h[0], 0.0]) / np.linalg.norm(h)
            ang, second = _tip(pts, first, axis, [first], statics, field, bury, 75.0)
            pts = _rotate(pts, first, axis, ang)
            M = _rot_matrix(first, axis, ang) @ M
            note += f", tipped {math.degrees(ang):.1f} deg"
            if second is not None:
                note += f" onto ({second[0]:.0f}, {second[1]:.0f}, {second[2]:.0f})"
                a2 = (second - first) / max(np.linalg.norm(second - first), 1e-6)
                lever = float(np.cross(a2, pts.mean(0) - first)[2])
                if abs(lever) > 1.5:
                    if lever > 0:
                        a2 = -a2
                    ang2, third = _tip(pts, first, a2, [first, second], statics, field, bury, max_roll)
                    M = _rot_matrix(first, a2, ang2) @ M
                    note += f", rolled {math.degrees(ang2):.1f} deg" + (
                        f" onto ({third[0]:.0f}, {third[1]:.0f}, {third[2]:.0f})" if third is not None else " (free)")
    M = Matrix.Translation((0.0, 0.0, -bite * MM)) @ M
    obj.matrix_world = M @ obj.matrix_world
    return note


def _place_stones(bed, field, rng):
    rocks = []
    for k, (name, model, piece, pre, rot, x, y, how) in enumerate(STONES):
        obj = _import_piece(model, piece)
        obj.name = name
        if how[0] == "bed":
            _, h, buried = how
            full = h / (1 - buried)
            assets.place(obj, bed.at(x, y), height_mm=full, pre_rot_deg=pre, rot_deg=rot, sink_mm=full * buried)
            lo, hi = assets.world_bounds(obj)
            dx = max(0.0, 9 * MM - lo.x) - max(0.0, hi.x - 291 * MM)       # keep clear of the glass
            dy = max(0.0, 9 * MM - lo.y) - max(0.0, hi.y - 291 * MM)
            obj.location += Vector((dx, dy, 0.0))
            note = "bedded"
        else:
            longest, bite = how[1], how[2]
            assets.place(obj, (x * MM, y * MM, 0.3), longest_mm=longest, pre_rot_deg=pre, rot_deg=rot)
            note = _settle(obj, rocks, field, how[0] == "lean", bite, max_roll=how[3] if len(how) > 3 else 12.0)
        _grade(obj, model, seed=k * 1.7 + 0.3, tone=rng.uniform(0.90, 1.08))
        rock = _Rock(name, model, obj).bake()
        rocks.append(rock)
        print(f"[cove] {name}: x {rock.lo[0]:.0f}..{rock.hi[0]:.0f} y {rock.lo[1]:.0f}..{rock.hi[1]:.0f} "
              f"z {rock.lo[2]:.0f}..{rock.hi[2]:.0f}; {note}")
    return rocks


def _joint_distance(rock, others, limit=9.0):
    """For each vertex of a stone, how far the nearest other stone is (mm, capped)."""
    d = np.full(len(rock.co), 99.0)
    for s in others:
        if s is rock or np.any(s.hi < rock.lo - limit) or np.any(s.lo > rock.hi + limit):
            continue
        m = np.all((rock.co > s.lo - limit) & (rock.co < s.hi + limit), axis=1)
        for i in np.nonzero(m)[0]:
            hit = s.bvh.find_nearest(Vector(rock.co[i].tolist()), limit)
            if hit[0] is not None and hit[3] < d[i]:
                d[i] = hit[3]
    return d


def _rock_moss(rock, joint, field, seed):
    """Where moss grows on a stone: in the joints between stones, along the
    foot where it meets the sand, and on a few flat ledges; in ragged patches,
    with most of the rock left bare."""
    co, nr = rock.co, rock.nr
    hz = co[:, 2] - field(co[:, 0], co[:, 1])
    crack = 1.0 - _np_smooth(1.0, 7.5, joint)
    foot = 1.0 - _np_smooth(2.0, 13.0, hz)
    up = _np_smooth(0.0, 0.55, nr[:, 2])
    patch = _np_smooth(-0.15, 0.35, _wobble(co, 0.06, seed))
    ledge = _np_smooth(0.82, 0.96, nr[:, 2]) * _np_smooth(0.05, 0.45, _wobble(co, 0.17, seed + 5.0))
    w = np.maximum(np.maximum(crack * (0.30 + 0.70 * up), 0.22 * foot * up), 0.45 * ledge) * (0.08 + 0.92 * patch)
    return w * (hz > -0.5)


def _weighted_area(co, tris, w):
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * MM * MM
    return float((area * w[tris].mean(axis=1)).sum())


def _rubble_sources(seed0):
    """Small broken pieces of the same stone, baked at 10 mm, for crumbs and pebbles."""
    out = []
    for k, (model, piece, pre) in enumerate((("moon_rock_03", 0, (0, 0, 0)), ("moon_rock_04", 0, (0, 0, 0)),
                                             ("moon_rock_07", 0, (0, 0, 0)), ("moon_rock_06", 0, (0, 0, 0)),
                                             ("moon_rock_01", 0, (0, 0, 0)), ("rock_moss_set_02", 2, (0, 0, 0)),
                                             ("moon_rock_02", 0, (0, 90, 0)))):
        obj = _import_piece(model, piece)
        obj.name = f"CoveRubble_{k}"
        assets.place(obj, (0.0, 0.0, 0.0), longest_mm=10.0, pre_rot_deg=pre)
        bpy.context.view_layer.update()
        me = obj.data
        me.transform(Matrix.Scale(obj.scale.x, 4))
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)                               # (the object's bound_box is stale here)
        co = co.reshape(-1, 3)
        me.transform(Matrix.Translation((-(co.min(0) + co.max(0)) / 2).tolist()))   # origin at the middle, so any face can be down
        me.update()
        obj.scale = (1.0, 1.0, 1.0)
        obj.location = (0.0, 0.0, 0.0)
        _grade(obj, model, seed=seed0 + k * 2.3, tone=random.Random(k).uniform(0.80, 1.12))
        out.append(obj)
    return out


def _moss_sources():
    """Moss in two greens, both darker than the kit's default, which glows like lawn under this lamp."""
    deep = plants.leaf_material("CoveMossDeep", base=(0.010, 0.028, 0.006), tip=(0.064, 0.165, 0.026), translucency=0.30,
                                roughness=0.55, variation=0.3)
    olive = plants.leaf_material("CoveMossOlive", base=(0.020, 0.034, 0.008), tip=(0.105, 0.165, 0.032), translucency=0.3,
                                 roughness=0.55, variation=0.3)
    return collection_of("CoveMossSrc", plants.moss_fronds(variants=6, seed=13, material=deep) +
                         plants.moss_fronds(variants=4, seed=29, length_mm=(4, 8), material=olive))


# --- Bucephalandra -------------------------------------------------------------------

def _buce_sources(variants=4, seed=11):
    """Small rhizome plants for the cracks: a short creeping stem and a rosette
    of dark, glossy, pointed-oval leaves on thin stalks, the newest leaves
    smallest. Built at real size (a clump is about 35 mm across)."""
    mat = plants.leaf_material("BuceLeaf", base=(0.010, 0.036, 0.012), tip=(0.050, 0.140, 0.036), translucency=0.22,
                               roughness=0.36, variation=0.3)
    # Under water a leaf's sheen is weak (water and leaf bend light almost alike);
    # at the kit's default a leaf turned toward the lamp burns out to a white fleck.
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Specular IOR Level"].default_value = 0.12
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        ra = rng.uniform(0, 2 * math.pi)
        rl = rng.uniform(7, 13) * MM
        rz = [(math.cos(ra) * rl * (t - 0.5), math.sin(ra) * rl * (t - 0.5), 0.9 * MM) for t in (0.0, 0.5, 1.0)]
        sv, sf, su = plants._tube_strip(rz, [1.1 * MM, 1.3 * MM, 0.9 * MM], sides=5)
        parts.append((sv, sf, [(u, 0.1 * w) for u, w in su]))
        n = rng.randint(7, 11)
        for i in range(n):
            t = i / (n - 1)
            base = (math.cos(ra) * rl * (t - 0.5), math.sin(ra) * rl * (t - 0.5), 1.4 * MM)
            az = ra + 2.4 * i + rng.uniform(-0.45, 0.45)
            young = t
            rise = rng.uniform(0.70, 1.25)
            pet = rng.uniform(5, 11) * (1 - 0.45 * young) * MM
            d = (math.cos(az) * math.cos(rise), math.sin(az) * math.cos(rise), math.sin(rise))
            tip = tuple(b + c * pet for b, c in zip(base, d))
            mid = tuple(b + c * pet * 0.5 for b, c in zip(base, d))
            sv, sf, su = plants._tube_strip([base, mid, tip], [0.34 * MM, 0.30 * MM, 0.27 * MM])
            parts.append((sv, sf, [(u, 0.05 + 0.2 * w) for u, w in su]))
            flat = rise - rng.uniform(0.35, 0.85)                       # the blade levels out from its stalk
            bd = (math.cos(az) * math.cos(flat), math.sin(az) * math.cos(flat), math.sin(flat))
            up = (-math.cos(az) * math.sin(flat), -math.sin(az) * math.sin(flat), math.cos(flat))
            length = rng.uniform(11, 17) * (1 - 0.45 * young) * MM
            width = length * rng.uniform(0.42, 0.56)
            parts.append(plants._leaf(tip, bd, up, length, width, 0.30 + 0.2 * young, 0.80 + 0.2 * young,
                                      fold=rng.uniform(0.10, 0.26), curl=rng.uniform(0.08, 0.30), belly=0.42, segs=6))
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"Buce_{v}", verts, faces, uvs, mat))
    return out


def _buce_spots(rocks, joints, field, rng, count):
    """Planting spots: points in a joint between stones or at a stone's foot,
    on a face the camera can see, well apart from each other."""
    cam = Vector((152.0, -555.0, 143.0))
    cand = []
    for rock, joint in zip(rocks, joints):
        if rock.name.startswith("S_"):
            continue
        co, nr = rock.co, rock.nr
        hz = co[:, 2] - field(co[:, 0], co[:, 1])
        ok = ((joint < 3.5) | ((hz > 1.0) & (hz < 5.0))) & (hz > 1.0) & (nr[:, 1] < 0.25) & (nr[:, 2] > -0.3) & (co[:, 1] < 262)
        for i in np.nonzero(ok)[0]:
            cand.append((rock, int(i)))
    rng.shuffle(cand)
    picked = []
    for rock, i in cand:
        p = Vector(rock.co[i].tolist())
        n = Vector(rock.nr[i].tolist())
        if any((p - q).length < 30.0 for q, _ in picked):
            continue
        to_cam = (cam - p).normalized()
        start = p + n * 1.5 + to_cam * 1.5
        if any(r.bvh.ray_cast(start, to_cam, 400.0)[0] is not None for r in rocks):
            continue
        picked.append((p, n))
        if len(picked) == count:
            break
    return picked


# --- build -------------------------------------------------------------------------

def build(ctx):
    if os.environ.get("COVE_SHEET"):
        return _sheet(ctx)
    made = []
    rng = random.Random(8)

    bed = substrate.build_bed(ground, "CoveSand")
    bed.obj.data.materials.append(_sand_material())
    made.append(bed.obj)
    field = _Field()

    rocks = _place_stones(bed, field, rng)
    made += [r.obj for r in rocks]
    print("[cove] cylinders " + ", ".join(
        f"{r.name} ({(r.lo[0] + r.hi[0]) / 2:.0f}, {(r.lo[1] + r.hi[1]) / 2:.0f}, "
        f"{max(r.hi[0] - r.lo[0], r.hi[1] - r.lo[1]) / 2 * 0.8:.0f}, {r.hi[2]:.0f})" for r in rocks))

    # how far each point of the sand is from rock, and whether it lies under one
    n_all = len(bed.obj.data.vertices)
    near = np.full(len(bed.top_xy_mm), 40.0)
    under = np.zeros(len(bed.top_xy_mm), dtype=bool)
    top = np.array([(x, y, ground(x, y)) for x, y in bed.top_xy_mm])
    for r in rocks:
        m = np.all((top > r.lo - 40.0) & (top < r.hi + 40.0), axis=1)
        for i in np.nonzero(m)[0]:
            p = Vector(top[i].tolist())
            loc, nrm, _, dist = r.bvh.find_nearest(p, 40.0)
            if loc is None:
                continue
            if (p - loc).dot(nrm) < -0.8 * dist and dist > 0.3:
                under[i] = True
            near[i] = min(near[i], dist)
    wall = np.minimum.reduce([top[:, 0] - 5, 295 - top[:, 0], top[:, 1] - 5, 295 - top[:, 1]])
    open_sand = (~under) * _np_smooth(6.0, 12.0, wall)
    pad = [0.0] * (n_all - len(top))

    def paint(name, values, skirt=0.0):
        set_point_attribute(bed.obj, name, list(values) + [skirt] * len(pad))

    pa = np.array([_n2(x, y, 0.07, 5.0) for x, y in bed.top_xy_mm])
    pb = np.array([_n2(x, y, 0.045, 3.0) for x, y in bed.top_xy_mm])
    pc = np.array([_n2(x, y, 0.09, 12.0) for x, y in bed.top_xy_mm])
    fan = np.array([_fan(x, y) for x, y in bed.top_xy_mm])
    paint("pale", np.clip(_np_smooth(0.10, 0.55, fan + 0.12 * pb) + 0.25 * _np_smooth(0.0, 0.5, pc), 0.0, 1.0), skirt=1.0)
    paint("foot", np.exp(-(near / 13.0) ** 2) * (0.5 + 0.5 * _np_smooth(-0.4, 0.3, pb)))
    paint("grit", 0.55 + 0.45 * _np_smooth(-0.4, 0.4, pa), skirt=0.6)
    paint("crumb", np.maximum(np.exp(-(near / 7.0) ** 2) * (0.35 + 0.65 * _np_smooth(-0.3, 0.3, pa)),
                              0.10 * np.exp(-(near / 22.0) ** 2)) * open_sand)
    paint("pebble", np.maximum(np.exp(-((near - 5.0) / 10.0) ** 2) * _np_smooth(-0.15, 0.35, pb),
                               0.05 * _np_smooth(0.25, 0.5, pc)) * open_sand)
    _grit(bed, made)
    made += substrate.add_cut_face(
        bed, substrate.sand_mix("CoveFace", pale=(0.72, 0.67, 0.57), tan=(0.62, 0.53, 0.41), brown=(0.44, 0.34, 0.23)),
        size_mm=1.2, seed=8, tone=(1.3, 0.6), room=0.2, vary=0.35)

    rubble = collection_of("CoveRubbleSrc", _rubble_sources(40.0))
    scatter(bed.obj, rubble, name="Crumbs", min_distance=3.0 * MM, weight_attr="crumb", seed=51,
            scale=(0.16, 0.50), align="NORMAL", tilt=math.pi, sink=0.5 * MM)
    scatter(bed.obj, rubble, name="Pebbles", min_distance=9.0 * MM, weight_attr="pebble", seed=52,
            scale=(0.55, 1.50), align="NORMAL", tilt=math.pi, sink=2.0 * MM)

    # moss in the joints, along the feet and on a few ledges: one carrier mesh for all the rock
    joints, weights = [], []
    cv, cf, cw = [], [], []
    off = 0
    for k, r in enumerate(rocks):
        joint = _joint_distance(r, rocks)
        w = _rock_moss(r, joint, field, seed=k * 3.1)
        if r.name.startswith("S_"):
            w = w * 0.4
        set_point_attribute(r.obj, "moss", w)
        _moss_floor(r.obj)
        joints.append(joint)
        weights.append(w)
        cv.append(r.co * MM)
        cf.append(r.tris + off)
        cw.append(w)
        off += len(r.co)
    area = sum(_weighted_area(r.co, r.tris, w) for r, w in zip(rocks, weights))
    dens = min(1.1e6, MOSS_POINTS / max(area, 1e-9))
    print(f"[cove] moss area {area * 1e4:.1f} cm2 -> density {dens:.0f}/m2")
    carrier = new_mesh_object("CoveMossCarrier", np.concatenate(cv).tolist(), np.concatenate(cf).tolist())
    set_point_attribute(carrier, "moss", np.concatenate(cw))
    made.append(carrier)
    scatter(carrier, _moss_sources(), name="CoveMoss", density=dens, weight_attr="moss", seed=80,
            scale=(0.45, 1.0), align="NORMAL", tilt=0.55, keep_surface=False, sway=(0.04, 30.0, 0.3))

    # Bucephalandra in the cracks
    sources = _buce_sources()
    plants_made = []
    for k, (p, n) in enumerate(_buce_spots(rocks, joints, field, random.Random(31), BUCE)):
        src = sources[k % len(sources)]
        obj = bpy.data.objects.new(f"BucePlant_{k}", src.data)
        bpy.context.scene.collection.objects.link(obj)
        axis = (n * 0.55 + Vector((0.0, -0.25, 0.75))).normalized()
        rot = Vector((0.0, 0.0, 1.0)).rotation_difference(axis).to_matrix().to_4x4()
        size = rng.uniform(0.50, 0.82)
        base = Matrix.Translation(p * MM - n * 0.8 * MM) @ rot @ Matrix.Rotation(rng.uniform(0, 6.283), 4, "Z") @ Matrix.Scale(size, 4)
        obj.matrix_world = base
        plants_made.append((obj, base, rng.uniform(0, 6.283)))
        made.append(obj)
        print(f"[cove] buce {k} at ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) size {size:.2f}")

    # two red shrimp: one grazing its way across a low stone of the left arm, one out on the beach
    perch_rock = next(r for r in rocks if r.name == SHRIMP_ROCK)
    w = weights[rocks.index(perch_rock)]
    hz = perch_rock.co[:, 2] - field(perch_rock.co[:, 0], perch_rock.co[:, 1])
    cand = np.nonzero((perch_rock.nr[:, 2] > 0.8) & (w < 0.2) & (hz > 6.0))[0]
    if len(cand) == 0:
        cand = np.nonzero(perch_rock.nr[:, 2] > 0.6)[0]
    best = cand[np.argmax(perch_rock.co[cand, 2] - 0.25 * perch_rock.co[cand, 1])]
    rock_shrimp = shrimp.Shrimp("Shrimp_rock", size=0.9)
    sand_shrimp = shrimp.Shrimp("Shrimp_sand", size=1.0)
    made += [rock_shrimp.obj, sand_shrimp.obj]

    school = fish.School()
    made += school.objects

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["cove"] = dict(rock_shrimp=rock_shrimp, sand_shrimp=sand_shrimp, perch=perch_rock.co[best].copy(),
                       perch_up=(Vector(perch_rock.nr[best].tolist()) + Vector((0.0, 0.0, 1.0))).normalized(),
                       perch_bvh=perch_rock.bvh, buce=plants_made)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    if ctx.get("school") is None:
        return
    ctx["school"].pose(frame)
    c = ctx["cove"]
    t = frame / 30.0
    t0 = 2850 / 30.0                                   # when the scape comes on screen

    # the rock shrimp creeps along its ledge, kept on the stone's surface
    s = c["rock_shrimp"]
    w = 0.2
    along = (t - t0 - 3.0) * shrimp.Shrimp.WALK_SPEED * w * s.size / MM
    p = Vector(c["perch"].tolist()) + Vector((along, -0.3 * along, 3.0))
    loc = c["perch_bvh"].find_nearest(p, 30.0)[0]
    if loc is not None:
        up = c["perch_up"]
        s.pose(loc * MM + up * 0.2 * MM, yaw=-0.3 + 0.12 * math.sin(t / 2.9), t=t, up=tuple(up), walk=w)

    # the other picks its way across the beach toward the left, a millimetre a second
    s = c["sand_shrimp"]
    w = 0.35
    x = 206.0 - (t - t0) * shrimp.Shrimp.WALK_SPEED * w * s.size / MM
    y = 58.0 + 3.0 * math.sin(t * 0.3)
    s.pose(Vector(ctx["bed"].at(x, y)), yaw=math.pi + 0.05 * math.cos(t * 0.3), t=t, up=tuple(_ground_normal(x, y)), walk=w)

    # the Bucephalandra nod a little in the current
    for obj, base, ph in c["buce"]:
        obj.matrix_world = base @ Matrix.Rotation(0.035 * math.sin(t * 0.9 + ph), 4, "X") @ Matrix.Rotation(
            0.025 * math.sin(t * 0.63 + 1.7 * ph), 4, "Y")


# --- stone sheet (COVE_SHEET=a) -------------------------------------------------------

SHEET = [("moon_rock_01", 0), ("moon_rock_02", 0), ("moon_rock_03", 0), ("moon_rock_04", 0),
         ("moon_rock_05", 0), ("moon_rock_06", 0), ("moon_rock_07", 0), ("boulder_01", 0),
         ("rock_09", 0), ("namaqualand_boulder_02", 0), ("namaqualand_boulder_04", 0), ("rock_moss_set_02", 0),
         ("rock_moss_set_02", 1), ("rock_moss_set_02", 2), ("rock_moss_set_02", 3), ("rock_moss_set_02", 4),
         ("rock_moss_set_02", 5), ("rock_moss_set_02", 6)]


def _sheet(ctx):
    """Candidate scans hung in the water in a grid, graded as in the scape, to
    choose which stands which way up. COVE_SHEET_ROT="rx,ry,rz" turns them all."""
    made = []
    pre = tuple(float(v) for v in os.environ.get("COVE_SHEET_ROT", "0,0,0").split(","))
    for k, (model, piece) in enumerate(SHEET):
        obj = _import_piece(model, piece)
        cx, cz = 45 + 70 * (k % 4), 232 - 46 * (k // 4)
        assets.place(obj, (cx * MM, 150 * MM, cz * MM), longest_mm=56, pre_rot_deg=pre)
        lo, hi = assets.world_bounds(obj)
        obj.location.z += cz * MM - (lo.z + hi.z) / 2
        d = hi - lo
        print(f"[sheet] {k} {model}#{piece}: verts {len(obj.data.vertices)} dims x {d.x / MM:.0f} y {d.y / MM:.0f} z {d.z / MM:.0f}")
        _grade(obj, model, seed=k * 1.7)
        made.append(obj)
    assets.mark_underwater(made, ctx)
    ctx["school"] = None
    return ctx
