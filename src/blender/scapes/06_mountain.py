"""Scape 6: Mountains.

A mountain range in miniature. A ridge of jagged, warm tan stone climbs from the
front left to a summit a third of the way across and two thirds of the way
back; a lower, shorter ridge answers it on the right. Between them a valley of
pale sand runs from the front glass, wide as a river plain, and winds back
until it is lost in a pass between small far peaks. Everything shrinks with
distance (stones, rubble, plants), so the valley seems to recede for miles.
The lower slopes carry moss and a fine carpet like alpine meadow; above the
tree line the rock is bare. The water over the valley and above the ridges is
left open for the shoal. Coordinates are millimetres: X left to right, Y front
to back.
"""
import math
import os
import random

import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

import assets
import fish
import materials
import plants
import shrimp
import substrate
from bl_common import MM
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "06_mountain"

# A little less soft fill than the default: the pale valley throws plenty of
# light back up, and the ridges need their shaded faces to read as relief.
LIGHT = dict(fill=0.012, bounce=0.035)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py):
# the open water over the valley and above the ridges. Only the stones that
# reach up into it are listed.
SWIM = {
    "box": (40, 262, 28, 200, 136, 238),
    "floor": (62, 100),
    "obstacles": [(104, 212, 32, 197), (70, 178, 38, 162), (132, 238, 20, 172), (42, 240, 34, 167),
                  (46, 130, 26, 123), (240, 172, 40, 149), (266, 236, 18, 147)],
}

# The valley's centre line as (y, x): it enters right of centre, bends left
# round the right-hand foothills, then right round a spur of the main ridge,
# and closes in a pass at the back.
PATH = [(-40, 172), (0, 172), (55, 168), (110, 150), (170, 184), (225, 196), (262, 182), (300, 172), (340, 166)]

# name, model, x, y, height above the bed, how the scan is stood up,
# (lean, lean, heading), fraction buried. One craggy scan (boulder_01) serves
# for every stone, as one kind of rock does in a real layout; stood on five
# different faces and turned, no two show the camera the same outline. Its
# underside is smooth, so each pose keeps that face down or turned to the back:
# (90,0,0) hides it at headings near 0, (-90,0,0) near 180, (0,90,0) near 270
# for stones left of the camera and (0,-90,0) near 90 for stones right of it.
STONES = [
    # the main ridge: foothills at the front left, up to the summit, and on behind it
    ("L_summit",   "boulder_01",   104, 212, 78, (90, 0, 0),   (0, -4, 20),   0.20),
    ("L_shoulder", "boulder_01",   70,  178, 50, (0, 90, 0),   (0, 5, 324),   0.25),
    ("L_back",     "boulder_01",   132, 238, 46, (-90, 0, 0),  (0, -5, 140),  0.25),
    ("L_corner",   "boulder_01",   42,  240, 44, (0, -90, 0),  (4, 0, 60),    0.25),
    ("L_spur",     "boulder_01",   130, 160, 24, (0, 0, 0),    (0, -6, 200),  0.30),
    ("L_foot",     "boulder_01",   46,  130, 30, (0, -90, 0),  (0, 5, 150),   0.30),
    ("L_front",    "boulder_01",   62,  84,  21, (0, 0, 0),    (0, -5, 70),   0.34),
    ("L_toe",      "boulder_01",   28,  52,  12, (0, 90, 0),   (0, 4, 300),   0.38),
    ("L_far",      "boulder_01",   152, 268, 26, (90, 0, 0),   (0, 0, 330),   0.28),
    # the lower ridge on the right
    ("R_peak",     "boulder_01",   240, 172, 55, (0, -90, 0),  (0, 4, 40),    0.25),
    ("R_mid",      "boulder_01",   262, 124, 28, (-90, 0, 0),  (0, -6, 160),  0.30),
    ("R_spur",     "boulder_01",   210, 110, 19, (0, 0, 0),    (5, 0, 120),   0.32),
    ("R_back",     "boulder_01",   222, 228, 36, (90, 0, 0),   (5, 0, 320),   0.28),
    ("R_corner",   "boulder_01",   266, 236, 40, (-90, 0, 0),  (0, 5, 150),   0.28),
    ("R_front",    "boulder_01",   250, 78,  15, (0, -90, 0),  (0, 4, 60),    0.36),
    ("R_toe",      "boulder_01",   272, 44,  9,  (0, 0, 0),    (0, 0, 220),   0.40),
    ("R_far",      "boulder_01",   212, 272, 22, (0, 90, 0),   (0, 0, 200),   0.30),
    ("Far_pass",   "boulder_01",   184, 287, 18, (-90, 0, 0),  (0, 0, 130),   0.30),
]

# The stone: warm tan to red-brown, like dragon stone (ohko). Per scan, the
# brightness range of its photographed colour that is spread from dark to light.
STONE_DARK = (0.070, 0.034, 0.016)
STONE_LIGHT = (0.66, 0.43, 0.24)
LUM = {"boulder_01": (0.05, 0.42), "rock_09": (0.03, 0.30), "rock_moss_set_02": (0.10, 0.62)}
HAZE_DARK = (0.085, 0.080, 0.074)       # what distance does to the stone: paler, greyer, flatter
HAZE_LIGHT = (0.52, 0.49, 0.44)

MOSS_POINTS = 46000           # fronds across all stones; the scene's heaviest item


def _smooth(e0, e1, v):
    t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _np_smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _n2(x, y, scale, off=0.0):
    """Perlin noise -1..1 at a point in mm."""
    return noise.noise(Vector((x * scale + off, y * scale - off * 0.7, off * 1.3)))


# --- the valley ------------------------------------------------------------------

def path_x(y):
    """Centre of the valley at depth y: Catmull-Rom through PATH, so the curve has no kinks."""
    for k in range(1, len(PATH) - 2):
        y1, y2 = PATH[k][0], PATH[k + 1][0]
        if y <= y2 or k == len(PATH) - 3:
            p0, p1, p2, p3 = (PATH[k + i][1] for i in (-1, 0, 1, 2))
            t = max(0.0, min(1.0, (y - y1) / (y2 - y1)))
            return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
    return PATH[-1][1]


def path_half(y):
    """Half-width: a broad plain at the glass, a thread at the pass. It narrows
    far faster than real perspective would, which is what stretches the distance."""
    w = 66.0 * (1 - 0.88 * min(1.0, max(0.0, y) / 275) ** 0.75)
    return w * (1 + 0.09 * math.sin(y * 0.067 + 0.6) + 0.05 * math.sin(y * 0.171 + 2.1))


def _edge_noise(x, y):
    """In-and-out along the valley's edge, so it is a shoreline and not a ruled
    line; finer toward the back, like everything else."""
    k = 1.0 - 0.6 * min(1.0, max(0.0, y) / 280)
    return k * (4.5 * math.sin(x * 0.11 + 1.3 * math.sin(y * 0.083)) * math.cos(y * 0.097 + 0.4)
                + 2.2 * math.sin(x * 0.31 + y * 0.27) + 1.0 * math.sin(x * 0.9 - y * 0.7))


def _front_shore(x):
    """How far back from the front glass the sand reaches at x: the valley opens
    onto a plain that runs the whole width of the glass, so the bed's cut face
    is one band of sand and the dark soil starts behind it."""
    return 30.0 + 9.0 * math.sin(x * 0.027 + 0.9) + 5.0 * math.sin(x * 0.083 + 2.0) + 2.0 * math.sin(x * 0.23)


def path_dist(x, y):
    """Signed distance in mm from the edge of the sand: negative on it."""
    e = _edge_noise(x, y)
    dv = abs(x - path_x(y)) - path_half(y) + e
    df = (y - _front_shore(x)) * 0.9 + 0.5 * e
    k = 16.0                                   # the plain and the valley join in a curve, not a corner
    h = max(0.0, min(1.0, 0.5 + 0.5 * (df - dv) / k))
    return df + (dv - df) * h - k * h * (1 - h)


def path_fac(x, y, soft=2.5):
    """1 on the sand, 0 on the slopes; the far end fades out where the valley is lost in the pass."""
    return (1.0 - _smooth(-soft, soft, path_dist(x, y))) * (1.0 - _smooth(262, 284, y))


def ground(x, y):
    """Bed height in mm. The valley floor climbs gently from the front glass to
    the pass; the slopes bank up either side of it, higher on the left and
    higher toward the back, and heap a little under every stone."""
    yy = min(max(y, 0.0), 300.0) / 290.0
    d = path_dist(x, y)
    h = 60.0 + 30.0 * yy ** 1.15
    amp = (5.0 + 37.0 * yy ** 1.2) if x < path_x(y) else (4.0 + 23.0 * yy ** 1.2)
    bank = _smooth(-5.0, 62.0 + 16.0 * math.sin(x * 0.031 + y * 0.023 + 0.7), d)
    h += amp * bank
    for name, _, sx, sy, sh, *_ in STONES:
        r = 7.0 + 0.42 * sh
        h += (1.5 + 0.045 * sh) * math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (r * r))
    # the sand lies in soft swells; the slopes are lumpier
    h += (1.0 - bank) * (1.3 * math.sin(x * 0.047 + 0.8 + 0.9 * math.sin(y * 0.031)) * math.cos(y * 0.043 - 0.3)
                         + 0.5 * math.sin(x * 0.13 + y * 0.09 + 1.0))
    h += bank * (2.0 * math.sin(x * 0.061 + 1.3) * math.cos(y * 0.052) + 1.0 * math.sin(x * 0.17) * math.sin(y * 0.13 + 0.5))
    # against the front glass the sand is never level: it shows as a wavering line
    h += math.exp(-max(y, 0.0) / 38.0) * (2.6 * math.sin(x * 0.036 + 2.2) + 1.3 * math.sin(x * 0.097 + 0.4) + 0.6 * math.sin(x * 0.29))
    return h


def _ground_normal(x, y):
    e = 1.0
    gx = (ground(x + e, y) - ground(x - e, y)) / (2 * e)
    gy = (ground(x, y + e) - ground(x, y - e)) / (2 * e)
    return Vector((-gx, -gy, 1.0)).normalized()


class _Field:
    """ground() sampled on a grid, for the many thousand lookups the stones' moss needs."""

    def __init__(self, step=2.0):
        self.step = step
        self.n = int(300 / step) + 1
        self.h = np.array([[ground(i * step, j * step) for i in range(self.n)] for j in range(self.n)])

    def __call__(self, x, y):
        fx = np.clip(x / self.step, 0, self.n - 1.001)
        fy = np.clip(y / self.step, 0, self.n - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        h = self.h
        return (h[j, i] * (1 - u) * (1 - v) + h[j, i + 1] * u * (1 - v) + h[j + 1, i] * (1 - u) * v + h[j + 1, i + 1] * u * v)


def _far(y):
    """0 at the front glass .. 1 at the back: how far away a thing pretends to be."""
    return min(1.0, max(0.0, y / 290.0))


# --- stone -----------------------------------------------------------------------

_SRC = {}


def _source(model, ratio=1.0, piece=0):
    """A scan, imported once and kept out of the scene; every stone is a copy of
    it, so its textures are loaded once. ratio < 1 is a thinned-out version for
    the small and far stones, which do not need a hundred thousand faces."""
    key = (model, ratio, piece)
    if key not in _SRC:
        if ratio >= 1.0:
            objs = assets.import_model(model)
            src = objs[piece]
            for o in objs:
                if o is not src:
                    bpy.data.objects.remove(o, do_unlink=True)
        else:
            src = _copy(_source(model, 1.0, piece))
            src.modifiers.new("Thin", "DECIMATE").ratio = ratio
            dg = bpy.context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
            old = src.data
            src.modifiers.clear()
            src.data = me
            bpy.data.meshes.remove(old)
        for c in list(src.users_collection):
            c.objects.unlink(src)
        _SRC[key] = src
    return _SRC[key]


def _copy(src):
    obj = src.copy()
    obj.data = src.data.copy()
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj


def _stone(model, height_mm=100.0):
    """A fresh copy of a scan, as detailed as a stone of this height needs."""
    ratio = 1.0
    if model == "boulder_01":
        ratio = 1.0 if height_mm >= 55 else 0.4 if height_mm >= 30 else 0.15
    return _copy(_source(model, ratio))


def _bake(obj):
    """Move the object's placement into its mesh, so the mesh is in world metres:
    scatters on it are then measured in real millimetres."""
    bpy.context.view_layer.update()
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    obj.data.update()
    bpy.context.view_layer.update()


def _mixc(a, b, t):
    return tuple(p + (q - p) * t for p, q in zip(a, b))


def _grade(obj, model, seed, far=0.0, tone=1.0):
    """One kind of stone for every scan: warm tan with darker red-brown in the
    hollows, hardly veined, rust-stained, a thin algae film on faces that look
    up at the lamp. Far stones are a little paler and greyer, as haze makes them."""
    dark = _mixc(STONE_DARK, HAZE_DARK, 0.55 * far)
    light = tuple(c * tone for c in _mixc(STONE_LIGHT, HAZE_LIGHT, 0.45 * far))
    assets.grade_stone(obj, dark=dark, light=light, lum_range=LUM.get(model, (0.16, 0.58)), vein_amount=0.05,
                       rust=0.55, algae=0.30, bump=0.5, normal_strength=2.4, seed=seed)
    # grade_stone's procedural layers are sized for a scan about 10 cm across;
    # the bigger scans are metres across in their own space, so bring them into line
    lo, hi = Vector(obj.bound_box[0]), Vector(obj.bound_box[6])
    m = max(hi - lo) / 0.10
    for slot in obj.material_slots:
        for n in slot.material.node_tree.nodes:
            if n.type in {"TEX_NOISE", "TEX_WAVE"}:
                n.inputs["Scale"].default_value /= m
            if n.type == "TEX_WAVE":
                # this stone is not bedded like Seiryu: faint, broken banding only
                n.inputs["Distortion"].default_value = 14.0
                rng_node = n.outputs["Fac"].links[0].to_node
                rng_node.inputs["To Min"].default_value = 0.86
                rng_node.inputs["To Max"].default_value = 1.10


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
        fac.inputs["To Max"].default_value = 0.88
        L.new(attr.outputs["Fac"], fac.inputs["Value"])
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["B"].default_value = (0.020, 0.030, 0.010, 1.0)
        L.new(fac.outputs["Result"], mix.inputs["Factor"])
        L.new(src, mix.inputs["A"])
        L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


def _mesh_arrays(me):
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    nr = np.empty(n * 3)
    me.vertices.foreach_get("normal", nr)
    return co.reshape(n, 3) / MM, nr.reshape(n, 3)


def _weighted_area(me, w):
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    idx = np.empty(n * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", idx)
    area = np.empty(n)
    me.loop_triangles.foreach_get("area", area)
    return float((area * w[idx.reshape(n, 3)].mean(axis=1)).sum())


def _set_mesh_attribute(me, name, values):
    attr = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))
    me.update()


def _wobble(co, f, seed):
    """Cheap smooth noise -1..1 over an array of points (mm)."""
    x, y, z = co[:, 0] * f, co[:, 1] * f, co[:, 2] * f
    return (np.sin(x + seed + 1.7 * np.sin(y * 0.83 + z * 0.6)) * np.cos(y * 1.1 + 1.3 * np.sin(z * 0.9 + x * 0.57 + seed))
            + 0.5 * np.sin(z * 1.9 + x * 1.3 - seed) * np.sin(y * 2.3 + seed * 0.7)) / 1.5


def _stone_moss(co, nr, field, height, seed):
    """Where moss grows on a stone: on what faces up, low down. It thins out
    toward a ragged tree line about two fifths of the way up; above that only
    the odd flat ledge holds a tuft, and the summit is bare."""
    hz = co[:, 2] - field(co[:, 0], co[:, 1])                      # mm above the bed just here
    a = _wobble(co, 0.09, seed)
    b = _wobble(co, 0.31, seed + 5.0)
    line = height * (0.40 + 0.14 * a) + 3.0
    low = 1.0 - _np_smooth(line * 0.55, line, hz)
    up = _np_smooth(0.20 + 0.15 * b, 0.62, nr[:, 2])
    patch = 0.25 + 0.75 * _np_smooth(-0.45, 0.25, _wobble(co, 0.055, seed + 9.0))
    w = low * up * patch
    ledge = _np_smooth(0.86, 0.97, nr[:, 2]) * (1.0 - _np_smooth(line, line * 1.7, hz)) * _np_smooth(0.15, 0.5, b)
    return np.maximum(w, 0.5 * ledge)


# --- the bed ---------------------------------------------------------------------

def _paint_all(bed, name, fn):
    """Like Bed.set_weight but over every vertex, the skirt included, so the cut
    face at the glass can be painted too. fn takes (x, y, depth below the top) in mm."""
    co = [v.co for v in bed.obj.data.vertices]
    set_point_attribute(bed.obj, name, [fn(c.x / MM, c.y / MM, ground(c.x / MM, c.y / MM) - c.z / MM) for c in co])


def _floor_material():
    """One material for the whole bed: warm pale sand where the painted `path`
    attribute is 1, dark soil elsewhere, with a noisy, uneven join. In the cut
    face at the glass the sand shows faint settled layers, as poured sand does."""
    mat = materials.white_sand("MountainFloor")
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    for n in nt.nodes:
        # pure white reads as snow or plaster under this lamp; real sand is ivory to beige
        if n.type == "HUE_SAT":
            n.inputs["Value"].default_value = 1.12
            n.inputs["Saturation"].default_value = 0.75
        elif n.type == "MIX" and n.blend_type == "MULTIPLY" and not n.inputs["B"].links:
            n.inputs["B"].default_value = (0.97, 0.89, 0.76, 1.0)
        elif n.type == "BUMP":
            n.inputs["Strength"].default_value = 0.8
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    sand = bsdf.inputs["Base Color"].links[0].from_socket
    coord = N.new("ShaderNodeTexCoord")

    # settled layers and damp patches in the sand
    lay_map = N.new("ShaderNodeMapping")
    lay_map.inputs["Scale"].default_value = (18.0, 18.0, 420.0)
    L.new(coord.outputs["Object"], lay_map.inputs["Vector"])
    lay = N.new("ShaderNodeTexNoise")
    lay.inputs["Scale"].default_value = 1.0
    lay.inputs["Detail"].default_value = 5.0
    lay.inputs["Roughness"].default_value = 0.6
    L.new(lay_map.outputs["Vector"], lay.inputs["Vector"])
    lay_v = N.new("ShaderNodeMapRange")
    lay_v.inputs["From Min"].default_value = 0.30
    lay_v.inputs["From Max"].default_value = 0.70
    lay_v.inputs["To Min"].default_value = 0.72
    lay_v.inputs["To Max"].default_value = 1.06
    L.new(lay.outputs["Fac"], lay_v.inputs["Value"])
    sand_l = N.new("ShaderNodeMix")
    sand_l.data_type = "RGBA"
    sand_l.blend_type = "MULTIPLY"
    sand_l.inputs["Factor"].default_value = 1.0
    L.new(sand, sand_l.inputs["A"])
    L.new(lay_v.outputs["Result"], sand_l.inputs["B"])

    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "path"
    brk = N.new("ShaderNodeTexNoise")
    brk.inputs["Scale"].default_value = 170.0
    brk.inputs["Detail"].default_value = 6.0
    brk.inputs["Roughness"].default_value = 0.65
    L.new(coord.outputs["Object"], brk.inputs["Vector"])
    add = N.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 0.7
    L.new(brk.outputs["Fac"], add.inputs[0])
    L.new(attr.outputs["Fac"], add.inputs[2])
    fac = N.new("ShaderNodeMapRange")
    fac.inputs["From Min"].default_value = 0.62
    fac.inputs["From Max"].default_value = 0.92
    L.new(add.outputs[0], fac.inputs["Value"])

    # soil between the granules: near black, a little uneven
    sn = N.new("ShaderNodeTexNoise")
    sn.inputs["Scale"].default_value = 420.0
    sn.inputs["Detail"].default_value = 3.0
    L.new(coord.outputs["Object"], sn.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.012, 0.009, 0.006, 1)
    ramp.color_ramp.elements[1].color = (0.052, 0.038, 0.025, 1)
    L.new(sn.outputs["Fac"], ramp.inputs["Fac"])

    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L.new(fac.outputs["Result"], mix.inputs["Factor"])
    L.new(ramp.outputs["Color"], mix.inputs["A"])
    L.new(sand_l.outputs["Result"], mix.inputs["B"])
    L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    rough = N.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.88
    rough.inputs["To Max"].default_value = 0.58
    L.new(fac.outputs["Result"], rough.inputs["Value"])
    L.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def _granules(bed, material, size_mm, weight, seed, name, spacing=0.72):
    grains = [substrate._granule(f"{name}_{k}", size_mm * MM / 2, seed * 10 + k, material) for k in range(4)]
    return scatter(bed.obj, collection_of(f"{name}_src", grains), name=name, min_distance=size_mm * MM * spacing,
                   seed=seed, weight_attr=weight, scale=(0.55, 1.3), align="NORMAL", tilt=math.pi, sink=size_mm * MM * 0.2)


def _front_glow(ctx, objs, watts):
    """Room light on the bed's cut face. The lamp is overhead, so grains pressed
    against the front glass catch it only on their upper sides and the face
    reads far darker than the same sand does on top; in the filmed room the
    glass front is lit softly from the camera's side. A low, wide source in
    front of the tank, linked to the face grains alone, puts that back."""
    col = bpy.data.collections.new("MtCutFace")
    for o in objs:
        col.objects.link(o)
    ld = bpy.data.lights.new("MtFrontGlow", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = 0.34, 0.10
    ld.energy = watts
    ld.color = (1.0, 0.97, 0.92)
    ob = bpy.data.objects.new("MtFrontGlow", ld)
    ctx["scene"].collection.objects.link(ob)
    ob.location = (0.15, -0.28, 0.11)
    aim = Vector((0.15, 0.0, 0.045)) - ob.location
    ob.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    ob.visible_camera = False
    ob.visible_glossy = False
    ob.visible_transmission = False
    ob.light_linking.receiver_collection = col
    ob.light_linking.blocker_collection = ctx["underwater"]
    return ob


def _loose(bed, made, name, source, weight, density, scale, seed, sink_mm):
    """Grains strewn at random on a copy of the bed. An even spacing of equal
    grains reads as a perforated sheet; real soil and sand lie anyhow, in heaps."""
    obj = bpy.data.objects.new(name, bed.obj.data)
    bpy.context.scene.collection.objects.link(obj)
    made.append(obj)
    scatter(obj, source, name=name, density=density, weight_attr=weight, seed=seed,
            scale=scale, align="NORMAL", tilt=math.pi, keep_surface=False, sink=sink_mm * MM)
    return obj


def _sand_grains(name, cols):
    """A collection of sand grains in the given (colour, how many) mix."""
    grains = []
    for k, (c, count) in enumerate(cols):
        m = substrate.sand_material(f"{name}_{k}", colour=c)
        for j in range(count):
            grains.append(substrate._granule(f"{name}_{k}_{j}", 0.6 * MM, 70 + k * 5 + j, m))
    return collection_of(f"{name}_src", grains)


def _rubble_sources(seed0):
    """Small broken pieces of the same stone, baked at 10 mm, for scree."""
    out = []
    rng = random.Random(3)
    for k, (model, ratio, pre) in enumerate((("boulder_01", 0.03, (0, 0, 0)), ("boulder_01", 0.03, (90, 0, 0)),
                                             ("boulder_01", 0.03, (0, 90, 0)), ("boulder_01", 0.03, (180, 0, 0)),
                                             ("moon_rock_03", 0.25, (0, 0, 0)), ("moon_rock_07", 0.25, (0, 0, 0)))):
        obj = _copy(_source(model, ratio))
        obj.name = f"Rubble_{k}"
        assets.place(obj, (0.0, 0.0, 0.0), longest_mm=10.0, pre_rot_deg=pre)
        _bake(obj)
        lo, hi = Vector(obj.bound_box[0]), Vector(obj.bound_box[6])
        obj.data.transform(Matrix.Translation(-(lo + hi) / 2))          # origin at the middle, so any face can be down
        _grade(obj, model, seed=seed0 + k * 2.3, tone=rng.uniform(0.85, 1.1))
        out.append(obj)
    return out


def _moss_sources():
    """Moss in two greens, both darker than the kit's default, which glows like lawn under this lamp."""
    deep = plants.leaf_material("MtMossDeep", base=(0.010, 0.028, 0.006), tip=(0.066, 0.170, 0.026), translucency=0.30,
                                roughness=0.55, variation=0.3)
    olive = plants.leaf_material("MtMossOlive", base=(0.020, 0.034, 0.008), tip=(0.110, 0.170, 0.032), translucency=0.3,
                                 roughness=0.55, variation=0.3)
    return collection_of("MtMossSrc", plants.moss_fronds(variants=6, seed=13, material=deep) +
                         plants.moss_fronds(variants=4, seed=29, length_mm=(4, 8), material=olive))


BANDS = (("near", lambda y: 1.0 - _smooth(80, 125, y)),
         ("mid", lambda y: _smooth(80, 125, y) * (1.0 - _smooth(170, 215, y))),
         ("far", lambda y: _smooth(170, 215, y)))


# --- build -----------------------------------------------------------------------

def build(ctx):
    if os.environ.get("MOUNTAIN_SHEET"):
        return _sheet(ctx)
    made = []
    rng = random.Random(6)

    # bed: sand and soil in one mesh, split by a painted attribute
    bed = substrate.build_bed(ground, "MountainFloor")
    bed.obj.data.materials.append(_floor_material())
    _paint_all(bed, "path", lambda x, y, d: path_fac(x, y))
    made.append(bed.obj)
    field = _Field()

    # stones
    stones = []
    for k, (name, model, x, y, h, pre, rot, buried) in enumerate(STONES):
        obj = _stone(model, h)
        obj.name = name
        full = h / (1 - buried)
        assets.place(obj, bed.at(x, y), height_mm=full, pre_rot_deg=pre, rot_deg=rot, sink_mm=full * buried)
        lo, hi = assets.world_bounds(obj)
        # keep clear of the glass
        dx = max(0.0, 9 * MM - lo.x) - max(0.0, hi.x - 291 * MM)
        dy = max(0.0, 9 * MM - lo.y) - max(0.0, hi.y - 291 * MM)
        obj.location += Vector((dx, dy, 0.0))
        _bake(obj)
        lo, hi = assets.world_bounds(obj)
        _grade(obj, model, seed=k * 1.7 + 0.3, far=_far(y), tone=rng.uniform(0.9, 1.08))
        _moss_floor(obj)
        cx, cy = (lo.x + hi.x) / 2 / MM, (lo.y + hi.y) / 2 / MM
        stones.append(dict(name=name, obj=obj, x=cx, y=cy, r=assets.footprint_radius_mm(obj), top=hi.z / MM,
                           h=hi.z / MM - ground(cx, cy), size=(hi - lo) / MM))
        made.append(obj)
    for s in stones:
        print(f"[stone] {s['name']:10s} at ({s['x']:.0f}, {s['y']:.0f})  size {s['size'].x:.0f} x {s['size'].y:.0f}  "
              f"top {s['top']:.0f}  ({s['h']:.0f} above the bed)  swim ({s['x']:.0f}, {s['y']:.0f}, {s['r'] * 0.8:.0f}, {s['top']:.0f})")

    # per top-vertex facts about the bed, worked out once for all the weights below
    facts = []
    for x, y in bed.top_xy_mm:
        d = path_dist(x, y)
        gap, ring = 1.0, 0.0
        for s in stones:
            q = math.hypot(x - s["x"], y - s["y"]) / s["r"]
            gap = min(gap, _smooth(0.62, 0.90, q))
            ring = max(ring, math.exp(-((q - 0.92) / 0.30) ** 2) * min(1.0, 0.35 + s["h"] / 45.0))
        wall = min(x - 5, 295 - x, y - 5, 295 - y)
        facts.append((x, y, d, gap, ring, wall))

    # granules: soil on the slopes (and down their cut face), grit on the sand
    _paint_all(bed, "soil", lambda x, y, d: (1.0 - path_fac(x, y, soft=5.0)) * (d < 40))
    _paint_all(bed, "grit", lambda x, y, d: path_fac(x, y, soft=6.0) * (0.55 + 0.45 * _smooth(-0.4, 0.4, _n2(x, y, 0.06, 8.0)))
               * (d < 26))
    # where each kind meets the glass its cut face is a wall of grains, not a flat sheet
    edge = lambda x, y: min(x - 6.2, 293.8 - x, y - 6.2, 293.8 - y) < 0.6
    _paint_all(bed, "soil_face", lambda x, y, d: (1.0 - path_fac(x, y, soft=5.0)) * edge(x, y) * (d < 30))
    _paint_all(bed, "sand_face", lambda x, y, d: path_fac(x, y, soft=5.0) * edge(x, y) * (d < 30))
    soil = substrate.soil_material("MtSoil")
    _granules(bed, soil, 2.4, "soil", 5, "SoilGrains")
    fine_soil = collection_of("MtSoilFine_src", [substrate._granule(f"MtSoilFine_{k}", 0.8 * MM, 300 + k, soil) for k in range(4)])
    _loose(bed, made, "SoilFine", fine_soil, "soil", 2.6e5, (0.6, 1.5), 7, 0.3)
    faces = [_loose(bed, made, "SoilFace", fine_soil, "soil_face", 5.0e5, (0.8, 1.7), 8, 0.5)]
    # pale grains with some tan and a few dark ones, as in any bag of real sand
    grit = _sand_grains("MtGrit", [((0.70, 0.65, 0.56), 4), ((0.62, 0.54, 0.43), 3), ((0.44, 0.35, 0.25), 2), ((0.15, 0.12, 0.10), 1)])
    _loose(bed, made, "SandGrit", grit, "grit", 1.6e5, (0.55, 1.5), 17, 0.25)
    # the cut face stands in shade, so its grains are the palest of the mix
    face = _sand_grains("MtFace", [((0.86, 0.80, 0.69), 6), ((0.74, 0.66, 0.54), 2), ((0.40, 0.32, 0.24), 1)])
    faces.append(_loose(bed, made, "SandFace", face, "sand_face", 3.4e6, (0.45, 1.1), 18, 0.3))
    # a second layer set further back closes the gaps between them, so no flat bed shows through
    faces.append(_loose(bed, made, "SandFaceBack", face, "sand_face", 2.4e6, (0.8, 1.4), 19, 0.7))

    # scree: broken stone round every foot and along the valley's shore, smaller with distance
    def _patch(x, y):
        """Where the meadow grows, 0..1: in drifts, with bare ground between."""
        return _smooth(-0.22, 0.20, _n2(x, y, 0.032, 21.0) + 0.4 * _n2(x, y, 0.10, 23.0))

    def scree(x, y, d, gap, ring, wall):
        shore = math.exp(-((d - 2.0) / 6.0) ** 2) * _smooth(-0.15, 0.35, _n2(x, y, 0.045, 3.0))
        stray = 0.05 * _smooth(0.2, 0.5, _n2(x, y, 0.09, 12.0))
        bare = 0.75 * (1.0 - _patch(x, y)) * _smooth(2.0, 10.0, d)              # where no meadow grows, loose stone lies
        if d > 6.0:
            stray = 0.16 * _smooth(0.0, 0.4, _n2(x, y, 0.08, 14.0))            # the odd stone lying in the meadow
        w = max(ring * (0.45 + 0.55 * _smooth(-0.3, 0.3, _n2(x, y, 0.07, 5.0))), 0.8 * shore, stray, bare)
        return w * gap * _smooth(7, 13, wall) * (1.0 - _smooth(268, 286, y) * 0.6)
    rubble = collection_of("MtRubbleSrc", _rubble_sources(40.0))
    sizes = dict(near=((0.40, 1.50), 6.5), mid=((0.26, 0.85), 4.2), far=((0.14, 0.42), 2.4))
    for k, (band, bw) in enumerate(BANDS):
        set_point_attribute(bed.obj, f"scree_{band}", [scree(*f) * bw(f[1]) for f in facts] + [0.0] * (len(bed.obj.data.vertices) - len(facts)))
        sc, gap_mm = sizes[band]
        scatter(bed.obj, rubble, name=f"Scree_{band}", min_distance=gap_mm * MM, weight_attr=f"scree_{band}", seed=50 + k,
                scale=sc, align="NORMAL", tilt=math.pi, sink=sc[0] * 2.0 * MM)

    # alpine meadow on the slopes: a fine carpet and moss in patches, thinning to bare soil and scree
    def meadow(x, y, d, gap, ring, wall):
        creep = 6.0 * _smooth(-0.2, 0.5, _n2(x, y, 0.05, 2.0)) - 2.0
        on = _smooth(creep, creep + 8.0, d)
        return on * _patch(x, y) * gap * _smooth(9, 18, wall)

    def split(x, y):
        return _smooth(-0.25, 0.25, _n2(x, y, 0.045, 31.0))          # 1: carpet, 0: moss

    pad = [0.0] * (len(bed.obj.data.vertices) - len(facts))
    green = [meadow(*f) for f in facts]
    share = [split(f[0], f[1]) for f in facts]
    sprigs_a = plants.carpet_sprigs(variants=5, seed=7, leaf_mm=3.0, material=plants.leaf_material(
        "MtCarpetFresh", base=(0.026, 0.085, 0.016), tip=(0.115, 0.26, 0.040), translucency=0.4, roughness=0.32))
    sprigs_b = plants.carpet_sprigs(variants=4, seed=19, leaf_mm=2.6, material=plants.leaf_material(
        "MtCarpetDark", base=(0.020, 0.055, 0.012), tip=(0.080, 0.17, 0.034), translucency=0.38, roughness=0.32))
    sprigs_c = plants.carpet_sprigs(variants=2, seed=43, leaf_mm=3.2, material=plants.leaf_material(
        "MtCarpetGold", base=(0.036, 0.080, 0.014), tip=(0.15, 0.23, 0.038), translucency=0.4, roughness=0.34))
    sprig_src = collection_of("MtCarpetSrc", sprigs_a + sprigs_b + sprigs_c)
    moss_src = _moss_sources()
    carpet_sz = dict(near=((0.50, 0.95), 4.2), mid=((0.36, 0.66), 3.1), far=((0.22, 0.42), 2.1))
    moss_sz = dict(near=((0.8, 1.7), 9.0e5), mid=((0.6, 1.2), 1.5e6), far=((0.4, 0.8), 2.6e6))
    for k, (band, bw) in enumerate(BANDS):
        set_point_attribute(bed.obj, f"carpet_{band}", [g * s * bw(f[1]) for g, s, f in zip(green, share, facts)] + pad)
        set_point_attribute(bed.obj, f"moss_{band}", [g * (1.0 - s) * bw(f[1]) for g, s, f in zip(green, share, facts)] + pad)
        sc, gap_mm = carpet_sz[band]
        scatter(bed.obj, sprig_src, name=f"Carpet_{band}", min_distance=gap_mm * MM, weight_attr=f"carpet_{band}",
                seed=31 + k, scale=sc, align="NORMAL", tilt=0.3, sway=(0.06, 12.0, 0.35))
        sc, dens = moss_sz[band]
        scatter(bed.obj, moss_src, name=f"BedMoss_{band}", density=dens, weight_attr=f"moss_{band}", seed=61 + k,
                scale=sc, align="NORMAL", tilt=0.6, sway=(0.04, 30.0, 0.3))

    # moss on the stones' lower slopes; bare rock above the tree line
    grown = []
    for k, s in enumerate(stones):
        me = s["obj"].data
        co, nr = _mesh_arrays(me)
        w = _stone_moss(co, nr, field, s["h"], seed=k * 3.1)
        _set_mesh_attribute(me, "moss", w)
        grown.append((s, me, co, nr, w, _weighted_area(me, w)))
    area = sum(g[5] for g in grown)
    dens = min(1.6e6, MOSS_POINTS / max(area, 1e-9))
    print(f"[mountain] stone moss area {area * 1e4:.1f} cm2 -> density {dens:.0f}/m2")
    for k, (s, me, co, nr, w, _) in enumerate(grown):
        f = 1.0 - 0.5 * _far(s["y"])                    # finer moss on the further stones
        scatter(s["obj"], moss_src, name=f"Moss_{s['name']}", density=dens / f, weight_attr="moss", seed=80 + k,
                scale=(0.7 * f, 1.5 * f), align="NORMAL", tilt=0.55, sway=(0.04, 30.0, 0.3))

    # a red shrimp grazing on the front foothill, where the camera can see it
    s, me, co, nr, w, _ = next(g for g in grown if g[0]["name"] == "L_front")
    cand = np.nonzero((nr[:, 2] > 0.78) & (w < 0.2))[0]
    if len(cand) == 0:
        cand = np.nonzero(nr[:, 2] > 0.6)[0]
    best = cand[np.argmax(co[cand, 2] - 0.25 * co[cand, 1])]
    perch = (Vector(co[best]) * MM, Vector(nr[best]).normalized())
    rock_shrimp = shrimp.Shrimp("Shrimp_rock", size=0.9)
    sand_shrimp = shrimp.Shrimp("Shrimp_sand", size=1.0)
    made += [rock_shrimp.obj, sand_shrimp.obj]

    school = fish.School()
    made += school.objects
    for src in _SRC.values():
        bpy.data.objects.remove(src, do_unlink=True)
    _SRC.clear()

    assets.mark_underwater(made, ctx)
    _front_glow(ctx, faces, 0.7)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["shrimp"] = dict(rock=(rock_shrimp, perch), sand=sand_shrimp)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    if ctx.get("school") is None:
        return
    ctx["school"].pose(frame)
    t = frame / 30.0
    rock_shrimp, (pos, up) = ctx["shrimp"]["rock"]
    rock_shrimp.pose(pos + up * 0.2 * MM, yaw=2.6 + 0.2 * math.sin(t / 2.9), t=t, up=tuple(up), walk=0.0)
    # the other picks its way along the valley mouth, a few millimetres a second
    s = ctx["shrimp"]["sand"]
    w = 0.35
    x = 196.0 + (t - 43.8) * shrimp.Shrimp.WALK_SPEED * w * s.size / MM
    y = 40.0 + 3.0 * math.sin(t * 0.3)
    s.pose(Vector(ctx["bed"].at(x, y)), yaw=0.05 * math.cos(t * 0.3), t=t, up=tuple(_ground_normal(x, y)), walk=w)


# --- stone sheet (MOUNTAIN_SHEET=<name>) --------------------------------------------

SHEETS = {
    "a": [("boulder_01", 0, (0, 0, 0)), ("boulder_01", 0, (90, 0, 0)), ("boulder_01", 0, (-90, 0, 0)),
          ("boulder_01", 0, (0, 90, 0)), ("boulder_01", 0, (0, -90, 0)), ("boulder_01", 0, (180, 0, 0)),
          ("rock_09", 0, (0, 0, 0)), ("namaqualand_boulder_02", 0, (0, 0, 0)), ("rock_09", 0, (90, 0, 0))],
    "b": [("rock_moss_set_02", 0, (0, 0, 0)), ("rock_moss_set_02", 1, (0, 0, 0)), ("rock_moss_set_02", 2, (0, 0, 0)),
          ("rock_moss_set_02", 3, (0, 0, 0)), ("rock_moss_set_02", 4, (0, 0, 0)), ("rock_moss_set_02", 5, (0, 0, 0)),
          ("moon_rock_05", 0, (90, 0, 0)), ("moon_rock_03", 0, (0, 0, 0)), ("moon_rock_07", 0, (0, 90, 0))],
}


def _sheet(ctx):
    """Candidate scans hung in the water in a grid, graded as in the scape, to
    choose which stands which way up."""
    made = []
    cells = SHEETS[os.environ["MOUNTAIN_SHEET"]]
    for k, (model, piece, pre) in enumerate(cells):
        obj = _copy(_source(model, 1.0, piece))
        cx, cz = 60 + 90 * (k % 3), 205 - 80 * (k // 3)
        assets.place(obj, (cx * MM, 150 * MM, cz * MM), longest_mm=72, pre_rot_deg=pre)
        lo, hi = assets.world_bounds(obj)
        obj.location.z += cz * MM - (lo.z + hi.z) / 2
        _bake(obj)
        _grade(obj, model, seed=k * 1.7)
        made.append(obj)
    for src in _SRC.values():
        bpy.data.objects.remove(src, do_unlink=True)
    _SRC.clear()
    assets.mark_underwater(made, ctx)
    ctx["school"] = None
    return ctx
