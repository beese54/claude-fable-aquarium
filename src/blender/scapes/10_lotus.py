"""Scape 10: Lotus pond.

A lily pond seen from within. Three red tiger lotus (Nymphaea zenkeri) of
different ages are rooted in dark gravel: each is a low crown of arrow-shaped
submerged leaves, bronze-red with darker flecks, a few of them green. From the
crowns long thin stalks wander up through clear water, each on its own curve,
to round notched pads lying in the surface film. The camera is below the
surface, so the pads are seen from underneath, lit through like stained glass.
A young leaf is still on its way up, rolled like a scroll; another has almost
arrived; a bud follows. The pads lie under the lamp, so their shadows fall as
round dapples on the pale sand at the front, which is the cue that sells the
picture. The water between the stalks is open for the shoal.
Coordinates are millimetres: X left to right, Y front to back.
"""
import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import assets
import fish
import materials
import plants
import snail
import substrate
from bl_common import MM, new_material
from fish import _math, _mix
from fish import _smooth as _ramp
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "10_lotus"

# Crisp light: the pads' shadows are the point, so the soft fill is kept low.
LIGHT = dict(watts=40.0, colour=(0.92, 0.96, 1.0), fill=0.004, bounce=0.012)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py):
# the open middle, above the crowns and below the pads. The stalks are a
# millimetre or two thick and are not obstacles; the cylinders are the crowns
# (as printed by the build's [crowns] line) and the rolled leaf in mid water.
SWIM = {
    "box": (44, 258, 28, 196, 146, 228),
    "floor": (60, 110),
    "obstacles": [(82, 196, 62, 162), (238, 226, 56, 156), (58, 112, 36, 116), (268, 170, 21, 230, 186), (132, 198, 8, 230, 204)],
}

LEVEL = 268.0            # water surface
PAD_Z = LEVEL - 1.2      # pads lie in the surface film; with air under the surface the mirror there would hide them
STILL_S = 2664 / 30.0    # the design still, in seconds: the snail's path is laid out around it

# Crowns: name, x, y, leaves, blade length range, leaf-stalk length range (mm), mean redness, seed,
# radius of the gravel pocket it is rooted in. Three ages, placed unevenly: the
# old plant at the back left, a middle-aged one in the back right corner, and a
# youngster that has come up in the sand at the front left.
CROWNS = [
    ("A", 82, 196, 10, (30, 46), (28, 68), 0.80, 11, 60),
    ("B", 238, 226, 7, (24, 38), (22, 50), 0.58, 23, 48),
    ("C", 58, 112, 5, (16, 25), (12, 26), 0.30, 37, 15),
]

# Floating pads: crown, x, y, radius, direction the tip points (deg), notch width (deg), redness, age.
# The lamp is a small bar of LEDs close above the water, so a pad's shadow on
# the bed is twice the pad's size and thrown outward from under the lamp: only
# the pads near the middle shade the bed (the rest shade the glass). The two
# largest are placed so that their shadows fall on the open sand at the front
# left and the middle right, with lit sand between and in front of them.
PADS = [
    ("A", 116, 164, 29.0, 205, 16, 0.75, 0.55),
    ("A", 146, 218, 19.0, 320, 24, 0.45, 0.15),
    ("A", 62, 204, 17.5, 100, 12, 0.85, 0.75),
    ("B", 191, 186, 24.0, 60, 20, 0.60, 0.35),
    ("B", 236, 204, 20.0, 255, 28, 0.30, 0.05),
    ("C", 38, 138, 16.0, 150, 18, 0.55, 0.0),
]

# Leaves still rising: crown, x, y, z, radius, how rolled (0 open .. 1 scroll), tip heading (deg), tip pitch (deg), redness.
RISING = [
    ("B", 268, 170, 205, 17.0, 0.80, 115, 56, 0.70),
    ("A", 100, 232, 247, 18.0, 0.15, -80, 24, 0.50),
]

# Flower buds on their way to the surface: crown, x, y, z of the tip, length (mm).
BUDS = [("B", 246, 242, 256, 19.0), ("A", 132, 198, 226, 13.0)]

# Pebbles: x, y, semi-axes (mm), heading (deg), fraction buried, seed. The first carries the snail.
PEBBLES = [
    (182, 52, (16.0, 12.0, 8.5), 20, 0.40, 1),
    (70, 62, (9.0, 6.5, 4.5), 75, 0.35, 2),
    (118, 84, (6.0, 4.5, 3.2), 140, 0.30, 3),
    (246, 66, (10.5, 8.0, 5.0), 110, 0.38, 4),
    (212, 96, (5.0, 4.2, 3.0), 30, 0.30, 5),
    (38, 118, (7.5, 5.5, 4.0), 160, 0.35, 6),
    (156, 104, (4.2, 3.4, 2.6), 60, 0.25, 7),
    (136, 236, (11.0, 7.5, 5.5), 95, 0.40, 8),
]


def _smooth(e0, e1, v):
    t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _patch(x, y, a=0.0, f=1.0):
    """Smooth 0..1 patchiness."""
    x, y = x * f, y * f
    n = math.sin(x * 0.071 + a + 1.6 * math.sin(y * 0.053 + a)) * math.cos(y * 0.064 + 1.2 * math.sin(x * 0.045 - a))
    n2 = math.sin(x * 0.19 + 0.7 + a) * math.sin(y * 0.23 + 1.9 - a)
    return max(0.0, min(1.0, 0.5 + 0.42 * n + 0.18 * n2))


def _sandy(x, y):
    """1 where pale sand lies (the open front, where the pads' shadows fall),
    0 on the dark gravel the plants are rooted in. Ragged, with gravel tongues."""
    s = 1.0 - _smooth(128, 172, y + 26 * (_patch(x, y, 1.3) - 0.5) + 0.08 * abs(x - 152))
    for c in CROWNS:
        cx, cy, r = c[1], c[2], c[8]
        s *= _smooth(r * 0.75, r * 1.15, math.hypot(x - cx, y - cy) + 0.3 * r * (_patch(x, y, 4.0, 2.0) - 0.5))
    s *= _smooth(0.20, 0.40, _patch(x, y, 2.6, 0.8) + 0.25 * _smooth(60, 20, y))
    return s


def ground(x, y):
    """Bed height in mm: 60 at the front glass, banked to about 110 at the back
    (which also hides the bright floor behind the tank), mounded a little where
    each plant is rooted; the sand has settled a couple of millimetres lower
    than the gravel."""
    h = 62 + 40 * (max(y, 0) / 295) ** 1.1 + 8 * _smooth(228, 292, y)
    h += 6 * math.exp(-(((x - 82) / 68) ** 2 + ((y - 196) / 58) ** 2))
    h += 5 * math.exp(-(((x - 238) / 52) ** 2 + ((y - 226) / 48) ** 2))
    h += 2 * math.exp(-(((x - 58) / 26) ** 2 + ((y - 112) / 26) ** 2))
    h += 1.7 * math.sin(x * 0.043 + 0.7) * math.cos(y * 0.037 + 0.2) + 0.7 * math.sin(x * 0.15 + y * 0.09)
    return h - 2.2 * _sandy(x, y)


def _ground_normal(x, y, d=1.5):
    n = Vector((ground(x - d, y) - ground(x + d, y), ground(x, y - d) - ground(x, y + d), 2 * d))
    return n.normalized()


def _edge_dist(x, y):
    return min(x - 5, 295 - x, y - 5, 295 - y)


# --- materials -------------------------------------------------------------------

def _attr(nt, name):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_name = name
    return n.outputs["Fac"]


def _vec(nt, x, y, z):
    n = nt.nodes.new("ShaderNodeCombineXYZ")
    for sock, v in zip(n.inputs, (x, y, z)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            nt.links.new(v, sock)
    return n.outputs[0]


def _noise(nt, vector, scale, detail=2.0, rough=0.5):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    nt.links.new(vector, n.inputs["Vector"])
    return n.outputs["Fac"]


def _value(nt, colour, value, saturation=1.0):
    n = nt.nodes.new("ShaderNodeHueSaturation")
    nt.links.new(colour, n.inputs["Color"])
    for name, v in (("Value", value), ("Saturation", saturation)):
        if isinstance(v, (int, float)):
            n.inputs[name].default_value = v
        else:
            nt.links.new(v, n.inputs[name])
    return n.outputs["Color"]


def _lotus_material(name, green, red, fleck, vein, under=None, veins=10, fleck_amount=0.5, translucency=0.45,
                    roughness=0.34, vein_width=0.055, midrib=1.0, vein_mix=0.34):
    """A lotus leaf. The mesh's UV is the flat leaf with the stalk at (0.5, 0.5)
    and the tip toward +u, so the shader can draw what a real leaf has: veins
    fanning from the stalk and forking toward the margin, a midrib, the tiger
    flecks (darker blotches drawn out along the veins), a paler heart, a darker
    rim, and on old leaves brown rot at the edge and a few holes. Each leaf
    carries its own redness, age and a random number as mesh attributes. Much
    of the light goes through, which is what makes a leaf glow against the lamp."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    du = _math(nt, "SUBTRACT", sep.outputs["X"], 0.5)
    dv = _math(nt, "SUBTRACT", sep.outputs["Y"], 0.5)
    ang = _math(nt, "ARCTAN2", dv, du)
    rho = _math(nt, "MULTIPLY", _math(nt, "SQRT", _math(nt, "ADD", _math(nt, "MULTIPLY", du, du), _math(nt, "MULTIPLY", dv, dv))), 2.0)
    tone, rnd, age, edge = (_attr(nt, k) for k in ("tone", "rnd", "age", "edge"))

    flat = _vec(nt, sep.outputs["X"], sep.outputs["Y"], _math(nt, "MULTIPLY", rnd, 31.0))
    polar = _vec(nt, _math(nt, "MULTIPLY", rho, 0.5), _math(nt, "MULTIPLY", ang, 0.55), _math(nt, "MULTIPLY", rnd, 17.0))
    mott = _noise(nt, flat, 3.5, 4.0, 0.6)
    fl = _math(nt, "ADD", _math(nt, "MULTIPLY", _noise(nt, polar, 6.0, 2.0, 0.6), 0.62), _math(nt, "MULTIPLY", _noise(nt, flat, 26.0, 2.0, 0.6), 0.38))
    lo = 0.575 - 0.06 * fleck_amount
    flecks = _ramp(nt, fl, lo, lo + 0.035)

    col = _mix(nt, _ramp(nt, tone, 0.22, 0.58), green, red)       # a leaf is green or red, seldom the mud in between
    heart = tuple(min(1.0, c * 1.5 + 0.02) for c in green)
    col = _mix(nt, _ramp(nt, rho, 0.0, 0.38, 0.28, 0.0), col, heart)
    col = _mix(nt, _math(nt, "MULTIPLY", flecks, 0.85), col, fleck)
    col = _mix(nt, _ramp(nt, edge, 0.80, 1.0, 0.0, 0.45), col, tuple(c * 0.4 for c in red))

    # veins: constant width on the leaf (the angular distance is scaled by the radius)
    def lines(count, phase, width):
        s = _math(nt, "ABSOLUTE", _math(nt, "SINE", _math(nt, "ADD", _math(nt, "MULTIPLY", ang, count / 2.0), phase)))
        return _ramp(nt, _math(nt, "MULTIPLY", s, rho), 0.0, width, 1.0, 0.0)
    main = _math(nt, "MULTIPLY", lines(veins, 0.25, vein_width), _ramp(nt, rho, 0.55, 1.0, 1.0, 0.35))
    fork = _math(nt, "MULTIPLY", lines(veins * 2, 1.3, vein_width * 0.8), _ramp(nt, rho, 0.42, 0.62, 0.0, 0.55))
    mid = _ramp(nt, _math(nt, "MULTIPLY", _math(nt, "ABSOLUTE", ang), rho), 0.0, vein_width * 0.75, 1.0, 0.0)
    vmask = _math(nt, "MAXIMUM", _math(nt, "MAXIMUM", main, fork), _math(nt, "MULTIPLY", mid, midrib))
    col = _mix(nt, _math(nt, "MULTIPLY", vmask, vein_mix), col, vein)

    # every leaf its own brightness; within a leaf, soft mottling
    val = _math(nt, "MULTIPLY", _ramp(nt, mott, 0.3, 0.7, 0.72, 1.28), _ramp(nt, rnd, 0.0, 1.0, 0.82, 1.18))
    col = _value(nt, col, val, 1.04)

    # old leaves: brown rot creeping in from the margin
    rot = _noise(nt, flat, 5.0, 3.0, 0.6)
    rot = _ramp(nt, _math(nt, "ADD", rot, _math(nt, "MULTIPLY", edge, 0.38)), 0.72, 0.82)
    col = _mix(nt, _math(nt, "MULTIPLY", rot, age), col, (0.15, 0.085, 0.028))

    if under is not None:           # the underside of a pad is a deeper red than its top
        geo = N.new("ShaderNodeNewGeometry")
        ucol = _mix(nt, _math(nt, "MULTIPLY", tone, 0.9), col, under)
        ucol = _mix(nt, _math(nt, "MULTIPLY", vmask, vein_mix), ucol, vein)
        col = _mix(nt, geo.outputs["Backfacing"], col, ucol)

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.12       # under water a leaf's sheen is weak
    fine = _noise(nt, flat, 42.0, 2.0, 0.5)
    height = _math(nt, "ADD", _math(nt, "MULTIPLY", vmask, 0.7), _math(nt, "ADD", _math(nt, "MULTIPLY", mott, 0.5), _math(nt, "MULTIPLY", fine, 0.3)))
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    bump.inputs["Distance"].default_value = 0.0004
    L.new(height, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(_value(nt, col, 1.15, 1.1), trans.inputs["Color"])
    L.new(bump.outputs["Normal"], trans.inputs["Normal"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = translucency
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    # a few holes eaten into the oldest leaves
    hole = _math(nt, "MULTIPLY", _ramp(nt, _noise(nt, flat, 9.0, 2.0, 0.5), 0.725, 0.74), _math(nt, "GREATER_THAN", age, 0.6))
    clear = N.new("ShaderNodeBsdfTransparent")
    holes = N.new("ShaderNodeMixShader")
    L.new(hole, holes.inputs["Fac"])
    L.new(mix.outputs[0], holes.inputs[1])
    L.new(clear.outputs[0], holes.inputs[2])
    L.new(holes.outputs[0], out.inputs["Surface"])
    return mat


def _stalk_material(name="LotusStalk"):
    """Leaf stalks: u carries each stalk's own redness, v runs base to tip."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    col = _mix(nt, sep.outputs["X"], (0.13, 0.17, 0.045), (0.30, 0.075, 0.040))
    coord = N.new("ShaderNodeTexCoord")
    streak = _noise(nt, coord.outputs["Object"], 700.0, 2.0, 0.5)
    val = _math(nt, "MULTIPLY", _ramp(nt, sep.outputs["Y"], 0.0, 1.0, 0.62, 1.12), _ramp(nt, streak, 0.3, 0.7, 0.8, 1.2))
    col = _value(nt, col, val)
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(col, trans.inputs["Color"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.28
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _bud_material(name="LotusBud"):
    """A closed bud: bronze-green sepals, their seams darker, blushing pink at the tip."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    col = _mix(nt, _ramp(nt, sep.outputs["Y"], 0.35, 1.0), (0.12, 0.15, 0.04), (0.50, 0.22, 0.20))
    seam = _math(nt, "ABSOLUTE", _math(nt, "SINE", _math(nt, "MULTIPLY", sep.outputs["X"], 4 * math.pi)))
    col = _mix(nt, _ramp(nt, seam, 0.0, 0.25, 0.6, 0.0), col, (0.10, 0.03, 0.03))
    coord = N.new("ShaderNodeTexCoord")
    col = _value(nt, col, _ramp(nt, _noise(nt, coord.outputs["Object"], 500.0, 3.0), 0.3, 0.7, 0.8, 1.2))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(col, trans.inputs["Color"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.25
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _stone_material(name, value=1.0, algae=0.3, speck_scale=1400.0):
    """Water-worn stone: every pebble or grain takes its own colour from a
    range of greys and browns, speckled and mottled, with a film of algae on
    whatever faces up."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    info = N.new("ShaderNodeObjectInfo")
    ramp = N.new("ShaderNodeValToRGB")
    stops = [(0.0, (0.030, 0.031, 0.036)), (0.2, (0.095, 0.090, 0.085)), (0.4, (0.16, 0.12, 0.085)),
             (0.58, (0.060, 0.050, 0.042)), (0.78, (0.27, 0.24, 0.19)), (1.0, (0.12, 0.068, 0.045))]
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = stops[0][0], (*stops[0][1], 1)
    e[1].position, e[1].color = stops[-1][0], (*stops[-1][1], 1)
    for p, c in stops[1:-1]:
        e.new(p).color = (*c, 1)
    L.new(info.outputs["Random"], ramp.inputs["Fac"])
    coord = N.new("ShaderNodeTexCoord")
    speck = _noise(nt, coord.outputs["Object"], speck_scale, 2.0, 0.6)
    mottle = _noise(nt, coord.outputs["Object"], speck_scale / 6.0, 3.0, 0.55)
    val = _math(nt, "MULTIPLY", _ramp(nt, speck, 0.35, 0.65, 0.70, 1.25), _ramp(nt, mottle, 0.3, 0.7, 0.8 * value, 1.2 * value))
    col = _value(nt, ramp.outputs["Color"], val, 0.85)
    geo = N.new("ShaderNodeNewGeometry")
    nz = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], nz.inputs[0])
    film = _math(nt, "MULTIPLY", _ramp(nt, nz.outputs["Z"], 0.2, 0.9, 0.0, algae), _ramp(nt, mottle, 0.35, 0.6))
    col = _mix(nt, film, col, (0.050, 0.062, 0.022))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    bump.inputs["Distance"].default_value = 0.0002
    L.new(speck, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# --- lotus geometry ----------------------------------------------------------------

class _Parts:
    """All the lotus pieces gathered into one mesh. Besides position and UV,
    each vertex carries its leaf's shading attributes and how it sways: which
    sway group it belongs to, how much of the group's motion it takes (0 at the
    root of a stalk .. 1 at its top), and whether it turns with the pad."""

    def __init__(self):
        self.co, self.uv, self.faces, self.fmat = [], [], [], []
        self.attrs = {k: [] for k in ("tone", "rnd", "age", "edge")}
        self.w, self.grp, self.turn = [], [], []
        self.groups = []
        self.n = 0

    def group(self, **kw):
        self.groups.append(kw)
        return len(self.groups) - 1

    def add(self, co, faces, uv, mat, tone=0.5, rnd=0.0, age=0.0, edge=0.0, w=0.0, grp=-1, turn=0.0):
        co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
        n = len(co)
        self.co.append(co)
        self.uv.append(np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        self.faces += [tuple(i + self.n for i in f) for f in faces]
        self.fmat += [mat] * len(faces)
        for k, v in (("tone", tone), ("rnd", rnd), ("age", age), ("edge", edge)):
            self.attrs[k].append(np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)).copy())
        self.w.append(np.broadcast_to(np.asarray(w, dtype=np.float64), (n,)).copy())
        self.grp.append(np.full(n, grp, dtype=np.int64))
        self.turn.append(np.full(n, turn, dtype=np.float64))
        self.n += n

    def finish(self, name, mats):
        co = np.concatenate(self.co) * MM
        me = bpy.data.meshes.new(name)
        me.from_pydata(co.tolist(), [], self.faces)
        uv = np.concatenate(self.uv)
        li = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", li)
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[li].astype(np.float32).ravel())
        me.polygons.foreach_set("material_index", self.fmat)
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        for k, parts in self.attrs.items():
            me.attributes.new(k, "FLOAT", "POINT").data.foreach_set("value", np.concatenate(parts).astype(np.float32))
        for m in mats:
            me.materials.append(m)
        me.update()
        obj = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(obj)
        grp, w, turn = np.concatenate(self.grp), np.concatenate(self.w), np.concatenate(self.turn)
        groups = []
        for g, spec in enumerate(self.groups):
            idx = np.nonzero(grp == g)[0]
            groups.append(dict(spec, idx=idx, w=w[idx][:, None], turn=turn[idx]))
        return dict(obj=obj, rest=co, groups=groups)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _tube(pts, radii, sides=7, u=0.5):
    """A round tube through pts. The cross-section frame is carried along the
    curve (never re-chosen), so a stalk that turns from flat to upright does not twist."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    n1 = np.cross(tan[0], (1.0, 0.0, 0.0) if abs(tan[0, 0]) < 0.9 else (0.0, 1.0, 0.0))
    a = 2 * np.pi * np.arange(sides) / sides
    ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
    co = np.empty((n, sides, 3))
    for k in range(n):
        n1 = _unit(n1 - tan[k] * np.dot(n1, tan[k]))
        n2 = np.cross(tan[k], n1)
        co[k] = pts[k] + radii[k] * (ca * n1 + sa * n2)
    faces = [(k * sides + s, k * sides + (s + 1) % sides, (k + 1) * sides + (s + 1) % sides, (k + 1) * sides + s)
             for k in range(n - 1) for s in range(sides)]
    t = np.repeat(np.linspace(0.0, 1.0, n), sides)
    uv = np.stack([np.full(n * sides, u) if u is not None else np.tile(np.arange(sides) / sides, n), t], 1)
    return co.reshape(-1, 3), faces, uv, t


def _blade(outline, thmax, R, J, K):
    """A flat leaf as a polar grid about the point where the stalk joins:
    x, y (mm; the tip is toward +x), each vertex's fraction of the way to the
    margin, its angle, and the faces. The grid is open at the back, where the
    leaf is split to the stalk."""
    th = np.linspace(-thmax, thmax, J + 1)
    r = outline(th) * R
    f = np.linspace(0.0, 1.0, K + 1)[1:]
    X = np.concatenate([[0.0], (f[:, None] * (r * np.cos(th))[None, :]).ravel()])
    Y = np.concatenate([[0.0], (f[:, None] * (r * np.sin(th))[None, :]).ravel()])
    F = np.concatenate([[0.0], np.repeat(f, J + 1)])
    TH = np.concatenate([[0.0], np.tile(th, K)])
    faces = [(0, 1 + j, 2 + j) for j in range(J)]
    for k in range(K - 1):
        a = 1 + k * (J + 1)
        faces += [(a + j, a + J + 1 + j, a + J + 2 + j, a + j + 1) for j in range(J)]
    return X, Y, F, TH, faces


def _arrow_outline(rng):
    """The submerged leaf: an arrowhead, broad, with two back-swept lobes and a
    wavy margin. No two alike: width, lobe length and sweep differ leaf to leaf
    and side to side."""
    wid = rng.uniform(0.50, 0.64)
    wav, n, ph = rng.uniform(0.015, 0.04), rng.randint(7, 11), rng.uniform(0, 6.28)

    def keys():
        a, r = rng.uniform(144, 156), rng.uniform(0.60, 0.82)
        w = wid * rng.uniform(0.94, 1.06)
        return ([0, 16, 40, 72, 104, 128, a, a + 12, 171, 177], [1.0, 0.90, 0.72, w, w * 0.98, w * 1.08, r, r * 0.74, 0.24, 0.07])
    left, right = keys(), keys()

    def outline(th):
        deg = np.abs(np.degrees(th))
        r = np.where(th < 0, np.interp(deg, *left), np.interp(deg, *right))
        for _ in range(2):
            r[1:-1] = (r[:-2] + 2 * r[1:-1] + r[2:]) / 4
        return r * (1 + wav * np.sin(n * th + ph))
    return outline, math.radians(177.0)


def _pad_outline(rng, notch_deg):
    """The floating leaf: round but not a circle, split to the stalk by a narrow
    notch, its margin faintly toothed."""
    p1, p2, p3 = (rng.uniform(0, 6.28) for _ in range(3))
    teeth = rng.randint(20, 27)
    thmax = math.pi - math.radians(notch_deg / 2)

    def outline(th):
        r = 1 + 0.035 * np.sin(2 * th + p1) + 0.02 * np.sin(3 * th + p2)
        r *= 1 + 0.016 * np.abs(np.sin(teeth * th / 2 + p3))
        return r * (1 + 0.05 * np.clip((np.abs(th) - thmax + 0.3) / 0.3, 0, 1) ** 2)      # the notch's corners stand out a little
    return outline, thmax


def _frame(az_deg, pitch_deg, roll_deg=0.0, toward=None):
    """Axes of a leaf: ex along the midrib to the tip, en out of the upper face."""
    az, p, r = math.radians(az_deg), math.radians(pitch_deg), math.radians(roll_deg)
    h, up = np.array([math.cos(az), math.sin(az), 0.0]), np.array([0.0, 0.0, 1.0])
    ex = math.cos(p) * h + math.sin(p) * up
    en = -math.sin(p) * h + math.cos(p) * up
    en = math.cos(r) * en + math.sin(r) * np.cross(en, ex)
    if toward is not None:
        en = _unit(en + np.asarray(toward))
        ex = _unit(ex - en * np.dot(ex, en))
    return ex, np.cross(en, ex), en


def _roll(Y, Z, R, amount):
    """Roll a leaf's two halves in over its upper face, as a lily leaf is
    rolled while it rises. amount 0 leaves it flat; 1 is a tight double scroll
    of more than a full turn each side. The curl tightens as it winds, so the
    sheet spirals inside itself instead of lying on itself."""
    for y0 in np.linspace(0.12, 0.45, 12) * R:      # the flat strip along the midrib: just wide enough that the two rolls do not meet
        S = 1.2 * R - y0
        s = np.linspace(0.0, S, 200)
        kappa = 1.0 + 1.4 * s / S
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(s))])
        phi *= max(amount, 1e-3) * 2.7 * math.pi / phi[-1]
        ds = np.diff(s)
        yy = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(phi[1:]) + np.cos(phi[:-1])) * ds)])
        zz = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(phi[1:]) + np.sin(phi[:-1])) * ds)])
        if y0 + yy.min() > 0.03 * R:
            break
    over = np.clip(np.abs(Y) - y0, 0.0, S)
    return np.sign(Y) * (np.minimum(np.abs(Y), y0) + np.interp(over, s, yy)), Z + np.interp(over, s, zz)


def _add_blade(P, at, frame, X, Y, Z, F, faces, R, mat, grp, turn=0.0, uv_xy=None, **attrs):
    ex, ey, en = frame
    co = np.asarray(at)[None, :] + X[:, None] * ex + Y[:, None] * ey + Z[:, None] * en
    ux, uy = uv_xy if uv_xy is not None else (X, Y)
    uv = np.stack([0.5 + 0.5 * ux / (1.12 * R), 0.5 + 0.5 * uy / (1.12 * R)], 1)
    P.add(co, faces, uv, mat, edge=F, w=1.0, grp=grp, turn=turn, **attrs)
    return co


def _grow_crown(P, spec):
    """One plant's submerged leaves, oldest first: long-stalked and low on the
    outside, short and upright in the heart. Returns its base point and its
    measured reach and height, for the shoal's keep-out cylinder."""
    name, cx, cy, n, (l0, l1), (p0, p1), red, seed, _ = spec
    rng = random.Random(seed)
    base = np.array([cx, cy, ground(cx, cy)])
    up = np.array([0.0, 0.0, 1.0])
    reach, top = 0.0, 0.0
    az0 = rng.uniform(0, 360)
    greens = set(rng.sample(range(n), 2 if n > 6 else 1))
    limit = 20 + 0.9 * l1                                          # how far a leaf may reach from the crown, mm
    for i in range(n):
        rank = i / max(n - 1, 1)                                   # 0 the oldest leaf .. 1 the youngest
        L = (l1 + (l0 - l1) * rank ** 1.2) * rng.uniform(0.85, 1.12)
        R = L / 1.6
        tone = 0.12 + 0.2 * rng.random() if i in greens else min(1.0, max(0.05, rng.gauss(red, 0.14)))
        if rank > 0.8:
            tone = 0.5 * tone + 0.3                                # the newest leaves are a paler bronze
        age = rng.uniform(0.45, 1.0) if rank < 0.3 else rng.uniform(0.0, 0.3)

        outline, thmax = _arrow_outline(rng)
        X, Y, F, TH, faces = _blade(outline, thmax, R, 84, 9)
        cup, droop, lift = rng.uniform(0.05, 0.16), rng.uniform(0.0, 0.28), rng.uniform(0.04, 0.26)
        ruffle, m, ph = rng.uniform(0.03, 0.085), rng.randint(5, 8), rng.uniform(0, 6.28)
        rho = np.hypot(X, Y) / R
        Z = (cup * R * rho ** 2 - droop * R * np.clip(X / R, 0, None) ** 2 + lift * R * np.clip(-X / R, 0, None) ** 1.5
             + ruffle * R * F ** 3 * np.sin(m * TH + ph) - 0.035 * R * np.exp(-(Y / (0.07 * R)) ** 2) * np.clip(X / R, 0, 1))
        Lp = (p1 + (p0 - p1) * rank ** 0.8) * rng.uniform(0.6, 1.22)
        for attempt in range(12):
            az = az0 + i * 137.5 + rng.uniform(-22, 22) + 47.0 * attempt
            h = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
            elev = math.radians(50 + 28 * rank + rng.uniform(-9, 9))
            # the blade lies between flat and tilted up, turned a little to the front so its face is seen
            frame = _frame(az + rng.uniform(-75, 75), rng.uniform(-22, 40) * (1 - 0.5 * rank) + 22 * rank, rng.uniform(-34, 34),
                           toward=(0.0, -rng.uniform(0.0, 0.7), 0.0))
            a = base + h * rng.uniform(2.5, 4.0 + 0.7 * n) + up * 0.5
            e = a + Lp * (math.cos(elev) * h + math.sin(elev) * up)
            co = e[None, :] + X[:, None] * frame[0] + Y[:, None] * frame[1] + Z[:, None] * frame[2]
            if (_edge_dist(co[:, 0].min(), co[:, 1].min()) > 9 and _edge_dist(co[:, 0].max(), co[:, 1].max()) > 9
                    and np.hypot(co[:, 0] - cx, co[:, 1] - cy).max() < limit):
                break
            Lp *= 0.93
        low = max(ground(x, y) + 2.5 - z for x, y, z in co[::7])    # never through the bed
        if low > 0:
            e = e + up * low
        g = P.group(az=rng.uniform(0, 6.28), amp=rng.uniform(0.7, 1.5), ampz=rng.uniform(0.3, 0.7), T=rng.uniform(4.0, 7.0),
                    ph=rng.uniform(0, 6.28), pivot=(e[0], e[1]), yaw=rng.uniform(0.015, 0.04), x=cx)
        co = _add_blade(P, e, frame, X, Y, Z, F, faces, R, 0, g, turn=1.0, tone=tone, rnd=rng.random(), age=age)
        side = np.cross(h, up)
        c = a + up * Lp * 0.18 * math.sin(elev) + h * Lp * 0.92 * math.cos(elev) + side * rng.uniform(-0.14, 0.14) * Lp
        t = np.linspace(0.0, 1.0, 13)[:, None]
        pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * e
        r0 = 0.45 + 0.009 * L
        tco, tf, tuv, tt = _tube(pts, r0 * (1.0 - 0.3 * t[:, 0]), sides=6, u=min(1.0, tone + 0.15))
        P.add(tco, tf, tuv, 2, w=tt ** 1.5, grp=g)
        reach = max(reach, float(np.hypot(co[:, 0] - cx, co[:, 1] - cy).max()))
        top = max(top, float(co[:, 2].max()))

    # a new leaf in the heart of the two older plants: a tight scroll on a short stalk
    if n > 5:
        R = l0 * 0.32
        outline, thmax = _arrow_outline(rng)
        X, Y, F, TH, faces = _blade(outline, thmax, R, 60, 8)
        Y2, Z = _roll(Y, 0.08 * R * (np.hypot(X, Y) / R) ** 2, R, 0.95)
        frame = _frame(rng.uniform(0, 360), 72, 0)
        a = base + up * 1.0
        e = a + up * p0 * 0.8 + np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])
        g = P.group(az=rng.uniform(0, 6.28), amp=0.6, ampz=0.2, T=5.0, ph=rng.uniform(0, 6.28), pivot=None, yaw=0.0, x=cx)
        _add_blade(P, e, frame, X, Y2, Z, F, faces, R, 0, g, uv_xy=(X, Y), tone=0.45, rnd=rng.random(), age=0.0)
        t = np.linspace(0.0, 1.0, 7)[:, None]
        tco, tf, tuv, tt = _tube(a + (e - a) * t, np.full(7, 0.6), sides=6, u=0.6)
        P.add(tco, tf, tuv, 2, w=tt ** 1.5, grp=g)

    # the tuber, nearly buried
    t = np.linspace(0.0, 1.0, 9)
    pts = base[None, :] + np.outer(t, [1.5, 0.8, 9.0]) + np.array([0.0, 0.0, -7.5])
    tco, tf, tuv, _ = _tube(pts, (4.0 + 0.45 * n) * np.sin(np.pi * (0.05 + 0.9 * t)) ** 0.7, sides=10, u=0.5)
    P.add(tco, tf, tuv, 3)
    return base, reach, top


def _stalk(P, rng, a, e, radius, tone, grp, slack=1.0):
    """A long leaf stalk from the crown to a leaf: it leaves the crown steeply,
    wanders (it is longer than the water is deep), and meets the leaf from below."""
    a, e = np.asarray(a, dtype=np.float64), np.asarray(e, dtype=np.float64)
    d = e - a
    flat = np.array([d[0], d[1], 0.0])
    side = _unit(np.cross(flat if np.linalg.norm(flat) > 1e-6 else (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    up = np.array([0.0, 0.0, 1.0])
    c1 = a + up * d[2] * rng.uniform(0.28, 0.45) + flat * rng.uniform(-0.25, 0.15) + side * rng.uniform(-1, 1) * 24 * slack
    c2 = e - up * d[2] * rng.uniform(0.22, 0.40) + flat * rng.uniform(-0.10, 0.35) + side * rng.uniform(-1, 1) * 26 * slack
    pts = _bezier(a, c1, c2, e, 56)
    t = np.linspace(0.0, 1.0, 57)
    wob, k, ph = rng.uniform(1.5, 4.0) * slack, rng.uniform(1.2, 2.2), rng.uniform(0, 6.28)
    pts += (wob * np.sin(np.pi * t) * np.sin(2 * np.pi * k * t + ph))[:, None] * side
    pts += (0.6 * wob * np.sin(np.pi * t) * np.sin(2 * np.pi * (k + 0.7) * t + 2 * ph))[:, None] * _unit(flat + 1e-6)
    tco, tf, tuv, tt = _tube(pts, radius * (1.0 - 0.28 * t), sides=8, u=tone)
    P.add(tco, tf, tuv, 2, w=tt ** 1.6, grp=grp)


def _grow_lotus():
    rng = random.Random(10)
    P = _Parts()
    bases, measured = {}, []
    for spec in CROWNS:
        base, reach, top = _grow_crown(P, spec)
        bases[spec[0]] = base
        measured.append((spec[1], spec[2], reach, top))

    def root(crown):
        return bases[crown] + np.array([rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0), 0.5])

    # pads in the surface film
    for crown, x, y, R, tip, notch, tone, age in PADS:
        outline, thmax = _pad_outline(rng, notch)
        X, Y, F, TH, faces = _blade(outline, thmax, R, 132, 10)
        m, ph = rng.randint(3, 5), rng.uniform(0, 6.28)
        Z = -rng.uniform(0.4, 0.8) * F ** 2 * (0.5 + 0.5 * np.sin(m * TH + ph)) - 0.35 * np.exp(-(F / 0.18) ** 2)
        g = P.group(az=rng.uniform(0, 6.28), amp=rng.uniform(1.8, 3.2), ampz=0.0, T=rng.uniform(5.0, 8.0), ph=rng.uniform(0, 6.28),
                    pivot=(x, y), yaw=rng.uniform(0.02, 0.05), x=x)
        at = np.array([x, y, PAD_Z])
        _add_blade(P, at, _frame(tip, 0.0), X, Y, Z, F, faces, R, 1, g, turn=1.0, tone=tone, rnd=rng.random(), age=age)
        _stalk(P, rng, root(crown), at - np.array([0.0, 0.0, 0.5]), 0.62 + 0.018 * R, min(1.0, tone + rng.uniform(-0.2, 0.3)), g,
               slack=rng.choice((0.45, 0.8, 1.1, 1.5, 1.9)))

    # leaves still on their way up
    for crown, x, y, z, R, rolled, tip, pitch, tone in RISING:
        outline, thmax = _pad_outline(rng, 26)
        X, Y, F, TH, faces = _blade(outline, thmax, R, 120, 12)
        Z = 0.26 * R * (np.hypot(X, Y) / R) ** 1.6 + 0.07 * R * F ** 3 * np.sin(5 * TH + 1.0 + x)
        Y2, Z = _roll(Y, Z, R, rolled)
        g = P.group(az=rng.uniform(0, 6.28), amp=rng.uniform(2.0, 3.0), ampz=0.5, T=rng.uniform(5.0, 7.5), ph=rng.uniform(0, 6.28),
                    pivot=(x, y), yaw=0.03, x=x)
        at = np.array([x, y, z])
        _add_blade(P, at, _frame(tip, pitch, rng.uniform(-12, 12)), X, Y2, Z, F, faces, R, 1, g, turn=1.0, uv_xy=(X, Y),
                   tone=tone, rnd=rng.random(), age=0.0)
        _stalk(P, rng, root(crown), at, 0.95, tone, g, slack=0.7)

    # the buds
    for crown, x, y, z, length in BUDS:
        tip = np.array([x, y, z])
        axis = _unit((rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.1), 1.0))
        neck = tip - axis * length
        g = P.group(az=rng.uniform(0, 6.28), amp=2.2, ampz=0.3, T=rng.uniform(5.5, 7.0), ph=rng.uniform(0, 6.28), pivot=None, yaw=0.0, x=x)
        _stalk(P, rng, root(crown), neck + axis * 1.0, 0.06 * length, 0.55, g, slack=0.6)
        k = np.linspace(0.0, 1.0, 16)
        radii = 0.23 * length * np.sin(np.pi * np.clip(0.04 + 0.96 * k ** 0.72, 0, 1)) ** 0.85 + 0.05 * length * (1 - k) ** 6
        bco, bf, buv, _ = _tube(neck[None, :] + np.outer(k, axis * length), radii, sides=12, u=None)
        P.add(bco, bf, buv, 4, w=1.0, grp=g)

    leaf = _lotus_material("LotusLeaf", green=(0.050, 0.110, 0.022), red=(0.165, 0.014, 0.016), fleck=(0.022, 0.004, 0.010),
                           vein=(0.26, 0.085, 0.05), veins=10, fleck_amount=0.85, translucency=0.40, roughness=0.42, vein_width=0.045)
    pad = _lotus_material("LotusPad", green=(0.050, 0.15, 0.028), red=(0.15, 0.065, 0.028), fleck=(0.07, 0.012, 0.014),
                          vein=(0.30, 0.30, 0.10), under=(0.27, 0.022, 0.042), veins=15, fleck_amount=0.5,
                          translucency=0.5, roughness=0.3, vein_width=0.06, midrib=0.5, vein_mix=0.38)
    tuber = _stone_material("LotusTuber", value=0.45, algae=0.0, speck_scale=900.0)
    lotus = P.finish("Lotus", [leaf, pad, _stalk_material(), tuber, _bud_material()])
    return lotus, measured


def _sway(lotus, t):
    """Pads drift a few millimetres on the surface and turn a little, each in
    its own time over a slow swell they all share; the stalks follow, held at
    the root; the submerged leaves nod."""
    co = lotus["rest"].copy()
    tau = 2 * math.pi
    for g in lotus["groups"]:
        idx = g["idx"]
        s1 = 0.65 * math.sin(tau * t / g["T"] + g["ph"]) + 0.35 * math.sin(tau * t / 8.7 + 0.021 * g["x"] + 0.4)
        s2 = math.sin(tau * t / (g["T"] * 1.37) + 2.1 * g["ph"])
        ca, sa = math.cos(g["az"]), math.sin(g["az"])
        d = np.array([g["amp"] * (ca * s1 - 0.45 * sa * s2), g["amp"] * (sa * s1 + 0.45 * ca * s2),
                      g["ampz"] * math.sin(tau * t / (g["T"] * 0.8) + 1.3 * g["ph"])]) * MM
        p = co[idx]
        if g["pivot"] is not None and g["yaw"]:
            a = g["yaw"] * math.sin(tau * t / (g["T"] * 1.6) + 0.7 * g["ph"])
            px, py = g["pivot"][0] * MM, g["pivot"][1] * MM
            rx, ry = p[:, 0] - px, p[:, 1] - py
            k = g["turn"]
            p[:, 0] += k * (rx * math.cos(a) - ry * math.sin(a) - rx)
            p[:, 1] += k * (rx * math.sin(a) + ry * math.cos(a) - ry)
        co[idx] = p + g["w"] * d
    me = lotus["obj"].data
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()


# --- the bed and what grows on it --------------------------------------------------

def _pebble(name, semi_mm, seed, material, subdiv=4):
    """A water-worn stone: an ellipsoid with a few broad lumps and a flatter underside."""
    rng = random.Random(seed)
    lumps = [(rng.uniform(1.5, 4.5), rng.uniform(0, 6.28), Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized(),
              rng.uniform(0.03, 0.10)) for _ in range(5)]
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    a, b, c = semi_mm
    for v in bm.verts:
        p = v.co.copy()
        s = 1.0 + sum(amp * math.sin(fq * p.dot(d) + ph) for fq, ph, d, amp in lumps)
        v.co = Vector((p.x * a * s, p.y * b * s, p.z * c * s * (0.8 if p.z < 0 else 1.0))) * MM
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(material)
    return bpy.data.objects.new(name, me)


def _ribbon_tufts(name, material, variants, seed, leaves, length_mm, width_mm, rise_deg, curl, belly, fold):
    """Rosette plants for scattering: leaves fanning from a point (plants.py's leaf)."""
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for _ in range(rng.randint(*leaves)):
            L = rng.uniform(*length_mm) * MM
            az, rise = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(*rise_deg))
            d = (math.cos(az) * math.cos(rise), math.sin(az) * math.cos(rise), math.sin(rise))
            u = (-math.cos(az) * math.sin(rise), -math.sin(az) * math.sin(rise), math.cos(rise))
            o = (rng.uniform(-1.5, 1.5) * MM, rng.uniform(-1.5, 1.5) * MM, -1.0 * MM)
            parts.append(plants._leaf(o, d, u, L, rng.uniform(*width_mm) * MM, 0.0, 1.0, fold=fold,
                                      curl=rng.uniform(*curl), belly=belly, segs=8))
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"{name}_{v}", verts, faces, uvs, material))
    return out


def _curled(objs, seed):
    """Fallen leaves, some cupped and some rolled, so they do not lie like chips."""
    rng = random.Random(seed)
    for o in objs:
        kind = rng.random()
        xs = [v.co.x for v in o.data.vertices]
        half = (max(xs) - min(xs)) / 2 or 1.0
        a = rng.uniform(0.15, 0.4)
        for v in o.data.vertices:
            if kind < 0.55:
                v.co.z += a * half * (v.co.x / half) ** 2
            else:
                v.co.z += a * 1.1 * v.co.y ** 2 / (0.3 * half)
        o.data.update()
    return objs


def build(ctx):
    made = []

    # --- bed: dark fine gravel, with pale sand settled in the open front ----------
    bed = substrate.build_bed(ground, "PondBed")
    me = bed.obj.data
    me.materials.append(materials.pbr_box("PondGravel", "pebbles", scale=15.0, saturation=0.8, value=0.72,
                                          bump=1.0, bump_distance=0.0012))
    me.materials.append(materials.pbr_box("PondSand", "damp_beach_sand", scale=8.0, tint=(0.96, 0.90, 0.78), saturation=0.75,
                                          value=1.32, bump=0.5, bump_distance=0.0006, roughness=0.6))
    for p in me.polygons:
        if p.normal.z > 0.5 and _sandy(p.center.x / MM, p.center.y / MM) > 0.5:
            p.material_index = 1
    made.append(bed.obj)

    # Real grains over the gravel (and thinly strewn on the sand), so the bed's
    # edge is grainy; its section at the glass is the kit's wall of grains.
    def gravel(x, y):
        s = _sandy(x, y)
        return max(1.0 - _smooth(0.25, 0.6, s), 0.10 * _patch(x, y, 5.1, 2.2) * _smooth(20, 70, y))
    ntop = len(bed.top_xy_mm)
    set_point_attribute(bed.obj, "gravel", [gravel(x, y) for x, y in bed.top_xy_mm] + [0.0] * (len(me.vertices) - ntop))
    grain_mat = _stone_material("PondGrain", value=0.85, algae=0.12, speck_scale=2200.0)
    # in section: gravel throughout, under a skin of sand where sand lies on top
    face = [_pebble(f"FaceGrain_{k}", (1.45 * random.Random(k).uniform(0.8, 1.2), 1.15, 1.0), 300 + k, grain_mat, subdiv=2)
            for k in range(7)]
    skin = lambda x, y: min(1.0, 2.0 * _sandy(x, y))
    made += substrate.add_cut_face(
        bed, sources=face, size_mm=2.8, thick_mm=2.0, seed=3, name="FaceGravel", tone=(1.0, 0.5), room=0.7, spacing=0.62,
        weight=lambda x, y, d: 1.0 - skin(x, y) * (1.0 - _smooth(4.0, 7.0, d)), backing=(0.012, 0.011, 0.010))
    made += substrate.add_cut_face(
        bed, substrate.sand_mix("FaceSand", pale=(0.64, 0.58, 0.48)), size_mm=1.2, seed=5, name="FaceSand", tone=(1.3, 0.6),
        weight=lambda x, y, d: skin(x, y) * (1.0 - _smooth(5.0, 9.0, d)))
    grains = [_pebble(f"Grain_{k}", (1.35 * random.Random(k).uniform(0.8, 1.2), 1.05, 0.8 * random.Random(k + 9).uniform(0.8, 1.2)),
                      100 + k, grain_mat, subdiv=2) for k in range(7)]
    scatter(bed.obj, collection_of("GrainSrc", grains), name="Gravel", min_distance=2.0 * MM, weight_attr="gravel", seed=3,
            scale=(0.6, 1.2), align="NORMAL", tilt=math.pi, sink=0.5 * MM)
    bed.set_weight("strewn", lambda x, y: (0.25 + 0.75 * _patch(x, y, 3.3, 1.7)) * (1.0 - 0.7 * _sandy(x, y)) * float(_edge_dist(x, y) > 6))
    stones = [_pebble(f"Stone_{k}", (2.6 * random.Random(k).uniform(0.8, 1.3), 2.0, 1.4), 200 + k,
                      _stone_material("PondStone", value=1.0, algae=0.25), subdiv=3) for k in range(6)]
    scatter(bed.obj, collection_of("StoneSrc", stones), name="Stones", density=1400.0, weight_attr="strewn", seed=8,
            scale=(0.6, 1.5), align="NORMAL", tilt=0.5, sink=0.6 * MM)

    # a few pebbles, each its own stone, a third or so buried
    pebble_mat = _stone_material("PondPebble", value=1.05, algae=0.35)
    pebbles = []
    for x, y, semi, yaw, buried, seed in PEBBLES:
        o = _pebble(f"Pebble_{seed}", semi, 300 + seed, pebble_mat)
        bpy.context.scene.collection.objects.link(o)
        o.location = bed.at(x, y, lift_mm=semi[2] * (1 - 2 * buried))
        o.rotation_euler = (math.radians(random.Random(seed).uniform(-8, 8)), math.radians(random.Random(seed + 1).uniform(-8, 8)),
                            math.radians(yaw))
        pebbles.append(o)
    made += pebbles

    # --- the lotus -----------------------------------------------------------------
    lotus, measured = _grow_lotus()
    made.append(lotus["obj"])
    print("[crowns] " + ", ".join(f"({x}, {y}, {r:.0f}, {top:.0f})" for x, y, r, top in measured))

    # --- low green plants for contrast: dwarf sagittaria and a few small crypts ----
    def clear_of_crowns(x, y, k=1.0):
        return min(_smooth(r * 0.55 * k, r * 0.8 * k, math.hypot(x - cx, y - cy)) for cx, cy, r, _ in measured)

    def blob(x, y, cx, cy, rx, ry):
        return math.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))

    def sag_back(x, y):
        w = max(blob(x, y, 30, 270, 34, 26), blob(x, y, 150, 280, 70, 13), blob(x, y, 284, 150, 12, 30), 0.7 * blob(x, y, 20, 215, 12, 30))
        return _smooth(0.3, 0.6, w * (0.6 + 0.6 * _patch(x, y, 2.2, 2.0))) * clear_of_crowns(x, y) * float(_edge_dist(x, y) > 9)

    def sag_low(x, y):
        w = max(blob(x, y, 22, 52, 10, 13), blob(x, y, 150, 256, 40, 12), blob(x, y, 274, 100, 11, 20), blob(x, y, 160, 196, 14, 11))
        return _smooth(0.35, 0.65, w * (0.6 + 0.6 * _patch(x, y, 6.2, 2.0))) * clear_of_crowns(x, y) * float(_edge_dist(x, y) > 8)

    def crypt(x, y):
        w = max(blob(x, y, 168, 238, 22, 14), blob(x, y, 150, 150, 12, 12), blob(x, y, 268, 140, 12, 14), blob(x, y, 106, 96, 9, 9))
        return _smooth(0.35, 0.6, w) * clear_of_crowns(x, y, 0.8) * float(_edge_dist(x, y) > 14)

    for name, fn in (("sag_back", sag_back), ("sag_low", sag_low), ("crypt", crypt)):
        bed.set_weight(name, fn)
    sag_mat = plants.leaf_material("PondSag", base=(0.016, 0.060, 0.010), tip=(0.10, 0.27, 0.045), translucency=0.42, roughness=0.32,
                                   variation=0.35)
    sag = collection_of("SagSrc", _ribbon_tufts("Sag", sag_mat, 8, 5, (7, 12), (22, 66), (1.7, 2.8), (50, 88), (0.05, 0.75), 0.30, 0.12))
    scatter(bed.obj, sag, name="SagBack", min_distance=8.0 * MM, weight_attr="sag_back", seed=31, scale=(0.7, 1.15),
            align="UP", tilt=0.12, sway=(0.06, 7.0, 0.3))
    scatter(bed.obj, sag, name="SagLow", min_distance=7.0 * MM, weight_attr="sag_low", seed=37, scale=(0.4, 0.72),
            align="UP", tilt=0.15, sway=(0.05, 7.0, 0.3))
    crypt_mat = plants.leaf_material("PondCrypt", base=(0.030, 0.065, 0.014), tip=(0.12, 0.25, 0.045), translucency=0.35,
                                     roughness=0.36, variation=0.3)
    crypts = collection_of("CryptSrc", _ribbon_tufts("Crypt", crypt_mat, 5, 9, (6, 10), (16, 30), (6.0, 9.5), (22, 62), (0.15, 0.5), 0.45, 0.28))
    scatter(bed.obj, crypts, name="Crypts", min_distance=17.0 * MM, weight_attr="crypt", seed=41, scale=(0.75, 1.15),
            align="UP", tilt=0.1, sway=(0.03, 7.0, 0.3))

    # --- a little leaf litter, drifted against the back and into the corners --------
    def litter(x, y):
        w = 0.25 + 0.75 * _smooth(150, 270, y) + 0.5 * max(blob(x, y, 15, 40, 40, 50), blob(x, y, 285, 30, 30, 40))
        return min(1.0, w) * _patch(x, y, 0.9, 1.6) * float(_edge_dist(x, y) > 22)
    bed.set_weight("litter", litter)
    fallen = _curled(plants.dead_leaves(variants=8, seed=61, length_mm=(13, 27)), 61)
    scatter(bed.obj, collection_of("LitterSrc", fallen), name="Litter", density=900.0, weight_attr="litter", seed=67,
            scale=(0.65, 1.05), align="NORMAL", tilt=0.2, sink=-0.6 * MM)

    # --- animals -------------------------------------------------------------------
    school = fish.School()
    made += school.objects

    # A nerite on the big pebble at the front, side-on so its striped shell shows.
    nerite = snail.Snail("Nerite")
    made.append(nerite.obj)
    bpy.context.view_layer.update()
    perch = pebbles[0]
    pm = perch.matrix_world
    tree = BVHTree.FromPolygons([pm @ v.co for v in perch.data.vertices], [tuple(p.vertices) for p in perch.data.polygons])

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["lotus"] = lotus
    ctx["snail"] = (nerite, tree, PEBBLES[0][0] - 3.0, PEBBLES[0][1] - 1.5)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    t = frame / 30.0
    _sway(ctx["lotus"], t)

    # the nerite creeps over the pebble at about half a millimetre a second
    nerite, tree, x0, y0 = ctx["snail"]
    x = x0 + 0.5 * (t - STILL_S)
    hit, normal, _, _ = tree.ray_cast(Vector((x * MM, y0 * MM, 0.25)), Vector((0, 0, -1)))
    if hit is None:
        hit, normal = Vector(ctx["bed"].at(x, y0)), _ground_normal(x, y0)
    nerite.pose(hit + normal * 0.15 * MM, yaw=0.12, t=t, up=tuple(normal))
