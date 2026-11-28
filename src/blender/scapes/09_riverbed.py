"""Scape 9: Riverbed.

The bed of a clear hill stream, with the current running from left to right.
Water-worn stones in every size, laid as moving water leaves them: each larger
stone has a scour hollow on its upstream side with coarse pebbles jammed in it
and a tail of fine gravel streaming away downstream; flat cobbles lean on one
another with their downstream ends raised; pebbles gather in drifts on the
bars, and a ribbon of clean rippled sand winds through the quiet channel
between them. A mother plant of ribbon grass (Vallisneria) at the back left has
sent runners along the bed; every leaf is bent over to the right and streams in
the flow. It is the most open of the ten scapes: nothing but the grass stands
more than a few centimetres off the bed.
Coordinates are millimetres: X left to right, Y front to back.
"""
import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise
from mathutils.bvhtree import BVHTree

import assets
import fish
import materials
import snail
import substrate
import wood
from bl_common import MM, new_material, new_mesh_object
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "09_riverbed"

# A little less soft fill than the default: the pale sand throws light back up
# onto the stones, and their rounded forms need the shaded side to read.
LIGHT = dict(fill=0.005, bounce=0.02)
# A stream: clear and a touch cool, with a livelier surface than a still tank.
# The fine ripples are shorter than the kit's default, which brings their focus
# (about wavelength^2 / (13 x height)) down toward the bed: the lamp's light
# lands dappled, with caustic threads running over the sand and the stones.
WATER = dict(tint=(0.86, 0.975, 0.985), density=1.3, ripple_mm=0.50, fine_mm=0.32, fine_scale=0.011)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py).
# Nearly all of it is free; the cylinders are the two largest stones and the
# ribbon-grass plants (the build prints the measured ones as [stones] and [vallis]).
SWIM = {
    "box": (40, 262, 28, 232, 116, 236),
    "floor": (62, 108),
    "obstacles": [(110, 160, 40, 128), (216, 208, 34, 130), (52, 236, 30, 262), (96, 258, 26, 262),
                  (150, 242, 24, 250), (38, 124, 26, 150)],
}

SURFACE_MM = 268.0
STILL_T = 1854 / 30.0          # the snail's track is laid out around this scape's still frame

# name, x, y, longest side, width / length, thickness / length, heading of the
# long axis (deg from X), dip (deg; the downstream end is raised, as flat stones
# stack in a current), fraction buried, rock type, seed.
# Sized for a 30 cm cube: the largest is 68 mm long and stands under 30 mm off the bed.
STONES = [
    # main group, left of centre: a boulder with cobbles jammed against its upstream side
    ("main",          110, 160, 68, 0.76, 0.60, 20,  6,  0.36, "granite",    11),
    ("main_stoss_a",  64,  156, 36, 0.72, 0.50, 80,  16, 0.38, "jasper",     12),
    ("main_stoss_b",  76,  186, 30, 0.80, 0.55, 105, 14, 0.40, "gneiss",     13),
    ("main_stoss_c",  86,  132, 24, 0.70, 0.50, 60,  12, 0.42, "basalt",     14),
    ("main_side",     138, 200, 33, 0.75, 0.52, 150, 10, 0.40, "greenstone", 15),
    ("main_lee",      166, 190, 20, 0.80, 0.62, 30,  8,  0.45, "quartz",     16),
    # second group, right and further back
    ("second",        216, 208, 56, 0.72, 0.58, -30, 8,  0.38, "basalt",     21),
    ("second_stoss_a", 180, 224, 31, 0.75, 0.50, 95, 15, 0.40, "granite",    22),
    ("second_stoss_b", 184, 195, 24, 0.80, 0.55, 70, 12, 0.42, "limestone",  23),
    ("second_side",   252, 234, 28, 0.70, 0.50, 20,  6,  0.42, "sandstone",  24),
    # front left: the pale stone the snail is on
    ("front_left",    60,  82,  47, 0.80, 0.62, 65,  5,  0.40, "limestone",  31),
    ("front_left_a",  30,  70,  26, 0.75, 0.50, 100, 14, 0.42, "greenstone", 32),
    ("front_left_b",  92,  62,  22, 0.80, 0.55, 40,  10, 0.45, "gneiss",     33),
    # right, behind the channel's way out
    ("right",         246, 124, 40, 0.74, 0.55, -15, 7,  0.40, "gneiss",     41),
    ("right_a",       216, 110, 24, 0.78, 0.50, 85,  14, 0.42, "sandstone",  42),
    ("right_b",       270, 146, 27, 0.72, 0.55, 110, 9,  0.42, "granite",    43),
    # loners; the first sits alone in the sand so its hollow and tail show clean
    ("lone",          158, 106, 30, 0.80, 0.60, 12,  3,  0.46, "jasper",     51),
    ("mid",           198, 146, 26, 0.76, 0.52, 100, 12, 0.45, "sandstone",  52),
    ("back_left",     28,  204, 32, 0.75, 0.55, 75,  10, 0.40, "sandstone",  53),
    ("back_mid",      126, 264, 36, 0.70, 0.50, 15,  6,  0.40, "granite",    54),
    ("back_right",    268, 270, 30, 0.80, 0.60, 130, 8,  0.40, "limestone",  55),
]
SNAIL_STONE = "front_left"

# Ribbon grass: x, y, leaves, shortest and longest leaf, leaf width. The mother
# plant and the daughters on its runners, smaller with distance; one young plant
# nearer the glass. Every leaf streams to the right.
VALLIS = [
    (52,  236, 17, 130, 238, 4.5),
    (96,  258, 10, 110, 196, 4.1),
    (150, 242, 7,  80,  134, 3.7),
    (224, 262, 5,  36,  62,  3.1),
    (38,  124, 6,  45,  88,  3.3),
]

# A water-worn branch: its butt jammed in behind the main group, the shaft
# propped on a cobble of the second group, the thin end free and pointing
# downstream. Butt (x, y), a point it rests on (x, y, height above the bed), length, butt radius.
BRANCH = ((128, 218), (184, 195, 13), 108, 5.2)

# Rock types, as base colours of wet stone: two tones that mottle together,
# mineral flecks, bedding bands, quartz veins.
LITH = {
    "granite":    dict(a=(0.25, 0.25, 0.26), b=(0.43, 0.42, 0.41), speck=0.80, bands=0.0, vein=0.0, rough=0.46),
    "basalt":     dict(a=(0.040, 0.045, 0.055), b=(0.095, 0.10, 0.11), speck=0.15, bands=0.0, vein=0.45, rough=0.40),
    "sandstone":  dict(a=(0.34, 0.25, 0.155), b=(0.50, 0.40, 0.27), speck=0.20, bands=0.50,
                       band_col=(0.23, 0.155, 0.09), vein=0.0, rough=0.58),
    "jasper":     dict(a=(0.19, 0.075, 0.042), b=(0.32, 0.15, 0.085), speck=0.10, bands=0.45,
                       band_col=(0.11, 0.035, 0.02), vein=0.25, rough=0.40),
    "greenstone": dict(a=(0.080, 0.105, 0.082), b=(0.165, 0.195, 0.15), speck=0.20, bands=0.25,
                       band_col=(0.045, 0.07, 0.055), vein=0.40, rough=0.42),
    "quartz":     dict(a=(0.52, 0.50, 0.47), b=(0.72, 0.70, 0.67), speck=0.0, bands=0.18,
                       band_col=(0.42, 0.30, 0.18), vein=0.0, rough=0.34, sss=0.25),
    "limestone":  dict(a=(0.38, 0.35, 0.29), b=(0.54, 0.51, 0.45), speck=0.10, bands=0.12,
                       band_col=(0.28, 0.26, 0.22), vein=0.30, rough=0.50),
    "gneiss":     dict(a=(0.19, 0.18, 0.19), b=(0.41, 0.35, 0.33), speck=0.40, bands=0.80,
                       band_col=(0.09, 0.09, 0.10), vein=0.15, rough=0.45),
}
# What a handful of the gravel holds: mostly greys and tans, some dark, the odd red, green or white one.
LITH_MIX = ["granite", "sandstone", "basalt", "granite", "limestone", "gneiss", "jasper", "sandstone",
            "greenstone", "granite", "quartz", "limestone"]


def _smooth(e0, e1, v):
    t = min(1.0, max(0.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


# --- the bed: where the water runs fast and slow -----------------------------------

# (x, y, footprint radius, size) of every placed stone, for the hollows and tails they make
_STONE_FIELD = [(x, y, 0.25 * a * (1 + flat), a) for _, x, y, a, flat, *_ in STONES]
_BARS = [(115, 180, 105, 50, 1.0), (228, 212, 58, 42, 0.9), (62, 74, 48, 30, 0.9), (246, 126, 40, 26, 0.8)]
_cache = {}


def _fields(x, y):
    """The flow at a point of the bed, as the things it leaves behind:
    energy (0 quiet water and sand .. 1 fast water and cobbles), the relief the
    stones cut and build (mm), the fine-gravel tail behind a stone (0..1), the
    sand channel (0..1) and the gravel bars (0..1)."""
    hit = _cache.get((x, y))
    if hit is not None:
        return hit
    relief = stoss = lee = tail = 0.0
    for sx, sy, R, size in _STONE_FIELD:
        dx, dy = x - sx, y - sy
        if abs(dy) > 3.0 * R + 10 or dx < -(2.6 * R + 10) or dx > 8.0 * R:
            continue
        d = math.hypot(dx, dy) + 1e-6
        up = 0.5 - 0.5 * dx / d                       # 1 on the upstream side, 0 downstream
        ring = math.exp(-((d - 1.05 * R) / (0.5 * R)) ** 2)
        relief -= 0.12 * size * ring * (0.15 + 0.85 * up ** 1.3)       # horseshoe scour, deepest upstream
        k = min(1.0, size / 38.0)
        stoss = max(stoss, k * math.exp(-((d - 1.2 * R) / (0.55 * R + 4)) ** 2) * up ** 0.8)
        if dx > 0:
            u, lat = dx / R, dy / R
            f = (u / 1.7) * math.exp(1 - u / 1.7)
            relief += 0.09 * size * f * math.exp(-(lat / 0.55) ** 2 * (1 + 0.2 * u))    # the tail's low ridge
            lee = max(lee, k * math.exp(-(lat / 0.9) ** 2) * _smooth(0.7, 1.2, u) * (1 - _smooth(3.5, 6.5, u)))
            tail = max(tail, math.exp(-(lat / 0.34) ** 2 * (1 + 0.25 * u)) * _smooth(0.9, 1.25, u) * (1 - _smooth(2.6, 5.5, u)))
    yc = 128 - 0.2 * (x - 5) + 12 * math.sin((x - 30) / 55)
    chan = math.exp(-((y - yc) / 27) ** 2)
    bars = max(w * math.exp(-(((x - bx) / rx) ** 2 + ((y - by) / ry) ** 2)) for bx, by, rx, ry, w in _BARS)
    bars = max(bars, 0.55 * math.exp(-((y - 272) / 20) ** 2))
    streak = noise.noise(Vector((x * 0.0065, y * 0.050, 3.3)))         # drawn out along the flow
    fine = noise.noise(Vector((x * 0.06, y * 0.06, 9.1)))
    energy = 0.46 + 0.32 * bars - 0.62 * chan + 0.60 * stoss - 0.55 * lee + 0.60 * streak + 0.10 * fine
    out = (energy, relief, tail, chan, bars)
    _cache[(x, y)] = out
    return out


def _sand(x, y):
    energy, _, tail, _, _ = _fields(x, y)
    return (1 - _smooth(0.30, 0.50, energy)) * (1 - 0.85 * tail)


def ground(x, y):
    """Bed height in mm: 62 at the front glass, banked to about 108 at the back
    and a little higher upstream; gravel bars, the sand channel between them,
    a hollow and a tail at every stone, and current ripples on the sand."""
    _, relief, _, chan, bars = _fields(x, y)
    v = max(y, 0.0) / 290.0
    h = 62 + 44 * v + 4 * (1 - x / 300) * v
    h += 6.0 * bars - 5.0 * chan
    h += 1.5 * math.sin(x * 0.021 + 0.7) * math.cos(y * 0.034 + 0.3) + 0.7 * math.sin(x * 0.052 + y * 0.083)
    h += relief
    # current ripples on the sand: crests across the flow, wandering, coming and going
    ph = (x + 7 * math.sin(y * 0.045) + 9 * noise.noise(Vector((x * 0.015, y * 0.02, 1.7)))) / 11.0
    amp = 0.75 * (0.3 + 0.7 * _smooth(-0.3, 0.25, noise.noise(Vector((x * 0.02, y * 0.03, 6.2)))))
    h += _sand(x, y) * amp * (math.sin(2 * math.pi * ph) + 0.35 * math.sin(4 * math.pi * ph + 0.6))
    return max(h, 50.0)


# --- stones ------------------------------------------------------------------------

def _relax(co, faces, iters=3, lam=0.5):
    """A few rounds of smoothing: rounds off the edges of the worn flats."""
    f = np.array(faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    cnt = np.bincount(e[:, 0], minlength=len(co)).astype(float)
    for _ in range(iters):
        acc = np.zeros_like(co)
        np.add.at(acc, e[:, 0], co[e[:, 1]])
        co = co + lam * (acc / cnt[:, None] - co)
    return co


_ICO = {}


def _stone_shape(seed, dims_mm, subdiv):
    """A water-worn stone: vertices (mm, about its middle; X long, Z thin) and faces.
    Not an ellipsoid: squarer or rounder stone to stone, one end blunter, low
    lumps, and one to three flats worn on it with their edges rounded over."""
    if subdiv not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
        bm.verts.ensure_lookup_table()
        d = np.array([tuple(v.co) for v in bm.verts])
        _ICO[subdiv] = (d / np.linalg.norm(d, axis=1, keepdims=True), [tuple(v.index for v in f.verts) for f in bm.faces])
        bm.free()
    d, faces = _ICO[subdiv]
    rng = random.Random(seed)
    p = rng.uniform(2.1, 3.0)
    co = d * ((np.abs(d) ** p).sum(1) ** (-1.0 / p))[:, None]
    co[:, 1] *= 1.0 + rng.uniform(-0.22, 0.22) * co[:, 0]
    co[:, 2] *= 1.0 + rng.uniform(-0.25, 0.25) * co[:, 0] + rng.uniform(-0.15, 0.15) * co[:, 1]
    off = rng.uniform(0, 100)
    amp = rng.uniform(0.05, 0.11)
    shift = Vector((off, off * 0.7, -off))
    lump = np.array([noise.noise(Vector(q) * 1.15 + shift) for q in d])
    co = co * (1.0 + amp * 2.0 * lump)[:, None]
    for _ in range(rng.randint(1, 3)):
        n = np.array(Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized())
        h = rng.uniform(0.74, 0.92)
        co = co - np.clip(co @ n - h, 0.0, None)[:, None] * n[None, :] * 0.8
    co = _relax(co, faces, iters=3 if subdiv >= 3 else 1)
    return co * (np.array(dims_mm) / 2.0), faces


def _stone_material(name, lith, size_mm, algae=0.4, seed=0.0):
    """Wet river stone. Patterns are solid (3D), sized to the stone, and shifted
    stone to stone, so no two instances of one mesh show the same face."""
    p = LITH[lith]
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links

    def mth(op, a, b=None, c=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for k, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                L.new(v, n.inputs[k])
        return n.outputs[0]

    def rmap(v, a, b, c=0.0, d=1.0):
        n = N.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        L.new(v, n.inputs["Value"])
        n.inputs["From Min"].default_value = a
        n.inputs["From Max"].default_value = b
        n.inputs["To Min"].default_value = c
        n.inputs["To Max"].default_value = d
        return n.outputs["Result"]

    def mix(f, a, b, blend="MIX"):
        n = N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        for key, v in (("Factor", f), ("A", a), ("B", b)):
            if isinstance(v, (int, float)):
                n.inputs[key].default_value = v
            elif isinstance(v, tuple):
                n.inputs[key].default_value = (*v, 1.0)
            else:
                L.new(v, n.inputs[key])
        return n.outputs["Result"]

    coord = N.new("ShaderNodeTexCoord")
    info = N.new("ShaderNodeObjectInfo")
    jit = N.new("ShaderNodeVectorMath")
    jit.operation = "SCALE"
    jit.inputs[0].default_value = (3.1, 5.7, 7.9)
    L.new(info.outputs["Random"], jit.inputs["Scale"])
    add = N.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    L.new(coord.outputs["Object"], add.inputs[0])
    L.new(jit.outputs["Vector"], add.inputs[1])
    add2 = N.new("ShaderNodeVectorMath")
    add2.operation = "ADD"
    add2.inputs[1].default_value = (seed * 1.3, seed * 0.7, seed * 2.1)
    L.new(add.outputs["Vector"], add2.inputs[0])
    pos = add2.outputs["Vector"]

    def noise_tex(scale, detail=3.0, rough=0.55, vector=None, distortion=0.0):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        L.new(vector or pos, n.inputs["Vector"])
        return n.outputs["Fac"]

    def mapped(rot, scale):
        m = N.new("ShaderNodeMapping")
        m.inputs["Rotation"].default_value = rot
        m.inputs["Scale"].default_value = scale
        L.new(pos, m.inputs["Vector"])
        return m.outputs["Vector"]

    k = 1000.0 / size_mm
    col = mix(rmap(noise_tex(2.6 * k, 3.0), 0.36, 0.64), p["a"], p["b"])
    if p["bands"] > 0:
        # bedding: tone runs in layers across the thin axis, as flat stones split along it
        bed_vec = mapped((0.25 + 0.3 * math.sin(seed), 0.2, 0.0), (0.35, 0.35, 3.2))
        bn = noise_tex(3.0 * k, 2.0, vector=bed_vec, distortion=0.6)
        col = mix(rmap(bn, 0.44, 0.60, 0.0, p["bands"]), col, p["band_col"])
    if p["speck"] > 0:
        # crystals a millimetre or two across: dark mica and pale feldspar
        sn = noise_tex(max(950.0, 24.0 * k), 1.0, 0.7)
        col = mix(rmap(sn, 0.31, 0.36, p["speck"], 0.0), col, (0.04, 0.04, 0.045))
        col = mix(rmap(sn, 0.64, 0.69, 0.0, 0.8 * p["speck"]), col, (0.62, 0.60, 0.57))
    if p["vein"] > 0:
        vv = mapped((0.5, 0.35 + seed, 0.2), (0.30, 1.0, 1.0))
        ridge = mth("ABSOLUTE", mth("SUBTRACT", noise_tex(2.6 * k, 2.5, vector=vv, distortion=0.4), 0.5))
        thin = rmap(ridge, 0.002, 0.006, 1.0, 0.0)
        gate = rmap(noise_tex(1.2 * k, 1.0), 0.50, 0.60)
        col = mix(mth("MULTIPLY", mth("MULTIPLY", thin, gate), p["vein"]), col, (0.70, 0.70, 0.67))
    # the grain of the rock itself, and broad darker staining: no stone is one flat tone
    col = mix(1.0, col, rmap(noise_tex(1700.0, 2.0, 0.7), 0.30, 0.70, 0.84, 1.16), "MULTIPLY")
    col = mix(1.0, col, rmap(noise_tex(1.5 * k, 3.0), 0.38, 0.72, 1.0, 0.74), "MULTIPLY")
    # sparse dark pits, a millimetre across
    col = mix(rmap(noise_tex(820.0, 0.0), 0.69, 0.75, 0.0, 0.55), col, tuple(0.35 * c for c in p["a"]))
    # stone to stone, a little lighter or darker
    col = mix(1.0, col, rmap(info.outputs["Random"], 0.0, 1.0, 0.80, 1.20), "MULTIPLY")
    # a brown-green film of algae and settled silt on what faces up at the lamp
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], sep.inputs[0])
    upw = rmap(sep.outputs["Z"], 0.15, 0.90)
    patch = rmap(noise_tex(3.5 * k, 4.0, 0.6), 0.36, 0.60)
    film = mth("MULTIPLY", upw, mth("MULTIPLY_ADD", patch, 0.8 * algae, 0.2 * algae))
    col = mix(film, col, (0.060, 0.075, 0.025))

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    L.new(mth("MULTIPLY_ADD", film, 0.25, p["rough"]), bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.40
    if p.get("sss"):
        bsdf.inputs["Subsurface Weight"].default_value = p["sss"]
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.9, 0.8)
        bsdf.inputs["Subsurface Scale"].default_value = 0.0015
    height = mth("MULTIPLY_ADD", noise_tex(2200.0, 2.0), 0.4, noise_tex(9.0 * k, 1.0))
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0005
    L.new(height, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def _mesh_from(name, co_mm, faces, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata((np.asarray(co_mm) * MM).tolist(), [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.materials.append(material)
    me.update()
    return me


def _place_stones():
    """Build every placed stone in world space. Returns their records."""
    out = []
    for k, (name, x, y, a, flat, thick, heading, dip, buried, lith, seed) in enumerate(STONES):
        co, faces = _stone_shape(seed, (a, a * flat, a * thick), 6 if a >= 45 else 5)
        rng = random.Random(seed + 500)
        R = (Matrix.Rotation(math.radians(-dip), 3, "Y") @ Matrix.Rotation(math.radians(heading), 3, "Z")
             @ Matrix.Rotation(math.radians(rng.uniform(-8, 8)), 3, "X"))
        co = co @ np.array(R).T
        zlo, zhi = co[:, 2].min(), co[:, 2].max()
        g = ground(x, y)
        co += np.array([x, y, g - zlo - buried * (zhi - zlo)])
        mat = _stone_material(f"Stone_{name}", lith, a, algae=0.75 if a >= 45 else 0.55, seed=seed * 0.37)
        me = _mesh_from(f"Stone_{name}", co, faces, mat)
        obj = bpy.data.objects.new(f"Stone_{name}", me)
        bpy.context.scene.collection.objects.link(obj)
        r = max(co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min()) / 2
        out.append(dict(name=name, obj=obj, x=x, y=y, r=r, top=co[:, 2].max(), above=co[:, 2].max() - g, co=co, faces=faces))
    return out


def _sources(prefix, count, size_mm, subdiv, lift, algae, seed, flat=(0.62, 0.90), thick=(0.38, 0.62), thick_mm=None):
    """Loose stones of one size class for scattering: a handful of shapes, in
    the rock types of LITH_MIX. Z is the thin axis, so they lie flat."""
    rng = random.Random(seed)
    mats, out = {}, []
    for k in range(count):
        lith = LITH_MIX[k % len(LITH_MIX)]
        if lith not in mats:
            mats[lith] = _stone_material(f"{prefix}_{lith}", lith, size_mm, algae=algae, seed=seed + 0.1 * k)
        a = size_mm * rng.uniform(0.85, 1.15)
        dims = (a, a * rng.uniform(*flat), thick_mm or a * rng.uniform(*thick))
        co, faces = _stone_shape(seed * 100 + k, dims, subdiv)
        co[:, 2] += lift * dims[2]
        out.append(bpy.data.objects.new(f"{prefix}_{k}", _mesh_from(f"{prefix}_{k}", co, faces, mats[lith])))
    return out


# --- bed surface ---------------------------------------------------------------------

def _bed_material():
    """Sand where the water is quiet, fine river gravel where it is not, blended
    by the `sand` attribute painted on the bed. In the cut face against the
    glass the sand is only a skin: below it the bed is gravel."""
    mat, nt, out = new_material("RiverBed")
    N, L = nt.nodes, nt.links
    coord = N.new("ShaderNodeTexCoord")

    def tex(asset, kind, scale, colour_data):
        files = materials.texture_files(asset)
        mp = N.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (scale, scale, scale)
        L.new(coord.outputs["Object"], mp.inputs["Vector"])
        t = N.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(files[kind]), check_existing=True)
        if not colour_data:
            t.image.colorspace_settings.name = "Non-Color"
        t.projection = "BOX"
        t.projection_blend = 0.3
        L.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t.outputs["Color"]

    def graded(col, saturation, value, tint):
        hsv = N.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = saturation
        hsv.inputs["Value"].default_value = value
        L.new(col, hsv.inputs["Color"])
        m = N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = "MULTIPLY"
        m.inputs["Factor"].default_value = 1.0
        m.inputs["B"].default_value = (*tint, 1.0)
        L.new(hsv.outputs["Color"], m.inputs["A"])
        return m.outputs["Result"]

    sand_col = graded(tex("damp_beach_sand", "colour", 8.0, True), 0.65, 1.50, (0.97, 0.94, 0.87))
    # single grains are near a pixel across: light and dark speckle, and the odd dark grain
    grains = N.new("ShaderNodeTexNoise")
    grains.inputs["Scale"].default_value = 1900.0
    grains.inputs["Detail"].default_value = 1.0
    L.new(coord.outputs["Object"], grains.inputs["Vector"])
    gr = N.new("ShaderNodeMapRange")
    gr.inputs["From Min"].default_value = 0.30
    gr.inputs["From Max"].default_value = 0.70
    gr.inputs["To Min"].default_value = 0.62
    gr.inputs["To Max"].default_value = 1.30
    L.new(grains.outputs["Fac"], gr.inputs["Value"])
    sp = N.new("ShaderNodeMix")
    sp.data_type = "RGBA"
    sp.blend_type = "MULTIPLY"
    sp.inputs["Factor"].default_value = 1.0
    L.new(sand_col, sp.inputs["A"])
    L.new(gr.outputs["Result"], sp.inputs["B"])
    sand_col = sp.outputs["Result"]
    grav_col = graded(tex("ganges_river_pebbles", "colour", 8.0, True), 0.90, 0.95, (0.96, 0.95, 0.93))
    ao = N.new("ShaderNodeMix")
    ao.data_type = "RGBA"
    ao.blend_type = "MULTIPLY"
    ao.inputs["Factor"].default_value = 0.8
    L.new(grav_col, ao.inputs["A"])
    L.new(tex("ganges_river_pebbles", "ao", 8.0, False), ao.inputs["B"])
    grav_col = ao.outputs["Result"]

    def attr(name):
        a = N.new("ShaderNodeAttribute")
        a.attribute_type = "GEOMETRY"
        a.attribute_name = name
        return a.outputs["Fac"]

    edge = N.new("ShaderNodeTexNoise")              # a ragged edge between sand and gravel
    edge.inputs["Scale"].default_value = 260.0
    edge.inputs["Detail"].default_value = 3.0
    L.new(coord.outputs["Object"], edge.inputs["Vector"])
    wob = N.new("ShaderNodeMath")
    wob.operation = "MULTIPLY_ADD"
    wob.inputs[1].default_value = 0.30
    wob.inputs[2].default_value = -0.15
    L.new(edge.outputs["Fac"], wob.inputs[0])
    top = N.new("ShaderNodeMath")
    top.operation = "ADD"
    L.new(attr("sand"), top.inputs[0])
    L.new(wob.outputs[0], top.inputs[1])
    deep = N.new("ShaderNodeMath")                  # in the cut face: gravel below the sand skin
    deep.operation = "MULTIPLY_ADD"
    deep.inputs[1].default_value = -3.0
    L.new(attr("depth"), deep.inputs[0])
    L.new(top.outputs[0], deep.inputs[2])
    fac = N.new("ShaderNodeMapRange")
    fac.interpolation_type = "SMOOTHSTEP"
    fac.inputs["From Min"].default_value = 0.38
    fac.inputs["From Max"].default_value = 0.62
    L.new(deep.outputs[0], fac.inputs["Value"])
    f = fac.outputs["Result"]

    col = N.new("ShaderNodeMix")
    col.data_type = "RGBA"
    L.new(f, col.inputs["Factor"])
    L.new(grav_col, col.inputs["A"])
    L.new(sand_col, col.inputs["B"])
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col.outputs["Result"], bsdf.inputs["Base Color"])
    rough = N.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.50
    rough.inputs["To Max"].default_value = 0.62
    L.new(f, rough.inputs["Value"])
    L.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.35

    hs = N.new("ShaderNodeMix")                      # sand's relief is a third of the gravel's
    hs.data_type = "RGBA"
    L.new(f, hs.inputs["Factor"])
    L.new(tex("ganges_river_pebbles", "height", 8.0, False), hs.inputs["A"])
    dim = N.new("ShaderNodeMix")
    dim.data_type = "RGBA"
    dim.blend_type = "MULTIPLY"
    dim.inputs["Factor"].default_value = 1.0
    dim.inputs["B"].default_value = (0.3, 0.3, 0.3, 1.0)
    L.new(tex("damp_beach_sand", "height", 8.0, False), dim.inputs["A"])
    L.new(dim.outputs["Result"], hs.inputs["B"])
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.9
    bump.inputs["Distance"].default_value = 0.0018
    L.new(hs.outputs["Result"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def _dress_bed(bed, made):
    """Paint the flow onto the bed and scatter the loose stone it sorts:
    pebbles in drifts where the water is fastest, gravel around them, fine
    gravel in the tails, grit nearly everywhere, and bare sand in the channel."""
    top = bed.top_xy_mm
    w_pebble, w_gravel, w_fine, w_grit, sand = [], [], [], [], []
    for x, y in top:
        energy, _, tail, _, _ = _fields(x, y)
        wall = min(x - 6.2, 293.8 - x, y - 6.2, 293.8 - y)       # distance to the bed's edge at the glass
        n2 = noise.noise(Vector((x * 0.035, y * 0.05, 5.5)))
        peb = _smooth(0.62, 0.85, energy + 0.25 * n2)
        w_pebble.append(peb * _smooth(8.0, 12.0, wall))
        w_gravel.append(_smooth(0.40, 0.58, energy) * (1 - 0.35 * peb) * _smooth(3.5, 6.0, wall))
        w_fine.append(max(_smooth(0.30, 0.46, energy) * (1 - 0.5 * _smooth(0.6, 0.8, energy)), 0.9 * tail)
                      * _smooth(1.8, 3.2, wall))
        w_grit.append((0.22 + 0.78 * _smooth(0.2, 0.45, energy)) * _smooth(0.8, 1.6, wall))
        sand.append(_sand(x, y))
    n_all = len(bed.obj.data.vertices)
    pad = [0.0] * (n_all - len(top))
    for name, vals in (("w_pebble", w_pebble), ("w_gravel", w_gravel), ("w_fine", w_fine), ("w_grit", w_grit)):
        set_point_attribute(bed.obj, name, vals + pad)
    # the skirt takes the sand value of the edge above it, and a depth of 1 at the floor
    co = [(v.co.x / MM, v.co.y / MM) for v in bed.obj.data.vertices]
    set_point_attribute(bed.obj, "sand", sand + [_sand(x, y) for x, y in co[len(top):]])
    set_point_attribute(bed.obj, "depth", [0.0] * len(top) + [1.0] * len(pad))

    scatter(bed.obj, collection_of("PebbleSrc", _sources("Pebble", 12, 14.0, 3, 0.04, 0.30, seed=3)),
            name="Pebbles", min_distance=9.5 * MM, weight_attr="w_pebble", seed=3, scale=(0.60, 1.35),
            align="NORMAL", tilt=0.30)
    scatter(bed.obj, collection_of("GravelSrc", _sources("Gravel", 12, 6.5, 2, 0.10, 0.15, seed=5)),
            name="Gravel", min_distance=4.4 * MM, weight_attr="w_gravel", seed=5, scale=(0.60, 1.30),
            align="NORMAL", tilt=0.35)
    scatter(bed.obj, collection_of("FineSrc", _sources("Fine", 12, 3.2, 1, 0.10, 0.0, seed=7)),
            name="FineGravel", min_distance=2.5 * MM, weight_attr="w_fine", seed=7, scale=(0.60, 1.30),
            align="NORMAL", tilt=0.5)
    scatter(bed.obj, collection_of("GritSrc", _sources("Grit", 12, 1.6, 1, 0.0, 0.0, seed=9)),
            name="Grit", density=1.1e5, weight_attr="w_grit", seed=9, scale=(0.60, 1.40),
            align="NORMAL", tilt=math.pi)

    # Against the front glass the bed is seen in section. Real gravel shows there
    # as stones pressed flat to the pane, so flat pebbles are set into the cut
    # face, standing proud of it by under a millimetre (the gap to the glass is 1.2).
    xs = [7.0 + 2.0 * i for i in range(144)]
    rows = 12
    verts, faces, weight = [], [], []
    for x in xs:
        ztop = ground(x, 6.2) - 1.5
        for j in range(rows + 1):
            z = 30.0 + (ztop - 30.0) * j / rows
            verts.append((x * MM, 6.2 * MM, z * MM))
            d = (ztop - z) / 30.0
            weight.append(max(0.25, 1.0 - _sand(x, 6.2) * (1 - _smooth(0.1, 0.35, d))))
    for i in range(len(xs) - 1):
        for j in range(rows):
            a = i * (rows + 1) + j
            faces.append((a, a + rows + 1, a + rows + 2, a + 1))
    face = new_mesh_object("FrontFace", verts, faces)
    set_point_attribute(face, "w", weight)
    scatter(face, collection_of("FaceSrc", _sources("Face", 12, 6.0, 2, 0.0, 0.0, seed=13, thick_mm=2.6)),
            name="FacePebbles", min_distance=3.4 * MM, weight_attr="w", seed=13, scale=(0.55, 1.25),
            align="NORMAL", tilt=0.10, keep_surface=False, sink=0.65 * MM)
    made.append(face)


# --- ribbon grass --------------------------------------------------------------------

def _ribbon_material():
    """A ribbon-grass leaf: thin and translucent, pale where it leaves the crown,
    fine veins along it and a paler midrib. Blade to blade the green differs (the
    `tone` attribute), and the oldest blades brown toward the tip."""
    mat, nt, out = new_material("Vallis")
    N, L = nt.nodes, nt.links

    def rmap(v, a, b, c, d, smooth=False):
        n = N.new("ShaderNodeMapRange")
        if smooth:
            n.interpolation_type = "SMOOTHSTEP"
        L.new(v, n.inputs["Value"])
        for key, val in (("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            n.inputs[key].default_value = val
        return n.outputs["Result"]

    def mth(op, a, b=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for k, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                L.new(v, n.inputs[k])
        return n.outputs[0]

    def mix(f, a, b, blend="MIX"):
        n = N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        for key, v in (("Factor", f), ("A", a), ("B", b)):
            if isinstance(v, (int, float)):
                n.inputs[key].default_value = v
            elif isinstance(v, tuple):
                n.inputs[key].default_value = (*v, 1.0)
            else:
                L.new(v, n.inputs[key])
        return n.outputs["Result"]

    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    across, along = sep.outputs["X"], sep.outputs["Y"]
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "tone"
    tone = attr.outputs["Fac"]

    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.0, (0.11, 0.17, 0.05, 1)
    e[1].position, e[1].color = 1.0, (0.15, 0.33, 0.055, 1)
    e.new(0.20).color = (0.042, 0.13, 0.022, 1)
    L.new(along, ramp.inputs["Fac"])
    hsv = N.new("ShaderNodeHueSaturation")
    L.new(rmap(tone, 0.0, 1.0, 0.47, 0.53), hsv.inputs["Hue"])
    L.new(rmap(tone, 0.0, 1.0, 0.65, 1.25), hsv.inputs["Value"])
    L.new(ramp.outputs["Color"], hsv.inputs["Color"])
    col = hsv.outputs["Color"]
    veins = rmap(mth("SINE", mth("MULTIPLY", across, 2 * math.pi * 9)), -1.0, 1.0, 0.84, 1.06)
    rib = rmap(mth("ABSOLUTE", mth("SUBTRACT", across, 0.5)), 0.0, 0.08, 1.25, 1.0)
    col = mix(1.0, col, mth("MULTIPLY", veins, rib), "MULTIPLY")
    old = mth("MULTIPLY", rmap(along, 0.50, 1.0, 0.0, 1.0, True), rmap(tone, 0.10, 0.30, 0.9, 0.0, True))
    col = mix(old, col, (0.17, 0.125, 0.04))

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.26
    L.new(col, bsdf.inputs["Base Color"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(col, trans.inputs["Color"])
    sh = N.new("ShaderNodeMixShader")
    sh.inputs["Fac"].default_value = 0.5
    L.new(bsdf.outputs[0], sh.inputs[1])
    L.new(trans.outputs[0], sh.inputs[2])
    L.new(sh.outputs[0], out.inputs["Surface"])
    return mat


class _Vallis:
    """Every leaf of every plant in one mesh, re-posed each frame. A leaf starts
    up out of the crown, is turned over by the current within a hand's breadth
    and streams away to the right, rising a little; waves run down it toward
    the tip, it twists as it goes, and the whole stand lifts and settles as the
    flow gusts. Leaves stop short of the surface and the glass."""

    SEG = 28

    def __init__(self, name, clumps, seed=4):
        rng = random.Random(seed)
        rows = []
        for ci, (cx, cy, n, lmin, lmax, wmm) in enumerate(clumps):
            for j in range(n):
                az = 2 * math.pi * (j + rng.uniform(-0.3, 0.3)) / n
                rr = rng.uniform(0.5, 3.0)
                bx, by = cx + rr * math.cos(az), cy + rr * math.sin(az)
                lean = rng.uniform(0.08, 0.40)
                rows.append(dict(
                    base=(bx, by, ground(bx, by) - 4.0),
                    L=lmin + (lmax - lmin) * rng.random() ** 0.7,
                    W=wmm * rng.uniform(0.8, 1.15),
                    d0=(math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)),
                    kap=1.0 / rng.uniform(30, 58), buoy=rng.uniform(0.07, 0.30),
                    amp=rng.uniform(0.22, 0.42), kw=2 * math.pi / rng.uniform(55, 100),
                    om=2 * math.pi * rng.uniform(0.35, 0.60),
                    ph=(rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28)),
                    tw0=rng.uniform(0, 6.28), twr=rng.choice((-1, 1)) * rng.uniform(1.5, 6.0),
                    gph=ci * 1.9 + rng.uniform(-0.3, 0.3)))
        arr = lambda key: np.array([r[key] for r in rows], dtype=np.float64)
        self.base, self.L, self.W, self.d0 = arr("base"), arr("L"), arr("W"), arr("d0")
        self.kap, self.buoy, self.amp, self.kw, self.om = arr("kap"), arr("buoy"), arr("amp"), arr("kw"), arr("om")
        self.ph, self.tw0, self.twr, self.gph = arr("ph"), arr("tw0"), arr("twr"), arr("gph")
        B, S = len(rows), self.SEG
        u = np.linspace(0.0, 1.0, S + 1)
        sm = lambda a, b, v: (lambda t: t * t * (3 - 2 * t))(np.clip((v - a) / (b - a), 0.0, 1.0))
        self.profile = (0.55 + 0.45 * sm(0.0, 0.12, u)) * (1.0 - 0.85 * sm(0.88, 1.0, u))

        faces, uvs = [], []
        for b in range(B):
            for i in range(S + 1):
                uvs += [(0.0, i / S), (0.5, i / S), (1.0, i / S)]
            for i in range(S):
                a = (b * (S + 1) + i) * 3
                faces += [(a, a + 1, a + 4, a + 3), (a + 1, a + 2, a + 5, a + 4)]
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0.0, 0.0, 0.0)] * (B * (S + 1) * 3), [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        for loop in me.loops:
            uvl.data[loop.index].uv = uvs[loop.vertex_index]
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.materials.append(_ribbon_material())
        tone = me.attributes.new("tone", "FLOAT", "POINT")
        tone.data.foreach_set("value", np.repeat([rng.random() for _ in range(B)], (S + 1) * 3))
        me.update()
        self.obj = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(self.obj)
        self.pose(0.0)

    def centrelines(self, t):
        """Points (B, S+1, 3) in mm and unit tangents along every leaf at time t (seconds)."""
        B, S = len(self.L), self.SEG
        ds = self.L / S
        p = self.base.copy()
        d = self.d0.copy()
        P = np.empty((B, S + 1, 3))
        F = np.empty((B, S + 1, 3))
        for i in range(S + 1):
            u = i / S
            s = ds * i
            gust = (1.0 + 0.18 * np.sin(0.85 * t + self.gph - 0.006 * p[:, 0])
                    + 0.08 * np.sin(2.1 * t + 1.7 * self.gph + 0.010 * p[:, 0]))
            tgt = np.stack([gust, np.zeros(B), self.buoy], 1)
            tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
            d = d + np.minimum(self.kap * ds, 1.0)[:, None] * (tgt - d)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            a = self.amp * u ** 1.2
            f = d.copy()
            f[:, 2] += a * np.sin(self.kw * s - self.om * t + self.ph[:, 0])
            f[:, 1] += (0.7 * a * np.sin(0.8 * self.kw * s - 0.83 * self.om * t + self.ph[:, 1])
                        + 0.10 * u * np.sin(0.45 * t + self.ph[:, 2]))
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            # under the surface, inside the glass
            fz = np.minimum(f[:, 2], np.clip((SURFACE_MM - 7.0 - p[:, 2]) / ds, -0.3, 1.0))
            fy = np.where((p[:, 1] > 283.0) & (f[:, 1] > 0), 0.0, f[:, 1])
            fx = np.where(p[:, 0] > 284.0, np.minimum(f[:, 0], 0.0), f[:, 0])
            h = np.sqrt(np.clip(1.0 - fz * fz, 0.0, 1.0)) / np.maximum(np.hypot(fx, fy), 1e-6)
            f = np.stack([fx * h, fy * h, fz], 1)
            P[:, i] = p
            F[:, i] = f
            p = p + f * ds[:, None]
        return P, F

    def pose(self, t):
        P, F = self.centrelines(t)
        B, S = len(self.L), self.SEG
        u = np.linspace(0.0, 1.0, S + 1)[None, :]
        n1 = np.stack([-F[..., 2], np.zeros_like(F[..., 0]), F[..., 0]], -1)          # F x Y
        n1 /= np.maximum(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-6)
        n2 = np.cross(F, n1)
        s = u * self.L[:, None]
        psi = (self.tw0[:, None] + self.twr[:, None] * u
               + 0.5 * u * np.sin(1.3 * self.kw[:, None] * s - 1.1 * self.om[:, None] * t + self.ph[:, 3:4]))
        wv = np.cos(psi)[..., None] * n2 + np.sin(psi)[..., None] * n1
        nr = np.cross(F, wv)
        half = (0.5 * self.W[:, None] * self.profile[None, :])[..., None]
        co = np.stack([P - wv * half, P - nr * half * 0.12, P + wv * half], 2)        # (B, S+1, 3 across, 3)
        self.obj.data.vertices.foreach_set("co", (co * MM).astype(np.float32).ravel())
        self.obj.data.update()
        return co


def _branch_material():
    """Wood that has been in the river a long time: bark gone, grey-brown, smooth."""
    # the kit's driftwood stretches its grain along Z, for standing wood; this piece lies along X
    return materials.pbr_box("RiverBranch", "bark_brown_02", scale=16.0, stretch=(0.30, 1.0, 1.0), tint=(0.66, 0.53, 0.41),
                             saturation=0.5, value=0.85, bump=0.5, bump_distance=0.0008, roughness=0.7, use_ao=False)


# --- build ---------------------------------------------------------------------------

def build(ctx):
    made = []

    bed = substrate.build_bed(ground, "RiverBed")
    bed.obj.data.materials.append(_bed_material())
    made.append(bed.obj)
    _dress_bed(bed, made)

    stones = _place_stones()
    made += [s["obj"] for s in stones]

    vallis = _Vallis("Vallis", VALLIS)
    made.append(vallis.obj)

    (bx, by), (rx, ry, rlift), blen, brad = BRANCH
    butt = Vector(bed.at(bx, by, -1.5))
    skin = wood.grow(butt, (Vector(bed.at(rx, ry, rlift)) - butt).normalized(), blen, brad, seed=6, depth=1,
                     children=(2, 2), wander=0.14, child_length=0.26, child_radius=0.5, spread=(0.5, 0.9), taper=0.32)
    branch = wood.finish(skin, "Branch", _branch_material(), subdivide=1)
    branch.modifiers["Knots"].strength = 0.0012          # the kit's knots are sized for trunks
    zs = [v[2] / MM for v in skin.verts]
    print(f"[branch] x {min(v[0] for v in skin.verts) / MM:.0f}..{max(v[0] for v in skin.verts) / MM:.0f} "
          f"y {min(v[1] for v in skin.verts) / MM:.0f}..{max(v[1] for v in skin.verts) / MM:.0f} z {min(zs):.0f}..{max(zs):.0f}")
    made.append(branch)

    # The nerite grazes the pale stone at the front left, side-on to the camera.
    host = next(s for s in stones if s["name"] == SNAIL_STONE)
    host_bvh = BVHTree.FromPolygons([Vector(v) * MM for v in host["co"].tolist()], host["faces"])
    nerite = snail.Snail("Nerite")
    made.append(nerite.obj)

    # The film's one shoal; SWIM above says where it may go in this scape. These
    # are the measured stones and plants, printed so SWIM's cylinders can be kept true.
    print("[stones] " + ", ".join(f"{s['name']} ({s['x']}, {s['y']}, {s['r']:.0f}, {s['top']:.0f}; {s['above']:.0f} above bed)"
                                  for s in stones if s["above"] > 14))
    for t in (57.0, 61.8, 66.0):
        co = vallis.pose(t).reshape(-1, 3)
        print(f"[vallis] t={t}: x {co[:, 0].min():.0f}..{co[:, 0].max():.0f} y {co[:, 1].min():.0f}..{co[:, 1].max():.0f} "
              f"z ..{co[:, 2].max():.0f}")
    school = fish.School()
    made += school.objects

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["vallis"] = vallis
    ctx["snail"] = (nerite, host, host_bvh)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    t = frame / 30.0
    ctx["vallis"].pose(t)

    # the snail creeps upstream across the front shoulder of its stone, well under a millimetre a second
    nerite, host, bvh = ctx["snail"]
    x = host["x"] + 3.0 - 0.6 * (t - STILL_T)
    y = host["y"] - 7.0
    hit, normal, _, _ = bvh.ray_cast(Vector((x * MM, y * MM, 0.30)), Vector((0, 0, -1)))
    if hit is not None:
        nerite.pose(hit - normal * 0.3 * MM, yaw=math.pi, t=t, up=tuple(normal))
