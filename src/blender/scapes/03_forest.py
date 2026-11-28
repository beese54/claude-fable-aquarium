"""Scape 3: Forest.

A miniature woodland seen from its edge. A pale sand path enters right of
centre at the front glass and winds back between driftwood trees until it
disappears behind them. Every tree is its own individual, grown by code with
its own habit: an old thick tree that forks low under a broad cushion of moss,
a tall sparse one with a small fine-leaved crown, a young sapling carrying
round-leaved epiphytes, a loner on the right leaning out across the path under
flat fern-like sprays, and two smaller ones far back in olive and fresh green.
Three of them stand together on the left; the right is lighter. The bark is
fluted, knotted and flared into its roots. The front of the tank is left as
open water for the shoal. Coordinates are millimetres: X left to right, Y front
to back.
"""
import math
import random

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import assets
import fish
import materials
import plants
import shrimp
import snail
import substrate
import wood
from bl_common import MM, shot_frames
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "03_forest"

# Water the film's shoal may use while this scape is on screen (src/sim/school.py).
# The front two thirds are open water from the carpet to well above mid height;
# the trees are covered by the cylinders a build prints as [trees] (trunks
# standing on the bed, crowns hanging in the water).
SWIM = {
    "box": (36, 264, 26, 168, 118, 206),
    "floor": (82, 114),
    "obstacles": [
        (90, 157, 29, 119), (89, 155, 19, 138, 110), (101, 148, 30, 160, 132), (87, 156, 64, 195, 154), (73, 169, 61, 212, 176), (94, 147, 51, 225, 198), (100, 129, 4, 229, 220),     # Tree_old
        (122, 217, 17, 120), (125, 221, 12, 142, 114), (128, 225, 11, 164, 136), (134, 238, 33, 186, 158), (140, 241, 37, 213, 180), (135, 237, 34, 235, 202), (125, 236, 15, 245, 224),     # Tree_tall
        (259, 122, 13, 115), (252, 125, 24, 141, 109), (261, 114, 24, 170, 131), (257, 113, 20, 170, 153),     # Tree_young
        (236, 181, 29, 127), (219, 179, 25, 145, 117), (199, 165, 48, 182, 139), (187, 175, 55, 209, 161), (182, 169, 50, 218, 183), (182, 158, 19, 218, 205),     # Tree_lean
        (198, 263, 11, 128), (198, 264, 9, 150, 122), (189, 265, 18, 173, 144), (192, 264, 27, 194, 166), (197, 272, 16, 218, 188), (192, 270, 15, 227, 210),     # Tree_far_r
        (59, 257, 24, 132), (52, 267, 20, 153, 126), (57, 267, 26, 181, 148), (61, 254, 33, 203, 170), (60, 243, 36, 218, 192),     # Tree_far_l
    ],
}

# The path's centre line as (y, x): it enters right of centre, swings left
# between the trees, then bends back right and fades out behind the far ones.
PATH = [(-40, 200), (0, 190), (50, 178), (100, 162), (150, 148), (195, 150), (235, 165), (270, 172), (320, 166)]


def _tree(name, x, y, h, r, lean, seed, **kw):
    """One tree's habit. h is the height above the soil, r the trunk radius (mm).
    trunk          length of the bare trunk before it divides, as a fraction of h
    leaders        what the trunk divides into: (heading deg, angle off the trunk deg, length / h, radius / trunk top)
    up             how far the leaders turn back toward vertical (a leaning trunk's crown still reaches up)
    reach          widest the wood may spread from the foot, as a fraction of h
    crown          where on the tree (fraction of h) foliage starts
    clump          size of the gaps in the foliage: >1 finer, <1 bigger cushions
    foliage        (layer, share) of the crown
    epi            spots of epiphytes: (fraction along the trunk, radius mm)"""
    d = dict(name=name, x=x, y=y, h=h, r=r, lean=lean, seed=seed,
             trunk=0.3, trunk_taper=0.7, trunk_wander=0.3, trunk_gravity=-0.04, up=0.0,
             leaders=[(0, 0, 0.7, 0.85)], depth=3, children=(2, 3), spread=(0.7, 1.1),
             child_length=0.64, child_radius=0.6, gravity=-0.07, wander=0.5, taper=0.32,
             reach=0.42, crown=0.48, clump=1.0, gap=0.15, foliage=[("moss_mid", 1.0)], epi=[], epi_n=10, roots=(3, 5))
    d.update(kw)
    return d


TREES = [
    # the old one: thick, short trunk forking low into two leaders, broad cushions of deep moss
    _tree("Tree_old", 90, 158, 136, 11.5, (-0.06, 0.05), 3, trunk=0.27, trunk_taper=0.72, trunk_wander=0.45,
          leaders=[(170, 34, 0.52, 0.74), (30, 14, 0.6, 0.68)], depth=3, children=(2, 3), spread=(0.6, 1.05),
          reach=0.44, crown=0.5, clump=0.65, gap=0.08,
          foliage=[("moss_deep", 0.7), ("moss_olive", 0.3)], epi=[(1.0, 13), (0.55, 7)], epi_n=12, roots=(4, 6)),
    # tall and sparse: a long bare stem, a few short branches and a small fine-leaved crown
    _tree("Tree_tall", 120, 216, 146, 5.2, (0.05, -0.02), 15, trunk=0.56, trunk_taper=0.6, trunk_wander=0.25,
          trunk_gravity=-0.02, leaders=[(20, 38, 0.3, 0.78), (200, 52, 0.27, 0.62), (115, 58, 0.2, 0.5)],
          depth=2, children=(2, 3), spread=(0.6, 1.2), child_length=0.6, gravity=-0.04, reach=0.36, crown=0.62,
          clump=1.4, gap=0.18, foliage=[("fine", 1.0)]),
    # the young one: a thin sapling at the front with round-leaved epiphytes for a crown
    _tree("Tree_young", 262, 124, 70, 3.0, (-0.12, -0.04), 41, trunk=0.36, trunk_taper=0.8,
          leaders=[(-25, 18, 0.55, 0.85), (150, 42, 0.34, 0.62)], depth=2, children=(2, 3), spread=(0.6, 1.1),
          gravity=-0.08, reach=0.5, crown=0.55, clump=1.3, gap=0.25, foliage=[("epi", 1.0)], epi_n=46, roots=(3, 4)),
    # the loner: leans out from the right over the path, curving up into flat tiers of fern-like moss
    _tree("Tree_lean", 248, 180, 110, 7.6, (-0.78, 0.12), 12, trunk=0.78, trunk_taper=0.6, trunk_wander=0.3,
          trunk_gravity=-0.11, up=0.9, leaders=[(180, 24, 0.42, 0.72), (95, 48, 0.3, 0.55), (-70, 40, 0.26, 0.5)],
          depth=2, children=(2, 3), spread=(0.85, 1.35), child_length=0.62, gravity=0.0, reach=0.9, crown=0.55,
          clump=0.9, gap=0.12, foliage=[("fern", 1.0)], epi=[(0.42, 8)], epi_n=6),
    # far right: small, upright, olive moss
    _tree("Tree_far_r", 198, 262, 124, 3.9, (0.06, 0.03), 33, trunk=0.55,
          leaders=[(90, 15, 0.5, 0.8), (250, 40, 0.34, 0.6)], depth=3, children=(2, 3), spread=(0.7, 1.15),
          reach=0.4, crown=0.5, clump=1.1, foliage=[("moss_olive", 1.0)]),
    # far back behind the cluster: mid green moss mixed with fine growth
    _tree("Tree_far_l", 58, 256, 104, 4.2, (0.1, 0.04), 27, trunk=0.18,
          leaders=[(0, 0, 0.76, 0.9)], depth=3, children=(2, 3), spread=(0.75, 1.15),
          reach=0.38, crown=0.46, clump=1.2, foliage=[("moss_mid", 0.6), ("fine", 0.4)]),
]

# foliage layers: points across all trees, most points per m2 of fully covered bark, size range
LAYERS = {
    "moss_deep":  dict(points=24000, cap=1.8e6, scale=(1.0, 1.9), tilt=0.5),
    "moss_olive": dict(points=16000, cap=2.2e6, scale=(0.9, 1.6), tilt=0.5),
    "moss_mid":   dict(points=12000, cap=2.0e6, scale=(0.9, 1.6), tilt=0.5),
    "fine":       dict(points=16000, cap=7.0e6, scale=(0.8, 1.3), tilt=0.5),
    "fern":       dict(points=5000,  cap=0.7e6, scale=(0.8, 1.25), tilt=0.3, align="UP"),
}
FROND_MM = {"moss_deep": 14, "moss_olive": 10, "moss_mid": 11, "fine": 11, "fern": 20, "epi": 14}

# model, x, y, longest side, how the scan is stood up, (lean, lean, heading), fraction buried
STONES = [
    ("moon_rock_03", 136, 40,  28, (0, 0, 0),  (0, -6, 30),   0.35),
    ("moon_rock_07", 232, 66,  20, (0, 0, 0),  (4, 0, 120),   0.40),
    ("moon_rock_04", 108, 150, 22, (0, 0, 0),  (0, 5, 250),   0.38),
    ("moon_rock_03", 62,  114, 15, (90, 0, 0), (0, 0, 80),    0.40),
    ("moon_rock_07", 240, 108, 22, (0, 90, 0), (-5, 0, 200),  0.42),
    ("moon_rock_04", 226, 172, 17, (0, 0, 0),  (0, 0, 10),    0.45),
    ("moon_rock_04", 168, 116, 11, (0, 0, 0),  (0, 0, 300),   0.40),
]


def _smooth(e0, e1, v):
    t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def path_x(y):
    """Centre of the path at depth y: Catmull-Rom through PATH, so the curve has no kinks."""
    for k in range(1, len(PATH) - 2):
        y1, y2 = PATH[k][0], PATH[k + 1][0]
        if y <= y2 or k == len(PATH) - 3:
            p0, p1, p2, p3 = (PATH[k + i][1] for i in (-1, 0, 1, 2))
            t = max(0.0, min(1.0, (y - y1) / (y2 - y1)))
            return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
    return PATH[-1][1]


def path_half(y):
    """Half-width: 38 mm at the glass, narrowing as it recedes, with a wavering edge."""
    w = 38.0 * (1 - 0.76 * min(1.0, max(0.0, y) / 265) ** 0.85)
    return w * (1 + 0.10 * math.sin(y * 0.071 + 1.0) + 0.06 * math.sin(y * 0.19 + 2.3))


def _edge_noise(x, y):
    """A few millimetres of in-and-out along the path's edge so it is not ruled."""
    return (2.6 * math.sin(x * 0.21 + 1.3 * math.sin(y * 0.13)) * math.cos(y * 0.17 + 0.4)
            + 1.4 * math.sin(x * 0.57 + y * 0.43) + 0.8 * math.sin(x * 1.3 - y * 0.9)
            + 0.6 * math.sin(x * 2.3 + y * 1.7))


def path_dist(x, y):
    """Signed distance in mm from the path's edge: negative on the sand."""
    return abs(x - path_x(y)) - path_half(y) + _edge_noise(x, y)


def path_fac(x, y, soft=2.5):
    """1 on the sand, 0 on soil; the far end fades out as the path disappears."""
    f = 1.0 - _smooth(-soft, soft, path_dist(x, y))
    return f * (1.0 - _smooth(255, 285, y))


def sand_depth(x, y):
    """How deep the sand lies over the soil, mm: a lens, thickest mid-path and
    thinning to nothing at the edges, so the cut at the front glass is a
    wedge of sand settling into the soil, not a block."""
    d = -path_dist(x, y)
    lens = 8.5 * _smooth(-2.0, 18.0, d) ** 0.8
    return lens * (0.7 + 0.5 * _patch(x * 1.7, y * 1.7, 4.0)) * (1.0 - _smooth(240, 280, y))


def ground(x, y):
    """Bed height in mm. Deep enough at the front glass to show a band above the
    tray, rising toward the back and sides, with the path sunk a little into
    it and soil heaped round each trunk."""
    side = ((x - 150) / 145) ** 2
    h = 67 + 34 * (max(y, 0) / 295) ** 1.2 + 14 * side * (0.5 + 0.5 * y / 295)
    valley = 1.0 - _smooth(-8, 22, path_dist(x, y))
    h -= (3.0 + 2.5 * _smooth(20, 90, y)) * valley * (1.0 - _smooth(240, 285, y))
    for t in TREES:
        h += (4.0 + 0.25 * t["r"]) * math.exp(-((x - t["x"]) ** 2 + (y - t["y"]) ** 2) / (12 + 1.6 * t["r"]) ** 2)
    h += 1.8 * math.sin(x * 0.05 + 1.3) * math.cos(y * 0.041) + 0.8 * math.sin(x * 0.17) * math.sin(y * 0.13 + 0.5)
    return h


def _patch(x, y, a=0.0):
    """Low-frequency patchiness, 0..1."""
    n = math.sin(x * 0.061 + a + 1.7 * math.sin(y * 0.049)) * math.cos(y * 0.067 + 1.3 * math.sin(x * 0.041 + a))
    n2 = math.sin(x * 0.23 + 0.5 + a) * math.sin(y * 0.21 + 1.1)
    return max(0.0, min(1.0, 0.5 + 0.45 * n + 0.2 * n2))


def _paint_all(bed, name, fn):
    """Like Bed.set_weight but over every vertex, the skirt included, so the
    cut face at the glass can be painted too. fn takes (x, y, depth below the top) in mm."""
    co = [v.co for v in bed.obj.data.vertices]
    set_point_attribute(bed.obj, name, [fn(c.x / MM, c.y / MM, ground(c.x / MM, c.y / MM) - c.z / MM) for c in co])


def _split_skirt(bed, cuts=30):
    """The kit's bed drops each border vertex straight to the floor in one
    face. Cut those faces into rows so the face at the glass can carry a
    painted layering (sand over soil) and granules in it."""
    me = bed.obj.data
    n_top = len(bed.top_xy_mm)
    bm = bmesh.new()
    bm.from_mesh(me)
    tall = [e for e in bm.edges
            if abs(e.verts[0].co.x - e.verts[1].co.x) < 1e-7 and abs(e.verts[0].co.y - e.verts[1].co.y) < 1e-7
            and abs(e.verts[0].co.z - e.verts[1].co.z) > 3 * MM]
    bmesh.ops.subdivide_edges(bm, edges=tall, cuts=cuts, use_grid_fill=True)
    bm.to_mesh(me)
    bm.free()
    me.update()
    # new vertices come after the old, so the top surface keeps its order
    for i in range(0, n_top, max(1, n_top // 50)):
        (x, y), v = bed.top_xy_mm[i], me.vertices[i]
        assert abs(v.co.x / MM - x) < 1e-3 and abs(v.co.y / MM - y) < 1e-3, "bed vertex order changed"
    set_point_attribute(bed.obj, "all", [1.0] * len(me.vertices))


# --- materials ------------------------------------------------------------------

def _floor_material():
    """One material for the whole bed: the white sand scan where the painted
    `path` attribute is 1, dark soil elsewhere, with a noisy, uneven join.
    Below the surface the sand is a layer `sand_mm` deep with a ragged floor."""
    mat = materials.white_sand("ForestFloor")
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    sand = bsdf.inputs["Base Color"].links[0].from_socket
    coord = N.new("ShaderNodeTexCoord")

    def attr(name):
        a = N.new("ShaderNodeAttribute")
        a.attribute_type = "GEOMETRY"
        a.attribute_name = name
        return a.outputs["Fac"]

    def math_node(op, a, b=None, c=None):
        m = N.new("ShaderNodeMath")
        m.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    def map_range(v, a, b, c=0.0, d=1.0):
        m = N.new("ShaderNodeMapRange")
        m.inputs["From Min"].default_value = a
        m.inputs["From Max"].default_value = b
        m.inputs["To Min"].default_value = c
        m.inputs["To Max"].default_value = d
        L.new(v, m.inputs["Value"])
        return m.outputs["Result"]

    def noise(scale, detail=4.0, rough=0.6):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        L.new(coord.outputs["Object"], n.inputs["Vector"])
        return n.outputs["Fac"]

    brk = noise(160.0, 6.0, 0.65)
    fac = map_range(math_node("MULTIPLY_ADD", brk, 0.7, attr("path")), 0.62, 0.92)

    # depth below the surface here, mm, against the sand's depth with a ragged floor
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sep.inputs[0])
    depth = math_node("MULTIPLY_ADD", sep.outputs["Z"], -1000.0, attr("top_mm"))
    rag = math_node("MULTIPLY_ADD", noise(260.0, 5.0, 0.7), 7.0, -3.5)
    below = math_node("SUBTRACT", math_node("ADD", depth, rag), attr("sand_mm"))
    layer = map_range(below, -0.8, 0.8, 1.0, 0.0)
    surface = map_range(depth, 0.4, 1.6, 1.0, 0.0)          # the top itself follows the path alone
    sand_fac = math_node("MULTIPLY", fac, math_node("MAXIMUM", layer, surface))

    # soil between the granules: near black, a little uneven
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.012, 0.009, 0.006, 1)
    ramp.color_ramp.elements[1].color = (0.050, 0.036, 0.024, 1)
    L.new(noise(420.0, 3.0), ramp.inputs["Fac"])
    # where sand has settled into soil the two mix: a greyed band at the join
    mixed = N.new("ShaderNodeMix")
    mixed.data_type = "RGBA"
    mixed.inputs["Factor"].default_value = 0.35
    L.new(ramp.outputs["Color"], mixed.inputs["A"])
    L.new(sand, mixed.inputs["B"])
    join = math_node("MULTIPLY", fac, map_range(below, 0.0, 4.0, 1.0, 0.0))
    join = math_node("MULTIPLY", join, map_range(noise(900.0, 2.0), 0.45, 0.6))

    soil = N.new("ShaderNodeMix")
    soil.data_type = "RGBA"
    L.new(join, soil.inputs["Factor"])
    L.new(ramp.outputs["Color"], soil.inputs["A"])
    L.new(mixed.outputs["Result"], soil.inputs["B"])
    # against the glass the sand is packed, damp and shaded: greyer and darker than on top
    packed = N.new("ShaderNodeMix")
    packed.data_type = "RGBA"
    packed.blend_type = "MULTIPLY"
    L.new(map_range(depth, 1.0, 3.0, 0.0, 1.0), packed.inputs["Factor"])
    L.new(sand, packed.inputs["A"])
    packed.inputs["B"].default_value = (0.62, 0.60, 0.57, 1.0)
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L.new(sand_fac, mix.inputs["Factor"])
    L.new(soil.outputs["Result"], mix.inputs["A"])
    L.new(packed.outputs["Result"], mix.inputs["B"])
    L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    L.new(map_range(sand_fac, 0.0, 1.0, 0.88, 0.55), bsdf.inputs["Roughness"])
    return mat


def _granules(bed, material, size_mm, weight, seed, name, spacing=0.72):
    grains = [substrate._granule(f"{name}_{k}", size_mm * MM / 2, seed * 10 + k, material) for k in range(4)]
    col = collection_of(f"{name}_src", grains)
    return scatter(bed.obj, col, name=name, min_distance=size_mm * MM * spacing, seed=seed, weight_attr=weight,
                   scale=(0.6, 1.15), align="NORMAL", tilt=math.pi, sink=size_mm * MM * 0.2)


# --- foliage sources -------------------------------------------------------------

def _rosettes(name, variants, seed, leaf_mm, width, leaves, material, petiole=(2.0, 5.0), rise=(0.35, 1.0),
              belly=0.45, fold=0.18):
    """Rhizome epiphytes (Anubias nana petite, Bucephalandra): a short creeping
    rhizome with a rosette of stalked leaves held up and out from it."""
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        rl = rng.uniform(4, 9) * MM
        rh = [(-rl / 2 + rl * t, 0.0, (0.5 + 0.3 * math.sin(t * 3)) * MM) for t in (0, 0.33, 0.66, 1.0)]
        parts.append(plants._tube_strip(rh, [0.9 * MM, 0.85 * MM, 0.8 * MM, 0.55 * MM], sides=5))
        for _ in range(rng.randint(*leaves)):
            ox = -rl / 2 + rl * rng.uniform(0.1, 1.0)
            az = rng.uniform(0, 2 * math.pi)
            el = rng.uniform(*rise)
            pl = rng.uniform(*petiole) * MM
            dp = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))
            o0 = (ox, 0.0, 0.6 * MM)
            o1 = tuple(o0[i] + dp[i] * pl for i in range(3))
            parts.append(plants._tube_strip([o0, tuple((o0[i] + o1[i]) / 2 for i in range(3)), o1],
                                            [0.42 * MM, 0.36 * MM, 0.3 * MM], sides=3))
            el2 = el * 0.45 + rng.uniform(-0.2, 0.2)
            d = (math.cos(az) * math.cos(el2), math.sin(az) * math.cos(el2), math.sin(el2))
            up = (-math.cos(az) * math.sin(el2), -math.sin(az) * math.sin(el2), math.cos(el2))
            ln = rng.uniform(*leaf_mm) * MM
            parts.append(plants._leaf(o1, d, up, ln, ln * rng.uniform(*width), 0.3, 1.0, fold=fold,
                                      curl=rng.uniform(0.04, 0.22), belly=belly, segs=6))
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"{name}_{v}", verts, faces, uvs, material))
    return out


def _fern_sprays(name, variants, seed, length_mm, material):
    """Christmas moss: flat, triangular, feathery sprays held out level and
    drooping at the tips, so a crown of it builds up in tiers."""
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for _ in range(rng.randint(1, 3)):
            az = rng.uniform(-0.7, 0.7)
            Lm = rng.uniform(*length_mm) * MM
            rise, droop = rng.uniform(0.05, 0.3), rng.uniform(0.12, 0.32)
            segs = 8
            pts = [(Lm * t * math.cos(az), Lm * t * math.sin(az), Lm * (rise * t - droop * t * t) - 0.3 * MM)
                   for t in (k / segs for k in range(segs + 1))]
            parts.append(plants._tube_strip(pts, [(0.24 - 0.17 * k / segs) * MM for k in range(segs + 1)]))
            m = rng.randint(8, 11)
            for j in range(1, m):
                t = j / m
                o = pts[min(segs, int(round(t * segs)))]
                side = 1 if j % 2 else -1
                pa = az + side * rng.uniform(0.85, 1.15)
                pl = Lm * (0.48 * (1 - t) ** 0.8 + 0.07)
                d = Vector((math.cos(pa), math.sin(pa), rng.uniform(-0.12, 0.04) - droop * 0.3)).normalized()
                parts.append(plants._leaf(o, tuple(d), (0, 0, 1), pl, 1.0 * MM, t * 0.7, min(1.0, t + 0.15),
                                          fold=0.05, curl=0.12, belly=0.5, segs=3))
                # pinnules on the longer side branches
                for q in range(1, int(pl / (1.6 * MM))):
                    tq = q * 1.6 * MM
                    oq = (o[0] + d.x * tq, o[1] + d.y * tq, o[2] + d.z * tq)
                    qa = pa + (1 if q % 2 else -1) * 1.0
                    qd = (math.cos(qa), math.sin(qa), -0.1)
                    parts.append(plants._leaf(oq, qd, (0, 0, 1), max(0.6 * MM, 0.32 * (pl - tq)), 0.6 * MM,
                                              t * 0.7, min(1.0, t + 0.2), fold=0.0, curl=0.1, belly=0.5, segs=2))
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"{name}_{v}", verts, faces, uvs, material))
    return out


def _fine_fronds(name, variants, seed, length_mm, material):
    """Fine-leaved moss (flame / weeping kinds): long thin shoots, rising and
    twisting a little, clothed in needle leaves a millimetre or two long, so a
    crown of it is a soft haze rather than a set of leaves."""
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for _ in range(rng.randint(2, 4)):
            az = rng.uniform(0, 2 * math.pi)
            tilt = rng.uniform(0.15, 0.8)
            Lm = rng.uniform(*length_mm) * MM
            curl = rng.uniform(-0.35, 0.35)
            segs = 6
            pts = []
            for k in range(segs + 1):
                t = k / segs
                a = az + curl * t * 3
                r = math.sin(tilt) * Lm * t
                pts.append((math.cos(a) * r, math.sin(a) * r, math.cos(tilt) * Lm * t - 0.4 * MM))
            parts.append(plants._tube_strip(pts, [(0.16 - 0.1 * k / segs) * MM for k in range(segs + 1)]))
            n = int(Lm / (0.45 * MM))
            for j in range(1, n):
                t = j / n
                i = min(segs - 1, int(t * segs))
                f = t * segs - i
                o = tuple(pts[i][c] * (1 - f) + pts[i + 1][c] * f for c in range(3))
                la = rng.uniform(0, 2 * math.pi)
                d = (math.cos(la) * 0.8, math.sin(la) * 0.8, 0.55)
                parts.append(plants._leaf(o, d, (0, 0, 1), rng.uniform(1.0, 2.0) * MM * (1 - 0.4 * t), 0.32 * MM,
                                          t * 0.8, t, fold=0.0, curl=0.1, belly=0.4, segs=2))
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"{name}_{v}", verts, faces, uvs, material))
    return out


def _foliage_sources():
    leaf = plants.leaf_material
    src = {
        # big soft cushions, deep blue-green
        "moss_deep": plants.moss_fronds(variants=6, seed=13, length_mm=(6, 12), material=leaf(
            "MossDeep", base=(0.007, 0.030, 0.010), tip=(0.035, 0.135, 0.040), translucency=0.3, roughness=0.5)),
        # smaller, olive to brownish
        "moss_olive": plants.moss_fronds(variants=5, seed=29, length_mm=(4, 8), material=leaf(
            "MossOlive", base=(0.030, 0.032, 0.008), tip=(0.17, 0.17, 0.035), translucency=0.3, roughness=0.55)),
        # mid green with fresh tips
        "moss_mid": plants.moss_fronds(variants=5, seed=37, length_mm=(5, 9), material=leaf(
            "MossMid", base=(0.020, 0.060, 0.012), tip=(0.13, 0.32, 0.05), translucency=0.35, roughness=0.45)),
        # very fine, pale yellow-green
        "fine": _fine_fronds("FineMoss", variants=6, seed=51, length_mm=(5.0, 10.0), material=leaf(
            "MossFine", base=(0.045, 0.085, 0.013), tip=(0.30, 0.42, 0.07), translucency=0.42, roughness=0.42)),
        # flat fern-like sprays, a cooler green
        "fern": _fern_sprays("Xmas", variants=6, seed=61, length_mm=(13, 22), material=leaf(
            "MossXmas", base=(0.006, 0.034, 0.020), tip=(0.045, 0.17, 0.085), translucency=0.35, roughness=0.45)),
    }
    anubias = _rosettes("AnubiasPetite", 4, 71, (7, 12), (0.55, 0.72), (4, 7), leaf(
        "Anubias", base=(0.008, 0.028, 0.008), tip=(0.03, 0.085, 0.020), translucency=0.12, roughness=0.22,
        variation=0.15), belly=0.42)
    buce = _rosettes("Bucephalandra", 3, 83, (8, 13), (0.28, 0.38), (5, 8), leaf(
        "Buce", base=(0.030, 0.028, 0.012), tip=(0.075, 0.095, 0.028), translucency=0.15, roughness=0.3,
        variation=0.3), petiole=(1.0, 2.5), rise=(0.6, 1.25), belly=0.5, fold=0.3)
    src["epi"] = anubias + buce
    return {k: collection_of(f"Src_{k}", v) for k, v in src.items()}


# --- trees ----------------------------------------------------------------------

def _resample(pts, radii, n):
    """n + 1 points along a Catmull-Rom curve through pts, radii linear between them."""
    m = len(pts) - 1
    P, R = [], []
    for j in range(n + 1):
        u = m * j / n
        i = min(int(u), m - 1)
        t = u - i
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, m)]
        P.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
        R.append(radii[i] * (1 - t) + radii[i + 1] * t)
    return P, R


class _Skin(wood._Skin):
    """The kit's tube skin, rebuilt finer: rings every millimetre or two, more
    sides on thick wood, frames carried along the branch so it never twists,
    and for every vertex the centre-line point under it and where it lies on
    its branch, so bark relief can be laid on afterwards."""

    def __init__(self, k=1.0):
        super().__init__(8)
        self.k = k              # the scale the skin will be enlarged by, so sizes are chosen in final mm
        self.kind = "trunk"
        self.centres, self.meta = [], []
        self.nb = 0

    def tube(self, pts, radii):
        pts = [Vector(p) for p in pts]
        k = self.k
        length = sum((b - a).length for a, b in zip(pts, pts[1:])) / MM * k
        r0 = radii[0] / MM * k
        n = max(len(pts) - 1, int(length / min(3.0, max(1.1, 0.5 * r0))))
        P, R = _resample(pts, list(radii), n)
        s = max(8, min(44, 2 * round(math.pi * r0 / 1.3)))
        off = len(self.verts)
        bid = self.nb
        self.nb += 1
        n1, _ = wood._frame(P[1] - P[0])
        along = 0.0
        for j, (p, r) in enumerate(zip(P, R)):
            t = (P[min(j + 1, n)] - P[max(j - 1, 0)]).normalized()
            n1 = (n1 - t * n1.dot(t)).normalized()
            n2 = t.cross(n1)
            if j:
                along += (P[j] - P[j - 1]).length / MM * k
            for i in range(s):
                ang = 2 * math.pi * i / s
                self.verts.append(tuple(p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * r))
                self.centres.append(p)
                self.meta.append((bid, along, ang, r / MM * k, self.kind))
        for j in range(n):
            for i in range(s):
                a = off + j * s + i
                b = off + j * s + (i + 1) % s
                self.faces.append((a, b, b + s, a + s))
        root = len(self.verts)
        self.verts.append(tuple(P[0]))
        for i in range(s):
            self.faces.append((off + (i + 1) % s, off + i, root))
        tip = len(self.verts)
        self.verts.append(tuple(P[-1] + (P[-1] - P[-2]).normalized() * R[-1] * 0.6))
        base = off + n * s
        for i in range(s):
            self.faces.append((base + i, base + (i + 1) % s, tip))
        self.centres += [P[0], Vector(self.verts[tip])]
        self.meta += [None, None]


def _wood(t, k):
    """Grow the tree's wood with lengths as given and radii divided by k, so
    that enlarging the result by k gives the intended height and trunk."""
    skin = _Skin(k)
    zb = ground(t["x"], t["y"])
    start = Vector((t["x"] * MM, t["y"] * MM, (zb - 9) * MM))
    d = Vector((t["lean"][0], t["lean"][1], 1.0)).normalized()
    H, seed = t["h"], t["seed"]
    wood.grow(start, d, H * t["trunk"], t["r"] / k, seed=seed, depth=0, wander=t["trunk_wander"],
              gravity=t["trunk_gravity"], taper=t["trunk_taper"], skin=skin)
    top, td, tr = skin.tips.pop()
    skin.kind = "limb"
    base_dir = (td + Vector((0, 0, 1)) * t["up"]).normalized()
    for j, (az, tilt, ln, rf) in enumerate(t["leaders"]):
        az, tilt = math.radians(az), math.radians(tilt)
        horiz = Vector((math.cos(az), math.sin(az), 0.0))
        horiz = (horiz - base_dir * horiz.dot(base_dir)).normalized()
        dl = (base_dir * math.cos(tilt) + horiz * math.sin(tilt)).normalized()
        rl = tr * rf
        wood.grow(top - dl * rl * 0.5, dl, H * ln, rl / MM, seed=seed * 31 + j, depth=t["depth"],
                  children=t["children"], wander=t["wander"], gravity=t["gravity"], spread=t["spread"],
                  child_length=t["child_length"], child_radius=t["child_radius"], taper=t["taper"], skin=skin)
    return skin, start, zb, top


def _grow_tree(t):
    """Grow one tree to exactly its height above the soil, kept inside the
    glass, with roots spreading from a flared foot and bark relief laid on."""
    H, R = t["h"], t["r"]
    trial, start, zb, _ = _wood(t, 1.0)
    k = H * MM / (max(v[2] for v in trial.verts) - zb * MM)
    skin, start, zb, fork = _wood(t, k)
    skin.verts = [start + (Vector(v) - start) * k for v in skin.verts]
    skin.centres = [start + (c - start) * k for c in skin.centres]
    fork = start + (fork - start) * k
    tips = [(start + (p - start) * k, dd, r * k) for p, dd, r in skin.tips]

    rng = random.Random(t["seed"] + 101)
    d = Vector((t["lean"][0], t["lean"][1], 1.0)).normalized()
    foot = start + d * ((zb + 3) * MM - start.z) / d.z
    skin.k, skin.kind = 1.0, "root"
    n_roots = rng.randint(*t["roots"])
    a0 = rng.uniform(0, 2 * math.pi)
    root_az = []
    for i in range(n_roots):
        az = a0 + 2 * math.pi * i / n_roots + rng.uniform(-0.45, 0.45)
        root_az.append(az)
        out = Vector((math.cos(az), math.sin(az), 0.0))
        rd = (out * 0.9 + Vector((0, 0, -0.30))).normalized()
        wood.grow(foot + out * R * 0.7 * MM, rd, rng.uniform(18, 38) * (R / 6.0) ** 0.5, R * rng.uniform(0.28, 0.42),
                  seed=t["seed"] * 7 + i, depth=1, children=(1, 2), wander=0.6, gravity=0.22, spread=(0.4, 0.9), skin=skin)
    skin.verts = [Vector(v) for v in skin.verts]

    # A crown no wider than the tree's own reach, so a sprawling seed cannot
    # roof over the scape; then the side that crosses the glass limit is drawn
    # in. Both move the branches' centre lines toward the foot; bark keeps its thickness.
    wood_v = [v for v, m in zip(skin.verts, skin.meta) if m is None or m[4] != "root"]
    reach = max(math.hypot(v.x - start.x, v.y - start.y) for v in wood_v)
    c = min(1.0, t["reach"] * H * MM / reach)
    lo, hi = 24 * MM, 276 * MM
    s = [[c, c], [c, c]]
    for v in skin.verts:
        for a in (0, 1):
            off = v[a] - start[a]
            if start[a] + off * c > hi and off > 0:
                s[a][1] = min(s[a][1], (hi - start[a]) / off)
            elif start[a] + off * c < lo and off < 0:
                s[a][0] = min(s[a][0], (lo - start[a]) / off)

    def squeeze(p):
        q = [start[a] + (p[a] - start[a]) * s[a][p[a] > start[a]] for a in (0, 1)]
        return Vector((q[0], q[1], p[2]))
    new_c = [squeeze(cn) for cn in skin.centres]
    skin.verts = [nc + (v - cn) for v, cn, nc in zip(skin.verts, skin.centres, new_c)]
    skin.centres = new_c
    tips = [(squeeze(p), dd, r) for p, dd, r in tips]
    fork = squeeze(fork)
    _bark_relief(skin, t, zb, root_az)
    trunk_line = sorted(((m[1], cn) for cn, m in zip(skin.centres, skin.meta) if m and m[4] == "trunk"),
                        key=lambda e: e[0])
    return skin, tips, zb, fork, trunk_line


def _bark_relief(skin, t, zb, root_az):
    """Driftwood is not a pipe: shallow flutes that wander and twist along each
    branch, an oval, uneven section, knots with a dimple at their heart, and a
    trunk that flares into buttresses where it meets its roots."""
    R = t["r"]
    par = {}
    for i, (v, cn, m) in enumerate(zip(skin.verts, skin.centres, skin.meta)):
        if m is None:
            continue
        bid, s, ang, r, kind = m
        if bid not in par:
            q = random.Random(t["seed"] * 7919 + bid)
            par[bid] = (q.uniform(0, 1), q.uniform(-0.012, 0.012) * (1 if q.random() < 0.7 else -1) * 4,
                        q.uniform(0, 6.3), q)
        ph, twist, ph2, _ = par[bid]
        rr = max(r, 0.3)
        disp = 0.0
        if rr > 0.8:
            nf = max(3, round(2 * math.pi * rr / 3.6))
            u = nf * (ang + twist * s) / (2 * math.pi) + ph + 0.3 * math.sin(s * 0.11 + ph * 5)
            g = 0.5 - 0.5 * math.cos(2 * math.pi * u)
            disp += (g ** 0.55 - 0.62) * min(0.17 * rr, 1.2)
            disp += 0.07 * rr * math.sin(2 * ang + s * 0.04 + ph2)
            # knots: one cell every few radii along the branch may hold one
            cell = max(12.0, 6.0 * rr)
            c0 = int(s // cell)
            for c in (c0 - 1, c0, c0 + 1):
                if c < 0:
                    continue
                q = random.Random(t["seed"] * 1000003 + bid * 1009 + c)
                if q.random() > 0.5:
                    continue
                sk, ak, hk = (c + q.random()) * cell, q.uniform(0, 2 * math.pi), rr * q.uniform(0.22, 0.45)
                ds = (s - sk) / (1.0 * rr + 1.5)
                da = (ang - ak + math.pi) % (2 * math.pi) - math.pi
                e = math.exp(-ds * ds - (da / 0.6) ** 2)
                dimple = math.exp(-(ds / 0.32) ** 2 - (da / 0.2) ** 2)
                disp += hk * e * (1.0 - 1.5 * dimple)
        scale = (rr + disp) / rr
        if kind == "trunk":
            h = v.z / MM - zb
            if h < 4.5 * R:
                az = math.atan2(v.y - cn.y, v.x - cn.x)
                lob = max(max(0.0, math.cos(az - ra)) ** 3 for ra in root_az) if root_az else 0.0
                scale *= 1.0 + math.exp(-max(h, 0.0) / (0.9 * R)) * (0.3 + 0.9 * lob)
        skin.verts[i] = cn + (v - cn) * scale


def _foliage(t, skin, tips, zb, fork, trunk_line):
    """Per-vertex weights for each foliage layer this tree carries, plus the
    epiphyte spots. Foliage sits on the upper branches only: zero on the trunk,
    rising through the crown, in clumps with real gaps, thin at branch ends."""
    kd = KDTree(len(tips))
    for i, (p, _, _) in enumerate(tips):
        kd.insert(p, i)
    kd.balance()
    rng = random.Random(t["seed"])
    H, q = t["h"], t["clump"]
    tip_k = min(1.0, H / 120.0)
    c0 = t["crown"] + rng.uniform(-0.02, 0.02)
    ph = rng.uniform(0, 6.3)
    spots = []
    for frac, sig in t["epi"]:
        if frac >= 1.0:
            spots.append((fork, sig))
        elif trunk_line:
            spots.append((trunk_line[min(len(trunk_line) - 1, int(frac * len(trunk_line)))][1], sig))
    layers = {name: [0.0] * len(skin.verts) for name, _ in t["foliage"]}
    if "epi" not in layers:
        layers["epi"] = [0.0] * len(skin.verts)
    for i, (v, cn, m) in enumerate(zip(skin.verts, skin.centres, skin.meta)):
        if m is None or m[4] == "root":
            continue
        x, y, z = (c / MM for c in v)
        rel = (z - zb) / H
        up = (v - cn).normalized().z if (v - cn).length > 1e-9 else 0.0
        for p, sig in spots:
            e = math.exp(-((v - p).length / MM / sig) ** 2) * _smooth(-0.1, 0.6, up)
            layers["epi"][i] = max(layers["epi"][i], e)
        if rel < c0 - 0.02:
            continue
        n = math.sin(x * 0.21 * q + ph) * math.sin(y * 0.19 * q + z * 0.17 * q + ph) + 0.5 * math.sin(z * 0.31 * q + x * 0.11 * q - ph)
        patch = t["gap"] + (1 - t["gap"]) * _smooth(-0.55, 0.35, n)
        _, _, dt = kd.find(v)
        w = _smooth(c0, c0 + 0.16, rel) * patch
        if t["foliage"][0][0] == "epi":
            w *= _smooth(-0.5, 0.4, up)
        else:
            w *= _smooth(3.0 * MM * tip_k, 12 * MM * tip_k, dt)
        if len(t["foliage"]) == 1:
            name = t["foliage"][0][0]
            layers[name][i] = max(layers[name][i], w)
            continue
        split = 0.5 + 0.5 * math.sin(x * 0.09 + ph * 2) * math.cos(y * 0.08 + z * 0.07 - ph)
        a = t["foliage"][0][1]
        f = _smooth(a - 0.12, a + 0.12, split)
        layers[t["foliage"][0][0]][i] = w * (1 - f)
        layers[t["foliage"][1][0]][i] = w * f
    return layers


def _weighted_area(skin, w):
    total = 0.0
    for f in skin.faces:
        if len(f) < 3:
            continue
        a, b, c = (Vector(skin.verts[i]) for i in f[:3])
        area = (b - a).cross(c - a).length * (0.5 if len(f) == 3 else 1.0)
        total += area * sum(w[i] for i in f) / len(f)
    return total


def _obstacles(t, skin, layers, zb, band=22.0, margin=4.0):
    """Cylinders covering this tree for SWIM: the lowest band stands on the
    bed, the rest hang in the water; foliage counts with its frond length."""
    pts = []
    for i, (v, cn, m) in enumerate(zip(skin.verts, skin.centres, skin.meta)):
        if m is None or m[4] == "root":
            continue
        grow = max(FROND_MM[n] * w[i] for n, w in layers.items())
        p = v + (v - cn).normalized() * grow * MM if (v - cn).length > 1e-9 else v
        pts.append((p.x / MM, p.y / MM, max(v.z, p.z) / MM, min(v.z, p.z) / MM))
    top = max(p[2] for p in pts)
    out = []
    z0 = zb
    while z0 < top:
        z1 = z0 + band
        sel = [p for p in pts if p[3] < z1 and p[2] > z0]
        if sel:
            cx = sum(p[0] for p in sel) / len(sel)
            cy = sum(p[1] for p in sel) / len(sel)
            r = max(math.hypot(p[0] - cx, p[1] - cy) for p in sel) + margin
            zt = min(top, max(p[2] for p in sel)) + 3
            if not out:
                out.append((round(cx), round(cy), round(r), round(zt)))
            else:
                out.append((round(cx), round(cy), round(r), round(zt), round(z0 - 3)))
        z0 = z1
    return out


# --- livestock helpers ----------------------------------------------------------

def _ground_normal(x, y):
    e = 1.0
    gx = (ground(x + e, y) - ground(x - e, y)) / (2 * e)
    gy = (ground(x, y + e) - ground(x, y - e)) / (2 * e)
    return Vector((-gx, -gy, 1.0)).normalized()


def _perch(obj, verts, weights, pick, lift_mm):
    """A point on the evaluated wood (after smoothing and knots) near the best
    vertex chosen by `pick`, and the surface normal there."""
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(obj, dg)
    best = max(range(len(verts)), key=lambda i: pick(Vector(verts[i]), obj.data.vertices[i].normal, weights[i]))
    v = Vector(verts[best])
    n = obj.data.vertices[best].normal.normalized()
    hit, normal, _, _ = bvh.ray_cast(v + n * 6 * MM, -n, 20 * MM)
    if hit is None:
        hit, normal = v, n
    return hit + normal * lift_mm * MM, normal


# --- build ----------------------------------------------------------------------

def build(ctx):
    made = []

    # bed: soil and sand in one mesh, split by painted attributes
    bed = substrate.build_bed(ground, "ForestFloor")
    _split_skirt(bed)
    bed.obj.data.materials.append(_floor_material())
    _paint_all(bed, "path", lambda x, y, d: path_fac(x, y))
    _paint_all(bed, "top_mm", lambda x, y, d: ground(x, y))
    _paint_all(bed, "sand_mm", lambda x, y, d: sand_depth(x, y))

    def stray(x, y):
        """Clumps of grains carried a little way over the join, both ways."""
        return _smooth(0.42, 0.78, _patch(x * 3.1, y * 3.1, 1.7))

    def soil_w(x, y, d):
        if d > 45:
            return 0.0
        pf = path_fac(x, y, soft=4.0)
        if d < 0.8:
            pd = path_dist(x, y)
            return 1.0 - pf * (1.0 - 0.45 * stray(x + 7, y) * math.exp(min(0.0, pd) / 4.0))
        return 1.0 - pf * (0.88 - 0.88 * _smooth(sand_depth(x, y) - 1.0, sand_depth(x, y) + 2.5, d))

    def sand_w(x, y, d):
        pf = path_fac(x, y, soft=6.0)
        if d < 0.8:
            pd = path_dist(x, y)
            onto = stray(x, y) * math.exp(-max(0.0, pd) / 5.0) * (1.0 - _smooth(255, 285, y))
            return max(pf * (0.3 + 0.35 * _smooth(-14, -2, pd)), 0.75 * onto * _smooth(-3, 1, pd))
        return 0.25 * pf * (1.0 - _smooth(sand_depth(x, y) - 1.5, sand_depth(x, y) + 0.5, d))
    _paint_all(bed, "soil", soil_w)
    _paint_all(bed, "edge_sand", sand_w)
    soil = substrate.soil_material()
    _granules(bed, soil, 2.4, "soil", 5, "SoilGrains")
    pale = substrate.sand_material("SandGrain", colour=(0.66, 0.62, 0.55))
    _granules(bed, pale, 1.3, "edge_sand", 9, "SandGrains", spacing=0.9)
    made.append(bed.obj)

    # trees
    bark = {"Tree_old": materials.dark_bogwood("Bark_old"),
            "Tree_lean": materials.driftwood("Bark_lean", tint=(0.52, 0.40, 0.30), value=0.62)}
    bark_default = materials.dark_bogwood("Bark")
    sources = _foliage_sources()

    grown = []
    for t in TREES:
        skin, tips, zb, fork, trunk_line = _grow_tree(t)
        layers = _foliage(t, skin, tips, zb, fork, trunk_line)
        grown.append((t, skin, zb, layers))
        print(f"[foliage] {t['name']}: " + ", ".join(f"{n} {_weighted_area(skin, w) * 1e6:.0f} mm2" for n, w in layers.items())
              + f"; top {max(v.z for v in skin.verts) / MM - zb:.0f} mm")
    area = {name: 0.0 for name in LAYERS}
    for t, skin, zb, layers in grown:
        for name, w in layers.items():
            if name in area:
                area[name] += _weighted_area(skin, w)
    dens = {name: min(LAYERS[name]["cap"], LAYERS[name]["points"] / max(a, 1e-9)) for name, a in area.items()}

    trees, obstacles = {}, []
    skin_layers = {t["name"]: layers for t, skin, zb, layers in grown}
    for k, (t, skin, zb, layers) in enumerate(grown):
        obj = wood.finish(skin, t["name"], bark.get(t["name"], bark_default), subdivide=1)
        obj.modifiers["Knots"].strength = 0.0014
        total = [max(w[i] for w in layers.values()) for i in range(len(skin.verts))]
        for name, w in layers.items():
            set_point_attribute(obj, f"f_{name}", w)
        trees[t["name"]] = (t, obj, skin, total, zb)
        obstacles += _obstacles(t, skin, layers, zb)
        made.append(obj)
    print("[trees] obstacles: " + repr(obstacles))

    # shrimp perches are found on the bare wood before the foliage goes on
    perches = []
    t, s_obj, s_skin, s_w, zb = trees["Tree_lean"]
    # the leaning trunk's upper side, beyond the epiphytes and below the crown
    length = max(m[1] for m in s_skin.meta if m and m[4] == "trunk")
    free = [0.0 if (m and m[4] == "trunk" and 0.55 * length < m[1] < 0.75 * length and w < 0.01) else 1.0
            for m, w in zip(s_skin.meta, s_w)]
    perches.append(_perch(s_obj, s_skin.verts, free,
                          lambda v, n, w: n.z - (v.y / MM - t["y"]) / 40 - 999 * w, 0.0))
    t, m_obj, m_skin, m_w, zb = trees["Tree_old"]
    perches.append(_perch(m_obj, m_skin.verts, m_w,
                          lambda v, n, w: 2 * -n.y + n.z - (v.y / MM - t["y"]) / 8 - (v.z / MM - zb) / 40 - 999 * (w <= 0.6 or n.z < 0.2), 9.0))

    for k, (t, obj, skin, total, zb) in enumerate(trees.values()):
        size = (t["h"] / 150.0) ** 0.3
        for j, name in enumerate(l for l in ("moss_deep", "moss_olive", "moss_mid", "fine", "fern", "epi")
                                 if any(f[0] == l for f in t["foliage"]) or (l == "epi" and t["epi"])):
            if name == "epi":
                a = _weighted_area(skin, [max(0.0, v) for v in skin_layers[t["name"]]["epi"]])
                scatter(obj, sources[name], name=f"Epi_{t['name']}", density=t["epi_n"] / max(a, 1e-9),
                        weight_attr=f"f_{name}", seed=70 + k, scale=(0.75, 1.15), align="NORMAL", tilt=0.45,
                        sway=(0.03, 30.0, 0.3))
                continue
            L = LAYERS[name]
            scatter(obj, sources[name], name=f"{name}_{t['name']}", density=dens[name], weight_attr=f"f_{name}",
                    seed=40 + 7 * k + j, scale=(L["scale"][0] * size, L["scale"][1] * size),
                    align=L.get("align", "NORMAL"), tilt=L["tilt"], sway=(0.04, 30.0, 0.3))

    # stones along the path's edges and at the trunk bases
    for k, (model, x, y, size, pre, rot, buried) in enumerate(STONES):
        obj = assets.import_model(model)[0]
        obj.name = f"Stone_{k}"
        assets.place(obj, bed.at(x, y), longest_mm=size, pre_rot_deg=pre, rot_deg=rot)
        lo, hi = assets.world_bounds(obj)
        obj.location.z -= (hi.z - lo.z) * buried
        assets.grade_stone(obj, dark=(0.035, 0.034, 0.032), light=(0.36, 0.35, 0.32), vein_amount=0.15,
                           rust=0.3, algae=0.7, seed=k * 1.3)
        made.append(obj)

    # carpet: two greens in patches, thinning to nothing at the path, creeping onto the sand here and there
    def carpet(x, y, a):
        d = path_dist(x, y)
        creep = 5.0 * _patch(x, y, 2.0) - 3.0
        w = _smooth(creep - 2, creep + 9, d) * (0.55 + 0.45 * _smooth(0, 40, d))
        for t in TREES:
            r = t["r"] * 1.6
            w *= 0.35 + 0.65 * _smooth(r + 2, r + 9, math.hypot(x - t["x"], y - t["y"]))
        p = _patch(x, y, 0.0)
        # sprigs lean up to 15 mm, so they start a little in from the glass and leave a soil margin
        wall = _smooth(9, 19, min(x - 5, 295 - x, y - 5, 295 - y))
        return w * wall * (p if a else 1.0 - p) * (0.6 + 0.4 * min(1.0, y / 50))
    bed.set_weight("carpet_a", lambda x, y: carpet(x, y, True))
    bed.set_weight("carpet_b", lambda x, y: carpet(x, y, False))
    sprig_a = collection_of("CarpetSrcA", plants.carpet_sprigs(variants=5, seed=7, material=plants.leaf_material(
        "CarpetFresh", base=(0.035, 0.12, 0.02), tip=(0.15, 0.36, 0.05), translucency=0.4, roughness=0.32)))
    sprig_b = collection_of("CarpetSrcB", plants.carpet_sprigs(variants=5, seed=19, leaf_mm=2.8, material=plants.leaf_material(
        "CarpetDark", base=(0.022, 0.075, 0.014), tip=(0.09, 0.24, 0.035), translucency=0.38, roughness=0.32)))
    scatter(bed.obj, sprig_a, name="CarpetA", min_distance=5.0 * MM, weight_attr="carpet_a", seed=31,
            scale=(0.6, 1.15), align="NORMAL", tilt=0.25, sway=(0.06, 12.0, 0.35))
    scatter(bed.obj, sprig_b, name="CarpetB", min_distance=4.8 * MM, weight_attr="carpet_b", seed=37,
            scale=(0.55, 1.0), align="NORMAL", tilt=0.25, sway=(0.06, 12.0, 0.35))

    # the film's one shoal; SWIM above says where it may go in this scape
    school = fish.School()
    made += school.objects

    # shrimp on the sand, on the leaning trunk and on moss; a nerite on the sand at the path's mouth
    critters = []
    sx = path_x(48) + 12
    spots = [(Vector(bed.at(sx, 48)), _ground_normal(sx, 48), 2.6)] + [(p, n, y) for (p, n), y in zip(perches, (0.3, 3.4))]
    crew = []
    for i, (pos, up, yaw) in enumerate(spots):
        s = shrimp.Shrimp(f"Shrimp_{i}", size=random.Random(i).uniform(0.9, 1.05))
        crew.append((s, Vector(pos), tuple(up), yaw, 0.7 * i))
        critters.append(s.obj)
    nerite = snail.Snail("Nerite")
    critters.append(nerite.obj)
    made += critters

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["shrimp"] = crew
    ctx["snail"] = nerite
    animate(ctx, ctx["frame"])
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    f0, _ = shot_frames(SCAPE)
    secs = (frame - f0) / 30.0
    for i, (s, pos, up, yaw, t0) in enumerate(ctx["shrimp"]):
        s.pose(pos, yaw=yaw, t=t0 + secs, up=up, walk=0.0)        # grazing in place: legs and feelers work, body stays
    bed = ctx["bed"]
    y = 26.0
    x = path_x(y) - 26.0 + 0.55 * secs                            # about half a millimetre a second
    ctx["snail"].pose(bed.at(x, y, lift_mm=0.3), yaw=0.12, t=secs, up=tuple(_ground_normal(x, y)))
