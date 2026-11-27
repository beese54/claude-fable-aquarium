"""Scape 3: Forest.

A miniature woodland seen from its edge. A pale sand path starts wide at the
front glass and winds back between driftwood trees until it disappears behind
them. The trees are grown by code: bare bark trunks, moss only on the upper
branches so each crown reads as a soft green cloud. The largest stands a third
of the way across; trees shrink and thin toward the back to force perspective.
Dark soil banks up either side of the path under a low carpet, with small
stones and roots at the trunk bases. Coordinates are millimetres: X left to
right, Y front to back.
"""
import math
import random

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

# The path's centre line as (y, x): it enters left of centre, swings right,
# then turns back left behind the far trees.
PATH = [(-40, 136), (0, 140), (60, 150), (120, 170), (180, 182), (225, 174), (262, 152), (300, 128), (340, 110)]

# name, x, y, height above soil, trunk radius, lean (x, y), seed, levels of branching
TREES = [
    ("Tree_big",    96, 128, 160, 10.0, (-0.16, 0.06), 3,  4),
    ("Tree_right", 214, 104, 150, 8.0,  (0.14, 0.04),  12, 4),
    ("Tree_twin",   56, 172, 128, 5.0,  (-0.10, 0.04), 41, 4),
    ("Tree_back_r", 256, 190, 124, 4.8, (-0.06, 0.06), 21, 3),
    ("Tree_mid",   140, 204, 116, 4.4,  (-0.05, 0.08), 15, 3),
    ("Tree_back_l", 44, 240, 108, 4.2,  (0.06, 0.08),  27, 3),
    ("Tree_far",   198, 252, 96,  3.4,  (0.04, 0.10),  33, 3),
]
CROWN_REACH = 0.42            # widest branch from the foot, as a fraction of height
MOSS_POINTS = 50000           # cap across all crowns; moss is the heaviest thing in the scene
MOSS_DENSITY = 2000000.0      # fronds per m2 of fully mossy bark: stacking more adds nothing and slows the BVH

# model, x, y, longest side, how the scan is stood up, (lean, lean, heading), fraction buried
STONES = [
    ("moon_rock_03", 92,  36,  30, (0, 0, 0),  (0, -6, 30),   0.35),
    ("moon_rock_07", 190, 58,  22, (0, 0, 0),  (4, 0, 120),   0.40),
    ("moon_rock_04", 110, 104, 26, (0, 0, 0),  (0, 5, 250),   0.35),
    ("moon_rock_03", 206, 112, 18, (90, 0, 0), (0, 0, 80),    0.40),
    ("moon_rock_07", 236, 70,  30, (0, 90, 0), (-5, 0, 200),  0.40),
    ("moon_rock_04", 196, 160, 16, (0, 0, 0),  (0, 0, 10),    0.45),
    ("moon_rock_04", 52,  150, 20, (0, 0, 0),  (0, 0, 300),   0.40),
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
    """Half-width: 40 mm at the glass, narrowing as it recedes, with a wavering edge."""
    w = 40.0 * (1 - 0.78 * min(1.0, max(0.0, y) / 265) ** 0.85)
    return w * (1 + 0.10 * math.sin(y * 0.071 + 1.0) + 0.06 * math.sin(y * 0.19 + 2.3))


def _edge_noise(x, y):
    """A few millimetres of in-and-out along the path's edge so it is not ruled."""
    return (2.6 * math.sin(x * 0.21 + 1.3 * math.sin(y * 0.13)) * math.cos(y * 0.17 + 0.4)
            + 1.4 * math.sin(x * 0.57 + y * 0.43) + 0.8 * math.sin(x * 1.3 - y * 0.9))


def path_dist(x, y):
    """Signed distance in mm from the path's edge: negative on the sand."""
    return abs(x - path_x(y)) - path_half(y) + _edge_noise(x, y)


def path_fac(x, y, soft=2.5):
    """1 on the sand, 0 on soil; the far end fades out as the path disappears."""
    f = 1.0 - _smooth(-soft, soft, path_dist(x, y))
    return f * (1.0 - _smooth(255, 285, y))


def ground(x, y):
    """Bed height in mm. Deep enough at the front glass to show a band above the
    tray, rising toward the back and sides, with the path sunk a little into
    it and soil heaped round each trunk."""
    side = ((x - 150) / 145) ** 2
    h = 67 + 34 * (max(y, 0) / 295) ** 1.2 + 14 * side * (0.5 + 0.5 * y / 295)
    valley = 1.0 - _smooth(-4, 16, path_dist(x, y))
    h -= 6.5 * valley * (1.0 - _smooth(240, 285, y))
    for _, tx, ty, *_ in TREES:
        h += 5.5 * math.exp(-((x - tx) ** 2 + (y - ty) ** 2) / 22 ** 2)
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


# --- materials ------------------------------------------------------------------

def _floor_material():
    """One material for the whole bed: the white sand scan where the painted
    `path` attribute is 1, dark soil elsewhere, with a noisy, uneven join."""
    mat = materials.white_sand("ForestFloor")
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    sand = bsdf.inputs["Base Color"].links[0].from_socket
    coord = N.new("ShaderNodeTexCoord")

    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "path"
    brk = N.new("ShaderNodeTexNoise")
    brk.inputs["Scale"].default_value = 160.0
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

    # The sand is a layer about a centimetre deep laid over the soil, so the
    # cut face at the glass shows a pale band over dark, with a wavering line.
    top = N.new("ShaderNodeAttribute")
    top.attribute_type = "GEOMETRY"
    top.attribute_name = "top_mm"
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sep.inputs[0])
    depth = N.new("ShaderNodeMath")
    depth.operation = "MULTIPLY_ADD"
    depth.inputs[1].default_value = -1000.0
    L.new(sep.outputs["Z"], depth.inputs[0])
    L.new(top.outputs["Fac"], depth.inputs[2])
    wav = N.new("ShaderNodeMath")
    wav.operation = "MULTIPLY_ADD"
    wav.inputs[1].default_value = 9.0
    L.new(brk.outputs["Fac"], wav.inputs[0])
    L.new(depth.outputs[0], wav.inputs[2])
    layer = N.new("ShaderNodeMapRange")
    layer.inputs["From Min"].default_value = 13.0
    layer.inputs["From Max"].default_value = 16.0
    layer.inputs["To Min"].default_value = 1.0
    layer.inputs["To Max"].default_value = 0.0
    L.new(wav.outputs[0], layer.inputs["Value"])
    both = N.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    L.new(fac.outputs["Result"], both.inputs[0])
    L.new(layer.outputs["Result"], both.inputs[1])
    sand_fac = both.outputs[0]

    # soil between the granules: near black, a little uneven
    sn = N.new("ShaderNodeTexNoise")
    sn.inputs["Scale"].default_value = 420.0
    sn.inputs["Detail"].default_value = 3.0
    L.new(coord.outputs["Object"], sn.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.012, 0.009, 0.006, 1)
    ramp.color_ramp.elements[1].color = (0.050, 0.036, 0.024, 1)
    L.new(sn.outputs["Fac"], ramp.inputs["Fac"])

    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L.new(sand_fac, mix.inputs["Factor"])
    L.new(ramp.outputs["Color"], mix.inputs["A"])
    L.new(sand, mix.inputs["B"])
    L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    rough = N.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.88
    rough.inputs["To Max"].default_value = 0.55
    L.new(sand_fac, rough.inputs["Value"])
    L.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def _granules(bed, material, size_mm, weight, seed, name, spacing=0.72):
    grains = [substrate._granule(f"{name}_{k}", size_mm * MM / 2, seed * 10 + k, material) for k in range(4)]
    col = collection_of(f"{name}_src", grains)
    return scatter(bed.obj, col, name=name, min_distance=size_mm * MM * spacing, seed=seed, weight_attr=weight,
                   scale=(0.6, 1.15), align="NORMAL", tilt=math.pi, sink=size_mm * MM * 0.2)


# --- trees ----------------------------------------------------------------------

class _Skin(wood._Skin):
    """The kit's tube skin, also remembering the centre-line point under every
    vertex, so a grown tree can be reshaped without flattening its branches."""

    def __init__(self, sides=8):
        super().__init__(sides)
        self.centres = []

    def tube(self, pts, radii):
        super().tube(pts, radii)
        for p in pts:
            self.centres += [Vector(p)] * self.sides
        self.centres += [Vector(pts[0]), Vector(self.verts[-1])]      # base cap, tip cap


def _grow_tree(spec):
    """Grow one tree to exactly its height above the soil, kept inside the
    glass, with short roots spreading from its foot into the ground."""
    name, x, y, H, R, lean, seed, depth = spec
    zb = ground(x, y)
    start = Vector((x * MM, y * MM, (zb - 9) * MM))
    d = Vector((lean[0], lean[1], 1.0)).normalized()
    kw = dict(seed=seed, depth=depth, children=(2, 3), wander=0.5, gravity=-0.07, spread=(0.75, 1.15),
              child_length=0.66, child_radius=0.6, taper=0.32)
    L0 = H * 0.6
    trial = wood.grow(start, d, L0, R, **kw)
    k = H * MM / (max(v[2] for v in trial.verts) - zb * MM)
    # Same seed, same random draws: regrowing with the radius pre-divided and
    # then scaling the whole skin gives the intended height and trunk radius.
    skin = wood.grow(start, d, L0, R / k, skin=_Skin(), **kw)
    skin.verts = [start + (Vector(v) - start) * k for v in skin.verts]
    skin.centres = [start + (c - start) * k for c in skin.centres]
    tips = [(start + (p - start) * k, dd, r * k) for p, dd, r in skin.tips]
    n_wood = len(skin.verts)

    rng = random.Random(seed + 101)
    foot = start + d * ((zb + 3) * MM - start.z) / d.z
    for i in range(rng.randint(3, 5)):
        az = 2 * math.pi * i / 4.0 + rng.uniform(-0.6, 0.6)
        out = Vector((math.cos(az), math.sin(az), 0.0))
        rd = (out * 0.9 + Vector((0, 0, -0.30))).normalized()
        wood.grow(foot + out * R * 0.7 * MM, rd, rng.uniform(18, 38) * (R / 6.0) ** 0.5, R * rng.uniform(0.28, 0.4),
                  seed=seed * 7 + i, depth=1, children=(1, 2), wander=0.6, gravity=0.22, spread=(0.4, 0.9), skin=skin)
    skin.verts = [Vector(v) for v in skin.verts]

    # A crown no wider than a set fraction of the tree's height, so a sprawling
    # seed cannot roof over the whole scape. Then keep the wood, its roots and
    # the moss on it clear of the glass by drawing in only the side that
    # crosses the limit. Both move the branches' centre lines toward the foot;
    # the bark keeps its thickness.
    reach = max(math.hypot(v.x - start.x, v.y - start.y) for v in skin.verts[:n_wood])
    c = min(1.0, CROWN_REACH * H * MM / reach)
    lo, hi = 24 * MM, 276 * MM
    s = [[c, c], [c, c]]           # per axis: (toward lo, toward hi)
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
    skin.verts = [tuple(squeeze(cn) + (v - cn)) for v, cn in zip(skin.verts, skin.centres)]
    tips = [(squeeze(p), dd, r) for p, dd, r in tips]
    return skin, tips, n_wood, zb


def _moss_weights(skin, tips, n_wood, zb, H, seed):
    """Moss on the upper branches only: zero on the trunk, rising through the
    crown, patchy, and thin right at the branch ends so tips show through."""
    kd = KDTree(len(tips))
    for i, (p, _, _) in enumerate(tips):
        kd.insert(p, i)
    kd.balance()
    rng = random.Random(seed)
    # tall front trees keep long bare trunks under a high canopy; small back ones are mossy lower down
    c0 = 0.44 + 0.16 * min(1.0, max(0.0, (H - 95) / 70)) + rng.uniform(-0.03, 0.03)
    ph = rng.uniform(0, 6.3)
    w = []
    for i, v in enumerate(skin.verts):
        if i >= n_wood:
            w.append(0.0)
            continue
        rel = (v[2] / MM - zb) / H
        x, y, z = (c / MM for c in v)
        # clumps with real gaps between them, so light and branches show through the crown
        n = math.sin(x * 0.21 + ph) * math.sin(y * 0.19 + z * 0.17 + ph) + 0.5 * math.sin(z * 0.31 + x * 0.11 - ph)
        patch = 0.15 + 0.85 * _smooth(-0.55, 0.35, n)
        _, _, dt = kd.find(v)
        tip = _smooth(3.0 * MM, 12 * MM, dt)
        w.append(_smooth(c0, c0 + 0.16, rel) * patch * tip)
    return w


def _weighted_area(skin, w):
    total = 0.0
    for f in skin.faces:
        if len(f) < 3:
            continue
        a, b, c = (Vector(skin.verts[i]) for i in f[:3])
        area = (b - a).cross(c - a).length * (0.5 if len(f) == 3 else 1.0)
        total += area * sum(w[i] for i in f) / len(f)
    return total


# --- livestock helpers ----------------------------------------------------------

def _clear_of(kd, weights, p, base):
    """True when point p (metres) is `base` mm clear of bare wood and further
    clear of mossy wood, whose fronds stand out from the bark."""
    for _, idx, dist in kd.find_range(p, (base + 23) * MM):
        if dist < (base + 22 * weights[idx]) * MM:
            return False
    return True


CAMERA_MM = (152.0, -555.0, 143.0)


def _in_view(kd, weights, p):
    """True when the line of sight from the camera to p (metres) passes clear
    of the trees, so no fish is lost behind a trunk or inside a crown. The
    sight line in the water is bent toward the glass's normal, as refraction does."""
    to_cam = (Vector(CAMERA_MM) * MM - p).normalized()
    t = Vector((to_cam.x / 1.333, 0.0, to_cam.z / 1.333))
    d = Vector((t.x, math.sqrt(max(0.0, 1.0 - t.length_squared)) * (-1.0 if to_cam.y < 0 else 1.0), t.z))
    q = p.copy()
    for _ in range(80):
        q = q + d * 4 * MM
        if q.y < 6 * MM:
            return True
        for _, idx, dist in kd.find_range(q, 16 * MM):
            if dist < (4 + 11 * weights[idx]) * MM:
                return False
    return True


def _school(n, centre, spread, seed, heading_deg, kd, weights):
    """A loose shoal threaded between the trunks: each fish is tested along
    its whole body, snout to tail, against the wood and the moss."""
    rng = random.Random(seed)
    spots = []
    tries = 0
    while len(spots) < n and tries < 40000:
        tries += 1
        p = [rng.gauss(c, s) for c, s in zip(centre, spread)]
        if not (25 < p[0] < 275 and 25 < p[1] < 270 and p[2] < 238):
            continue
        if p[2] < ground(p[0], p[1]) + 28:
            continue
        if any(math.dist(p, q[0]) < 23 for q in spots):
            continue
        stray = rng.random() < 0.2
        yaw = math.radians(heading_deg + (rng.uniform(-150, 150) if stray else rng.gauss(0, 30)))
        pitch = rng.gauss(0, 0.15)
        f = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
        snout = Vector(p) * MM
        if not all(_clear_of(kd, weights, snout - f * (t * fish.TOTAL_MM * MM), 12) for t in (0.0, 0.35, 0.7, 1.0)):
            continue
        if not _in_view(kd, weights, snout - f * (0.45 * fish.TOTAL_MM * MM)):
            continue
        spots.append((p, f))
    out = []
    for i, (p, f) in enumerate(spots):
        t = fish.Tetra(f"Tetra_{i:02d}", size=rng.uniform(0.84, 1.06))
        t.pose(Vector(p) * MM, f, phase=rng.uniform(0, 6.283), amp=rng.uniform(0.25, 0.7),
               bend=rng.gauss(0, 0.3), roll=rng.gauss(0, 0.09))
        out.append(t)
    return out


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

    # bed: soil and sand in one mesh, split by a painted attribute
    bed = substrate.build_bed(ground, "ForestFloor")
    bed.obj.data.materials.append(_floor_material())
    _paint_all(bed, "path", lambda x, y, d: path_fac(x, y))
    _paint_all(bed, "top_mm", lambda x, y, d: ground(x, y))
    # soil grains everywhere but the path's surface; under the sand layer at the glass, soil again
    _paint_all(bed, "soil", lambda x, y, d: (1.0 - path_fac(x, y, soft=5.0) * (1.0 - _smooth(8, 20, d))) * (d < 40))
    _paint_all(bed, "edge_sand", lambda x, y, d: path_fac(x, y, soft=6.0) * (0.3 + 0.35 * _smooth(-14, -2, path_dist(x, y))) * (d < 4))
    soil = substrate.soil_material()
    _granules(bed, soil, 2.4, "soil", 5, "SoilGrains")
    pale = substrate.sand_material("SandGrain", colour=(0.66, 0.62, 0.55))
    _granules(bed, pale, 1.3, "edge_sand", 9, "SandGrains", spacing=0.9)
    made.append(bed.obj)

    # trees
    bark = [materials.dark_bogwood("Bark_a"), materials.dark_bogwood("Bark_b")]
    # three moss tones, mostly deep green, some olive, a little pale new growth,
    # all a step darker than the carpet so crowns and floor do not merge
    moss_src = (plants.moss_fronds(variants=5, seed=13, material=plants.leaf_material(
                    "MossDeep", base=(0.014, 0.045, 0.008), tip=(0.085, 0.22, 0.028), translucency=0.3, roughness=0.5)) +
                plants.moss_fronds(variants=3, seed=29, length_mm=(4, 8), material=plants.leaf_material(
                    "MossOlive", base=(0.024, 0.040, 0.009), tip=(0.13, 0.20, 0.035), translucency=0.3, roughness=0.5)) +
                plants.moss_fronds(variants=2, seed=37, length_mm=(5, 9), material=plants.leaf_material(
                    "MossNew", base=(0.03, 0.08, 0.014), tip=(0.19, 0.38, 0.06), translucency=0.35, roughness=0.45)))
    moss = collection_of("MossSrc", moss_src)

    grown = []
    for spec in TREES:
        skin, tips, n_wood, zb = _grow_tree(spec)
        w = _moss_weights(skin, tips, n_wood, zb, spec[3], spec[6])
        grown.append((spec, skin, tips, zb, w, _weighted_area(skin, w)))
    dens = min(MOSS_DENSITY, MOSS_POINTS / sum(g[5] for g in grown))
    trees, kd_pts, kd_w = [], [], []
    for k, (spec, skin, tips, zb, w, area) in enumerate(grown):
        obj = wood.finish(skin, spec[0], bark[k % 2], subdivide=1)
        set_point_attribute(obj, "moss", w)
        trees.append((spec, obj, skin, w, zb))
        kd_pts += skin.verts
        kd_w += w
        made.append(obj)
    kd = KDTree(len(kd_pts))
    for i, p in enumerate(kd_pts):
        kd.insert(p, i)
    kd.balance()

    # shrimp perches are found on the bare wood before the moss goes on
    perches = []
    spec, s_obj, s_skin, s_w, zb = trees[1]
    fork = zb + 0.42 * spec[3]           # a bare limb below the crown, on the camera side
    perches.append(_perch(s_obj, s_skin.verts, s_w,
                          lambda v, n, w: n.z - abs(v.z / MM - fork) / 15 - (v.y / MM - spec[2]) / 20 - 999 * (w > 0.01 or n.z < 0.6), 0.0))
    spec, m_obj, m_skin, m_w, zb = trees[0]
    perches.append(_perch(m_obj, m_skin.verts, m_w,
                          lambda v, n, w: 2 * -n.y + n.z - (v.y / MM - spec[2]) / 8 - (v.z / MM - zb) / 40 - 999 * (w <= 0.6 or n.z < 0.2), 9.0))

    for k, (spec, obj, skin, w, _) in enumerate(trees):
        f = (spec[4] / 10.0) ** 0.3          # finer moss on the smaller, further trees
        scatter(obj, moss, name=f"Moss_{spec[0]}", density=dens, weight_attr="moss", seed=40 + k,
                scale=(1.0 * f, 2.0 * f), align="NORMAL", tilt=0.5, sway=(0.04, 30.0, 0.3))

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
        for _, tx, ty, _, r, *_ in TREES:
            w *= 0.35 + 0.65 * _smooth(r + 2, r + 9, math.hypot(x - tx, y - ty))
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

    # tetras crossing between the trunks at mid height, like birds through trees
    # Trunks are solid from the floor up; crowns only from where the moss starts,
    # so the shoal can pass beneath them.
    keep = []
    for _, tx, ty, H, r, *_ in TREES:
        base = ground(tx, ty)
        keep.append((tx, ty, r + 4.0, base + H * 0.55))
        keep.append((tx, ty, CROWN_REACH * H * 0.85, base + H + 6.0, base + H * 0.48))
    f0, f1 = shot_frames(SCAPE)
    school = fish.School(SCAPE, 10, f0, f1, box=(34, 268, 30, 235, 118, 176), seed=6, obstacles=keep,
                         ground=ground, spread=(60.0, 48.0, 14.0))
    made += school.objects

    # shrimp on the sand, on a bare limb and on moss; a nerite low on the front glass
    critters = []
    sx = path_x(48) + 14
    spots = [(Vector(bed.at(sx, 48)), _ground_normal(sx, 48), 2.6)] + [(p, n, y) for (p, n), y in zip(perches, (0.3, 3.4))]
    crew = []
    for i, (pos, up, yaw) in enumerate(spots):
        s = shrimp.Shrimp(f"Shrimp_{i}", size=random.Random(i).uniform(0.9, 1.05))
        crew.append((s, Vector(pos), tuple(up), yaw, 0.7 * i))
        critters.append(s.obj)
    # The snail crosses the sand at the mouth of the path, side-on to the camera,
    # where its striped shell shows (on the front glass it only showed its foot).
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
    x = path_x(y) - 30.0 + 0.55 * secs                            # about half a millimetre a second
    ctx["snail"].pose(bed.at(x, y, lift_mm=0.3), yaw=0.12, t=secs, up=tuple(_ground_normal(x, y)))
