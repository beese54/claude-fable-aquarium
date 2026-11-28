"""Scape 2: Dutch garden.

No stone or wood shows: the picture is made of stem plants, each species in
its own loose drift, like plants in a cottage border rather than beds in a
park. Drifts differ clearly in size, height and outline, overlap and interlock
at their edges, and thin out into a few stray stems; inside a drift the stems
differ in height so no top is clipped flat. Between them the bed shows, under
a patchy carpet. A bay of open water runs from the front glass, right of
centre, deep into the tank toward a low group at the back, so the shoal swims
at mid height among the planting. Tall drifts stand back left and back right;
the skyline dips behind the bay. Coordinates are millimetres: X left to right,
Y front to back.
"""
import math
import os
import random

import numpy as np

import assets
import fish
import plants
import shrimp
import substrate
from bl_common import MM, shot_frames
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "02_dutch"

# Water the film's shoal may use while this scape is on screen (src/sim/school.py):
# the bay and the water above the low front planting. The cylinders cover every
# drift that stands into that water (printed by the build as [dutch-obstacles]).
SWIM = {
    "box": (52, 250, 24, 214, 118, 232),
    "floor": (62, 100),
    "obstacles": [(118, 106, 29, 133), (92, 100, 16, 137), (52, 149, 39, 190), (122, 176, 32, 216),
                  (252, 121, 27, 173), (262, 162, 25, 185), (84, 129, 11, 138), (228, 195, 24, 215),
                  (59, 234, 47, 254), (119, 242, 38, 257), (208, 257, 17, 215), (180, 257, 16, 208),
                  (248, 239, 43, 258), (149, 256, 16, 225)],
}

TOP_LIMIT = 257.0     # mm; tallest stem tip allowed, clear of the surface at 268 even with sway and leaves

# Species the kit lacks, plus kit species re-coloured for this tank. Same keys
# as plants.STEM_SPECIES, plus clump (stems per planting spot), bend, and turn:
# how far up the plant (0..1) the colour starts changing from base to tip. Red
# stem plants only colour up in the top, high-light part; below they are olive.
SPECIES = {
    # broad, round, mid green (Bacopa caroliniana grows tall when left)
    "bacopa":       dict(leaf=(11, 7.5), whorl=2, node=7.5, rise=32, belly=0.55, clump=(1, 3), bend=0.06, turn=0.15,
                         base=(0.025, 0.08, 0.022), tip=(0.11, 0.30, 0.07)),
    # fine, pink tops over olive stems (Rotala rotundifolia, high light)
    "rotala_red":   dict(leaf=(8.5, 2.8), whorl=2, node=3.4, rise=42, belly=0.50, clump=(1, 3), bend=0.09, turn=0.45,
                         base=(0.09, 0.10, 0.025), tip=(0.54, 0.11, 0.16)),
    # broad lanceolate, deep green (Hygrophila; not in the kit)
    "hygrophila":   dict(leaf=(26, 6.5), whorl=2, node=8.5, rise=30, belly=0.42, clump=(1, 2), bend=0.06, turn=0.15,
                         base=(0.02, 0.07, 0.016), tip=(0.09, 0.25, 0.045)),
    # feathery whorls, lime (Limnophila sessiliflora)
    "limnophila":   dict(leaf=(13, 0.9), whorl=8, node=5.5, rise=28, belly=0.40, clump=(1, 2), bend=0.07, turn=0.25,
                         base=(0.04, 0.12, 0.02), tip=(0.24, 0.46, 0.08)),
    # needles in tight whorls, pale green (Pogostemon erectus)
    "pogostemon":   dict(leaf=(16, 1.2), whorl=10, node=4.5, rise=22, belly=0.35, clump=(1, 2), bend=0.05, turn=0.25,
                         base=(0.04, 0.12, 0.02), tip=(0.21, 0.43, 0.07)),
    # small leaves, yellow-orange tips (Rotala 'H'ra'-type green form)
    "rotala_green": dict(leaf=(9, 2.6), whorl=2, node=3.8, rise=38, belly=0.45, clump=(2, 4), bend=0.09, turn=0.40,
                         base=(0.04, 0.12, 0.02), tip=(0.38, 0.40, 0.07)),
    # broad, wine red (Alternanthera reineckii)
    "alternanthera": dict(leaf=(16, 6.0), whorl=2, node=6.0, rise=30, belly=0.50, clump=(1, 2), bend=0.06, turn=0.10,
                          base=(0.08, 0.025, 0.025), tip=(0.30, 0.03, 0.065)),
    # low rosettes of broad leaves, mid green
    "staurogyne":   dict(leaf=(15, 4.8), whorl=2, node=3.6, rise=28, belly=0.50, clump=(1, 2), bend=0.08, turn=0.10,
                         base=(0.025, 0.08, 0.02), tip=(0.09, 0.25, 0.05)),
    # broad, light green (Lobelia cardinalis 'mini')
    "lobelia":      dict(leaf=(14, 9.0), whorl=2, node=5.5, rise=34, belly=0.60, clump=(1, 2), bend=0.06, turn=0.10,
                         base=(0.045, 0.14, 0.03), tip=(0.22, 0.46, 0.10)),
    # tiny round leaves, bright, bushy (pearlweed; not in the kit)
    "pearlweed":    dict(leaf=(4.6, 3.6), whorl=3, node=3.0, rise=36, belly=0.55, clump=(2, 4), bend=0.10, turn=0.15,
                         base=(0.045, 0.15, 0.028), tip=(0.20, 0.46, 0.08)),
}

# Drifts, in priority order: where two overlap, a later one creeps OVERLAP mm in
# under an earlier one's edge, so the two mingle instead of meeting on a line.
#   name, species, centre (mm), half-length and half-width (mm), angle of the
#   length from +X (deg), height above soil at the drift's front and back edge
#   (mm), Poisson spacing (mm), lean toward the light (rad), stray stems
#   (x, y, radius mm) standing apart from the drift, seed of its outline.
GROUPS = [
    # foreground: low, patchy, leaving the front centre open
    ("stauro",    "staurogyne",    (52, 46),   (42, 24), 12,  (16, 30),   4.6, 0.10, [(100, 30, 6), (114, 60, 5)], 1),
    ("pearl",     "pearlweed",     (262, 48),  (30, 30), -25, (18, 34),   4.2, 0.10, [(226, 22, 6)], 2),
    ("lobelia",   "lobelia",       (104, 104), (26, 17), 35,  (30, 50),   5.2, 0.10, [(140, 86, 6), (70, 120, 5)], 3),
    # mid ground: the bay's shores
    ("pogo",      "pogostemon",    (48, 148),  (44, 30), 22,  (50, 86),   4.7, 0.08, [(18, 104, 6), (30, 196, 6)], 4),
    ("rot_green", "rotala_green",  (124, 178), (25, 38), -28, (66, 104),  4.2, 0.07, [(150, 142, 5), (102, 214, 6)], 5),
    ("altern",    "alternanthera", (258, 138), (32, 42), -12, (46, 80),   5.0, 0.07, [(222, 106, 6), (240, 174, 5)], 6),
    ("alt_small", "alternanthera", (82, 126),  (14, 12), 0,   (32, 46),   5.0, 0.08, [], 7),
    ("rr_right",  "rotala_red",    (230, 198), (21, 27), 20,  (86, 112),  4.2, 0.05, [(206, 172, 6), (250, 220, 6)], 8),
    # background: tall to the sides, lower behind the bay
    ("bacopa",    "bacopa",        (46, 236),  (42, 56), 80,  (106, 124), 5.4, 0.04, [], 9),
    ("rot_red",   "rotala_red",    (108, 242), (34, 52), 105, (116, 136), 4.2, 0.04, [(88, 190, 6)], 10),
    ("hygro",     "hygrophila",    (196, 262), (32, 22), 6,   (72, 96),   5.6, 0.04, [], 11),
    ("limno",     "limnophila",    (254, 240), (38, 52), 75,  (118, 136), 4.8, 0.04, [(212, 228, 6)], 12),
    ("rg_back",   "rotala_green",  (150, 258), (22, 20), 30,  (98, 114),  4.2, 0.04, [], 13),
]
OVERLAP = 14.0     # mm a later drift may creep under an earlier one's edge
FADE = 6.0         # mm over which a drift thins out at its edge

# A path of open soil winds from the front glass into the bay, narrowing as it
# goes, where the shrimp graze; and a few small holes in the carpet elsewhere.
PATH = [(166, 0), (174, 36), (188, 80), (196, 118), (206, 158)]      # centre line, mm
PATH_WIDTH = (22.0, 6.0)                                              # half-width at front and far end, mm
BARE = [(34, 96, 10, 8), (150, 120, 9, 7), (232, 92, 8, 6)]           # centre x, y and radii, mm


def _path(x, y):
    """Distance from the path's centre line in half-widths (below 1 is on the path)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    d = np.full(np.broadcast(x, y).shape, np.inf)
    for (ax, ay), (bx, by) in zip(PATH[:-1], PATH[1:]):
        ex, ey = bx - ax, by - ay
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        d = np.minimum(d, np.hypot(x - ax - ex * t, y - ay - ey * t))
    f = np.clip(y / PATH[-1][1], 0.0, 1.0)
    return d / (PATH_WIDTH[0] + (PATH_WIDTH[1] - PATH_WIDTH[0]) * f)


def ground(x, y):
    """Soil height in mm: 62 at the front glass rising to about 100 at the back,
    with a soft roll so the bed is not a ramp, and a slight hollow along the path.
    Takes numbers or numpy arrays."""
    slope = 62 + 38 * (y / 290) ** 1.15
    roll = 2.0 * np.sin(x * 0.04 + 0.7) * np.cos(y * 0.035) + 0.8 * np.sin(x * 0.13) * np.sin(y * 0.09 + 0.5)
    dip = -2.0 * np.exp(-_path(x, y) ** 2) * (1.0 - _smooth(110.0, 170.0, y))
    z = slope + roll + dip
    return float(z) if np.ndim(z) == 0 else z


def _warp(x, y):
    """Bend every outline the same way, so drift borders wander like real
    planting instead of following smooth curves."""
    wx = x + 4.5 * np.sin(y * 0.071 + 1.3) + 2.0 * np.sin(y * 0.23 + x * 0.05)
    wy = y + 4.5 * np.sin(x * 0.063 + 0.4) + 2.0 * np.sin(x * 0.21 + y * 0.04)
    return wx, wy


def _smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _drift(x, y, centre, size, angle, seed):
    """Roughly the distance (mm, positive inside) to an irregular, lobed outline."""
    rng = random.Random(500 + seed)
    a, b = size
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    dx, dy = x - centre[0], y - centre[1]
    u = (dx * c + dy * s) / a
    v = (-dx * s + dy * c) / b
    q = np.sqrt(u * u + v * v) + 1e-9
    th = np.arctan2(v, u)
    r = (1.0 + 0.14 * np.sin(2 * th + rng.uniform(0, 6.3)) + 0.10 * np.sin(3 * th + rng.uniform(0, 6.3))
         + 0.06 * np.sin(5 * th + rng.uniform(0, 6.3)))
    return (r - q) * min(a, b)


def _regions(x, y):
    """For each drift: (distance inside its outline in mm after earlier drifts
    and the glass have taken theirs, distance inside its main body alone)."""
    wx, wy = _warp(x, y)
    glass = np.minimum.reduce([x - 15, 285 - x, y - 13, 271 - y])
    owned, out = [], {}
    for name, _, centre, size, angle, _, _, _, strays, seed in GROUPS:
        main = _drift(wx, wy, centre, size, angle, seed)
        d = main
        for sx, sy, sr in strays:
            d = np.maximum(d, sr - np.hypot(x - sx, y - sy))
        d = np.minimum(d, glass)
        eff = d
        for o in owned:
            eff = np.minimum(eff, OVERLAP - o)
        owned.append(d)
        out[name] = (eff, main)
    return out


def _leaf_material(name, base, tip, turn, translucency=0.32, roughness=0.36, variation=0.22):
    """The kit's leaf shading (base to tip along UV v, varied per stem, partly
    translucent) with three things a close look needs: colour that turns only
    above `turn`, leaf-sized mottling so neighbouring leaves on one stem differ,
    and a paler midrib."""
    from bl_common import new_material
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = turn, (*base, 1)
    e[1].position, e[1].color = 1.0, (*tip, 1)
    L.new(sep.outputs["Y"], ramp.inputs["Fac"])

    info = N.new("ShaderNodeObjectInfo")
    coord = N.new("ShaderNodeTexCoord")
    mottle = N.new("ShaderNodeTexNoise")
    mottle.inputs["Scale"].default_value = 260.0          # per metre: patches about one leaf across
    mottle.inputs["Detail"].default_value = 1.5
    L.new(coord.outputs["Object"], mottle.inputs["Vector"])
    # value: per stem (Random) times per leaf (mottle) times midrib
    v_stem = N.new("ShaderNodeMapRange")
    v_stem.inputs["To Min"].default_value = 1.0 - variation
    v_stem.inputs["To Max"].default_value = 1.0 + variation
    L.new(info.outputs["Random"], v_stem.inputs["Value"])
    v_leaf = N.new("ShaderNodeMapRange")
    v_leaf.inputs["From Min"].default_value = 0.3
    v_leaf.inputs["From Max"].default_value = 0.7
    v_leaf.inputs["To Min"].default_value = 0.78
    v_leaf.inputs["To Max"].default_value = 1.18
    L.new(mottle.outputs["Fac"], v_leaf.inputs["Value"])
    rib = N.new("ShaderNodeMath")
    rib.operation = "SUBTRACT"
    rib.inputs[1].default_value = 0.5
    L.new(sep.outputs["X"], rib.inputs[0])
    rib_a = N.new("ShaderNodeMath")
    rib_a.operation = "ABSOLUTE"
    L.new(rib.outputs[0], rib_a.inputs[0])
    rib_m = N.new("ShaderNodeMapRange")
    rib_m.inputs["From Min"].default_value = 0.0
    rib_m.inputs["From Max"].default_value = 0.12
    rib_m.inputs["To Min"].default_value = 1.22
    rib_m.inputs["To Max"].default_value = 1.0
    L.new(rib_a.outputs[0], rib_m.inputs["Value"])
    m1 = N.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    L.new(v_stem.outputs["Result"], m1.inputs[0])
    L.new(v_leaf.outputs["Result"], m1.inputs[1])
    m2 = N.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    L.new(m1.outputs[0], m2.inputs[0])
    L.new(rib_m.outputs["Result"], m2.inputs[1])
    hue = N.new("ShaderNodeMapRange")
    hue.inputs["To Min"].default_value = 0.5 - variation * 0.08
    hue.inputs["To Max"].default_value = 0.5 + variation * 0.08
    L.new(info.outputs["Random"], hue.inputs["Value"])
    hsv = N.new("ShaderNodeHueSaturation")
    L.new(hue.outputs["Result"], hsv.inputs["Hue"])
    L.new(m2.outputs[0], hsv.inputs["Value"])
    L.new(ramp.outputs["Color"], hsv.inputs["Color"])

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    L.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(hsv.outputs["Color"], trans.inputs["Color"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = translucency
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


# Each source clump is built at one of these fractions of the drift's height, so
# neighbouring stems differ by up to a third instead of sharing one trimmed top.
VARIANT_HEIGHTS = (0.70, 0.78, 0.85, 0.91, 0.96, 1.00, 1.04, 1.08)


def _stem_variants(key, H, variants=8, seed=0):
    """Clumps of 1..n stems of one species, each clump built at a different
    fraction of H mm. A grown-out stem sends up two or three side shoots of
    clearly different height, so one planting spot holds a small clump."""
    sp = SPECIES[key]
    mat = _leaf_material(f"Dutch_{key}_{seed}", sp["base"], sp["tip"], sp["turn"])
    rng = random.Random(1000 + seed)
    heights = list(VARIANT_HEIGHTS[:variants])
    rng.shuffle(heights)
    out = []
    for v in range(variants):
        parts = []
        H_v = H * heights[v]
        for s in range(rng.randint(*sp["clump"])):
            h = H_v * (1.0 if s == 0 else rng.uniform(0.62, 0.95))
            ox, oy = (0.0, 0.0) if s == 0 else (rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
            bend_az = rng.uniform(0, 2 * math.pi)
            bend = rng.uniform(0.2, 1.0) * sp["bend"]
            wob = rng.uniform(0, 6.28)

            def axis(t, h=h, ox=ox, oy=oy, bend_az=bend_az, bend=bend, wob=wob):
                off = bend * h * t * t + 0.010 * h * math.sin(t * 7 + wob)
                return ((ox + math.cos(bend_az) * off) * MM, (oy + math.sin(bend_az) * off) * MM, h * t * MM)

            segs = 10
            pts = [axis(k / segs) for k in range(segs + 1)]
            parts.append(plants._tube_strip(pts, [(0.55 - 0.25 * k / segs) * MM for k in range(segs + 1)]))
            z = rng.uniform(2, 6)
            turn = rng.uniform(0, 6.28)
            Lm, Wm = sp["leaf"]
            n = sp["whorl"]
            while z < h:
                t = z / h
                o = axis(t)
                crown = 1.0 - 0.38 * max(0.0, (t - 0.8) / 0.2)       # smaller leaves in the crown
                young = max(0.0, (t - 0.75) / 0.25)
                for i in range(n):
                    az = turn + 2 * math.pi * i / n + rng.uniform(-0.25, 0.25)
                    rise = math.radians(sp["rise"] + 32 * young + rng.uniform(-12, 12) - 25 * (1 - t) ** 2)
                    d = (math.cos(az) * math.cos(rise), math.sin(az) * math.cos(rise), math.sin(rise))
                    u = (-math.cos(az) * math.sin(rise), -math.sin(az) * math.sin(rise), math.cos(rise))
                    parts.append(plants._leaf(o, d, u, Lm * crown * rng.uniform(0.8, 1.15) * MM,
                                              Wm * crown * rng.uniform(0.85, 1.15) * MM, t, min(1.0, t + 0.06),
                                              fold=0.22, curl=rng.uniform(0.05, 0.30), belly=sp["belly"], segs=4))
                turn += math.pi / 2 if n == 2 else math.pi / n      # opposite pairs alternate at right angles
                z += sp["node"] * (1.0 - 0.55 * young) * rng.uniform(0.85, 1.15)
        verts, faces, uvs = plants._merge(parts)
        out.append(plants._object_from(f"Dutch_{key}_{seed}_{v}", verts, faces, uvs, mat))
    return out


def _shape(mod, size_attr, lean):
    """Extend a kit scatter: instance size follows a painted attribute (so a
    drift rises toward its back and falls away at its edges), and every
    instance leans a little toward the lamp at the back."""
    tree = mod.node_group
    N, L = tree.nodes, tree.links
    inst = next(n for n in N if n.bl_idname == "GeometryNodeInstanceOnPoints")
    rnd_scale = inst.inputs["Scale"].links[0].from_socket
    attr = N.new("GeometryNodeInputNamedAttribute")
    attr.data_type = "FLOAT"
    attr.inputs["Name"].default_value = size_attr
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(rnd_scale, mul.inputs[0])
    L.new(attr.outputs["Attribute"], mul.inputs[1])
    L.new(mul.outputs[0], inst.inputs["Scale"])
    rot = inst.inputs["Rotation"].links[0].from_socket
    toward = N.new("FunctionNodeEulerToRotation")
    toward.inputs["Euler"].default_value = (-lean, 0.0, 0.0)      # -X turn tips +Z toward +Y, the back
    rr = N.new("FunctionNodeRotateRotation")
    rr.rotation_space = "GLOBAL"
    L.new(rot, rr.inputs["Rotation"])
    L.new(toward.outputs["Rotation"], rr.inputs["Rotate By"])
    L.new(rr.outputs["Rotation"], inst.inputs["Rotation"])


SCALE_JITTER = (0.90, 1.10)
TALLEST = SCALE_JITTER[1] * VARIANT_HEIGHTS[-1] * 1.06     # tallest stem tip over the painted height


def _carrier(bed, name, made):
    """An object that shares the bed's mesh and carries one scatter. Stacking
    scatters on one object makes each one also distribute over the instances
    the earlier ones made, doubling the instance count per scatter (eleven
    stacked scatters ran out of memory), so each group gets its own carrier."""
    import bpy
    obj = bpy.data.objects.new(name, bed.obj.data)
    made.append(obj)
    return obj


def _cover(x, y, tops, sel, name):
    """Cylinders that cover one drift for the shoal: the drift is cut along its
    length into pieces about as long as it is wide, one cylinder each."""
    px, py, pt = x[sel], y[sel], tops[sel]
    if len(px) < 4:
        return []
    c = np.array([px.mean(), py.mean()])
    pts = np.column_stack([px, py]) - c
    w, vec = np.linalg.eigh(np.cov(pts.T))
    along = pts @ vec[:, 1]
    span = along.max() - along.min()
    width = 2.0 * math.sqrt(max(w[0], 1.0)) * 1.8
    k = max(1, int(round(span / max(width * 1.3, 1.0))))
    out = []
    edges = np.linspace(along.min(), along.max() + 1e-6, k + 1)
    for i in range(k):
        m = (along >= edges[i]) & (along < edges[i + 1])
        if m.sum() < 4:
            continue
        cx, cy = px[m].mean(), py[m].mean()
        r = float(np.percentile(np.hypot(px[m] - cx, py[m] - cy), 92))
        out.append((int(round(cx)), int(round(cy)), int(round(r)), int(round(float(pt[m].max()) + 4))))
    return out


def build(ctx):
    made = []

    soil = substrate.soil_material()
    bed = substrate.build_bed(ground, "Soil")
    bed.obj.data.materials.append(soil)
    substrate.add_granules(bed, soil, size_mm=2.4, seed=9)
    made.append(bed.obj)

    xy = np.array(bed.top_xy_mm, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    soil_z = ground(x, y)
    pad = [0.0] * (len(bed.obj.data.vertices) - len(x))
    regions = _regions(x, y)

    def paint(name, values):
        set_point_attribute(bed.obj, name, list(values) + pad)

    # Carpet wherever no drift stands, running a little way in under their
    # edges, patchy, and broken by open soil.
    wx, wy = _warp(x, y)
    taken = np.zeros_like(x)
    for eff, _ in regions.values():
        taken = np.maximum(taken, _smooth(2.0, 10.0, eff))
    bare = (1.0 - _smooth(0.75, 1.3, _path(wx, wy))) * (1.0 - _smooth(120.0, 175.0, y))
    for bx, by, brx, bry in BARE:
        q = np.sqrt(((wx - bx) / brx) ** 2 + ((wy - by) / bry) ** 2)
        bare = np.maximum(bare, 1.0 - _smooth(0.85, 1.25, q))
    # A carpet still spreading: thick runs, thin runs and holes where soil shows.
    n = (np.sin(x * 0.07 + 1.1 * np.sin(y * 0.05)) * np.cos(y * 0.06 + 0.9 * np.sin(x * 0.04))
         + 0.45 * np.sin(x * 0.23 + 0.4) * np.sin(y * 0.19 + 1.3))
    patchy = _smooth(-0.35, 0.45, n)
    wall = _smooth(1.0, 5.0, np.minimum.reduce([x - 9, 291 - x, y - 9, 285 - y]))
    carpet_w = (1.0 - taken) * (1.0 - bare) * patchy * wall
    paint("carpet", carpet_w)
    # deeper and more varied than the kit's carpet, so it reads as plants grown in, not a lawn
    carpet_mat = _leaf_material("DutchCarpetLeaf", (0.035, 0.12, 0.02), (0.15, 0.38, 0.06), 0.0,
                                translucency=0.40, roughness=0.32, variation=0.30)
    sprigs = collection_of("DutchCarpetSrc", plants.carpet_sprigs(variants=6, seed=12, material=carpet_mat))
    scatter(_carrier(bed, "DutchCarpet", made), sprigs, name="DutchCarpet", min_distance=3.0 * MM,
            weight_attr="carpet", seed=31, scale=(0.5, 1.1), align="UP", tilt=0.25, keep_surface=False,
            sway=(0.06, 9.0, 0.35))

    report, cover, sources = [], [], {}
    for k, (name, key, centre, size, angle, hts, spacing, lean, strays, seed) in enumerate(GROUPS):
        eff, main = regions[name]
        w = _smooth(0.0, FADE, eff)
        inside = w > 0.02
        body = inside & (main > 0)
        ys = y[body] if body.any() else y[inside]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0.0, 1.0)
        f = np.clip((y - y0) / max(y1 - y0, 1.0), 0.0, 1.0)
        target = hts[0] + (hts[1] - hts[0]) * f ** 0.8
        # tops that rise and fall across the drift, and fall away at its edge
        rng = random.Random(900 + seed)
        p1, p2, p3 = (rng.uniform(0, 6.3) for _ in range(3))
        target *= 1.0 + 0.13 * np.sin(x * 0.085 + p1) * np.cos(y * 0.07 + p2) + 0.06 * np.sin(x * 0.2 + y * 0.17 + p3)
        target *= 0.72 + 0.28 * _smooth(0.0, 16.0, eff)
        # never let a stem reach the surface; ease into the limit so no top is cut flat
        cap = (TOP_LIMIT - soil_z) / TALLEST
        knee = 0.85 * cap
        target = np.where(target < knee, target, knee + (cap - knee) * (1.0 - np.exp(-(target - knee) / (cap - knee))))
        H = float(np.median(target[inside])) if inside.any() else hts[1]
        paint(f"w_{name}", w)
        paint(f"s_{name}", np.where(inside, target / H, 1.0))

        src = _stem_variants(key, H, variants=8, seed=k * 17)
        sources.update({o.name: name for o in src})
        col = collection_of(f"Dutch_{name}_src", src)
        tall = hts[1] > 100
        sway = (0.030, 6.0, 0.28) if tall else (0.050, 8.0, 0.32)
        mod = scatter(_carrier(bed, f"Dutch_{name}", made), col, name=f"Dutch_{name}", min_distance=spacing * MM,
                      weight_attr=f"w_{name}", seed=40 + k, scale=SCALE_JITTER, align="UP", tilt=0.12,
                      keep_surface=False, sway=sway)
        _shape(mod, f"s_{name}", lean)

        if 40 < hts[1] < 115:
            # Young side shoots at half height, thickest round the drift's edge:
            # they hide bare lower stems and soften the sides.
            paint(f"u_{name}", w * (1.0 - 0.6 * _smooth(3.0, 16.0, eff)))
            under = _stem_variants(key, H * 0.5, variants=5, seed=k * 17 + 5)
            sources.update({o.name: name for o in under})
            mod = scatter(_carrier(bed, f"Dutch_{name}_under", made), collection_of(f"Dutch_{name}_under_src", under),
                          name=f"Dutch_{name}_under", min_distance=spacing * 1.15 * MM, weight_attr=f"u_{name}",
                          seed=70 + k, scale=(0.75, 1.15), align="UP", tilt=0.14, keep_surface=False, sway=sway)
            _shape(mod, f"s_{name}", lean)

        tops = soil_z + target * TALLEST
        report.append(f"{name}:{key} H={H:.0f} area={inside.sum() * 2.25 / 100:.0f}cm2 "
                      f"top_max={float(tops[inside].max()) if inside.any() else 0:.0f}")
        if inside.any() and float(tops[inside].max()) > 110:
            cover += _cover(x, y, tops, w > 0.25, name)

    print("[dutch] " + " | ".join(report))
    print("[dutch-obstacles] " + ", ".join(str(c) for c in cover))

    # The film's one shoal; SWIM above says where it may go in this scape.
    school = fish.School()
    made += school.objects

    # Red cherry shrimp grazing on the open soil, spread in depth as well as across.
    crew = []
    for i, (sx, sy, yaw, t, size) in enumerate([(160, 26, 2.6, 0.10, 0.95), (181, 60, -0.4, 0.35, 0.85),
                                                 (186, 34, 1.1, 0.55, 1.0), (199, 126, -2.3, 0.70, 0.9)]):
        s = shrimp.Shrimp(f"Shrimp_{i}", size=size)
        crew.append((s, sx, sy, yaw, t, size))
        made.append(s.obj)

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["shrimp"] = crew

    if os.environ.get("DUTCH_CHECK"):
        ctx["scene"].frame_set(ctx.get("frame", 1044))
        _check(ctx, sources, school.fish)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    f0, _ = shot_frames(SCAPE)
    secs = (frame - f0) / 30.0
    bed = ctx["bed"]
    for s, sx, sy, yaw, t0, size in ctx["shrimp"]:
        # Grazing: a slow creep along the heading, at the pace the legs are stepping.
        walk = 0.15
        d = shrimp.Shrimp.WALK_SPEED * size * walk * secs
        x, y = sx + math.cos(yaw) * d, sy + math.sin(yaw) * d
        s.pose(bed.at(x, y, lift_mm=0.9), yaw=yaw, t=t0 + secs, walk=walk)


def _check(ctx, sources, school):
    """Measure every plant vertex after Geometry Nodes have run, at several
    frames because the sway moves the tips: all must stay inside the water
    (X, Y 6..294 mm, top below 266 mm)."""
    import bpy
    scene = ctx["scene"]
    verts = {}
    for name in sources:
        me = bpy.data.objects[name].data
        v = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", v)
        verts[name] = v.reshape(-1, 3)
    frame0 = scene.frame_current
    for frame in (960, frame0, 1170):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        lo, hi, count = {}, {}, {}
        for inst in dg.object_instances:
            if not inst.is_instance:
                continue
            name = inst.instance_object.original.name
            grp = sources.get(name)
            if grp is None:
                continue
            m = np.array(inst.matrix_world)
            p = (verts[name] @ m[:3, :3].T + m[:3, 3]) / MM
            a, b = p.min(0), p.max(0)
            lo[grp] = np.minimum(lo.get(grp, a), a)
            hi[grp] = np.maximum(hi.get(grp, b), b)
            count[grp] = count.get(grp, 0) + 1
        if frame == frame0:
            for g in lo:
                print(f"[dutch-check] {g}: x {lo[g][0]:.1f}..{hi[g][0]:.1f}  y {lo[g][1]:.1f}..{hi[g][1]:.1f}  "
                      f"z {lo[g][2]:.1f}..{hi[g][2]:.1f}  n={count[g]}")
        bad = [g for g in lo if lo[g][0] < 6 or lo[g][1] < 6 or hi[g][0] > 294 or hi[g][1] > 294 or hi[g][2] > 266]
        top = max(hi[g][2] for g in hi)
        print(f"[dutch-check] frame {frame}: instances {sum(count.values())}, highest tip {top:.1f} mm, "
              f"outside water: {bad or 'none'}")
    scene.frame_set(frame0)
