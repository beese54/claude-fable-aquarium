"""Scape 7: Jungle.

A low-tech nature tank left to grow in for a year. Nothing is trimmed into
shape: Vallisneria rises along the back and one side and trails its ribbons
under the surface, fine and broad-leaved stems push up through each other in
the corners, Cryptocoryne spreads in loose stands that step down into an open
clearing, and three slender pieces of driftwood reach up through the water
carrying moss, Java fern, Anubias and small epiphytes. A raft of floating
plants hangs its roots over the right rear. The variety is in leaf size and
shape, not colour: ribbon, strap, lance, oval, needle, all green to olive.
The front and middle are kept as open water for the shoal.
Coordinates are millimetres: X left to right, Y front to back.
"""
import math
import random

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import assets
import fish
import materials
import plants
import shrimp
import snail
import substrate
import wood
from bl_common import MM, new_material, new_mesh_object, shot_frames
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "07_jungle"

# An older, softer tank: slightly warm light, water with a faint green-yellow cast.
LIGHT = dict(watts=56.0, colour=(1.0, 0.97, 0.90), fill=0.010, bounce=0.04)
WATER = dict(tint=(0.90, 0.975, 0.90), density=1.9, scatter=0.0)

# Water the film's shoal may use while this scape is on screen (src/sim/school.py).
# The cylinders are, in order: the wood (merged from the cells build() prints
# as "[jungle] wood"; those with a fifth value overhang the water and can be
# swum under), the sword plant, and the tall growth along both sides and the back.
SWIM = {
    "box": (62, 242, 30, 142, 122, 212),
    "floor": (62, 96),
    "obstacles": [(88, 176, 24, 214), (121, 176, 24, 240, 169), (148, 192, 16, 250, 188), (165, 176, 24, 207, 165),
                  (198, 165, 16, 213, 165), (198, 143, 16, 230, 201), (242, 178, 24, 182),
                  (46, 152, 36, 140),
                  (26, 85, 16, 200), (22, 125, 16, 225), (20, 165, 18, 240),
                  (278, 70, 14, 175), (281, 108, 16, 240), (282, 148, 16, 240),
                  (30, 215, 34, 245), (100, 238, 34, 250), (160, 240, 30, 245), (215, 232, 30, 245), (268, 218, 30, 245)],
}

LEVEL = 268.0                 # water surface
TOP = 254.0                   # no rigid plant may stand taller than this
CURRENT = math.radians(-128)  # the way the surface drift carries trailing leaves: toward the front left
Z = Vector((0.0, 0.0, 1.0))

# Driftwood: name, foot (x, y), direction, top (mm), trunk radius, seed, branching
# depth, wander, children per branch, widest reach from the foot (mm).
WOOD = [
    ("Wood_main",  (70, 168),  (0.62, 0.10, 0.78),   259, 12.0, 11, 3, 0.30, (2, 3), 130),
    ("Wood_right", (250, 168), (-0.50, 0.05, 0.86),  228, 8.6, 23, 3, 0.35, (2, 2), 85),
    ("Wood_back",  (150, 262), (0.12, -0.04, 1.00),  249, 4.6, 5,  2, 0.60, (2, 3), 40),
]

# Vallisneria, planted one by one: x, y, leaves. Uneven clusters along the back
# and a few up the right-hand glass.
VALLIS = [(36, 283, 8), (47, 274, 9), (58, 286, 10), (72, 279, 7), (88, 285, 9), (101, 274, 6),
          (128, 285, 7), (143, 278, 6), (166, 287, 8),
          (188, 286, 7), (201, 279, 10), (214, 287, 8), (226, 276, 9), (241, 283, 6), (262, 262, 7),
          (284, 150, 8), (286, 128, 7), (281, 104, 6), (15, 176, 7), (13, 154, 5)]

# Rooted and epiphytic specimens, each built as its own plant: kind, where, size.
# where is (x, y) on the soil, or (piece of wood, height along it 0..1) for a
# plant tied to the bark on the side the camera sees.
SPECIMENS = [
    ("sword",   (46, 152), 0.85),
    ("sword",   (266, 236), 0.75),
    ("anubias", (98, 158), 0.80),
    ("anubias", (236, 154), 0.90),
    ("anubias", (0, 0.22), 0.65),
    ("fern",    (0, 0.42), 0.80),
    ("fern",    (1, 0.36), 0.75),
    ("fern",    (2, 0.40), 0.70),
    ("fern",    (30, 236), 0.95),
]


def _smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(x, y, a=0.0, f=1.0):
    """Smooth 0..1 patchiness. Takes numbers or numpy arrays."""
    x, y = x * f, y * f
    n = np.sin(x * 0.071 + a + 1.6 * np.sin(y * 0.053 + a)) * np.cos(y * 0.064 + 1.2 * np.sin(x * 0.045 - a))
    n2 = np.sin(x * 0.19 + 0.7 + a) * np.sin(y * 0.23 + 1.9 - a)
    return np.clip(0.5 + 0.42 * n + 0.18 * n2, 0.0, 1.0)


def ground(x, y):
    """Soil height in mm: 62 at the front glass, banked up toward the back and
    into both rear corners (the left one higher), a little along the left
    glass, and hollowed slightly under the clearing. Numbers or numpy arrays."""
    h = 62 + 30 * (np.clip(y, 0, 300) / 295.0) ** 1.25
    h = h + 15 * np.exp(-(((x - 40) / 75) ** 2 + ((y - 225) / 85) ** 2))
    h = h + 10 * np.exp(-(((x - 265) / 60) ** 2 + ((y - 200) / 80) ** 2))
    h = h + 4 * np.exp(-(((x - 20) / 30) ** 2 + ((y - 90) / 70) ** 2))
    h = h - 3.5 * np.exp(-(((x - 150) / 65) ** 2 + ((y - 75) / 60) ** 2))
    return h + 1.8 * np.sin(x * 0.047 + 1.1) * np.cos(y * 0.039 + 0.3) + 0.8 * np.sin(x * 0.16 + y * 0.11)


def _clear_d(x, y):
    """Roughly the distance (mm) into the clearing: positive inside it. Its
    edge wanders, comes in from both sides toward the back, and is not centred."""
    xl = 66 + 14 * np.sin(y * 0.038 + 0.6) + 20 * (y / 160.0) ** 2
    xr = 240 - 12 * np.sin(y * 0.045 + 2.1) - 24 * (y / 160.0) ** 2
    yb = 150 + 12 * np.sin(x * 0.05 + 0.8) + 6 * np.sin(x * 0.13)
    return np.minimum(np.minimum(x - xl, xr - x), yb - y)


def _ground_normal(x, y):
    gx = (ground(x + 1.0, y) - ground(x - 1.0, y)) / 2.0
    gy = (ground(x, y + 1.0) - ground(x, y - 1.0)) / 2.0
    return Vector((-gx, -gy, 1.0)).normalized()


# --- leaves ---------------------------------------------------------------------

def _leaf_mat(name, young, old, rib=1.3, veins=0.2, vein_freq=55.0, stripes=0.0, translucency=0.35, rough=0.42,
              spots=0.3, mottle=240.0, variation=0.2):
    """A living leaf. UV is the leaf itself (u across, v base to tip) and the
    point attribute `age` runs from 0 (new growth) to 1 (old): colour goes from
    `young` to `old` with it, and old leaves pick up dark specks of algae.
    Each plant differs a little, each leaf is mottled, the midrib is paler,
    side veins (or lengthwise ones, `stripes`) show faintly, the margin is a
    shade darker, and light comes through the blade."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links

    def val(op, a, b=None, clamp=False):
        n = N.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for sock, v in zip(n.inputs, (a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                L.new(v, sock)
        return n.outputs[0]

    def remap(v, a, b, c, d):
        n = N.new("ShaderNodeMapRange")
        for key, q in (("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            n.inputs[key].default_value = q
        L.new(v, n.inputs["Value"])
        return n.outputs["Result"]

    def mix(kind, fac, a, b):
        n = N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = kind
        socks = [n.inputs[0]] + [s for s in n.inputs if s.type == "RGBA"][:2]
        for sock, v in zip(socks, (fac, a, b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            elif isinstance(v, tuple):
                sock.default_value = (*v, 1.0)
            else:
                L.new(v, sock)
        return next(s for s in n.outputs if s.type == "RGBA")

    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    u, v = sep.outputs["X"], sep.outputs["Y"]
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "age"
    info = N.new("ShaderNodeObjectInfo")
    r = info.outputs["Random"]
    coord = N.new("ShaderNodeTexCoord")

    age = val("ADD", attr.outputs["Fac"], val("MULTIPLY", val("SUBTRACT", r, 0.5), 0.35), clamp=True)
    col = mix("MIX", age, young, old)

    mott = N.new("ShaderNodeTexNoise")
    mott.noise_dimensions = "4D"
    mott.inputs["Scale"].default_value = mottle
    mott.inputs["Detail"].default_value = 3.0
    L.new(coord.outputs["Object"], mott.inputs["Vector"])
    L.new(val("MULTIPLY", r, 50.0), mott.inputs["W"])
    f = remap(mott.outputs["Fac"], 0.3, 0.7, 0.72, 1.24)
    across = val("ABSOLUTE", val("SUBTRACT", u, 0.5))
    f = val("MULTIPLY", f, remap(across, 0.0, 0.06, rib, 1.0))
    if veins:
        side = val("SINE", val("MULTIPLY", val("SUBTRACT", v, val("MULTIPLY", across, 0.8)), vein_freq))
        vm = val("MULTIPLY", val("GREATER_THAN", side, 0.86), val("LESS_THAN", across, 0.44))
        f = val("MULTIPLY", f, val("ADD", val("MULTIPLY", vm, veins), 1.0))
    if stripes:
        st = val("GREATER_THAN", val("SINE", val("MULTIPLY", u, 31.4)), 0.5)
        f = val("MULTIPLY", f, val("ADD", val("MULTIPLY", st, stripes), 1.0))
    f = val("MULTIPLY", f, remap(across, 0.38, 0.5, 1.0, 0.80))
    f = val("MULTIPLY", f, remap(r, 0.0, 1.0, 1.0 - variation, 1.0 + variation))
    col = mix("MULTIPLY", 1.0, col, f)
    if spots:
        sp = N.new("ShaderNodeTexNoise")
        sp.inputs["Scale"].default_value = 900.0
        sp.inputs["Detail"].default_value = 2.0
        L.new(coord.outputs["Object"], sp.inputs["Vector"])
        col = mix("MIX", val("MULTIPLY", remap(sp.outputs["Fac"], 0.56, 0.66, 0.0, spots), age), col, (0.030, 0.026, 0.010))
    hsv = N.new("ShaderNodeHueSaturation")
    L.new(remap(r, 0.0, 1.0, 0.485, 0.515), hsv.inputs["Hue"])
    L.new(col, hsv.inputs["Color"])
    col = hsv.outputs["Color"]

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = 0.3        # wet leaf under water: little surface shine
    L.new(col, bsdf.inputs["Base Color"])
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0003
    L.new(f, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(col, trans.inputs["Color"])
    sh = N.new("ShaderNodeMixShader")
    sh.inputs["Fac"].default_value = translucency
    L.new(bsdf.outputs[0], sh.inputs[1])
    L.new(trans.outputs[0], sh.inputs[2])
    L.new(sh.outputs[0], out.inputs["Surface"])
    return mat


def _blade(o, d, n, L, W, age, rng, belly=0.45, segs=7, nu=5, arch=0.5, fold=0.15, ruffle=0.0, waves=3.0,
           twist=0.0, tip=0.9):
    """One leaf blade (mm). o: where it starts; d: the way it points; n: its
    upper face. It arches over by `arch` radians toward its tip, is folded
    along the midrib, may have a wavy margin (`ruffle`, as a fraction of its
    width) and a twist. Returns verts, faces, uvs, ages."""
    d = Vector(d).normalized()
    n = Vector(n)
    n = (n - d * n.dot(d)).normalized()
    ph = rng.uniform(0, 6.283)
    verts, faces, uvs = [], [], []
    p = Vector(o)
    for k in range(segs + 1):
        t = k / segs
        th = arch * t ** 1.4
        dt = d * math.cos(th) - n * math.sin(th)
        nt = n * math.cos(th) + d * math.sin(th)
        if k:
            p = p + dt * (L / segs)
        st = dt.cross(nt)
        if twist:
            a = twist * t
            st, nt = st * math.cos(a) + nt * math.sin(a), nt * math.cos(a) - st * math.sin(a)
        if t <= belly:
            w = math.sin(0.5 * math.pi * t / belly) ** 0.8
        else:
            w = math.cos(0.5 * math.pi * (t - belly) / (1 - belly)) ** tip
        half = 0.5 * W * max(w, 0.03)
        env = math.sin(math.pi * min(1.0, t * 1.1))
        for j in range(nu):
            a = -1.0 + 2.0 * j / (nu - 1)
            lift = fold * half * abs(a) ** 1.5
            if ruffle:
                lift += ruffle * W * math.sin(waves * 6.283 * t + ph + (1.7 if a > 0 else 0.0)) * a * a * env
            verts.append(tuple(p + st * (half * a) + nt * lift))
            uvs.append(((a + 1.0) / 2.0, t))
    for k in range(segs):
        for j in range(nu - 1):
            a = k * nu + j
            faces.append((a, a + 1, a + nu + 1, a + nu))
    return verts, faces, uvs, [age] * len(verts)


def _tube(points, radii, age, sides=3):
    vv, ff, uu = plants._tube_strip([tuple(p) for p in points], radii, sides=sides)
    return vv, ff, uu, [age] * len(vv)


def _obj(name, parts, material):
    """Join parts (built in mm) into one object in metres, carrying `age`."""
    verts, faces, uvs, ages = [], [], [], []
    for vv, ff, uu, aa in parts:
        off = len(verts)
        verts += [(p[0] * MM, p[1] * MM, p[2] * MM) for p in vv]
        uvs += uu
        ages += aa
        faces += [tuple(i + off for i in f) for f in ff]
    obj = plants._object_from(name, verts, faces, uvs, material)
    set_point_attribute(obj, "age", ages)
    return obj


def _rosette(name, mat, variants, seed, leaves=(7, 11), length=(40, 70), ratio=(0.22, 0.30), stalk=(0.30, 0.45),
             elev=(30, 80), drop=12.0, arch=(0.4, 0.9), belly=0.42, ruffle=0.0, waves=3.0, twist=0.3, fold=0.16,
             segs=7, nu=5, rhizome=0.0, stalk_r=0.7, tip=0.9):
    """Plants that grow as a rosette from a crown or a creeping rhizome: crypts,
    swords, Anubias, Java fern. Leaves come out on a spiral; the newest stand
    upright in the middle and are small and bright, older ones lean out, are
    larger and darker. length is stalk plus blade (mm); elev is how steeply the
    oldest and newest leaves rise (degrees)."""
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        n = rng.randint(*leaves)
        az0 = rng.uniform(0, 6.283)
        rz = rng.uniform(0, 6.283)
        run = Vector((math.cos(rz), math.sin(rz), 0.0))
        if rhizome:
            pts = [run * (rhizome * (k / 4.0 - 0.5)) + Vector((0, 0, 1.2 + 0.8 * math.sin(k * 1.3))) for k in range(5)]
            parts.append(_tube(pts, [2.0, 2.4, 2.3, 2.0, 1.4], 0.9, sides=5))
        for i in range(n):
            f = i / max(n - 1, 1)
            age = min(1.0, max(0.0, 0.9 * f + rng.uniform(-0.12, 0.12)))
            az = az0 + i * 2.39996 + rng.uniform(-0.35, 0.35)
            e = math.radians(elev[1] + (elev[0] - elev[1]) * f ** 0.8 + rng.uniform(-8, 8))
            size = 0.6 + 0.4 * math.sin(math.pi * (0.15 + 0.75 * f))
            total = rng.uniform(*length) * size
            lp = total * rng.uniform(*stalk)
            h = Vector((math.cos(az), math.sin(az), 0.0))
            p = Vector((rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), -1.0)) + run * (rhizome * (rng.random() - 0.5))
            pts = [p.copy()]
            e0 = min(1.5, e + 0.4)
            for k in range(1, 5):
                ek = e0 + (e - e0) * k / 4.0
                p = p + (h * math.cos(ek) + Z * math.sin(ek)) * (lp / 4.0)
                pts.append(p.copy())
            parts.append(_tube(pts, [stalk_r * (1 - 0.35 * k / 4.0) for k in range(5)], min(1.0, age + 0.2)))
            eb = e - math.radians(drop + rng.uniform(-6, 6))
            d = h * math.cos(eb) + Z * math.sin(eb)
            nrm = (Z * math.cos(eb) - h * math.sin(eb) + h.cross(Z) * rng.uniform(-0.3, 0.3)).normalized()
            L = total - lp
            parts.append(_blade(p, d, nrm, L, L * rng.uniform(*ratio), age, rng, belly=belly, segs=segs, nu=nu,
                                arch=rng.uniform(*arch), fold=fold * rng.uniform(0.6, 1.4), ruffle=ruffle, waves=waves,
                                twist=rng.uniform(-twist, twist), tip=tip))
        out.append(_obj(f"{name}_{v}", parts, mat))
    return out


# Stem plants: leaf (length, width), leaves per node, node spacing, rise angle,
# belly, stems per planting spot, bend, colours of new and old growth.
STEMS = {
    # fine paired leaves, yellow-green tops (Rotala rotundifolia, green form)
    "rotala":   dict(leaf=(10, 2.9), whorl=2, node=4.3, rise=40, belly=0.48, clump=(2, 4), bend=0.10,
                     young=(0.22, 0.40, 0.06), old=(0.03, 0.10, 0.02)),
    # broad lance leaves, mid green (Hygrophila)
    "hygro":    dict(leaf=(27, 7.5), whorl=2, node=9.5, rise=32, belly=0.42, clump=(1, 2), bend=0.07,
                     young=(0.17, 0.36, 0.06), old=(0.02, 0.08, 0.02)),
    # oval leaves, olive with bronze tops (Ludwigia in low light): the one warm note
    "ludwigia": dict(leaf=(15, 7.0), whorl=2, node=7.5, rise=30, belly=0.50, clump=(1, 3), bend=0.08,
                     young=(0.34, 0.20, 0.05), old=(0.045, 0.10, 0.02)),
}


def _stems(key, H, variants, seed):
    """Clumps of one to four stems, H mm tall, each bending its own way."""
    sp = STEMS[key]
    mat = _leaf_mat(f"Jungle_{key}", sp["young"], sp["old"], veins=0.0, mottle=320.0, spots=0.25, translucency=0.36)
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for s in range(rng.randint(*sp["clump"])):
            h = H * (1.0 if s == 0 else rng.uniform(0.6, 0.95))
            ox, oy = (0.0, 0.0) if s == 0 else (rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 3.5))
            baz = rng.uniform(0, 6.283)
            bend = rng.uniform(0.3, 1.0) * sp["bend"]
            wob = rng.uniform(0, 6.283)

            def axis(t, h=h, ox=ox, oy=oy, baz=baz, bend=bend, wob=wob):
                off = bend * h * t * t + 0.014 * h * math.sin(t * 7 + wob)
                return Vector((ox + math.cos(baz) * off, oy + math.sin(baz) * off, h * t))

            parts.append(_tube([axis(k / 10.0) for k in range(11)], [0.6 - 0.25 * k / 10.0 for k in range(11)], 0.8))
            z = rng.uniform(3, 8)
            turn = rng.uniform(0, 6.283)
            Lm, Wm = sp["leaf"]
            while z < h:
                t = z / h
                o = axis(t)
                young = max(0.0, (t - 0.75) / 0.25)
                crown = 1.0 - 0.4 * young
                for i in range(sp["whorl"]):
                    az = turn + 2 * math.pi * i / sp["whorl"] + rng.uniform(-0.3, 0.3)
                    rise = math.radians(sp["rise"] + 32 * young + rng.uniform(-12, 12) - 25 * (1 - t) ** 2)
                    hz = Vector((math.cos(az), math.sin(az), 0.0))
                    parts.append(_blade(o, hz * math.cos(rise) + Z * math.sin(rise), Z * math.cos(rise) - hz * math.sin(rise),
                                        Lm * crown * rng.uniform(0.75, 1.15), Wm * crown * rng.uniform(0.85, 1.15),
                                        min(1.0, max(0.0, 1.05 - t ** 1.6 + rng.uniform(-0.1, 0.1))), rng,
                                        belly=sp["belly"], segs=4, nu=3, arch=rng.uniform(0.1, 0.6), fold=0.25))
                turn += math.pi / 2
                z += sp["node"] * (1.0 - 0.5 * young) * rng.uniform(0.85, 1.2)
        out.append(_obj(f"Jungle_{key}_{v}", parts, mat))
    return out


def _vallis(name, mat, x, y, n, seed):
    """One Vallisneria plant standing in the scene. Each ribbon climbs with a
    slow spiral; the long ones turn over a few millimetres under the surface
    and trail along it with the drift. Kept inside the glass point by point.
    Returns the object and what animate() needs to make the ribbons wave."""
    rng = random.Random(seed)
    base = Vector((x, y, float(ground(x, y)) - 1.5))
    column = LEVEL - base.z
    parts, arcs = [], []
    for i in range(n):
        length = column * (rng.uniform(0.5, 0.95) if rng.random() < 0.35 else rng.uniform(1.05, 1.85))
        width = rng.uniform(2.6, 6.0)
        under = rng.uniform(2.5, 9.0)
        az = rng.uniform(0, 6.283)
        lean = rng.uniform(0.06, 0.40)
        taz = CURRENT + rng.gauss(0, 0.28)
        drift = Vector((math.cos(taz), math.sin(taz), 0.0))
        across = Vector((-drift.y, drift.x, 0.0))
        d = Vector((math.cos(az) * lean, math.sin(az) * lean, 1.0)).normalized()
        p = base + Vector((rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0))
        a2 = rng.uniform(0, 6.283)
        s = d.cross(Vector((math.cos(a2), math.sin(a2), 0.0))).normalized()
        spiral = rng.choice((-1, 1)) * rng.uniform(0.003, 0.014)        # radians per mm of climb
        ph = rng.uniform(0, 6.283)
        step = 5.0
        segs = max(6, int(length / step))
        pts, sides, dirs = [], [], []
        for k in range(segs + 1):
            pts.append(p.copy())
            sides.append(s.copy())
            dirs.append(d.copy())
            room = (LEVEL - under) - p.z
            if room > 14.0:
                fade = 1.0 - k / segs
                target = Vector((math.cos(az) * lean * fade + 0.17 * math.sin(k * 0.31 + ph),
                                 math.sin(az) * lean * fade + 0.17 * math.cos(k * 0.23 + ph), 1.0)).normalized()
                tw = spiral * step
            else:
                target = (drift + across * (0.30 * math.sin(k * 0.33 + ph)) + Z * (room / 14.0)).normalized()
                tw = 0.0
            d = (d * 0.6 + target * 0.4).normalized()
            p = p + d * step
            p.x = min(max(p.x, 9.5), 290.5)
            p.y = min(max(p.y, 9.5), 290.5)
            p.z = min(p.z, LEVEL - 2.0)
            s = (s - d * s.dot(d)).normalized()
            if tw:
                s = (s * math.cos(tw) + d.cross(s) * math.sin(tw)).normalized()
            else:
                flat = d.cross(Z)
                if flat.length > 0.2:
                    flat.normalize()
                    if flat.dot(s) < 0:
                        flat = -flat
                    s = (s * 0.7 + flat * 0.3).normalized()
        verts, faces, uvs, ages = [], [], [], []
        for k in range(segs + 1):
            t = k / segs
            w = 0.5 * width * min(1.0, 0.4 + t * 10.0) * (1.0 if t < 0.92 else max(0.1, (1.0 - t) / 0.08) ** 0.7)
            nrm = dirs[k].cross(sides[k])
            for a in (-1.0, 0.0, 1.0):
                verts.append(tuple(pts[k] + sides[k] * (w * a) + nrm * (0.18 * w * (1.0 - abs(a)))))
                uvs.append(((a + 1.0) / 2.0, t))
                ages.append(min(1.0, 0.12 + 0.55 * t + (0.5 if t > 0.94 else 0.0)))
                arcs.append(t * length)
        for k in range(segs):
            for j in (0, 1):
                a = k * 3 + j
                faces.append((a, a + 1, a + 4, a + 3))
        parts.append((verts, faces, uvs, ages))
    obj = _obj(name, parts, mat)
    rest = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    return obj, dict(obj=obj, rest=rest, arc=np.array(arcs), column=column, phase=rng.uniform(0, 6.283))


# --- wood -----------------------------------------------------------------------

class _Skin(wood._Skin):
    """The kit's tube skin, also remembering the centre-line point under every
    vertex: for reshaping a grown piece and for knowing which way bark faces."""

    def __init__(self, sides=8):
        super().__init__(sides)
        self.centres = []

    def tube(self, pts, radii):
        super().tube(pts, radii)
        for p in pts:
            self.centres += [Vector(p)] * self.sides
        self.centres += [Vector(pts[0]), Vector(self.verts[-1])]


def _grow_wood(spec):
    """Grow one piece so its highest twig ends at `top`, with a few roots at
    its foot, drawn in toward the foot wherever it would spread too wide or
    reach the glass."""
    name, (x, y), direction, top, R, seed, depth, wander, kids, reach = spec
    zb = float(ground(x, y))
    start = Vector((x, y, zb - 10)) * MM
    d = Vector(direction).normalized()
    kw = dict(seed=seed, depth=depth, children=kids, wander=wander, gravity=-0.04, spread=(0.45, 1.0),
              child_length=0.72, child_radius=0.62, taper=0.44)
    L0 = (top - zb) * 0.62
    trial = wood.grow(start, d, L0, R, **kw)
    k = (top * MM - start.z) / (max(v[2] for v in trial.verts) - start.z)
    skin = wood.grow(start, d, L0, R / k, skin=_Skin(8), **kw)
    skin.verts = [start + (Vector(v) - start) * k for v in skin.verts]
    skin.centres = [start + (c - start) * k for c in skin.centres]
    skin.tips = [(start + (p - start) * k, dd, r * k) for p, dd, r in skin.tips]
    n_wood = len(skin.verts)

    rng = random.Random(seed + 101)
    foot = start + d * ((zb + 3) * MM - start.z) / d.z
    for i in range(rng.randint(2, 4)):
        az = 2 * math.pi * i / 3.0 + rng.uniform(-0.7, 0.7)
        out = Vector((math.cos(az), math.sin(az), 0.0))
        wood.grow(foot + out * R * 0.7 * MM, (out * 0.9 + Vector((0, 0, -0.28))).normalized(), rng.uniform(18, 40),
                  R * rng.uniform(0.3, 0.45), seed=seed * 7 + i, depth=1, children=(1, 2), wander=0.6, gravity=0.22,
                  spread=(0.4, 0.9), skin=skin)
    skin.verts = [Vector(v) for v in skin.verts]

    far = max(math.hypot(c.x - start.x, c.y - start.y) for c in skin.centres[:n_wood])
    c0 = min(1.0, reach * MM / far)
    lo, hi = 16 * MM, 284 * MM
    s = [[c0, c0], [c0, c0]]
    for c in skin.centres:
        for a in (0, 1):
            off = c[a] - start[a]
            if start[a] + off * c0 > hi and off > 0:
                s[a][1] = min(s[a][1], (hi - start[a]) / off)
            elif start[a] + off * c0 < lo and off < 0:
                s[a][0] = min(s[a][0], (lo - start[a]) / off)

    def squeeze(p):
        return Vector((start.x + (p.x - start.x) * s[0][p.x > start.x], start.y + (p.y - start.y) * s[1][p.y > start.y], p.z))
    new_c = [squeeze(c) for c in skin.centres]
    skin.verts = [nc + (v - c) for v, c, nc in zip(skin.verts, skin.centres, new_c)]
    skin.centres = new_c
    return skin, n_wood, zb


def _wood_weights(skin, n_wood, zb, top, seed):
    """Moss and small epiphytes. Moss sits in ragged clumps, mostly on the
    upper side of branches and off the lowest trunk; the epiphytes keep to the
    middle of the piece where moss is thin."""
    ph = random.Random(seed).uniform(0, 6.283)
    moss, epi = [], []
    for i, (v, c) in enumerate(zip(skin.verts, skin.centres)):
        if i >= n_wood:
            moss.append(0.0)
            epi.append(0.0)
            continue
        x, y, z = (q / MM for q in v)
        rel = (z - zb) / (top - zb)
        out = v - c
        up = out.z / max(out.length, 1e-6)
        n = math.sin(x * 0.19 + ph) * math.sin(y * 0.17 + z * 0.15 + ph) + 0.5 * math.sin(z * 0.29 + x * 0.11 - ph)
        m = float(_smooth(0.15, 0.75, n)) * (0.15 + 0.85 * float(_smooth(-0.3, 0.7, up))) * float(_smooth(0.08, 0.25, rel))
        m *= float(_smooth(1.6, 3.6, out.length / MM))            # thin twigs stay bare
        moss.append(m)
        n2 = math.sin(x * 0.13 - ph) * math.sin(z * 0.11 + y * 0.09 + 2 * ph)
        epi.append(float(_smooth(0.15, 0.6, n2)) * float(_smooth(0.12, 0.25, rel)) * float(1 - _smooth(0.60, 0.80, rel)) * (1.0 - m))
    return moss, epi


def _weighted_area(skin, w):
    total = 0.0
    for f in skin.faces:
        a, b, c = (skin.verts[i] for i in f[:3])
        total += (b - a).cross(c - a).length * (0.5 if len(f) == 3 else 1.0) * sum(w[i] for i in f) / len(f)
    return total


def _perch(obj, skin, n_wood, weights, pick, lift_mm=0.0):
    """A point on the finished wood (after smoothing and knots) near the best
    bare-bark vertex chosen by `pick`, and the surface normal there."""
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    best = max(range(n_wood), key=lambda i: pick(skin.verts[i], (skin.verts[i] - skin.centres[i]).normalized(), weights[i]))
    v = skin.verts[best]
    n = (v - skin.centres[best]).normalized()
    hit, normal, _, _ = bvh.ray_cast(v + n * 6 * MM, -n, 20 * MM)
    if hit is None:
        hit, normal = v, n
    return hit + normal * lift_mm * MM, normal


def _cells(skins, cell=22.0):
    """Keep-out cylinders covering the wood, for pasting into SWIM: the wood's
    points binned on a coarse grid, one cylinder per cell from its lowest to its
    highest point. A cell that starts well above the bed gets a bottom, so the
    shoal can pass under an overhanging branch."""
    cells = {}
    for skin, n_wood, _ in skins:
        for v in skin.verts[:n_wood]:
            p = v / MM
            k = (int(p.x // cell), int(p.y // cell))
            lo, hi = cells.get(k, (1e9, -1e9))
            cells[k] = (min(lo, p.z), max(hi, p.z))
    out = []
    for (i, j), (lo, hi) in sorted(cells.items()):
        x, y = (i + 0.5) * cell, (j + 0.5) * cell
        base = (round(x), round(y), round(cell * 0.6), round(hi + 4))
        out.append(base + (round(lo - 4),) if lo > float(ground(x, y)) + 45 else base)
    return out


# --- helpers for planting -------------------------------------------------------

def _size_by(mod, attr):
    """Extend a kit scatter so each instance's size also follows a painted
    attribute: a stand can then be tall in one place and low in another."""
    tree = mod.node_group
    N, L = tree.nodes, tree.links
    inst = next(n for n in N if n.bl_idname == "GeometryNodeInstanceOnPoints")
    rnd = inst.inputs["Scale"].links[0].from_socket
    a = N.new("GeometryNodeInputNamedAttribute")
    a.data_type = "FLOAT"
    a.inputs["Name"].default_value = attr
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(rnd, mul.inputs[0])
    L.new(a.outputs["Attribute"], mul.inputs[1])
    L.new(mul.outputs[0], inst.inputs["Scale"])


def _stand(obj, pos, up=(0, 0, 1), yaw=0.0, scale=1.0):
    """Stand a specimen on a point (metres), its own Z along `up`. Returns the
    pieces animate() needs to rock it gently about its foot."""
    rot = Z.rotation_difference(Vector(up).normalized()).to_matrix().to_4x4() @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(scale, 4)
    obj.matrix_world = Matrix.Translation(Vector(pos)) @ rot
    return obj, Vector(pos), rot


def _surface_helper(name, res_mm=4.0, z_mm=LEVEL - 1.0):
    """A flat grid just under the waterline for the floaters to be scattered on
    (the scatter drops it from the render)."""
    lo, hi = 9.0, 291.0
    n = int((hi - lo) / res_mm)
    verts = [((lo + (hi - lo) * i / n) * MM, (lo + (hi - lo) * j / n) * MM, z_mm * MM) for j in range(n + 1) for i in range(n + 1)]
    faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i) for j in range(n) for i in range(n)]
    return new_mesh_object(name, verts, faces)


def _raft(x, y):
    """Where the floating plants have gathered: the right rear, out of the
    drift, with a ragged edge and gaps that let light through."""
    edge = 176 + 16 * np.sin(y * 0.045 + 0.4) + 8 * np.sin(y * 0.13)
    c = _smooth(edge - 8, edge + 14, x) * _smooth(128, 156, y + 12 * np.sin(x * 0.05))
    c = np.maximum(c, _smooth(246, 262, x) * _smooth(84, 104, y))
    return c * _smooth(0.22, 0.34, _noise(x, y, 2.3, 1.2))


# --- build ----------------------------------------------------------------------

def build(ctx):
    made = []
    rng = random.Random(7)

    # Soil: the kit's granules, with a share of larger and paler ones mixed in so
    # the bed and its cut face at the glass are not one even grain.
    soil = substrate.soil_material()
    bed = substrate.build_bed(ground, "Soil")
    bed.obj.data.materials.append(soil)
    substrate.add_granules(bed, soil, size_mm=2.5, seed=7)
    brown = substrate.soil_material("AquaSoilBrown", colour=(0.060, 0.038, 0.022))
    grit = substrate.soil_material("AquaSoilGrit", colour=(0.15, 0.105, 0.065))       # a sprinkling of older, paler gravel
    coarse = [substrate._granule(f"Coarse_{k}", (1.9 if k < 3 else 1.3) * MM, 70 + k, (soil, brown, brown, grit, brown)[k]) for k in range(5)]
    scatter(bed.obj, collection_of("CoarseSrc", coarse), name="Coarse", density=26000.0, weight_attr="all", seed=3,
            scale=(0.7, 1.25), align="NORMAL", tilt=math.pi, sink=0.3 * MM)
    made.append(bed.obj)

    xy = np.array(bed.top_xy_mm, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    gz = ground(x, y)
    pad = [0.0] * (len(bed.obj.data.vertices) - len(x))

    def paint(name, values):
        set_point_attribute(bed.obj, name, list(np.broadcast_to(values, x.shape).astype(float)) + pad)

    def carrier(name):
        # shares the bed's mesh and carries one scatter, so every stand is its own object
        o = bpy.data.objects.new(name, bed.obj.data)
        made.append(o)
        return o

    # The lie of the land for planting.
    d = _clear_d(x, y)                                           # > 0 in the clearing
    wall = np.minimum.reduce([x - 5, 295 - x, y - 5, 295 - y])  # distance from the nearest glass
    box = np.minimum.reduce([x - 44, 258 - x, 166 - y])          # > 0 under the open water
    tall_ok = _smooth(0.0, 22.0, -box)
    cap = (108 - gz) + ((TOP - gz) / 1.2 - (108 - gz)) * tall_ok  # tallest a plant may stand here (mm above the soil)
    back = _smooth(160, 190, y)
    lstrip = _smooth(44, 30, x)
    rstrip = _smooth(256, 270, x)

    # --- wood ---
    barks = [materials.driftwood("JungleBark_a", tint=(0.80, 0.58, 0.40), value=0.85),
             materials.driftwood("JungleBark_b", tint=(0.66, 0.48, 0.33), value=0.70),
             materials.dark_bogwood("JungleBark_c")]
    skins, woods = [], []
    for k, spec in enumerate(WOOD):
        skin, n_wood, zb = _grow_wood(spec)
        obj = wood.finish(skin, spec[0], barks[k], subdivide=1)
        moss_w, epi_w = _wood_weights(skin, n_wood, zb, spec[3], spec[5])
        set_point_attribute(obj, "moss", moss_w)
        set_point_attribute(obj, "epi", epi_w)
        skins.append((skin, n_wood, zb))
        woods.append((obj, moss_w, epi_w))
        made.append(obj)
    print("[jungle] wood " + ", ".join(str(c) for c in _cells(skins)))

    # shrimp perches on bare bark, found before the moss goes on
    perches = []
    for k, rel in ((0, 0.40), (1, 0.45)):
        skin, n_wood, zb = skins[k]
        want = zb + rel * (WOOD[k][3] - zb)
        perches.append(_perch(woods[k][0], skin, n_wood, woods[k][1],
                              lambda v, n, w, want=want: n.z - 1.2 * n.y - abs(v.z / MM - want) / 14.0
                              - 999 * (w > 0.05 or n.z < 0.45)))

    moss_src = (plants.moss_fronds(variants=5, seed=13, material=plants.leaf_material(
                    "JMossDeep", base=(0.014, 0.045, 0.008), tip=(0.085, 0.22, 0.028), translucency=0.3, roughness=0.5)) +
                plants.moss_fronds(variants=3, seed=29, length_mm=(4, 8), material=plants.leaf_material(
                    "JMossOlive", base=(0.024, 0.040, 0.009), tip=(0.13, 0.20, 0.035), translucency=0.3, roughness=0.5)) +
                plants.moss_fronds(variants=2, seed=37, length_mm=(6, 11), material=plants.leaf_material(
                    "JMossNew", base=(0.03, 0.08, 0.014), tip=(0.17, 0.34, 0.06), translucency=0.35, roughness=0.45)))
    moss = collection_of("JMossSrc", moss_src)
    area = sum(_weighted_area(s[0], w[1]) for s, w in zip(skins, woods))
    dens = min(1500000.0, 14000 / max(area, 1e-9))
    buce_mat = _leaf_mat("JBuce", (0.10, 0.26, 0.07), (0.012, 0.045, 0.028), rough=0.34, veins=0.0, spots=0.2,
                         translucency=0.22, mottle=400.0)
    buce = collection_of("JBuceSrc", _rosette("JBuce", buce_mat, 6, 301, leaves=(5, 9), length=(11, 19), ratio=(0.30, 0.42),
                                              stalk=(0.25, 0.4), elev=(15, 70), arch=(0.3, 0.8), ruffle=0.05, waves=2.5,
                                              segs=5, nu=3, stalk_r=0.35))
    for k, (obj, _, _) in enumerate(woods):
        scatter(obj, moss, name=f"Moss_{k}", density=dens, weight_attr="moss", seed=40 + k, scale=(0.5, 1.1),
                align="NORMAL", tilt=0.6, sway=(0.04, 30.0, 0.3))
        scatter(obj, buce, name=f"Buce_{k}", min_distance=11 * MM, weight_attr="epi", seed=50 + k, scale=(0.6, 1.15),
                align="NORMAL", tilt=0.5, sway=(0.03, 20.0, 0.3))

    # --- specimens: swords, Anubias, Java fern, each its own plant ---
    kinds = {
        "sword": (_leaf_mat("JSword", (0.20, 0.42, 0.07), (0.035, 0.13, 0.03), veins=0.22, vein_freq=42.0, spots=0.3,
                            translucency=0.40, rough=0.40),
                  dict(leaves=(11, 15), length=(62, 104), ratio=(0.34, 0.44), stalk=(0.42, 0.55), elev=(22, 84), drop=14,
                       arch=(0.5, 1.1), belly=0.40, ruffle=0.035, waves=2.0, twist=0.35, segs=9, nu=5, stalk_r=0.9)),
        "anubias": (_leaf_mat("JAnubias", (0.10, 0.27, 0.05), (0.012, 0.055, 0.016), veins=0.3, vein_freq=70.0, spots=0.45,
                              translucency=0.18, rough=0.30, rib=1.4),
                    dict(leaves=(7, 11), length=(30, 50), ratio=(0.56, 0.70), stalk=(0.38, 0.52), elev=(32, 78), drop=38,
                         arch=(0.4, 0.9), belly=0.38, ruffle=0.03, waves=1.5, twist=0.3, segs=8, nu=5, rhizome=20.0,
                         stalk_r=0.9, tip=0.62, fold=0.22)),
        "fern": (_leaf_mat("JFern", (0.10, 0.28, 0.05), (0.014, 0.060, 0.018), veins=0.25, vein_freq=95.0, spots=0.5,
                           translucency=0.26, rough=0.42),
                 dict(leaves=(7, 11), length=(40, 84), ratio=(0.11, 0.16), stalk=(0.08, 0.16), elev=(25, 82), drop=4,
                      arch=(0.5, 1.2), belly=0.38, ruffle=0.10, waves=3.5, twist=0.9, segs=10, nu=5, rhizome=16.0,
                      stalk_r=0.5)),
    }
    specimens, keep = [], []
    for k, (kind, where, size) in enumerate(SPECIMENS):
        mat, kw = kinds[kind]
        obj = _rosette(f"J{kind}{k}", mat, 1, 700 + 13 * k, **kw)[0]
        if isinstance(where[0], int) and where[0] < len(WOOD) and where[1] <= 1.0:
            skin, n_wood, zb = skins[where[0]]
            want = zb + where[1] * (WOOD[where[0]][3] - zb)
            i = max(range(n_wood), key=lambda q: -abs(skin.verts[q].z / MM - want) / 6.0
                    + (skin.verts[q] - skin.centres[q]).normalized().dot(Vector((0.0, -0.8, 0.6))))
            pos, nrm = skin.verts[i], (skin.verts[i] - skin.centres[i]).normalized()
            up = (nrm * 0.7 + Z * 0.8).normalized()
        else:
            up, pos = Z, Vector(bed.at(where[0], where[1], lift_mm=-1.0))
        specimens.append(_stand(obj, pos, up, yaw=rng.uniform(0, 6.283), scale=size) + (rng.uniform(0, 6.283),))
        made.append(obj)
        reach = max(math.hypot(v.co.x, v.co.y) for v in obj.data.vertices) * size / MM
        top = max((obj.matrix_world @ v.co).z for v in obj.data.vertices) / MM
        keep.append((round(pos.x / MM), round(pos.y / MM), round(reach * 0.75), round(top)))
    print("[jungle] specimens " + ", ".join(str(c) for c in keep))

    # --- Vallisneria ---
    val_mat = _leaf_mat("JVallis", (0.22, 0.42, 0.08), (0.045, 0.16, 0.03), rib=1.0, veins=0.0, stripes=0.16, spots=0.4,
                        translucency=0.45, rough=0.38, mottle=160.0, variation=0.15)
    ribbons = []
    for k, (vx, vy, n) in enumerate(VALLIS):
        obj, info = _vallis(f"Vallis_{k}", val_mat, vx, vy, n, 900 + k)
        ribbons.append(info)
        made.append(obj)

    # --- stems: three kinds pushing up through each other at the back and sides ---
    # None is planted where it would stand in front of the wood and hide it.
    wpts = np.array([(c.x / MM, c.y / MM) for skin, n_wood, _ in skins[:2] for c in skin.centres[:n_wood:6] if c.z / MM < 225])
    shows = np.empty(len(x))
    for i0 in range(0, len(x), 4000):
        dx = x[i0:i0 + 4000, None] - wpts[None, :, 0]
        dy = y[i0:i0 + 4000, None] - wpts[None, :, 1]
        shows[i0:i0 + 4000] = np.where(dy < 10.0, np.hypot(dx, np.maximum(dy, 0.0) + 0.35 * np.minimum(dy, 0.0)), 1e3).min(axis=1)
    shows = _smooth(12.0, 30.0, shows)
    H = 112.0
    want = (92 + 66 * _noise(x, y, 5.0, 0.9)) * (0.72 + 0.28 * _smooth(160, 245, y)) * (1.0 - 0.32 * rstrip * (1 - back))
    tall = np.minimum(want, cap)
    grows = (cap > 60) & (tall > 50)
    everywhere = np.maximum.reduce([back, 0.9 * _smooth(52, 34, x) * _smooth(50, 85, y), 0.8 * rstrip])
    far = 0.6 * _smooth(236, 252, y)                           # the last few centimetres are planted solid
    for k, (key, zone, a, f, thr, gap, margin, lo) in enumerate((
            ("rotala", everywhere, 0.3, 1.9, (0.42, 0.58), 5.6, 19, 0.50),
            ("hygro", everywhere * (1 - 0.6 * lstrip), 2.1, 1.6, (0.46, 0.62), 11.0, 27, 0.55),
            ("ludwigia", np.exp(-(((x - 196) / 40) ** 2 + ((y - 218) / 30) ** 2)) * 1.3, 4.0, 1.6, (0.40, 0.58), 7.5, 16, 0.50))):
        patch = np.maximum(_smooth(thr[0], thr[1], _noise(x, y, a, f)), far * (k < 2))
        w = np.clip(zone, 0, 1) * patch * grows * _smooth(margin - 3, margin + 2, wall) * shows
        paint(f"w_{key}", w)
        paint(f"s_{key}", np.clip(tall * (0.62 + 0.62 * _noise(x, y, a + 7.0, 3.4)) / H, lo, cap / H))
        src = collection_of(f"J{key}Src", _stems(key, H, 8, 100 + 31 * k))
        mod = scatter(carrier(f"Stems_{key}"), src, name=f"Stems_{key}", min_distance=gap * MM, weight_attr=f"w_{key}",
                      seed=60 + k, scale=(0.80, 1.18), align="UP", tilt=0.10, keep_surface=False, sway=(0.035, 6.0, 0.28))
        _size_by(mod, f"s_{key}")

    # --- crypts: two forms in loose, overlapping stands that step down into the clearing ---
    crypt_kw = dict(leaves=(8, 13), length=(44, 74), ratio=(0.20, 0.28), stalk=(0.30, 0.44), elev=(24, 82), drop=10,
                    arch=(0.4, 1.0), belly=0.40, ruffle=0.07, waves=3.5, twist=0.5, segs=8, nu=5)
    crypt_size = np.clip(np.minimum.reduce([0.42 + 0.62 * _smooth(10, -28, d) + 0.25 * (_noise(x, y, 8.0, 2.2) - 0.5),
                                            cap / 58.0, (wall - 4) / 60.0]), 0.25, 1.15)
    band = _smooth(30, 6, d) * (wall > 17) * (1.0 - 0.55 * back * _smooth(200, 240, y))
    for k, (key, young, old, a, thr, gap) in enumerate((
            ("green", (0.17, 0.36, 0.06), (0.030, 0.105, 0.022), 1.1, (0.33, 0.55), 17.0),
            ("bronze", (0.20, 0.25, 0.07), (0.075, 0.062, 0.028), 3.6, (0.46, 0.64), 19.0))):
        mat = _leaf_mat(f"JCrypt_{key}", young, old, veins=0.18, vein_freq=60.0, spots=0.4, translucency=0.34, rough=0.44)
        paint(f"w_crypt_{key}", band * _smooth(thr[0], thr[1], _noise(x, y, a, 1.5)))
        paint(f"s_crypt_{key}", crypt_size * (1.0 if k == 0 else 0.88))
        src = collection_of(f"JCrypt_{key}Src", _rosette(f"JCrypt_{key}", mat, 7, 200 + 17 * k, **crypt_kw))
        mod = scatter(carrier(f"Crypt_{key}"), src, name=f"Crypt_{key}", min_distance=gap * MM, weight_attr=f"w_crypt_{key}",
                      seed=70 + k, scale=(0.78, 1.15), align="UP", tilt=0.14, keep_surface=False, sway=(0.045, 8.0, 0.3))
        _size_by(mod, f"s_crypt_{key}")

    # --- the clearing's floor: low growth in patches round a stretch of bare soil ---
    bare = _smooth(1.15, 0.80, np.sqrt(((x - 152) / 64.0) ** 2 + ((y - 60) / 56.0) ** 2) + 0.3 * (_noise(x, y, 7.7, 1.8) - 0.5))
    low_mat = _leaf_mat("JLow", (0.14, 0.34, 0.06), (0.03, 0.10, 0.025), veins=0.0, spots=0.2, translucency=0.36, mottle=380.0)
    low = _rosette("JLow", low_mat, 6, 400, leaves=(6, 10), length=(14, 26), ratio=(0.26, 0.36), stalk=(0.2, 0.35),
                   elev=(18, 75), arch=(0.3, 0.8), ruffle=0.04, segs=5, nu=3, stalk_r=0.4)
    paint("w_low", _smooth(-26, -8, d) * _smooth(44, 16, d) * _smooth(0.40, 0.60, _noise(x, y, 3.3, 1.7)) * (wall > 15) * (1 - bare))
    scatter(carrier("LowRosettes"), collection_of("JLowSrc", low), name="LowRosettes", min_distance=8.5 * MM, weight_attr="w_low",
            seed=81, scale=(0.6, 1.2), align="UP", tilt=0.2, keep_surface=False, sway=(0.05, 10.0, 0.32))
    paint("w_sprig", _smooth(0.50, 0.66, _noise(x, y, 4.4, 1.3)) * _smooth(-10, 8, d) * _smooth(8, 14, wall) * (1 - bare))
    sprigs = collection_of("JSprigSrc", plants.carpet_sprigs(variants=6, seed=17, material=plants.leaf_material(
        "JSprig", base=(0.03, 0.11, 0.02), tip=(0.14, 0.34, 0.05), translucency=0.4, roughness=0.34)))
    scatter(carrier("Sprigs"), sprigs, name="Sprigs", min_distance=4.2 * MM, weight_attr="w_sprig", seed=83,
            scale=(0.6, 1.2), align="NORMAL", tilt=0.3, keep_surface=False, sway=(0.06, 12.0, 0.35))
    paint("w_grass", _smooth(0.52, 0.68, _noise(x, y, 6.1, 2.1)) * _smooth(-34, -6, d) * _smooth(60, 30, d) * _smooth(8, 16, wall) * (1 - 0.8 * bare))
    tufts = collection_of("JGrassSrc", plants.hairgrass_tufts(variants=5, blades=(7, 13), height_mm=(16, 38), seed=5))
    scatter(carrier("Grass"), tufts, name="Grass", min_distance=7.0 * MM, weight_attr="w_grass", seed=85,
            scale=(0.55, 1.15), align="UP", tilt=0.3, keep_surface=False, sway=(0.07, 9.0, 0.35))
    paint("w_litter", _smooth(-20, 5, d) * (wall > 22) * (0.3 + 0.7 * _noise(x, y, 9.0, 1.4)) * (0.5 + 0.5 * bare))
    litter = collection_of("JLitterSrc", plants.dead_leaves(variants=6, seed=91, length_mm=(11, 22)))
    scatter(carrier("Litter"), litter, name="Litter", density=1100.0, weight_attr="w_litter", seed=87, scale=(0.7, 1.1),
            align="NORMAL", tilt=0.25, keep_surface=False)

    # --- floating plants over the right rear ---
    roots = plants.leaf_material("JFloaterRoot", base=(0.085, 0.075, 0.040), tip=(0.035, 0.030, 0.018), translucency=0.15,
                                 roughness=0.6, variation=0.35)
    for name, src, r_in, r_out, gap, seed in (("Floaters", plants.floaters(variants=6, seed=41, root_material=roots), 30, 1e9, 12.5, 71),
                                               ("FloatersSmall", plants.floaters(variants=5, seed=47, leaf_mm=(5, 9), root_material=roots), 17, 36, 7.0, 73)):
        for o in src:
            for v in o.data.vertices:
                if v.co.z < 0:
                    v.co.z *= 0.72
            o.data.update()
        helper = _surface_helper(name + "Raft")
        hx = np.array([v.co.x / MM for v in helper.data.vertices])
        hy = np.array([v.co.y / MM for v in helper.data.vertices])
        hw = np.minimum.reduce([hx - 5, 295 - hx, hy - 5, 295 - hy])
        set_point_attribute(helper, "raft", list((_raft(hx, hy) * ((hw > r_in) & (hw <= r_out))).astype(float)))
        scatter(helper, collection_of(name + "Src", src), name=name, min_distance=gap * MM, weight_attr="raft", seed=seed,
                scale=(0.75, 1.15), align="UP", tilt=0.05, keep_surface=False, sway=(0.015, 6.0, 0.15))
        made.append(helper)

    # --- animals ---
    school = fish.School()
    made += school.objects
    crew = []
    for i, (pos, up) in enumerate(perches):
        s = shrimp.Shrimp(f"Shrimp_{i}", size=(0.95, 0.85)[i])
        crew.append((s, Vector(pos), tuple(up), (2.7, 0.5)[i], 0.9 * i))
        made.append(s.obj)
    walker = shrimp.Shrimp("Shrimp_soil", size=1.0)
    nerite = snail.Snail("Nerite")
    made += [walker.obj, nerite.obj]

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["jungle"] = dict(ribbons=ribbons, specimens=specimens, crew=crew, walker=walker, snail=nerite)
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
    j = ctx["jungle"]
    f0, _ = shot_frames(SCAPE)
    t = (frame - f0) / 30.0

    # Vallisneria: a slow wave runs up each ribbon and out along the trailing part
    side = np.array([-math.sin(CURRENT), math.cos(CURRENT), 0.0])
    for r in j["ribbons"]:
        amp = np.minimum(4.0, 3.0 * (r["arc"] / r["column"]) ** 1.4) * MM
        co = r["rest"] + (amp * np.sin(r["arc"] / 85.0 - t * 1.5 + r["phase"]))[:, None] * side
        co[:, :2] = np.clip(co[:, :2], 7.5 * MM, 292.5 * MM)
        r["obj"].data.vertices.foreach_set("co", co.astype(np.float32).ravel())
        r["obj"].data.update()

    # specimens rock a degree or so about their feet
    for obj, pos, rot, ph in j["specimens"]:
        sway = Matrix.Rotation(0.022 * math.sin(t * 0.9 + ph), 4, "X") @ Matrix.Rotation(0.018 * math.sin(t * 0.7 + 2 * ph), 4, "Y")
        obj.matrix_world = Matrix.Translation(pos) @ sway @ rot

    for s, pos, up, yaw, t0 in j["crew"]:
        s.pose(pos, yaw=yaw + 0.15 * math.sin(t / 3.1 + t0), t=t0 + t, up=up, walk=0.0)
    bed = ctx["bed"]
    w = 0.18
    sx = 118.0 + t * shrimp.Shrimp.WALK_SPEED * w / MM
    sy = 38.0 + 3.0 * math.sin(t * 0.25)
    j["walker"].pose(Vector(bed.at(sx, sy, lift_mm=0.9)), yaw=0.08 * math.cos(t * 0.25), t=t, up=tuple(_ground_normal(sx, sy)), walk=w)
    nx, ny = 186.0 - 0.5 * t, 29.0
    j["snail"].pose(Vector(bed.at(nx, ny, lift_mm=0.3)), yaw=math.pi - 0.1, t=t, up=tuple(_ground_normal(nx, ny)))
