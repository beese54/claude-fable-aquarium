"""Scape 4: Blackwater Amazon.

A shaded forest stream in the Rio Negro basin. Roots and branches of a tree on
the bank reach down through tea-coloured water from the back and sides; two
heavier pieces of fallen wood lie on a sandy bed buried in leaf litter; a raft
of floating plants shades the back of the tank and hangs its roots into the
water. The middle and front are left open for a shoal of neon tetras, whose
stripes are the brightest colour in the picture. Coordinates are millimetres:
X left to right, Y front to back.
"""
import hashlib
import math
import random

from mathutils import Vector

import assets
import fish
import materials
import plants
import substrate
import timeline
import wood
from bl_common import MM, new_material, new_mesh_object
from scatter import collection_of, scatter, set_point_attribute

SCAPE = "04_blackwater"

# Canopy-filtered light: dimmer and warmer than the default white LED.
LIGHT = dict(watts=52.0, colour=(1.0, 0.93, 0.82), fill=0.03, bounce=0.045)
# Tannin-stained water. Absorption per metre is density * (1 - tint), so blue
# dies first: the back of the tank (30 cm away) goes amber-brown while fish
# 15 cm in keep most of their colour. Red is absorbed too, so depth darkens
# rather than only warms. Haze is kept faint and amber: any more, lit evenly by
# the broad lamp, greys the whole tank into murk instead of drawing shafts.
#
# Owner feedback on the first still: too dark; brighter, but still short of
# clear water. The stain is about a third weaker than it was (weak tea rather
# than strong) and the lamp a little stronger, so the bed and wood are read
# clearly and the far glass is amber, not brown.
WATER = dict(tint=(0.78, 0.52, 0.20), density=9.5, scatter=0.06, scatter_colour=(0.92, 0.68, 0.36))

# Water the film's shoal may use while this scape is on screen (src/sim/school.py):
# the open middle and front, below the floaters' roots (they reach down to about
# 205 mm). The cylinders are the wood, as printed by _obstacles() at build time.
SWIM = {
    "box": (45, 255, 28, 178, 104, 176),
    "floor": (62, 92),
    "obstacles": [
        (12, 132, 23, 276), (12, 156, 23, 280), (36, 132, 23, 280), (36, 156, 23, 283), (60, 108, 23, 113),
        (60, 156, 23, 256), (60, 180, 23, 246), (60, 204, 23, 213), (84, 108, 23, 151), (84, 132, 23, 213),
        (84, 156, 23, 238), (84, 180, 23, 238), (84, 204, 23, 232), (108, 84, 23, 111), (108, 108, 23, 151),
        (108, 132, 23, 196), (108, 156, 23, 210), (108, 180, 23, 134), (108, 204, 23, 230), (132, 132, 23, 149),
        (132, 156, 23, 130), (132, 180, 23, 158), (132, 204, 23, 166), (156, 132, 23, 133), (180, 204, 23, 259),
        (204, 204, 23, 268), (228, 204, 23, 277), (252, 180, 23, 262), (252, 204, 23, 253), (276, 156, 23, 278),
        (276, 180, 23, 276),
    ],
}

# A curtain of fine roots from the bank hangs along the back glass: x range,
# y range, how many, length range (mm).
CURTAIN = ((30, 270), (268, 286), 22, (50, 170))

LEVEL = 268.0           # water surface
LO, HI = 9.0, 291.0     # keep everything this far inside the outer walls

# Hanging pieces: name, start (x, y, z), direction, length, radius, seed, branching
# depth, gravity, wander. They start at or just above the surface at the back and
# sides, as roots and branches of a tree on the bank would.
HANGING = [
    ("Root_back_left",  (28, 260, 280),  (0.45, -0.35, -0.55), 185, 16.0, 5,  3, 0.08, 0.55),
    ("Root_back_right", (272, 268, 279), (-0.62, -0.30, -0.45), 185, 14.0, 17, 3, 0.09, 0.60),
    ("Root_left",       (20, 150, 276),  (0.75, 0.30, -0.45),  140, 9.0,  323, 3, 0.11, 0.65),
    ("Root_back",       (168, 284, 275), (-0.12, -0.45, -0.80), 160, 8.5, 31, 3, 0.14, 0.65),
    ("Root_right",      (285, 165, 272), (-0.78, 0.10, -0.45), 115, 7.0,  37, 3, 0.16, 0.70),
]

# Pieces lying on the bed: name, start (x, y), heading (deg), length, radius, seed.
# Grown rather than imported: the scanned trunks have flat saw-cut ends, which
# show as pale discs and no stream would have.
FALLEN = [
    ("Log_back",    (34, 238), -12, 190, 13.0, 3),
    ("Branch_left", (40, 112), 28,  120, 6.5,  8),
]


def _smooth(e0, e1, v):
    t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def ground(x, y):
    """Sand height in mm: 60 at the front glass rising gently to about 85 at the
    back, then a low bank against the back glass (as at a stream's edge) that
    hides the bright floor behind the tank, which otherwise shows through the
    back pane as a pale band just above the bed. Banked a little into the back
    corners and under the fallen wood."""
    h = 60 + 23 * (max(y, 0) / 295) ** 1.1 + 9 * _smooth(225, 292, y)
    h += 6 * math.exp(-(((x - 20) / 45) ** 2 + ((y - 280) / 50) ** 2))
    h += 5 * math.exp(-(((x - 280) / 45) ** 2 + ((y - 280) / 50) ** 2))
    h += 4 * math.exp(-(((x - 120) / 70) ** 2 + ((y - 222) / 30) ** 2))
    h += 1.6 * math.sin(x * 0.047 + 1.1) * math.cos(y * 0.039 + 0.3) + 0.7 * math.sin(x * 0.16 + y * 0.11)
    return h


def _noise(x, y, a=0.0, f=1.0):
    """Smooth 0..1 patchiness."""
    x, y = x * f, y * f
    n = math.sin(x * 0.071 + a + 1.6 * math.sin(y * 0.053 + a)) * math.cos(y * 0.064 + 1.2 * math.sin(x * 0.045 - a))
    n2 = math.sin(x * 0.19 + 0.7 + a) * math.sin(y * 0.23 + 1.9 - a)
    return max(0.0, min(1.0, 0.5 + 0.42 * n + 0.18 * n2))


def _ground_key():
    """The fish simulation is cached by key and cannot hash the ground function,
    so the key carries a fingerprint of the ground's shape."""
    s = ",".join(f"{ground(x, y):.2f}" for x in range(0, 301, 30) for y in range(0, 301, 30))
    return f"{SCAPE}_{hashlib.sha1(s.encode()).hexdigest()[:6]}"


# --- materials -------------------------------------------------------------------

def _sand():
    """Fine river sand, a little darker and warmer than beach sand: it is stained too."""
    return materials.pbr_box("BlackwaterSand", "damp_beach_sand", scale=8.0, tint=(0.95, 0.84, 0.66),
                             saturation=0.85, value=0.95, bump=0.5, bump_distance=0.0006, roughness=0.65)


def _litter_floor():
    """Scanned leaf litter for the bed under the modelled leaves, so the floor
    reads as solid litter however the modelled ones fall. Scaled so its leaves
    come out at two to four centimetres, darkened and reddened as if soaked."""
    return materials.pbr_box("LitterFloor", "dry_decay_leaves", scale=2.4, tint=(0.80, 0.55, 0.36), saturation=0.9,
                             value=0.42, bump=1.0, bump_distance=0.0015, roughness=0.55)


def _curled(objs, seed):
    """Bend some of the leaf variants: a few cupped (both ends lifting), a few
    rolled along the midrib as drying leaves do, the rest left nearly flat. The
    kit's leaves are all gently folded sheets, which en masse read as chips."""
    rng = random.Random(seed)
    for o in objs:
        kind = rng.random()
        if kind < 0.4:
            continue
        me = o.data
        xs = [v.co.x for v in me.vertices]
        half = (max(xs) - min(xs)) / 2 or 1.0
        a = rng.uniform(0.18, 0.40)
        for v in me.vertices:
            if kind < 0.7:                                   # cupped: ends lift
                v.co.z += a * half * (v.co.x / half) ** 2
            else:                                            # rolled across the midrib
                v.co.z += a * 1.1 * v.co.y ** 2 / (0.3 * half)
        me.update()
    return objs


def _leaf_material(name, dark, mid, light):
    """A soaked fallen leaf. Each leaf takes its own brown from the range given;
    within a leaf the colour is mottled, the edges have rotted darker, the
    midrib and side veins stand out paler, and the surface has a wet sheen.
    (The kit's dead-leaf shader is one flat colour per leaf, which reads as paper.)"""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links

    def math_node(op, a, b=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for sock, v in zip(n.inputs, (a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                L.new(v, sock)
        return n.outputs[0]

    def mul_rgb(a, b):
        m = N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = "MULTIPLY"
        m.inputs["Factor"].default_value = 1.0
        L.new(a, m.inputs["A"])
        L.new(b, m.inputs["B"])
        return m.outputs["Result"]

    info = N.new("ShaderNodeObjectInfo")
    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].color, e[1].color = (*dark, 1), (*light, 1)
    e.new(0.5).color = (*mid, 1)
    L.new(info.outputs["Random"], ramp.inputs["Fac"])

    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    across = math_node("ABSOLUTE", math_node("SUBTRACT", sep.outputs["X"], 0.5))      # 0 on the midrib .. 0.5 at the edge
    along = sep.outputs["Y"]

    # mottling, different on every leaf
    vec = N.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["X"], vec.inputs[0])
    L.new(math_node("MULTIPLY", along, 2.2), vec.inputs[1])
    L.new(math_node("MULTIPLY", info.outputs["Random"], 37.0), vec.inputs[2])
    mott = N.new("ShaderNodeTexNoise")
    mott.inputs["Scale"].default_value = 4.0
    mott.inputs["Detail"].default_value = 5.0
    mott.inputs["Roughness"].default_value = 0.6
    L.new(vec.outputs[0], mott.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    mr.inputs["To Min"].default_value = 0.55
    mr.inputs["To Max"].default_value = 1.35
    L.new(mott.outputs["Fac"], mr.inputs["Value"])
    col = mul_rgb(ramp.outputs["Color"], mr.outputs["Result"])

    # rotted, darker margin
    edge = N.new("ShaderNodeMapRange")
    edge.inputs["From Min"].default_value = 0.30
    edge.inputs["From Max"].default_value = 0.50
    edge.inputs["To Min"].default_value = 1.0
    edge.inputs["To Max"].default_value = 0.45
    L.new(math_node("ADD", across, math_node("MULTIPLY", mott.outputs["Fac"], 0.12)), edge.inputs["Value"])
    col = mul_rgb(col, edge.outputs["Result"])

    # midrib and side veins: thin pale lines sweeping out toward the tip
    rib = math_node("LESS_THAN", across, 0.022)
    side = math_node("SINE", math_node("MULTIPLY", math_node("SUBTRACT", along, math_node("MULTIPLY", across, 0.9)), 75.0))
    veins = math_node("MULTIPLY", math_node("GREATER_THAN", side, 0.93), math_node("LESS_THAN", across, 0.40))
    vein = math_node("MAXIMUM", rib, math_node("MULTIPLY", veins, 0.6))
    lift = N.new("ShaderNodeMapRange")
    lift.inputs["To Min"].default_value = 1.0
    lift.inputs["To Max"].default_value = 1.6
    L.new(vein, lift.inputs["Value"])
    col = mul_rgb(col, lift.outputs["Result"])

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.62
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0002
    L.new(math_node("ADD", vein, mott.outputs["Fac"]), bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(col, trans.inputs["Color"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.06
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _wet_wood(name, k):
    """Soaked wood: two bark scans, both pulled dark and a little red."""
    if k % 2 == 0:
        return materials.pbr_box(name, "jolcham_oak_bark_01", scale=10.0, stretch=(1.0, 1.0, 0.5), tint=(0.50, 0.36, 0.26),
                                 saturation=0.6, value=0.6, bump=1.0, bump_distance=0.002, roughness=0.65, use_ao=False)
    return materials.driftwood(name, tint=(0.56, 0.36, 0.22), value=0.52)


# --- wood ------------------------------------------------------------------------

class _Skin(wood._Skin):
    """The kit's tube skin, remembering each vertex's branch radius so the knot
    displacement can be scaled to it: knots a few millimetres high suit a
    trunk but would tear a rootlet a millimetre thick inside out."""

    def __init__(self, sides):
        super().__init__(sides)
        self.vr = []

    def tube(self, pts, radii):
        super().tube(pts, radii)
        self.vr += [r for r in radii for _ in range(self.sides)] + [radii[0], radii[-1]]
        # Thick ends are broken off, not sharpened: pull their cap almost flat.
        if radii[-1] > 1.5 * MM:
            d = (pts[-1] - pts[-2]).normalized()
            self.verts[-1] = tuple(pts[-1] + d * radii[-1] * 0.15)


def _finish(skin, name, material, subdivide=1):
    obj = wood.finish(skin, name, material, subdivide=subdivide)
    vg = obj.vertex_groups.new(name="Knots")
    for i, r in enumerate(skin.vr):
        vg.add([i], min(1.0, (r / MM / 6.0) ** 1.5), "REPLACE")
    obj.modifiers["Knots"].vertex_group = "Knots"
    return obj


def _keep_inside(skin, anchor):
    """Draw a piece in toward its anchor until it clears the glass, and keep any
    part that dives into the bed from going through the tank's floor."""
    s = 1.0
    for v in skin.verts:
        for a in (0, 1):
            off = v[a] - anchor[a]
            if v[a] > HI * MM and off > 0:
                s = min(s, (HI * MM - anchor[a]) / off)
            elif v[a] < LO * MM and off < 0:
                s = min(s, (LO * MM - anchor[a]) / off)
    out = []
    for v in skin.verts:
        x = anchor[0] + (v[0] - anchor[0]) * s
        y = anchor[1] + (v[1] - anchor[1]) * s
        z = max(v[2], (ground(x / MM, y / MM) - 14) * MM)
        out.append((x, y, z))
    skin.verts = out
    skin.tips = [(Vector((anchor[0] + (p[0] - anchor[0]) * s, anchor[1] + (p[1] - anchor[1]) * s, p[2])), d, r)
                 for p, d, r in skin.tips]
    return skin


def _grow_hanging(spec):
    name, start, direction, length, radius, seed, depth, gravity, wander = spec
    p0 = Vector(start) * MM
    skin = wood.grow(p0, Vector(direction).normalized(), length, radius, seed=seed, depth=depth,
                     children=(2, 3), wander=wander, gravity=gravity, spread=(0.45, 1.0),
                     child_length=0.62, child_radius=0.55, taper=0.26, skin=_Skin(10))
    # A few fine rootlets trail from the higher branch ends, hanging almost straight down.
    rng = random.Random(seed + 500)
    for i, (p, d, r) in enumerate(list(skin.tips)):
        if p.z < 160 * MM or rng.random() < 0.55:
            continue
        rd = (Vector((d.x, d.y, 0.0)) * 0.4 + Vector((0, 0, -1))).normalized()
        wood.grow(p - d * r, rd, rng.uniform(22, 60), max(0.7, r * 0.75), seed=seed * 31 + i, depth=1,
                  children=(1, 2), wander=0.7, gravity=0.6, spread=(0.3, 0.7), taper=0.4, skin=skin)
    skin = _keep_inside(skin, p0)
    if name == "Root_back":
        _curtain(skin, seed)
    return skin


def _curtain(skin, seed):
    """Fine roots from the bank hanging just in front of the back glass. They
    break up what would otherwise be a flat tinted wall behind the scape. Each
    is only a millimetre or two thick, so any that wander toward the glass are
    simply held off it point by point."""
    (x0, x1), (y0, y1), n, (l0, l1) = CURTAIN
    first = len(skin.verts)
    rng = random.Random(seed + 77)
    for i in range(n):
        x = x0 + (x1 - x0) * (i + rng.uniform(0.1, 0.9)) / n
        p = Vector((x, rng.uniform(y0, y1), rng.uniform(262, 272))) * MM
        d = Vector((rng.uniform(-0.25, 0.25), rng.uniform(-0.3, 0.05), -1.0)).normalized()
        L = rng.uniform(l0, l1) * (0.6 + 0.4 * math.sin(x * 0.03 + 1.0) ** 2)
        wood.grow(p, d, L, rng.uniform(0.8, 2.4), seed=seed * 101 + i, depth=2, children=(1, 2), wander=0.6,
                  gravity=0.5, spread=(0.25, 0.6), taper=0.35, child_length=0.5, skin=skin)
    lo, hi = (LO + 3) * MM, (HI - 3) * MM
    skin.verts[first:] = [(min(max(v[0], lo), hi), min(max(v[1], lo), hi), max(v[2], (ground(v[0] / MM, v[1] / MM) - 10) * MM))
                          for v in skin.verts[first:]]


def _twigs(rng, n=16):
    """Short thin twigs and a few seed pods lying flat among the leaves. Built
    as tubes whose centre line follows the bed, so they lie on it rather than
    poking out of it as a grown piece would."""
    skin = _Skin(6)

    def lying(x, y, az, L, R, bendy, pod):
        segs = max(4, int(L / 4))
        pts, radii = [], []
        for k in range(segs + 1):
            t = k / segs
            a = az + bendy * math.sin(t * 3.0)
            if k:
                x += math.cos(a) * L / segs
                y += math.sin(a) * L / segs
            if _edge_dist(x, y) < R + 6:
                return None                     # would reach the glass: leave this one out
            r = R * (math.sin(math.pi * (0.08 + 0.84 * t)) ** 0.5 if pod else 1 - 0.55 * t)
            pts.append(Vector((x * MM, y * MM, (ground(x, y) + 1.6 + R * 0.8) * MM)))
            radii.append(r * MM)
        skin.tube(pts, radii)
        return x, y, a

    for i in range(n):
        x, y = rng.uniform(30, 270), rng.uniform(25, 265)
        az = rng.uniform(0, 2 * math.pi)
        if rng.random() < 0.3:
            lying(x, y, az, rng.uniform(7, 13), rng.uniform(2.0, 3.2), 0.1, True)
            continue
        L, R = rng.uniform(22, 58), rng.uniform(0.8, 1.7)
        end = lying(x, y, az, L, R, rng.uniform(-0.4, 0.4), False)
        if end and rng.random() < 0.5:          # a side shoot
            ex, ey, ea = end
            t = rng.uniform(0.3, 0.6)
            lying(x + (ex - x) * t, y + (ey - y) * t, ea + rng.choice((-1, 1)) * rng.uniform(0.5, 0.9),
                  L * rng.uniform(0.25, 0.45), R * 0.6, 0.2, False)
    return skin


def _grow_fallen(spec):
    """A sunken log: a thick, uneven trunk whose centre line follows the bed,
    sunk to about half its girth, both ends dipping under the litter, with a few
    broken side branches. Built from tubes so it lies on the bed instead of
    floating off it as a freely grown piece would."""
    name, (x, y), heading, length, R, seed = spec
    rng = random.Random(seed)
    skin = _Skin(12)
    segs = int(length / 4)
    az = math.radians(heading)
    pts, radii, marks = [], [], []
    ph = rng.uniform(0, 6.3)
    for k in range(segs + 1):
        t = k / segs
        a = az + 0.25 * math.sin(t * 2.6 + ph)
        if k:
            x += math.cos(a) * length / segs
            y += math.sin(a) * length / segs
        r = R * (1 - 0.35 * t) * (1 + 0.10 * math.sin(k * 0.9 + ph) + 0.06 * math.sin(k * 2.3))
        dip = 0.55 * r * (1 - _smooth(0.0, 0.12, t) * _smooth(1.0, 0.88, t))   # ends sink under the litter
        pts.append(Vector((x * MM, y * MM, (ground(x, y) + 0.45 * r - dip) * MM)))
        radii.append(r * MM)
        marks.append((pts[-1], a, r))
    skin.tube(pts, radii)
    for i in range(rng.randint(2, 4)):
        p, a, r = marks[int(len(marks) * rng.uniform(0.15, 0.8))]
        side = rng.choice((-1, 1))
        d = Vector((math.cos(a + side * rng.uniform(0.6, 1.2)), math.sin(a + side * rng.uniform(0.6, 1.2)), rng.uniform(0.1, 0.5)))
        wood.grow(p, d.normalized(), rng.uniform(18, 45), r * rng.uniform(0.3, 0.45), seed=seed * 13 + i, depth=1,
                  children=(0, 1), wander=0.5, gravity=0.05, taper=0.55, skin=skin)
    return _keep_inside(skin, pts[len(pts) // 2])


def _obstacles(objs, cell=24.0, pad=6.0):
    """Fish keep-out cylinders covering the wood: its vertices binned on a
    coarse grid, one cylinder per occupied cell reaching to the highest point there."""
    cells = {}
    for o in objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co / MM
            k = (int(p.x // cell), int(p.y // cell))
            cells[k] = max(cells.get(k, -1e9), p.z)
    return [((i + 0.5) * cell, (j + 0.5) * cell, cell * 0.72 + pad, top) for (i, j), top in cells.items()]


# --- floating plants -------------------------------------------------------------

def _surface_helper(name, res_mm=4.0, z_mm=LEVEL - 1.0):
    """A flat grid at the waterline for the floaters to be scattered on; it is
    not itself rendered (the scatter drops it). It sits a millimetre under the
    surface so the leaves lie in the water film: with an air gap under them the
    underside of the surface, a mirror from the camera's low angle, hides them."""
    n = int((HI - LO) / res_mm)
    verts, faces = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append(((LO + (HI - LO) * i / n) * MM, (LO + (HI - LO) * j / n) * MM, z_mm * MM))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    return new_mesh_object(name, verts, faces)


def _edge_dist(x, y):
    """Distance in mm from the inside face of the nearest wall."""
    return min(x - 5, 295 - x, y - 5, 295 - y)


def _reach_mm(objs):
    """Largest horizontal distance from a source object's origin to any of its points."""
    r = 0.0
    for o in objs:
        for v in o.data.vertices:
            r = max(r, math.hypot(v.co.x, v.co.y))
    return r / MM


def _cover(x, y):
    """Where the floating raft lies: the back two thirds and the left side,
    with a ragged front edge and holes that let shafts of light through."""
    edge = 108 + 30 * math.sin(x * 0.031 + 0.6) + 14 * math.sin(x * 0.083 + 2.0)
    back = _smooth(edge - 12, edge + 18, y)
    side = 1.0 - _smooth(62, 96, x + 10 * math.sin(y * 0.05))
    c = max(back, side * _smooth(20, 60, y))
    holes = _smooth(0.26, 0.36, _noise(x, y, 2.3, 1.1))
    return c * holes


# --- build -----------------------------------------------------------------------

def build(ctx):
    made = []
    rng = random.Random(4)

    bed = substrate.build_bed(ground, "Sand")
    bed.obj.data.materials.append(_sand())
    made.append(bed.obj)

    # hanging roots and branches
    hanging = []
    for k, spec in enumerate(HANGING):
        obj = _finish(_grow_hanging(spec), spec[0], _wet_wood(f"Wood_{spec[0]}", k))
        hanging.append(obj)
    made += hanging

    # heavier pieces lying on the bottom, half sunk in sand and leaves
    fallen = []
    for k, spec in enumerate(FALLEN):
        fallen.append(_finish(_grow_fallen(spec), spec[0], _wet_wood(f"Wood_{spec[0]}", k + 1)))
    made += fallen

    twigs = _finish(_twigs(rng), "Twigs", _wet_wood("Wood_twigs", 1), subdivide=0)
    made.append(twigs)

    # Leaf litter in three layers so the bed has depth: a carpet over most of
    # the sand, a second layer where leaves drift together, and heaps against
    # the wood and in the back corners. Thin at the open front.
    near = []
    for o in fallen:
        lo, hi = assets.world_bounds(o)
        near.append(((lo.x + hi.x) / 2 / MM, (lo.y + hi.y) / 2 / MM, max(hi.x - lo.x, hi.y - lo.y) / 2 / MM))

    def pile(x, y):
        p = 0.0
        for cx, cy, r in near:
            p = max(p, 1.0 - _smooth(r * 0.4, r * 1.1, math.hypot(x - cx, y - cy)))
        corner = max(math.exp(-((x - 15) ** 2 + (y - 285) ** 2) / 55 ** 2), math.exp(-((x - 285) ** 2 + (y - 285) ** 2) / 55 ** 2))
        return max(p, corner, 0.6 * _smooth(200, 285, y))

    def depth(x, y):
        front = _smooth(10, 120, y)
        return (0.45 + 0.55 * front) * (0.5 + 0.5 * _noise(x, y, 0.7)) + 0.4 * pile(x, y)

    # The bed's top faces take the scanned litter wherever the litter is deep,
    # and stay sand in the open patches. The ragged join is per face (1.5 mm),
    # and the modelled leaves lie over it.
    me = bed.obj.data
    me.materials.append(_litter_floor())
    for p in me.polygons[:len(bed.top_xy_mm) - 1]:
        x, y = p.center.x / MM, p.center.y / MM
        if p.normal.z > 0.5 and depth(x, y) + 0.25 * _noise(x, y, 5.3, 3.1) > 0.78:
            p.material_index = 1

    leaf_mats = [
        _leaf_material("LeafBrown", (0.020, 0.011, 0.006), (0.10, 0.055, 0.025), (0.30, 0.19, 0.095)),
        _leaf_material("LeafCatappa", (0.030, 0.012, 0.005), (0.13, 0.052, 0.020), (0.36, 0.16, 0.060)),
        _leaf_material("LeafTan", (0.040, 0.032, 0.022), (0.13, 0.10, 0.065), (0.36, 0.29, 0.19)),
    ]
    small = _curled(plants.dead_leaves(variants=6, seed=83, length_mm=(12, 24), material=leaf_mats[2]), 83)
    leaves = _curled(plants.dead_leaves(variants=8, seed=31, length_mm=(16, 32), material=leaf_mats[0]) +
                     plants.dead_leaves(variants=7, seed=57, length_mm=(20, 38), material=leaf_mats[1]), 31) + small
    edge = _curled(plants.dead_leaves(variants=5, seed=97, length_mm=(9, 17), material=leaf_mats[0]) +
                   plants.dead_leaves(variants=4, seed=101, length_mm=(10, 18), material=leaf_mats[1]) +
                   plants.dead_leaves(variants=4, seed=103, length_mm=(10, 16), material=leaf_mats[2]), 97)
    # Instances turn freely, so a leaf can only lie as close to the glass as its
    # own reach. Big leaves keep that far off; small ones fill the band to the glass.
    r_all, r_edge = _reach_mm(leaves) * 1.05 + 2, _reach_mm(edge) * 1.05 + 2
    # weight, leaves per m2 at full weight, lift above the layer below (mm)
    layers = ((depth, 9000.0, 0.0),
              (lambda x, y: max(0.0, depth(x, y) - 0.45) * 1.6, 6300.0, 1.4),
              (lambda x, y: pile(x, y) * _noise(x, y, 4.1, 1.3) * 1.4, 3900.0, 2.8))
    litter = collection_of("LeafSrc", leaves)
    litter_edge = collection_of("LeafEdgeSrc", edge)
    for k, (fn, dens, lift) in enumerate(layers):
        bed.set_weight(f"litter_{k}", lambda x, y, fn=fn: min(1.0, fn(x, y)) * float(_edge_dist(x, y) > r_all))
        bed.set_weight(f"edge_{k}", lambda x, y, fn=fn: min(1.0, fn(x, y)) * float(r_edge < _edge_dist(x, y) <= r_all + 3))
        scatter(bed.obj, litter, name=f"Litter_{k}", density=dens, weight_attr=f"litter_{k}", seed=60 + k,
                scale=(0.6, 1.05), align="NORMAL", tilt=0.16 + 0.05 * k, sink=-lift * MM)
        scatter(bed.obj, litter_edge, name=f"LitterEdge_{k}", density=dens * 1.3, weight_attr=f"edge_{k}", seed=80 + k,
                scale=(0.75, 1.05), align="NORMAL", tilt=0.16 + 0.05 * k, sink=-lift * MM)

    # Floating plants: large rosettes in the open raft, small ones in the band at
    # the glass. Each set has its own helper, because a scatter that drops its
    # surface leaves nothing for a second scatter on the same object to use.
    # Roots darker than the kit's: pale ones lit from just above are the brightest
    # thing in the water and read as a fringe of hair.
    roots = plants.leaf_material("FloaterRootDark", base=(0.13, 0.085, 0.045), tip=(0.045, 0.030, 0.018),
                                 translucency=0.15, roughness=0.6, variation=0.35)
    big = plants.floaters(variants=6, seed=41, root_material=roots)
    tiny = plants.floaters(variants=5, seed=47, leaf_mm=(5, 9), root_material=roots)
    # Shorter beards than the kit's (25 to 85 mm), so the shoal's upper reach
    # stays clear of them: roots that fine cannot be obstacles for the fish.
    for o in big + tiny:
        for v in o.data.vertices:
            if v.co.z < 0:
                v.co.z *= 0.62
        o.data.update()
    # margin: reach at full scale, plus the sideways swing of 85 mm roots under tilt and sway
    r_big, r_tiny = _reach_mm(big) * 1.15 + 6, _reach_mm(tiny) * 1.1 + 6
    for name, src, r_in, r_out, gap, seed in (("Floaters", big, r_big, 1e9, 10.5, 71),
                                               ("FloatersSmall", tiny, r_tiny, r_big + 6, 7, 73)):
        helper = _surface_helper(name + "Raft")
        set_point_attribute(helper, "raft", [_cover(x, y) * float(r_in < _edge_dist(x, y) <= r_out)
                                             for x, y in ((v.co.x / MM, v.co.y / MM) for v in helper.data.vertices)])
        scatter(helper, collection_of(name + "Src", src), name=name, min_distance=gap * MM, weight_attr="raft", seed=seed,
                scale=(0.75, 1.15) if src is big else (0.8, 1.1), align="UP", tilt=0.05, keep_surface=False,
                sway=(0.015, 6.0, 0.15))
        made.append(helper)

    # The shoal keeps to the open water in the middle and front, clear of the
    # wood, and below the floaters' roots (they reach down to about 205 mm),
    # which are too fine to be obstacles but would show passing through a fish.
    # The wood where it could reach the shoal's water, for SWIM["obstacles"].
    bx = SWIM["box"]
    near_box = [o for o in _obstacles(hanging + fallen)
                if bx[0] - 40 < o[0] < bx[1] + 40 and o[1] < bx[3] + 40 and o[3] > bx[4] - 10]
    print("[obstacles] " + ", ".join(f"({x:.0f}, {y:.0f}, {r:.0f}, {top:.0f})" for x, y, r, top in sorted(near_box)))
    school = fish.School()
    made += school.objects

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["wood"] = hanging + fallen + [twigs]
    return ctx


def animate(ctx, frame):
    """Everything that moves by code rather than by keyframes."""
    ctx["school"].pose(frame)
