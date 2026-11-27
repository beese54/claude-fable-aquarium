"""Scape 2: Dutch garden.

No stone or wood shows: the picture is made of stem plants in groups of one
species each, like beds in a flower garden. Groups step up from a low front to
a tall back so each one shows, and neighbours contrast in colour and leaf
(fine next to broad, red next to green). A "street" of lobelia runs diagonally
from the front glass toward the back, narrowing, to give depth. A small open
patch of soil, ringed by a low carpet, keeps the front from reading as a wall.
Coordinates are millimetres: X left to right, Y front to back.
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

TOP_LIMIT = 257.0     # mm; tallest stem tip allowed, clear of the surface at 268 even with sway and leaves

# Species the kit lacks, plus kit species re-coloured for this tank. Same keys
# as plants.STEM_SPECIES, plus clump (stems per planting spot), bend, and turn:
# how far up the plant (0..1) the colour starts changing from base to tip. Red
# stem plants only colour up in the top, high-light part; below they are olive.
SPECIES = {
    # broad, round, mid green (Bacopa caroliniana grows tall when left)
    "bacopa":       dict(leaf=(11, 7.5), whorl=2, node=7.5, rise=32, belly=0.55, clump=(2, 3), bend=0.05, turn=0.15,
                         base=(0.025, 0.08, 0.022), tip=(0.11, 0.30, 0.07)),
    # fine, pink tops over olive stems (Rotala rotundifolia, high light)
    "rotala_red":   dict(leaf=(8.5, 2.8), whorl=2, node=3.4, rise=42, belly=0.50, clump=(2, 4), bend=0.07, turn=0.45,
                         base=(0.09, 0.10, 0.025), tip=(0.54, 0.11, 0.16)),
    # broad lanceolate, deep green (Hygrophila; not in the kit)
    "hygrophila":   dict(leaf=(26, 6.5), whorl=2, node=8.5, rise=30, belly=0.42, clump=(1, 2), bend=0.05, turn=0.15,
                         base=(0.02, 0.07, 0.016), tip=(0.09, 0.25, 0.045)),
    # feathery whorls, lime (Limnophila sessiliflora)
    "limnophila":   dict(leaf=(13, 0.9), whorl=8, node=5.5, rise=28, belly=0.40, clump=(1, 3), bend=0.05, turn=0.25,
                         base=(0.04, 0.12, 0.02), tip=(0.24, 0.46, 0.08)),
    # needles in tight whorls, pale green (Pogostemon erectus), trimmed short
    "pogostemon":   dict(leaf=(16, 1.2), whorl=10, node=4.5, rise=22, belly=0.35, clump=(1, 2), bend=0.04, turn=0.25,
                         base=(0.04, 0.12, 0.02), tip=(0.21, 0.43, 0.07)),
    # small leaves, yellow-orange tips (Rotala 'H'ra'-type green form)
    "rotala_green": dict(leaf=(9, 2.6), whorl=2, node=3.8, rise=38, belly=0.45, clump=(3, 5), bend=0.07, turn=0.40,
                         base=(0.04, 0.12, 0.02), tip=(0.38, 0.40, 0.07)),
    # broad, wine red (Alternanthera reineckii)
    "alternanthera": dict(leaf=(16, 6.0), whorl=2, node=6.0, rise=30, belly=0.50, clump=(1, 3), bend=0.05, turn=0.10,
                          base=(0.08, 0.025, 0.025), tip=(0.30, 0.03, 0.065)),
    # low rosettes of broad leaves, mid green
    "staurogyne":   dict(leaf=(15, 4.8), whorl=2, node=3.6, rise=28, belly=0.50, clump=(1, 2), bend=0.08, turn=0.10,
                         base=(0.025, 0.08, 0.02), tip=(0.09, 0.25, 0.05)),
    # broad, light green, the street (Lobelia cardinalis 'mini')
    "lobelia":      dict(leaf=(14, 9.0), whorl=2, node=5.5, rise=34, belly=0.60, clump=(1, 2), bend=0.06, turn=0.10,
                         base=(0.045, 0.14, 0.03), tip=(0.22, 0.46, 0.10)),
    # tiny round leaves, bright, bushy (pearlweed; not in the kit)
    "pearlweed":    dict(leaf=(4.6, 3.6), whorl=3, node=3.0, rise=36, belly=0.55, clump=(3, 5), bend=0.10, turn=0.15,
                         base=(0.05, 0.17, 0.03), tip=(0.24, 0.52, 0.09)),
}
# Groups, painted in priority order (an earlier shape wins where two overlap).
# name, species, outline (mm), height above soil at the group's front and back
# edge (mm), Poisson spacing (mm), lean toward the light (rad), glass margins
# (left, right, front, back) in mm.
GROUPS = [
    ("street",    "lobelia",       [(66, 4), (124, 4), (180, 208), (164, 208)],                 (26, 52), 4.9, 0.10, (10, 10, 9, 20)),
    ("open",      None,            [(96, 4), (242, 4), (254, 62), (234, 114), (190, 130), (140, 126)], None, None, None, (8, 8, 8, 20)),
    ("pearlweed", "pearlweed",     [(196, 4), (296, 4), (296, 104), (196, 104)],                (20, 32), 4.0, 0.10, (12, 13, 10, 20)),
    ("stauro",    "staurogyne",    [(4, 4), (112, 4), (112, 94), (4, 94)],                      (18, 32), 4.4, 0.10, (15, 10, 10, 20)),
    ("pogo",      "pogostemon",    [(4, 88), (150, 88), (150, 186), (4, 180)],                  (55, 82), 4.7, 0.08, (20, 10, 10, 20)),
    ("rot_green", "rotala_green",  [(140, 96), (210, 96), (204, 200), (150, 200)],              (60, 92), 4.2, 0.07, (10, 10, 10, 20)),
    ("altern",    "alternanthera", [(200, 98), (296, 98), (296, 196), (198, 200)],              (62, 90), 4.9, 0.07, (12, 18, 10, 20)),
    ("bacopa",    "bacopa",        [(4, 160), (80, 168), (80, 296), (4, 296)],                  (118, 140), 5.4, 0.04, (17, 10, 10, 28)),
    ("rot_red",   "rotala_red",    [(70, 168), (152, 176), (152, 296), (70, 296)],              (136, 158), 4.2, 0.04, (10, 10, 10, 28)),
    ("hygro",     "hygrophila",    [(146, 176), (222, 182), (222, 296), (146, 296)],            (126, 146), 5.8, 0.04, (10, 10, 10, 39)),
    ("limno",     "limnophila",    [(214, 180), (296, 180), (296, 296), (214, 296)],            (140, 160), 4.9, 0.04, (10, 24, 10, 30)),
]

BARE = (180, 36, 54, 36)      # open soil patch: centre x, y and radii (mm), inside the carpet


def ground(x, y):
    """Soil height in mm: 62 at the front glass rising to about 100 at the back,
    with a soft roll so the bed is not a ramp, and a slight dip at the open patch.
    Takes numbers or numpy arrays."""
    slope = 62 + 38 * (y / 290) ** 1.15
    roll = 2.0 * np.sin(x * 0.04 + 0.7) * np.cos(y * 0.035) + 0.8 * np.sin(x * 0.13) * np.sin(y * 0.09 + 0.5)
    dip = -3.0 * np.exp(-(((x - BARE[0]) / 50) ** 2 + ((y - BARE[1]) / 35) ** 2))
    return slope + roll + dip


def _warp(x, y):
    """Bend every outline the same way, so the groups still tile but their
    borders wander like real planting instead of following ruler lines."""
    wx = x + 4.5 * np.sin(y * 0.071 + 1.3) + 2.0 * np.sin(y * 0.23 + x * 0.05)
    wy = y + 4.5 * np.sin(x * 0.063 + 0.4) + 2.0 * np.sin(x * 0.21 + y * 0.04)
    return wx, wy


def _sdf_poly(px, py, poly):
    """Signed distance (mm, positive inside) from points to a polygon."""
    d = np.full(px.shape, np.inf)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        d = np.minimum(d, dx * dx + dy * dy)
        if ay != by:
            cross = ((ay > py) != (by > py)) & (px < ex * (py - ay) / (by - ay) + ax)
            inside ^= cross
    d = np.sqrt(d)
    return np.where(inside, d, -d)


def _smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _regions(x, y):
    """For each group: distance inside its own territory (mm), after higher
    priority groups have taken theirs and the glass margins are cut off."""
    wx, wy = _warp(x, y)
    owned, out = [], {}
    for name, _, poly, _, _, _, (ml, mr, mf, mb) in GROUPS:
        d = _sdf_poly(wx, wy, poly)
        # the glass margins use true coordinates: the warp must not push a plant into the glass
        d = np.minimum.reduce([d, x - (5 + ml), (295 - mr) - x, y - (5 + mf), (295 - mb) - y])
        eff = d
        for o in owned:
            eff = np.minimum(eff, -o)
        owned.append(d)
        out[name] = eff
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


def _stem_variants(key, H, variants=8, seed=0):
    """Clumps of 1..n stems of one species, built at H mm tall. A trimmed Dutch
    group is bushy: each cut stem sends up two or three shoots of slightly
    different height, so one planting spot holds a small clump."""
    sp = SPECIES[key]
    mat = _leaf_material(f"Dutch_{key}", sp["base"], sp["tip"], sp["turn"])
    rng = random.Random(1000 + seed)
    out = []
    for v in range(variants):
        parts = []
        for s in range(rng.randint(*sp["clump"])):
            h = H * (1.0 if s == 0 else rng.uniform(0.78, 0.97))
            ox, oy = (0.0, 0.0) if s == 0 else (rng.uniform(-2.6, 2.6), rng.uniform(-2.6, 2.6))
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
        out.append(plants._object_from(f"Dutch_{key}_{v}", verts, faces, uvs, mat))
    return out


def _shape(mod, size_attr, lean):
    """Extend a kit scatter: instance size follows a painted attribute (so a
    group is trimmed into a slope, low at its front and rounded at its edges),
    and every instance leans a little toward the lamp at the back."""
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


SCALE_JITTER = (0.88, 1.12)     # stems in a group differ by about 20%


def _carrier(bed, name, made):
    """An object that shares the bed's mesh and carries one scatter. Stacking
    scatters on one object makes each one also distribute over the instances
    the earlier ones made, doubling the instance count per scatter (eleven
    stacked scatters ran out of memory), so each group gets its own carrier."""
    import bpy
    obj = bpy.data.objects.new(name, bed.obj.data)
    made.append(obj)
    return obj


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

    # open area: carpet ringing a patch of bare soil
    bx, by, brx, bry = BARE
    wx, wy = _warp(x, y)
    q = np.sqrt(((wx - bx) / brx) ** 2 + ((wy - by) / bry) ** 2)
    bare = 1.0 - _smooth(0.85, 1.15, q)
    # The carpet runs a little way in under the neighbouring groups, so no strip
    # of soil shows between carpet and stems; it stops short of the glass.
    open_w = _smooth(-12.0, -6.0, regions["open"]) * _smooth(1.0, 4.0, np.minimum.reduce([x - 9, 291 - x, y - 9]))
    carpet_w = open_w * (1.0 - bare)
    # thinner and patchier at the patch, as a carpet that is still spreading
    carpet_w *= 0.6 + 0.4 * np.clip(q - 0.8, 0.0, 1.0) ** 0.5
    paint("carpet", carpet_w)
    sprigs = collection_of("DutchCarpetSrc", plants.carpet_sprigs(variants=6, seed=12))
    scatter(_carrier(bed, "DutchCarpet", made), sprigs, name="DutchCarpet", min_distance=3.0 * MM,
            weight_attr="carpet", seed=31, scale=(0.7, 1.2), align="UP", tilt=0.25, keep_surface=False,
            sway=(0.06, 9.0, 0.35))

    report, keep_out, sources = [], [], {}
    for k, (name, key, poly, hts, spacing, lean, _) in enumerate(GROUPS):
        if key is None:
            continue
        eff = regions[name]
        w = _smooth(1.5, 4.0, eff)
        inside = w > 0.01
        ys = y[inside]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0.0, 1.0)
        f = np.clip((y - y0) / max(y1 - y0, 1.0), 0.0, 1.0)
        target = hts[0] + (hts[1] - hts[0]) * f ** 0.8
        target *= 0.84 + 0.16 * _smooth(0.0, 14.0, eff)          # rounded off at the edges, as trimmed
        # never let a stem reach the surface: cap the tallest jittered stem below TOP_LIMIT
        target = np.minimum(target, (TOP_LIMIT - soil_z) / (SCALE_JITTER[1] * 1.06))
        H = float(np.median(target[inside])) if inside.any() else hts[1]
        paint(f"w_{name}", w)
        paint(f"s_{name}", np.where(inside, target / H, 1.0))

        src = _stem_variants(key, H, variants=8, seed=k * 17)
        sources.update({o.name: name for o in src})
        col = collection_of(f"Dutch_{name}_src", src)
        tall = hts[1] > 100
        sway = (0.030, 6.0, 0.28) if tall else (0.050, 8.0, 0.32)
        mod = scatter(_carrier(bed, f"Dutch_{name}", made), col, name=f"Dutch_{name}", min_distance=spacing * MM,
                      weight_attr=f"w_{name}", seed=40 + k, scale=SCALE_JITTER, align="UP", tilt=0.09,
                      keep_surface=False, sway=sway)
        _shape(mod, f"s_{name}", lean)

        if 45 < hts[1] < 100:
            # Young side shoots at half height, all through the group and thickest
            # round its edge: they hide the bare lower stems and the soil between
            # them, and soften the hedge-like sides. Only the mid-ground needs
            # them; the background's lower stems are hidden behind it.
            paint(f"u_{name}", w * (1.0 - 0.7 * _smooth(3.0, 16.0, eff)))
            under = _stem_variants(key, H * 0.5, variants=5, seed=k * 17 + 5)
            sources.update({o.name: name for o in under})
            mod = scatter(_carrier(bed, f"Dutch_{name}_under", made), collection_of(f"Dutch_{name}_under_src", under),
                          name=f"Dutch_{name}_under", min_distance=spacing * 1.1 * MM, weight_attr=f"u_{name}",
                          seed=70 + k, scale=(0.75, 1.15), align="UP", tilt=0.11, keep_surface=False, sway=sway)
            _shape(mod, f"s_{name}", lean)

        tops = soil_z + target * SCALE_JITTER[1]
        report.append(f"{name}:{key} H={H:.0f} area={inside.sum() * 2.25 / 100:.0f}cm2 "
                      f"top_max={float(tops[inside].max()) if inside.any() else 0:.0f}")
        # fish keep-out: cylinders over the group, each as tall as the plants under it
        for gx in range(10, 295, 18):
            for gy in range(10, 295, 18):
                sel = inside & (np.abs(x - gx) < 9) & (np.abs(y - gy) < 9)
                if sel.any():
                    keep_out.append((gx, gy, 10.0, float(tops[sel].max()) + 6.0))

    print("[dutch] " + " | ".join(report))

    # The shoal keeps to the open water above the low and mid-ground plants, in
    # front of the tall background.
    f0, f1 = shot_frames(SCAPE)
    school = fish.School(SCAPE, 10, f0, f1, box=(38, 262, 26, 150, 182, 242), seed=7,
                         obstacles=keep_out, ground=ground, spread=(55.0, 34.0, 16.0))
    made += school.objects

    # Red cherry shrimp grazing on the open soil, spread in depth as well as across.
    crew = []
    for i, (sx, sy, yaw, t, size) in enumerate([(140, 20, 2.6, 0.10, 0.95), (168, 60, -0.4, 0.35, 0.85),
                                                 (206, 28, 1.1, 0.55, 1.0), (219, 52, -2.3, 0.70, 0.9)]):
        s = shrimp.Shrimp(f"Shrimp_{i}", size=size)
        crew.append((s, sx, sy, yaw, t, size))
        made.append(s.obj)

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    ctx["shrimp"] = crew
    animate(ctx, ctx["frame"])

    if os.environ.get("DUTCH_CHECK"):
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
    (X, Y 6..294 mm, top below 266 mm). Then test each fish's box against
    every plant instance's box, which is stricter than a true intersection."""
    import bpy
    from mathutils import Vector
    scene = ctx["scene"]
    verts = {}
    for name in sources:
        me = bpy.data.objects[name].data
        v = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", v)
        verts[name] = v.reshape(-1, 3)
    frame0 = scene.frame_current
    for frame in (1, 600, frame0, 1500, 2400):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        lo, hi, count, boxes, per_scatter = {}, {}, {}, [], {}
        for inst in dg.object_instances:
            if not inst.is_instance:
                continue
            owner = inst.parent.original.name
            per_scatter[owner] = per_scatter.get(owner, 0) + 1
            name = inst.instance_object.original.name
            grp = sources.get(name)
            if grp is None:
                continue
            m = np.array(inst.matrix_world)
            p = (verts[name] @ m[:3, :3].T + m[:3, 3]) / MM
            a, b = p.min(0), p.max(0)
            boxes.append((grp, a, b))
            lo[grp] = np.minimum(lo.get(grp, a), a)
            hi[grp] = np.maximum(hi.get(grp, b), b)
            count[grp] = count.get(grp, 0) + 1
        if frame == frame0:
            print("[dutch-check] points per scatter: " + ", ".join(f"{k}={v}" for k, v in sorted(per_scatter.items())))
            for g in lo:
                print(f"[dutch-check] {g}: x {lo[g][0]:.1f}..{hi[g][0]:.1f}  y {lo[g][1]:.1f}..{hi[g][1]:.1f}  "
                      f"z {lo[g][2]:.1f}..{hi[g][2]:.1f}  n={count[g]}")
            for t in school:
                pts = np.array([tuple(t.obj.matrix_world @ Vector(c)) for c in t.obj.bound_box]) / MM
                a, b = pts.min(0), pts.max(0)
                hits = sorted({g for g, pa, pb in boxes if np.all(a < pb) and np.all(b > pa)})
                print(f"[dutch-check] {t.obj.name} at {((a + b) / 2).round(0)} plant-box overlaps: {hits or 'none'}")
        bad = [g for g in lo if lo[g][0] < 6 or lo[g][1] < 6 or hi[g][0] > 294 or hi[g][1] > 294 or hi[g][2] > 266]
        top = max(hi[g][2] for g in hi)
        print(f"[dutch-check] frame {frame}: instances {sum(count.values())}, highest tip {top:.1f} mm, "
              f"outside water: {bad or 'none'}")
    scene.frame_set(frame0)
