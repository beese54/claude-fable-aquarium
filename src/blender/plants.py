"""Aquatic plants, built as small instanceable meshes.

Each builder returns a list of variant objects (not linked to the scene) ready
to hand to scatter(). UV v runs from 0 at the base to 1 at the tip so materials
can shade along the length.
"""
import math
import random

import bpy

from bl_common import MM, new_material


def _tube_strip(points, radii, sides=3):
    """Vertices, faces and UVs for a tapering tube through `points`."""
    verts, faces, uvs = [], [], []
    n = len(points)
    for k, ((x, y, z), r) in enumerate(zip(points, radii)):
        # frame: tangent from neighbours, any stable perpendicular pair
        a = points[max(k - 1, 0)]
        b = points[min(k + 1, n - 1)]
        tx, ty, tz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
        tx, ty, tz = tx / tl, ty / tl, tz / tl
        ux, uy, uz = (1.0, 0.0, 0.0) if abs(tz) > 0.9 else (0.0, 0.0, 1.0)
        # n1 = t x u ; n2 = t x n1
        n1 = (ty * uz - tz * uy, tz * ux - tx * uz, tx * uy - ty * ux)
        l1 = math.sqrt(sum(c * c for c in n1)) or 1.0
        n1 = tuple(c / l1 for c in n1)
        n2 = (ty * n1[2] - tz * n1[1], tz * n1[0] - tx * n1[2], tx * n1[1] - ty * n1[0])
        for s in range(sides):
            ang = 2 * math.pi * s / sides
            c, sn = math.cos(ang) * r, math.sin(ang) * r
            verts.append((x + n1[0] * c + n2[0] * sn, y + n1[1] * c + n2[1] * sn, z + n1[2] * c + n2[2] * sn))
            uvs.append((s / sides, k / (n - 1)))
    for k in range(n - 1):
        for s in range(sides):
            a = k * sides + s
            b = k * sides + (s + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return verts, faces, uvs


def _object_from(name, verts, faces, uvs, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        uv.data[loop.index].uv = uvs[loop.vertex_index]
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(material)
    me.update()
    return bpy.data.objects.new(name, me)


def leaf_material(name, base, tip, translucency=0.35, roughness=0.35, variation=0.25):
    """Leaf shading: colour runs base to tip along UV v, varies plant to plant,
    and lets some light through, which is what makes backlit plants glow."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*base, 1)
    ramp.color_ramp.elements[1].color = (*tip, 1)
    L.new(sep.outputs["Y"], ramp.inputs["Fac"])
    info = N.new("ShaderNodeObjectInfo")
    hsv = N.new("ShaderNodeHueSaturation")
    val = N.new("ShaderNodeMapRange")
    val.inputs["To Min"].default_value = 1.0 - variation
    val.inputs["To Max"].default_value = 1.0 + variation
    L.new(info.outputs["Random"], val.inputs["Value"])
    L.new(val.outputs["Result"], hsv.inputs["Value"])
    hue = N.new("ShaderNodeMapRange")
    hue.inputs["To Min"].default_value = 0.5 - variation * 0.08
    hue.inputs["To Max"].default_value = 0.5 + variation * 0.08
    L.new(info.outputs["Random"], hue.inputs["Value"])
    L.new(hue.outputs["Result"], hsv.inputs["Hue"])
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


def hairgrass_tufts(variants=5, blades=(9, 15), height_mm=(18, 36), seed=3, material=None):
    """Dwarf hairgrass (Eleocharis): tufts of fine needle leaves fanning from a point."""
    material = material or leaf_material("Hairgrass", base=(0.018, 0.070, 0.014), tip=(0.115, 0.31, 0.055), translucency=0.38)
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        verts, faces, uvs = [], [], []
        for _ in range(rng.randint(*blades)):
            h = rng.uniform(*height_mm) * MM
            az = rng.uniform(0, 2 * math.pi)
            lean = rng.uniform(0.05, 0.50)            # how far the tip ends up from vertical, as a fraction of height
            curl = rng.uniform(-0.12, 0.12)
            bx, by = rng.uniform(-1.5, 1.5) * MM, rng.uniform(-1.5, 1.5) * MM
            pts, rad = [], []
            segs = 6
            for k in range(segs + 1):
                t = k / segs
                out_d = lean * h * t * t
                side = curl * h * math.sin(t * math.pi) * 0.5
                x = bx + math.cos(az) * out_d - math.sin(az) * side
                y = by + math.sin(az) * out_d + math.cos(az) * side
                z = h * (t - 0.12 * lean * t * t) - 1.0 * MM
                pts.append((x, y, z))
                rad.append((0.26 - 0.19 * t) * MM)
            vv, ff, uu = _tube_strip(pts, rad)
            off = len(verts)
            verts += vv
            uvs += uu
            faces += [tuple(i + off for i in f) for f in ff]
        out.append(_object_from(f"HairgrassTuft_{v}", verts, faces, uvs, material))
    return out


# --- broad-leaved plants --------------------------------------------------------

def _leaf(origin, direction, up, length, width, v0, v1, fold=0.25, curl=0.18, belly=0.42, segs=5):
    """One leaf blade as a small folded sheet.

    origin     where the leaf joins the stem
    direction  unit vector the leaf points along
    up         unit vector roughly normal to the blade's upper face
    v0, v1     UV v at the leaf's base and tip (v encodes height on the plant)
    fold       how far the edges rise above the midrib, as a fraction of width
    curl       how far the tip droops, as a fraction of length
    belly      where along the leaf it is widest (0 base .. 1 tip)
    Returns verts, faces, uvs.
    """
    dx, dy, dz = direction
    ux, uy, uz = up
    # side = direction x up
    sx, sy, sz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
    sl = math.sqrt(sx * sx + sy * sy + sz * sz) or 1.0
    sx, sy, sz = sx / sl, sy / sl, sz / sl
    verts, faces, uvs = [], [], []
    for k in range(segs + 1):
        t = k / segs
        # outline: zero at both ends, widest at `belly`
        if t <= belly:
            w = math.sin(0.5 * math.pi * t / belly) ** 0.8
        else:
            w = math.cos(0.5 * math.pi * (t - belly) / (1 - belly)) ** 0.9
        half = 0.5 * width * max(w, 0.04)
        droop = -curl * length * t * t
        cx = origin[0] + dx * length * t + ux * droop
        cy = origin[1] + dy * length * t + uy * droop
        cz = origin[2] + dz * length * t + uz * droop
        lift = fold * half
        for side, u in ((-1, 0.0), (0, 0.5), (1, 1.0)):
            e = abs(side)
            verts.append((cx + sx * half * side + ux * lift * e,
                          cy + sy * half * side + uy * lift * e,
                          cz + sz * half * side + uz * lift * e))
            uvs.append((u, v0 + (v1 - v0) * t))
    for k in range(segs):
        a = k * 3
        faces.append((a, a + 1, a + 4, a + 3))
        faces.append((a + 1, a + 2, a + 5, a + 4))
    return verts, faces, uvs


def _merge(parts):
    verts, faces, uvs = [], [], []
    for vv, ff, uu in parts:
        off = len(verts)
        verts += vv
        uvs += uu
        faces += [tuple(i + off for i in f) for f in ff]
    return verts, faces, uvs


# name: leaf length, leaf width (mm), leaves per node, node spacing (mm), stem
# height range (mm), leaf rise angle (deg), belly, colours base -> tip
STEM_SPECIES = {
    "rotala_green": dict(leaf=(9, 2.4), whorl=2, node=5.5, height=(90, 150), rise=35, belly=0.45,
                         base=(0.035, 0.12, 0.015), tip=(0.30, 0.50, 0.06)),
    "rotala_red":   dict(leaf=(10, 1.7), whorl=3, node=5.0, height=(100, 170), rise=40, belly=0.40,
                         base=(0.06, 0.10, 0.02), tip=(0.62, 0.07, 0.03)),
    "ludwigia_red": dict(leaf=(15, 7.0), whorl=2, node=9.0, height=(80, 140), rise=30, belly=0.50,
                         base=(0.10, 0.09, 0.02), tip=(0.55, 0.035, 0.03)),
    "bacopa":       dict(leaf=(9, 6.5), whorl=2, node=7.0, height=(60, 110), rise=30, belly=0.55,
                         base=(0.04, 0.14, 0.02), tip=(0.22, 0.48, 0.07)),
    "limnophila":   dict(leaf=(13, 0.9), whorl=8, node=6.0, height=(90, 160), rise=25, belly=0.40,
                         base=(0.05, 0.16, 0.02), tip=(0.36, 0.60, 0.10)),
    "lobelia":      dict(leaf=(15, 9.0), whorl=2, node=7.0, height=(35, 60), rise=35, belly=0.60,
                         base=(0.05, 0.16, 0.03), tip=(0.26, 0.52, 0.10)),
    "staurogyne":   dict(leaf=(17, 5.0), whorl=2, node=4.5, height=(22, 36), rise=30, belly=0.50,
                         base=(0.04, 0.13, 0.02), tip=(0.18, 0.42, 0.06)),
    "alternanthera": dict(leaf=(14, 5.5), whorl=2, node=7.5, height=(60, 110), rise=30, belly=0.50,
                          base=(0.16, 0.05, 0.05), tip=(0.60, 0.06, 0.16)),
    "pogostemon":   dict(leaf=(22, 1.3), whorl=10, node=5.0, height=(110, 180), rise=20, belly=0.35,
                         base=(0.05, 0.15, 0.02), tip=(0.30, 0.55, 0.08)),
}


def stem_plants(species, variants=5, seed=0, translucency=0.38):
    """Stem plants: a thin stem with leaves at regular nodes, crowded at the tip.
    Built at real size standing on the origin; scatter them in groups."""
    sp = STEM_SPECIES[species]
    mat = leaf_material(f"Leaf_{species}", base=sp["base"], tip=sp["tip"], translucency=translucency, roughness=0.32)
    rng = random.Random(hash(species) % 10007 + seed)
    out = []
    for v in range(variants):
        H = rng.uniform(*sp["height"])
        bend_az = rng.uniform(0, 2 * math.pi)
        bend = rng.uniform(0.03, 0.14)
        wob = rng.uniform(0, 6.28)

        def axis(t, H=H, bend_az=bend_az, bend=bend, wob=wob):
            off = bend * H * t * t + 0.012 * H * math.sin(t * 7 + wob)
            return (math.cos(bend_az) * off * MM, math.sin(bend_az) * off * MM, H * t * MM)

        parts = []
        segs = 10
        pts = [axis(k / segs) for k in range(segs + 1)]
        rad = [(0.55 - 0.25 * k / segs) * MM for k in range(segs + 1)]
        sv, sf, su = _tube_strip(pts, rad)
        parts.append((sv, sf, su))          # _tube_strip already gives v = 0 at base .. 1 at tip

        z = rng.uniform(4, 9)
        turn = rng.uniform(0, 6.28)
        Lm, Wm = sp["leaf"]
        while z < H:
            t = z / H
            ox, oy, oz = axis(t)
            n = sp["whorl"]
            crown = 1.0 - 0.35 * max(0.0, (t - 0.8) / 0.2)           # smaller leaves in the crown
            young = max(0.0, (t - 0.75) / 0.25)
            for i in range(n):
                az = turn + 2 * math.pi * i / n + rng.uniform(-0.2, 0.2)
                rise = math.radians(sp["rise"] + 30 * young + rng.uniform(-10, 10) - 25 * (1 - t) * (1 - t))
                d = (math.cos(az) * math.cos(rise), math.sin(az) * math.cos(rise), math.sin(rise))
                u = (-math.cos(az) * math.sin(rise), -math.sin(az) * math.sin(rise), math.cos(rise))
                ln = Lm * crown * rng.uniform(0.8, 1.15) * MM
                wd = Wm * crown * rng.uniform(0.85, 1.15) * MM
                parts.append(_leaf((ox, oy, oz), d, u, ln, wd, t, min(1.0, t + 0.06),
                                   fold=0.22, curl=rng.uniform(0.05, 0.28), belly=sp["belly"], segs=4))
            turn += math.pi / 2 if n == 2 else math.pi / n      # opposite pairs alternate at right angles
            z += sp["node"] * (1.0 - 0.55 * young) * rng.uniform(0.85, 1.15)
        verts, faces, uvs = _merge(parts)
        out.append(_object_from(f"{species}_{v}", verts, faces, uvs, mat))
    return out


def carpet_sprigs(variants=5, seed=7, leaf_mm=3.4, material=None):
    """Monte Carlo / dwarf baby tears: short creeping sprigs with pairs of tiny round leaves."""
    material = material or leaf_material("CarpetLeaf", base=(0.05, 0.20, 0.025), tip=(0.26, 0.58, 0.07), translucency=0.42, roughness=0.3)
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for _ in range(rng.randint(3, 5)):
            az = rng.uniform(0, 2 * math.pi)
            lean = rng.uniform(0.25, 0.9)
            L = rng.uniform(7, 15) * MM
            pts = []
            segs = 5
            for k in range(segs + 1):
                t = k / segs
                r = lean * L * t
                pts.append((math.cos(az) * r, math.sin(az) * r, L * t * (1 - 0.45 * lean) - 0.5 * MM))
            sv, sf, su = _tube_strip(pts, [0.18 * MM] * (segs + 1))
            parts.append((sv, sf, su))
            for k in range(1, segs + 1):
                t = k / segs
                ox, oy, oz = pts[k]
                for sgn in (1, -1):
                    la = az + sgn * math.pi / 2 + rng.uniform(-0.4, 0.4)
                    rise = rng.uniform(0.25, 0.75)
                    d = (math.cos(la) * math.cos(rise), math.sin(la) * math.cos(rise), math.sin(rise))
                    u = (-math.cos(la) * math.sin(rise), -math.sin(la) * math.sin(rise), math.cos(rise))
                    s = leaf_mm * rng.uniform(0.75, 1.2) * MM
                    parts.append(_leaf((ox, oy, oz), d, u, s, s * 0.92, t * 0.7, t, fold=0.18, curl=0.1, belly=0.58, segs=3))
        verts, faces, uvs = _merge(parts)
        out.append(_object_from(f"CarpetSprig_{v}", verts, faces, uvs, material))
    return out


def moss_fronds(variants=6, seed=13, length_mm=(5, 10), material=None):
    """Moss: tiny branching fronds. Scattered densely over wood or stone they
    give the soft, fuzzy outline that moss has; they are far too small to model leaf by leaf."""
    material = material or leaf_material("Moss", base=(0.020, 0.070, 0.010), tip=(0.13, 0.33, 0.035), translucency=0.30, roughness=0.5)
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts = []
        for _ in range(rng.randint(2, 4)):
            az = rng.uniform(0, 2 * math.pi)
            tilt = rng.uniform(0.1, 0.9)
            L = rng.uniform(*length_mm) * MM
            d = (math.cos(az) * math.sin(tilt), math.sin(az) * math.sin(tilt), math.cos(tilt))
            pts = [(d[0] * L * t, d[1] * L * t, d[2] * L * t - 0.4 * MM) for t in (0, 0.5, 1.0)]
            sv, sf, su = _tube_strip(pts, [0.22 * MM, 0.18 * MM, 0.08 * MM])
            parts.append((sv, sf, su))
            # side branchlets, alternating, like a fern in miniature
            for k in range(1, 6):
                t = k / 6
                o = (d[0] * L * t, d[1] * L * t, d[2] * L * t)
                sa = az + (1 if k % 2 else -1) * 1.25 + rng.uniform(-0.3, 0.3)
                bd = (math.cos(sa) * 0.8, math.sin(sa) * 0.8, 0.25 + 0.4 * d[2])
                bl = math.sqrt(sum(c * c for c in bd))
                bd = tuple(c / bl for c in bd)
                up = (0.0, 0.0, 1.0)
                parts.append(_leaf(o, bd, up, L * 0.38 * (1 - 0.5 * t), 0.9 * MM, t * 0.8, t, fold=0.1, curl=0.2, belly=0.4, segs=2))
        verts, faces, uvs = _merge(parts)
        out.append(_object_from(f"MossFrond_{v}", verts, faces, uvs, material))
    return out


def dead_leaf_material(name="DeadLeaf"):
    """Leaf litter: browns from tan to near black, matt, a little translucent."""
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    info = N.new("ShaderNodeObjectInfo")
    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].color = (0.030, 0.014, 0.006, 1)
    e[1].color = (0.33, 0.17, 0.06, 1)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.13, 0.055, 0.018, 1)
    L.new(info.outputs["Random"], ramp.inputs["Fac"])
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    # darker along the midrib and blotchy, like a leaf that has been soaking
    rib = N.new("ShaderNodeMath")
    rib.operation = "SUBTRACT"
    rib.inputs[1].default_value = 0.5
    L.new(sep.outputs["X"], rib.inputs[0])
    ab = N.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    L.new(rib.outputs[0], ab.inputs[0])
    ribm = N.new("ShaderNodeMapRange")
    ribm.inputs["From Min"].default_value = 0.0
    ribm.inputs["From Max"].default_value = 0.08
    ribm.inputs["To Min"].default_value = 0.55
    ribm.inputs["To Max"].default_value = 1.0
    L.new(ab.outputs[0], ribm.inputs["Value"])
    blot = N.new("ShaderNodeTexNoise")
    blot.inputs["Scale"].default_value = 140.0
    blot.inputs["Detail"].default_value = 4.0
    bm = N.new("ShaderNodeMapRange")
    bm.inputs["To Min"].default_value = 0.6
    bm.inputs["To Max"].default_value = 1.25
    L.new(blot.outputs["Fac"], bm.inputs["Value"])
    m1 = N.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    L.new(ribm.outputs["Result"], m1.inputs[0])
    L.new(bm.outputs["Result"], m1.inputs[1])
    mul = N.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], mul.inputs["A"])
    L.new(m1.outputs[0], mul.inputs["B"])
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.6
    L.new(mul.outputs["Result"], bsdf.inputs["Base Color"])
    trans = N.new("ShaderNodeBsdfTranslucent")
    L.new(mul.outputs["Result"], trans.inputs["Color"])
    mix = N.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.2
    L.new(bsdf.outputs[0], mix.inputs[1])
    L.new(trans.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def dead_leaves(variants=7, seed=31, length_mm=(22, 48), material=None):
    """Fallen leaves for a blackwater floor: broad, curled, lying roughly flat."""
    material = material or dead_leaf_material()
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        L = rng.uniform(*length_mm) * MM
        W = L * rng.uniform(0.36, 0.58)
        verts, faces, uvs = _leaf((-L / 2, 0.0, 0.8 * MM), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), L, W, 0.0, 1.0,
                                  fold=rng.uniform(-0.35, 0.5), curl=rng.uniform(-0.25, 0.3),
                                  belly=rng.uniform(0.38, 0.55), segs=8)
        # a twist along the length so no two lie alike
        tw = rng.uniform(-0.9, 0.9)
        twisted = []
        for (x, y, z) in verts:
            a = tw * (x / L)
            twisted.append((x, y * math.cos(a) - (z - 0.8 * MM) * math.sin(a), 0.8 * MM + y * math.sin(a) + (z - 0.8 * MM) * math.cos(a) + abs(y) * 0.05))
        out.append(_object_from(f"DeadLeaf_{v}", twisted, faces, uvs, material))
    return out


def floaters(variants=5, seed=41, leaf_mm=(9, 17), material=None, root_material=None):
    """Floating plants (frogbit-like): a rosette of round leaves lying on the
    surface with a beard of fine roots hanging below. Origin is at the waterline."""
    material = material or leaf_material("Floater", base=(0.07, 0.22, 0.03), tip=(0.22, 0.50, 0.08), translucency=0.45, roughness=0.25)
    root_material = root_material or leaf_material("FloaterRoot", base=(0.30, 0.22, 0.12), tip=(0.10, 0.07, 0.04), translucency=0.2, roughness=0.6, variation=0.3)
    rng = random.Random(seed)
    out = []
    for v in range(variants):
        parts, root_parts = [], []
        n = rng.randint(3, 6)
        for i in range(n):
            az = 2 * math.pi * i / n + rng.uniform(-0.3, 0.3)
            s = rng.uniform(*leaf_mm) * MM
            stalk = s * rng.uniform(0.25, 0.6)
            o = (math.cos(az) * stalk, math.sin(az) * stalk, 0.6 * MM)
            d = (math.cos(az), math.sin(az), rng.uniform(-0.02, 0.06))
            parts.append(_leaf(o, d, (0, 0, 1), s, s * 0.95, 0.3, 1.0, fold=rng.uniform(0.02, 0.12), curl=-0.03, belly=0.55, segs=5))
        for _ in range(rng.randint(5, 9)):
            L = rng.uniform(25, 85) * MM
            ox, oy = rng.uniform(-2, 2) * MM, rng.uniform(-2, 2) * MM
            dx, dy = rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)
            ph = rng.uniform(0, 6.28)
            pts = [(ox + dx * L * t + 1.2 * MM * math.sin(t * 9 + ph), oy + dy * L * t + 1.2 * MM * math.cos(t * 7 + ph), -L * t) for t in [k / 7 for k in range(8)]]
            root_parts.append(_tube_strip(pts, [(0.22 - 0.16 * k / 7) * MM for k in range(8)]))
        verts, faces, uvs = _merge(parts)
        obj = _object_from(f"Floater_{v}", verts, faces, uvs, material)
        rv, rf, ru = _merge(root_parts)
        # roots go in the same object on a second material slot
        me = obj.data
        off = len(me.vertices)
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()
        new_verts = [bm.verts.new(p) for p in rv]
        bm.verts.ensure_lookup_table()
        for f in rf:
            face = bm.faces.new([new_verts[i] for i in f])
            face.material_index = 1
            face.smooth = True
            for loop, i in zip(face.loops, f):
                loop[uv_layer].uv = ru[i]
        bm.to_mesh(me)
        bm.free()
        me.materials.append(root_material)
        out.append(obj)
    return out
