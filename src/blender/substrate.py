"""Substrate: a solid bed whose top follows a height function.

Heights are millimetres above the tank's outer base. The bed is a closed solid
that stops 1.2 mm short of the glass. That gap is not left empty: add_cut_face
packs it with grains pressed against the pane, which is what a real bed shows
in section through the front glass.
"""
import math
import os
import random

import bmesh
import bpy
from mathutils import Vector, noise

from bl_common import MM, new_material, new_mesh_object
from scatter import BED_TOP, collection_of, scatter, set_point_attribute
from tank import GLASS, TANK

GAP = 1.2 * MM       # between the bed's solid and the glass's inner face; add_cut_face packs it with grains


class Bed:
    """A built substrate: the object plus a way to ask how high the ground is."""

    def __init__(self, obj, height_fn, top_xy_mm, gap_mm=GAP / MM):
        self.obj = obj
        self.gap_mm = gap_mm              # between the solid's sides and the glass
        self._h = height_fn
        self.top_xy_mm = top_xy_mm        # (x, y) of each top-surface vertex, in vertex order

    def set_weight(self, name, fn):
        """Paint a 0..1 point attribute from fn(x_mm, y_mm) on the top surface; zero elsewhere."""
        vals = [fn(x, y) for x, y in self.top_xy_mm]
        vals += [0.0] * (len(self.obj.data.vertices) - len(vals))
        set_point_attribute(self.obj, name, vals)

    def z(self, x_mm, y_mm):
        return self._h(x_mm, y_mm)

    def at(self, x_mm, y_mm, lift_mm=0.0):
        """World position (metres) on the bed's surface."""
        return (x_mm * MM, y_mm * MM, (self._h(x_mm, y_mm) + lift_mm) * MM)


def build_bed(height_fn, name="Substrate", res_mm=1.5, collection=None, gap_mm=GAP / MM):
    """gap_mm: how far the solid stops short of the glass. add_cut_face fills it
    with grains, which can stand no further out of the solid than this, so a
    bed of pebbles needs about half a pebble (3 mm) for them to look round."""
    lo = GLASS + gap_mm * MM
    hi = TANK - GLASS - gap_mm * MM
    n = int(round((hi - lo) / (res_mm * MM)))
    z0 = GLASS + 0.2 * MM
    verts, faces = [], []
    for j in range(n + 1):
        y = lo + (hi - lo) * j / n
        for i in range(n + 1):
            x = lo + (hi - lo) * i / n
            verts.append((x, y, max(height_fn(x / MM, y / MM) * MM, z0 + 2 * MM)))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    top = len(verts)
    # skirt: drop every border vertex straight down to the floor
    border = ([i for i in range(n + 1)] + [j * (n + 1) + n for j in range(1, n + 1)] +
              [n * (n + 1) + i for i in range(n - 1, -1, -1)] + [j * (n + 1) for j in range(n - 1, 0, -1)])
    for k, vi in enumerate(border):
        x, y, _ = verts[vi]
        verts.append((x, y, z0))
    m = len(border)
    for k in range(m):
        a, b = border[k], border[(k + 1) % m]
        faces.append((b, a, top + k, top + (k + 1) % m))
    faces.append(tuple(top + k for k in range(m)))            # underside
    obj = new_mesh_object(name, verts, faces, collection)
    for p in obj.data.polygons[: n * n]:
        p.use_smooth = True
    bed = Bed(obj, height_fn, [(v[0] / MM, v[1] / MM) for v in verts[:top]], gap_mm)
    set_point_attribute(obj, "all", [1.0] * len(obj.data.vertices))
    # Mark the top. scatter() keeps to these faces on a bed, so nothing painted
    # on the top grows down the cut faces at the glass.
    flag = obj.data.attributes.new(BED_TOP, "BOOLEAN", "FACE")
    flag.data.foreach_set("value", [k < n * n for k in range(len(obj.data.polygons))])
    return bed


# --- materials ---------------------------------------------------------------

def soil_material(name="AquaSoil", colour=(0.030, 0.020, 0.013), variation=1.0):
    """Baked-clay aquarium soil: near-black brown, matt, varied granule to granule.
    `variation` scales how far granules differ; about 0.4 suits the cut face,
    where the full range under soft light reads as pale dots on black."""
    mat, nt, out = new_material(name)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    info = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    v = variation
    ramp.color_ramp.elements[0].color = tuple(c * (1 - 0.45 * v) for c in colour) + (1,)
    ramp.color_ramp.elements[1].color = (colour[0] * (1 + 0.9 * v), colour[1] * (1 + 0.7 * v), colour[2] * (1 + 0.5 * v), 1)
    nt.links.new(info.outputs["Random"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 900.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def sand_material(name="Sand", colour=(0.62, 0.56, 0.46)):
    """Fine pale sand: grains are below a pixel, so it is colour speckle plus fine bump."""
    mat, nt, out = new_material(name)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 2600.0
    fine.inputs["Detail"].default_value = 2.0
    nt.links.new(coord.outputs["Object"], fine.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.30, (colour[0] * 0.55, colour[1] * 0.52, colour[2] * 0.48, 1)
    e[1].position, e[1].color = 0.75, (min(colour[0] * 1.35, 1), min(colour[1] * 1.35, 1), min(colour[2] * 1.38, 1), 1)
    nt.links.new(fine.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    dunes = nt.nodes.new("ShaderNodeTexNoise")
    dunes.inputs["Scale"].default_value = 500.0
    nt.links.new(coord.outputs["Object"], dunes.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "ADD"
    nt.links.new(fine.outputs["Fac"], mix.inputs[0])
    nt.links.new(dunes.outputs["Fac"], mix.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(mix.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def sand_mix(name="SandMix", pale=(0.70, 0.66, 0.58), tan=(0.62, 0.55, 0.45), brown=(0.42, 0.34, 0.25),
             dark=(0.14, 0.12, 0.10), parts=(6, 3, 1, 1)):
    """Sand as it comes out of the bag: mostly pale grains, some tan, a few
    brown and the odd dark one. Returns a list of materials, each repeated by
    its share, for add_cut_face (or to build grains from)."""
    out = []
    for k, (colour, count) in enumerate(zip((pale, tan, brown, dark), parts)):
        out += [sand_material(f"{name}_{k}", colour=colour)] * count
    return out


# --- granules ----------------------------------------------------------------

def _granule(name, radius, seed, material):
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius)
    sx, sy, sz = (rng.uniform(0.75, 1.15) for _ in range(3))
    for v in bm.verts:
        j = 1.0 + rng.uniform(-0.12, 0.12)
        v.co.x *= sx * j
        v.co.y *= sy * j
        v.co.z *= sz * j
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(material)
    return bpy.data.objects.new(name, me)


def add_granules(bed, material, size_mm=2.4, seed=1, name="Granules"):
    """Cover the bed's top in real granules so its silhouette and shadows are
    grainy. The cut faces at the glass are add_cut_face's job."""
    grains = [_granule(f"{name}_{k}", size_mm * MM / 2, seed * 10 + k, material) for k in range(4)]
    col = collection_of(f"{name}_src", grains)
    return scatter(bed.obj, col, name=name, min_distance=size_mm * MM * 0.72, seed=seed, weight_attr="all",
                   scale=(0.7, 1.15), align="NORMAL", tilt=math.pi, sink=size_mm * MM * 0.15)


# --- the cut face at the glass ------------------------------------------------

CUT_FLOOR_MM = 33.0        # the tray hides everything below 38 mm (35 from the camera's height); the wall stops just under that
_SIDES = {                 # edge of the bed: (the axis that is fixed along it, which end, outward normal)
    "F": ("y", "lo", (0.0, -1.0, 0.0)), "B": ("y", "hi", (0.0, 1.0, 0.0)),
    "L": ("x", "lo", (-1.0, 0.0, 0.0)), "R": ("x", "hi", (1.0, 0.0, 0.0)),
}


def shade_for_cut_face(material, jitter=0.30):
    """Let a grain material take its tone from the wall it is instanced on.

    Instances made by add_cut_face carry `cut_dark` (0 = as on top of the bed,
    1 = black; set per point of the wall, so the face darkens with depth and in
    patches), `cut_face` (1) and `cut_room` (room light, see add_cut_face). The
    base colour is multiplied by 1 - cut_dark, and each grain on the face is
    shifted by up to +-jitter/2 so neighbours differ. Anywhere else they all
    read 0 and the material is unchanged, so one material serves the top of
    the bed and its face."""
    if material is None or not material.use_nodes or material.get("cut_shade"):
        return material
    nt = material.node_tree
    N, L = nt.nodes, nt.links

    def attr(name):
        a = N.new("ShaderNodeAttribute")
        a.attribute_type = "INSTANCER"
        a.attribute_name = name
        return a.outputs["Fac"]

    def math_node(op, a, b, c=None):
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

    # A tone above 1 stands in for room light the scene does not have, which
    # reaches the face from the front. Whatever of a grain is turned up to the
    # lamp is lit properly already (the top row of the face), so the lift is
    # kept off upward-facing surfaces; darkening applies everywhere.
    dark = attr("cut_dark")
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], sep.inputs[0])
    front = N.new("ShaderNodeMapRange")
    front.inputs["From Min"].default_value = 0.15
    front.inputs["From Max"].default_value = 0.60
    front.inputs["To Min"].default_value = 1.0
    front.inputs["To Max"].default_value = 0.0
    L.new(sep.outputs["Z"], front.inputs["Value"])
    lift = math_node("MULTIPLY", math_node("MINIMUM", dark, 0.0), front.outputs["Result"])
    tone = math_node("SUBTRACT", math_node("SUBTRACT", 1.0, lift), math_node("MAXIMUM", dark, 0.0))
    info = N.new("ShaderNodeObjectInfo")
    grain = math_node("MULTIPLY", math_node("MULTIPLY_ADD", info.outputs["Random"], jitter, -jitter / 2), attr("cut_face"))
    tone = math_node("MULTIPLY_ADD", tone, grain, tone)
    # Room light on the pane (`cut_room`, 0 anywhere but on a face that asks for
    # it): what a broad, soft source in front of the glass would add to each
    # grain, most where the grain is turned to the glass, least round its sides.
    # It is added as the grain's own glow, so it costs no samples and no lamp.
    out_n = N.new("ShaderNodeCombineXYZ")
    L.new(attr("cut_nx"), out_n.inputs["X"])
    L.new(attr("cut_ny"), out_n.inputs["Y"])
    dot = N.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    L.new(geo.outputs["Normal"], dot.inputs[0])
    L.new(out_n.outputs["Vector"], dot.inputs[1])
    facing = math_node("MULTIPLY_ADD", math_node("MAXIMUM", dot.outputs["Value"], 0.0), 0.75, 0.25)
    room = math_node("MULTIPLY", facing, attr("cut_room"))
    for bsdf in [n for n in N if n.type == "BSDF_PRINCIPLED"]:
        sock = bsdf.inputs["Base Color"]
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.clamp_result = True
        if sock.links:
            L.new(sock.links[0].from_socket, mix.inputs["A"])
        else:
            mix.inputs["A"].default_value = tuple(sock.default_value)
        L.new(tone, mix.inputs["B"])
        L.new(mix.outputs["Result"], sock)
        if not bsdf.inputs["Emission Strength"].links and bsdf.inputs["Emission Strength"].default_value == 0.0:
            L.new(mix.outputs["Result"], bsdf.inputs["Emission Color"])
            L.new(room, bsdf.inputs["Emission Strength"])
    # The glow is for the camera only. Left to itself Cycles would enter every
    # grain in its light tree as a lamp, which slows the whole frame by a fifth.
    material.cycles.emission_sampling = "NONE"
    material["cut_shade"] = True
    return material


def _pressed_grain(name, size, thick, seed, material):
    """A grain for the face: a granule no thicker than the gap it sits in, so
    stones wider than the gap lie with a flat side to the glass."""
    g = _granule(name, size / 2, seed, material)
    if thick < size:
        for v in g.data.vertices:
            v.co.z *= thick / size
    return g


def add_soil_face(bed, colours=((0.034, 0.026, 0.020),), size_mm=3.0, **options):
    """The cut face for dark aquarium soil: add_cut_face with what soil needs.
    Granules a little larger and far less varied than on top (at 3 px per mm a
    2.4 mm granule is mush, and the top's range of tones reads as pale dots on
    black), packed close, and room light, because the front face of a dark bed
    gets almost nothing from the scene's own lights. `colours` mixes several
    soils; other options are add_cut_face's."""
    mats = [soil_material("SoilAtGlass", colour=c, variation=0.4) for c in colours]
    opts = dict(tone=(1.0, 0.5), vary=0.3, room=0.9, spacing=0.6)
    opts.update(options)
    return add_cut_face(bed, mats, size_mm, **opts)


def _pore_material(colour):
    mat, nt, out = new_material("CutFacePores")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def add_cut_face(bed, material=None, size_mm=2.4, weight=None, sources=None, seed=1, name="CutFace", sides="FLR",
                 tone=(1.0, 0.42), fade_mm=30.0, vary=0.6, ragged_mm=None, spacing=None, scale=(0.55, 1.2),
                 tilt=None, thick_mm=None, layers=None, side_depth_mm=10.0, backing=None, top_mm=0.0, recess_mm=0.0,
                 room=0.0):
    """Pack grains against the glass over the bed's cut faces. Returns the new
    objects (add them to the scape's list for assets.mark_underwater).

    A bed seen in section through the pane is not a flat band: it is grains
    pressed against the glass, with an uneven top line where the top row sits,
    dimmer than the lit top and dimmer still toward the floor, in patches and
    faint settled layers. This builds that from instanced grains in the gap
    between the bed's solid and the glass (Bed.gap_mm): on the front face down
    to the tray line, and on the side faces as far down as can be seen (they
    show only by reflection in the side panes, where an empty gap reads as a
    dark trough).

    material    grain material, or a list of them to mix (see sand_mix); kit
                granules are made from it. Or give
    sources     your own grain objects instead (origin at the middle; if wider
                than about twice the bed's gap_mm, flat along local Z and
                `thick_mm` thick)
    size_mm     grain diameter: about 3 soil, 1.0 to 1.4 sand, 3 to 7 gravel
    weight      fn(x_mm, y_mm, depth_mm below the top) -> 0..1, where this kind
                of grain lies; call again with another material for a second
                kind (a sand lens in soil, sand over gravel). None: everywhere
    sides       any of "F" front, "L" left, "R" right, "B" back
    tone        brightness of the grains at the top of the face and at
                `fade_mm` below it, as multiples of the material's own colour
    vary        how much one grain differs from the next in brightness (0.6 is
                +-30%); fixed by the first call that uses a material
    ragged_mm   how far the top line wanders about the bed's edge (default
                about half a grain)
    spacing     distance between grains as a fraction of size_mm (default
                0.5 for sand, which must close up, 0.7 for anything coarser)
    layers      2 sets a second course behind the first to close the gaps
                (the default for grains of 2 mm and over; sand gets one close
                course, which renders a quarter faster than two). The side
                faces always get one
    backing     colour of a matt sheet behind the grains, for the pores between
                coarse grains; None for none
    top_mm      raise (or lower) the whole top line
    recess_mm   set this kind back from the glass, behind another (fine gravel
                in the pores between pebbles)
    room        soft light from the room on the pane, for scapes whose own
                lights leave the face black (dark soil under a dim fill). 1.0
                is about a third of the light the lamp puts on the bed's top.
                It follows `tone` down the face. It is the grains' own glow,
                not a lamp: no samples, no shadows, nothing else is lit by it
    """
    if os.environ.get("AQUARIUM_NO_CUT_FACE") == "1":      # to measure what the faces cost: render with and without
        return []
    lo, hi = GLASS / MM + bed.gap_mm, (TANK - GLASS) / MM - bed.gap_mm
    step = 2.0
    if spacing is None:
        spacing = 0.5 if size_mm < 2.0 else 0.7
    if layers is None:
        layers = 1 if size_mm < 2.0 else 2
    rag = max(0.45, 0.42 * size_mm) if ragged_mm is None else ragged_mm
    cols = int(math.ceil((hi - lo) / step))
    floor_mm = (GLASS + 0.2 * MM) / MM + 2.0
    thick = min(size_mm, 2.0 * bed.gap_mm - 0.1) if thick_mm is None else thick_mm
    if tilt is None:
        tilt = math.pi if size_mm <= 2.6 else 0.18
    if sources is None:
        mats = list(material) if isinstance(material, (list, tuple)) else [material]
        for m in mats:
            shade_for_cut_face(m, vary)
        sources = [_pressed_grain(f"{name}_grain{k}", size_mm * MM, thick * MM, seed * 10 + k, mats[k % len(mats)])
                   for k in range(max(5, len(mats)))]
    else:
        for o in sources:
            for m in o.data.materials:
                shade_for_cut_face(m, vary)
    src = collection_of(f"{name}_src", sources)
    made = []
    for side in sides:
        axis, end, normal = _SIDES[side]
        fixed = lo if end == "lo" else hi
        depth = None if side == "F" else side_depth_mm
        tops, pts = [], []
        for i in range(cols + 1):
            s = lo + (hi - lo) * i / cols
            x, y = (s, fixed) if axis == "y" else (fixed, s)
            wander = (0.65 * noise.noise(Vector((s / (3.0 * size_mm + 2.0), seed * 3.7, ord(side))))
                      + 0.35 * noise.noise(Vector((s / (0.9 * size_mm + 1.0), seed * 1.3 + 9.0, ord(side)))))
            tops.append(max(bed.z(x, y), floor_mm) + top_mm + rag * (0.3 + 1.6 * wander))
            pts.append((x, y))
        deepest = max(max(tops) - CUT_FLOOR_MM, 4.0) if depth is None else depth
        rows = max(2, int(math.ceil(deepest / step)))
        verts, faces, w, dark = [], [], [], []
        for (x, y), ztop in zip(pts, tops):
            zbot = min(CUT_FLOOR_MM, ztop - 4.0) if depth is None else ztop - depth
            s = x if axis == "y" else y
            for j in range(rows + 1):
                z = ztop + (zbot - ztop) * j / rows
                d = ztop - z
                verts.append((x * MM, y * MM, z * MM))
                w.append(1.0 if weight is None else max(0.0, min(1.0, weight(x, y, d))))
                # Light reaches the face from the lit top, through the pores and the
                # pane: less of it the deeper the grain. Over that, damp patches and
                # faint level bands where the bed settled.
                f = min(1.0, d / fade_mm)
                fall = tone[0] + (tone[1] - tone[0]) * (f * f * (3 - 2 * f)) ** 0.75
                if depth is not None:
                    # The side faces show only as a thin seam in the side panes' mirror,
                    # lit by almost nothing; paler grains keep that seam from going black.
                    fall = min(1.5, 1.3 * tone[0])
                patch = 1.0 + 0.16 * noise.noise(Vector((s * 0.045, z * 0.07, seed + 0.5)))
                strata = 1.0 + 0.10 * noise.noise(Vector((s * 0.012, z * 0.55, seed + 7.5)))
                dark.append(1.0 - fall * patch * strata)
        for i in range(cols):
            for j in range(rows):
                a = i * (rows + 1) + j
                quad = (a, a + 1, a + rows + 2, a + rows + 1)
                faces.append(quad if side in "FR" else quad[::-1])
        wall = new_mesh_object(f"{name}_{side}", verts, faces)
        set_point_attribute(wall, "w", w)
        set_point_attribute(wall, "cut_dark", dark)
        set_point_attribute(wall, "cut_face", [1.0] * len(verts))
        set_point_attribute(wall, "cut_room", [room] * len(verts))
        set_point_attribute(wall, "cut_nx", [normal[0]] * len(verts))
        set_point_attribute(wall, "cut_ny", [normal[1]] * len(verts))
        # the largest grain's front stands just clear of the glass
        r_max = 0.5 * (thick if tilt < 0.5 else size_mm) * scale[1]
        reach = bed.gap_mm - 0.10 - r_max - recess_mm
        if backing is not None:
            wall.data.materials.append(bpy.data.materials.get("CutFacePores") or _pore_material(backing))
            for v in wall.data.vertices:          # a hair proud of the bed's own skirt, which it hides
                v.co += Vector(normal) * 0.05 * MM
        for k in range(layers if side == "F" else 1):
            # one object per course, sharing the sheet; only the first may show the sheet itself
            carrier = wall if k == 0 else bpy.data.objects.new(f"{name}_{side}{k}", wall.data)
            if k:
                bpy.context.scene.collection.objects.link(carrier)
            scatter(carrier, src, name=f"{name}_{side}{k}", min_distance=size_mm * MM * spacing, weight_attr="w",
                    seed=seed * 7 + k * 13 + ord(side), scale=scale, align="NORMAL", tilt=tilt,
                    keep_surface=(k == 0 and backing is not None), lift=(reach - k * 0.45 * thick) * MM)
            made.append(carrier)
    return made
