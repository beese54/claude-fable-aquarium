"""Bring downloaded CC0 models into the scene at aquarium scale.

Models live in the work directory (see src/assets/fetch_assets.py). Several ship
all LODs in one file or as a row of separate pieces, so import returns the
individual LOD0 mesh objects and callers pick what they need.
"""
import math

import bpy
from mathutils import Matrix, Vector

from bl_common import MM
import paths


def import_model(asset_id, source="polyhaven"):
    """Import a glTF asset; return its mesh objects (highest detail only), unparented, at the origin's scale."""
    folder = paths.ASSETS / source / asset_id
    gltfs = sorted(folder.glob("*.gltf"))
    if not gltfs:
        raise FileNotFoundError(f"no glTF for {asset_id} in {folder}; run src/assets/fetch_assets.py")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(gltfs[0]))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = []
    for o in new:
        if o.type != "MESH":
            continue
        name = o.name.upper()
        if any(f"LOD{k}" in name for k in range(1, 6)):
            continue
        meshes.append(o)
    # bake parent transforms, then drop empties and lower LODs
    bpy.context.view_layer.update()
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    for o in new:
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    meshes.sort(key=lambda o: o.name)
    return meshes


def world_bounds(obj):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def normalise(obj):
    """Apply the object's transform and move its origin to the bottom centre of its bounds."""
    bpy.context.view_layer.update()
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    lo, hi = world_bounds(obj)
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    obj.data.transform(Matrix.Translation(-centre))
    obj.data.update()
    return obj


def place(obj, at, height_mm=None, longest_mm=None, rot_deg=(0, 0, 0), sink_mm=0.0, pre_rot_deg=(0, 0, 0)):
    """Scale and stand an object on a point.

    pre_rot_deg  re-orients the raw model first (e.g. lay it on another face);
                 the origin is then reset to the new bottom centre
    height_mm    make its height this (after pre-rotation), or
    longest_mm   make its longest side this
    rot_deg      final tilt and heading, applied about the bottom centre
    sink_mm      bury it this far
    """
    normalise(obj)
    if any(pre_rot_deg):
        e = Matrix.Rotation(math.radians(pre_rot_deg[2]), 4, "Z") @ Matrix.Rotation(math.radians(pre_rot_deg[1]), 4, "Y") @ Matrix.Rotation(math.radians(pre_rot_deg[0]), 4, "X")
        obj.data.transform(e)
        normalise(obj)
    lo, hi = world_bounds(obj)
    size = hi - lo
    if height_mm:
        k = height_mm * MM / size.z
    elif longest_mm:
        k = longest_mm * MM / max(size)
    else:
        k = 1.0
    obj.scale = (k, k, k)
    obj.rotation_euler = tuple(math.radians(a) for a in rot_deg)
    obj.location = (at[0], at[1], at[2] - sink_mm * MM)
    return obj


def footprint_radius_mm(obj):
    lo, hi = world_bounds(obj)
    return max(hi.x - lo.x, hi.y - lo.y) / 2 / MM


def grade_stone(obj, dark=(0.030, 0.036, 0.048), light=(0.40, 0.44, 0.50), lum_range=(0.16, 0.58),
                vein_amount=0.30, vein_scale=34.0, bump=0.45, normal_strength=2.2, algae=0.45, seed=0.0):
    """Turn a pale photoscanned rock into Seiryu-like stone.

    The scan's own brightness pattern is kept but re-coloured between a dark and
    a light blue-grey, which restores the contrast that makes the surface read as
    rock; crevices are deepened with the scan's ambient-occlusion map; thin pale
    calcite veins run in one direction, as they do in real Seiryu; and fine
    procedural bump is layered under the scan's normal map so the surface holds
    up when the stone is scaled beyond the scan's resolution.
    """
    for slot in obj.material_slots:
        mat = slot.material.copy()
        slot.material = mat
        nt = mat.node_tree
        N, L = nt.nodes, nt.links
        bsdf = next((n for n in N if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        base_in = bsdf.inputs["Base Color"]
        src = base_in.links[0].from_socket if base_in.links else None
        sep = next((n for n in N if n.type == "SEPARATE_COLOR"), None)
        nmap = next((n for n in N if n.type == "NORMAL_MAP"), None)

        lum = N.new("ShaderNodeRGBToBW")
        if src:
            L.new(src, lum.inputs["Color"])
        tone = N.new("ShaderNodeMapRange")
        tone.inputs["From Min"].default_value = lum_range[0]
        tone.inputs["From Max"].default_value = lum_range[1]
        L.new(lum.outputs["Val"], tone.inputs["Value"])
        col = N.new("ShaderNodeMix")
        col.data_type = "RGBA"
        col.inputs["A"].default_value = (*dark, 1.0)
        col.inputs["B"].default_value = (*light, 1.0)
        L.new(tone.outputs["Result"], col.inputs["Factor"])
        out_col = col.outputs["Result"]

        if sep is not None:                      # ambient occlusion lives in the red channel
            ao = N.new("ShaderNodeMath")
            ao.operation = "POWER"
            ao.inputs[1].default_value = 1.6
            L.new(sep.outputs["Red"], ao.inputs[0])
            mul = N.new("ShaderNodeMix")
            mul.data_type = "RGBA"
            mul.blend_type = "MULTIPLY"
            mul.inputs["Factor"].default_value = 1.0
            L.new(out_col, mul.inputs["A"])
            L.new(ao.outputs[0], mul.inputs["B"])
            out_col = mul.outputs["Result"]

        # Veins: the mid contour of a noise field stretched hard along one axis.
        coord = N.new("ShaderNodeTexCoord")
        mapping = N.new("ShaderNodeMapping")
        mapping.inputs["Location"].default_value = (seed * 3.1, seed * 1.7, seed * 0.9)
        mapping.inputs["Rotation"].default_value = (0.5, 0.35 + seed * 0.07, 0.2)
        mapping.inputs["Scale"].default_value = (0.30, 1.0, 1.0)
        L.new(coord.outputs["Object"], mapping.inputs["Vector"])
        noise = N.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = vein_scale
        noise.inputs["Detail"].default_value = 3.0
        noise.inputs["Distortion"].default_value = 0.35
        L.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        ridge = N.new("ShaderNodeMath")
        ridge.operation = "SUBTRACT"
        ridge.inputs[1].default_value = 0.5
        L.new(noise.outputs["Fac"], ridge.inputs[0])
        ab = N.new("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        L.new(ridge.outputs[0], ab.inputs[0])
        thin = N.new("ShaderNodeMapRange")
        thin.interpolation_type = "SMOOTHSTEP"
        thin.inputs["From Min"].default_value = 0.0015
        thin.inputs["From Max"].default_value = 0.0050
        thin.inputs["To Min"].default_value = 1.0
        thin.inputs["To Max"].default_value = 0.0
        L.new(ab.outputs[0], thin.inputs["Value"])
        patch = N.new("ShaderNodeTexNoise")       # veins come and go
        patch.inputs["Scale"].default_value = 9.0
        L.new(coord.outputs["Object"], patch.inputs["Vector"])
        gate = N.new("ShaderNodeMapRange")
        gate.inputs["From Min"].default_value = 0.56
        gate.inputs["From Max"].default_value = 0.70
        L.new(patch.outputs["Fac"], gate.inputs["Value"])
        amt = N.new("ShaderNodeMath")
        amt.operation = "MULTIPLY"
        L.new(thin.outputs["Result"], amt.inputs[0])
        L.new(gate.outputs["Result"], amt.inputs[1])
        amt2 = N.new("ShaderNodeMath")
        amt2.operation = "MULTIPLY"
        amt2.inputs[1].default_value = vein_amount
        L.new(amt.outputs[0], amt2.inputs[0])
        vein = N.new("ShaderNodeMix")
        vein.data_type = "RGBA"
        vein.inputs["B"].default_value = (0.62, 0.64, 0.64, 1.0)
        L.new(amt2.outputs[0], vein.inputs["Factor"])
        L.new(out_col, vein.inputs["A"])
        # Broad variation, and a faint olive film of algae on faces that look up
        # at the light, as on stone that has sat in a tank for months.
        vary = N.new("ShaderNodeTexNoise")
        vary.inputs["Scale"].default_value = 14.0
        vary.inputs["Detail"].default_value = 4.0
        L.new(mapping.outputs["Vector"], vary.inputs["Vector"])
        vr = N.new("ShaderNodeMapRange")
        vr.inputs["From Min"].default_value = 0.30
        vr.inputs["From Max"].default_value = 0.70
        vr.inputs["To Min"].default_value = 0.72
        vr.inputs["To Max"].default_value = 1.22
        L.new(vary.outputs["Fac"], vr.inputs["Value"])
        varied = N.new("ShaderNodeMix")
        varied.data_type = "RGBA"
        varied.blend_type = "MULTIPLY"
        varied.inputs["Factor"].default_value = 1.0
        L.new(vein.outputs["Result"], varied.inputs["A"])
        L.new(vr.outputs["Result"], varied.inputs["B"])
        geo = N.new("ShaderNodeNewGeometry")
        nsep = N.new("ShaderNodeSeparateXYZ")
        L.new(geo.outputs["Normal"], nsep.inputs[0])
        upw = N.new("ShaderNodeMapRange")
        upw.inputs["From Min"].default_value = 0.35
        upw.inputs["From Max"].default_value = 0.95
        L.new(nsep.outputs["Z"], upw.inputs["Value"])
        film_n = N.new("ShaderNodeTexNoise")
        film_n.inputs["Scale"].default_value = 30.0
        film_n.inputs["Detail"].default_value = 5.0
        L.new(coord.outputs["Object"], film_n.inputs["Vector"])
        fr = N.new("ShaderNodeMapRange")
        fr.inputs["From Min"].default_value = 0.42
        fr.inputs["From Max"].default_value = 0.68
        L.new(film_n.outputs["Fac"], fr.inputs["Value"])
        fa = N.new("ShaderNodeMath")
        fa.operation = "MULTIPLY"
        L.new(upw.outputs["Result"], fa.inputs[0])
        L.new(fr.outputs["Result"], fa.inputs[1])
        fb = N.new("ShaderNodeMath")
        fb.operation = "MULTIPLY"
        fb.inputs[1].default_value = algae
        L.new(fa.outputs[0], fb.inputs[0])
        film = N.new("ShaderNodeMix")
        film.data_type = "RGBA"
        film.inputs["B"].default_value = (0.085, 0.105, 0.040, 1.0)
        L.new(fb.outputs[0], film.inputs["Factor"])
        L.new(varied.outputs["Result"], film.inputs["A"])
        L.new(film.outputs["Result"], base_in)

        bsdf.inputs["Roughness"].default_value = 0.70
        for l in list(bsdf.inputs["Roughness"].links):
            L.remove(l)
        for l in list(bsdf.inputs["Metallic"].links):
            L.remove(l)
        bsdf.inputs["Metallic"].default_value = 0.0

        if nmap is not None:
            nmap.inputs["Strength"].default_value = normal_strength
            grit = N.new("ShaderNodeTexNoise")
            grit.inputs["Scale"].default_value = 260.0
            grit.inputs["Detail"].default_value = 6.0
            grit.inputs["Roughness"].default_value = 0.7
            L.new(coord.outputs["Object"], grit.inputs["Vector"])
            bmp = N.new("ShaderNodeBump")
            bmp.inputs["Strength"].default_value = bump
            bmp.inputs["Distance"].default_value = 0.0008
            L.new(grit.outputs["Fac"], bmp.inputs["Height"])
            L.new(nmap.outputs["Normal"], bmp.inputs["Normal"])
            L.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return obj

def mark_underwater(objs, ctx):
    """Objects below the surface receive the lamp's refracted light and its
    caustics, and join the collection that tank-only lights are linked to."""
    col = ctx["underwater"]
    for o in objs:
        o.cycles.is_caustics_receiver = True
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
