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
    material = material or leaf_material("Hairgrass", base=(0.030, 0.105, 0.012), tip=(0.19, 0.42, 0.045), translucency=0.30)
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
