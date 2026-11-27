"""Substrate: a solid bed whose top follows a height function.

Heights are millimetres above the tank's outer base. The bed stops a hair short
of the glass so it never shares a surface with it, and it is a closed solid so
its cut face shows through the front and side glass like real soil or sand.
"""
import math
import random

import bmesh
import bpy

from bl_common import MM, new_material, new_mesh_object
from scatter import collection_of, scatter
from tank import GLASS, TANK

GAP = 1.2 * MM       # between substrate and the glass's inner face


class Bed:
    """A built substrate: the object plus a way to ask how high the ground is."""

    def __init__(self, obj, height_fn, top_xy_mm):
        self.obj = obj
        self._h = height_fn
        self.top_xy_mm = top_xy_mm        # (x, y) of each top-surface vertex, in vertex order

    def set_weight(self, name, fn):
        """Paint a 0..1 point attribute from fn(x_mm, y_mm) on the top surface; zero elsewhere."""
        from scatter import set_point_attribute
        vals = [fn(x, y) for x, y in self.top_xy_mm]
        vals += [0.0] * (len(self.obj.data.vertices) - len(vals))
        set_point_attribute(self.obj, name, vals)

    def z(self, x_mm, y_mm):
        return self._h(x_mm, y_mm)

    def at(self, x_mm, y_mm, lift_mm=0.0):
        """World position (metres) on the bed's surface."""
        return (x_mm * MM, y_mm * MM, (self._h(x_mm, y_mm) + lift_mm) * MM)


def build_bed(height_fn, name="Substrate", res_mm=1.5, collection=None):
    lo = GLASS + GAP
    hi = TANK - GLASS - GAP
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
    bed = Bed(obj, height_fn, [(v[0] / MM, v[1] / MM) for v in verts[:top]])
    bed.set_weight("top", lambda x, y: 1.0)
    return bed


# --- materials ---------------------------------------------------------------

def soil_material(name="AquaSoil", colour=(0.030, 0.020, 0.013)):
    """Baked-clay aquarium soil: near-black brown, matt, slightly varied granule to granule."""
    mat, nt, out = new_material(name)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    info = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (colour[0] * 0.55, colour[1] * 0.55, colour[2] * 0.55, 1)
    ramp.color_ramp.elements[1].color = (colour[0] * 1.9, colour[1] * 1.7, colour[2] * 1.5, 1)
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
    """Cover the bed in real granules so its silhouette and shadows are grainy."""
    grains = [_granule(f"{name}_{k}", size_mm * MM / 2, seed * 10 + k, material) for k in range(4)]
    col = collection_of(f"{name}_src", grains)
    return scatter(bed.obj, col, name=name, min_distance=size_mm * MM * 0.72, seed=seed, weight_attr="top",
                   scale=(0.7, 1.15), align="NORMAL", tilt=math.pi, sink=size_mm * MM * 0.15)
