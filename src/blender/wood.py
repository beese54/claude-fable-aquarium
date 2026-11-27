"""Driftwood, grown by code.

The CC0 libraries have little that reads as aquarium wood, so branching pieces
(spiderwood trees, tangled roots, hanging rootlets) are generated: a trunk that
wanders and tapers, shedding thinner children that do the same. The skin is a
tube mesh with real bark laid on by box projection, so there are no seams to hide.
"""
import math
import random

import bpy
from mathutils import Vector

from bl_common import MM


def bark_material(name="Driftwood", **kw):
    """Scanned bark by box projection, pulled toward the warm grey-brown of soaked wood."""
    import materials
    return materials.driftwood(name, **kw)

def _frame(t):
    t = t.normalized()
    ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
    n1 = t.cross(ref).normalized()
    n2 = t.cross(n1).normalized()
    return n1, n2


class _Skin:
    def __init__(self, sides=8):
        self.sides = sides
        self.verts, self.faces = [], []
        self.tips = []          # (position, direction, radius) at each branch end

    def tube(self, pts, radii):
        s = self.sides
        off = len(self.verts)
        for k, (p, r) in enumerate(zip(pts, radii)):
            a = pts[max(k - 1, 0)]
            b = pts[min(k + 1, len(pts) - 1)]
            n1, n2 = _frame(b - a)
            for i in range(s):
                ang = 2 * math.pi * i / s
                self.verts.append(tuple(p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * r))
        for k in range(len(pts) - 1):
            for i in range(s):
                a = off + k * s + i
                b = off + k * s + (i + 1) % s
                self.faces.append((a, b, b + s, a + s))
        # close the cut end at the base, then cap the tip
        root = len(self.verts)
        self.verts.append(tuple(pts[0]))
        for i in range(s):
            self.faces.append((off + (i + 1) % s, off + i, root))
        tip = len(self.verts)
        self.verts.append(tuple(pts[-1] + (pts[-1] - pts[-2]).normalized() * radii[-1] * 0.6))
        base = off + (len(pts) - 1) * s
        for i in range(s):
            self.faces.append((base + i, base + (i + 1) % s, tip))


def grow(start, direction, length_mm, radius_mm, seed=0, depth=3, children=(2, 4), wander=0.35,
         gravity=0.0, child_length=0.62, child_radius=0.58, spread=(0.5, 1.1), sides=8, taper=0.35, skin=None):
    """Grow one branching piece. Returns the _Skin (use finish() to make an object).

    start, direction   metres and a unit vector
    wander             how much the branch meanders
    gravity            >0 bends branches downward (hanging roots), <0 upward (trees)
    spread             range of angle (radians) between a child and its parent
    taper              radius at a branch's tip as a fraction of its base
    """
    rng = random.Random(seed)
    skin = skin or _Skin(sides)

    def branch(p0, d0, L, R, level):
        segs = max(4, int(L / (6.0 * MM)))
        pts, radii = [Vector(p0)], [R]
        d = Vector(d0).normalized()
        p = Vector(p0)
        step = L / segs
        marks = []
        for k in range(1, segs + 1):
            n1, n2 = _frame(d)
            d = (d + (n1 * rng.uniform(-1, 1) + n2 * rng.uniform(-1, 1)) * wander * 0.35 + Vector((0, 0, -gravity * 0.25))).normalized()
            p = p + d * step
            t = k / segs
            pts.append(p.copy())
            radii.append(R * (1 - (1 - taper) * t) * (1 + 0.08 * math.sin(k * 2.1 + level)))
            marks.append((p.copy(), d.copy(), radii[-1], t))
        skin.tube(pts, radii)
        if level >= depth:
            skin.tips.append((pts[-1].copy(), d.copy(), radii[-1]))
            return
        n_children = rng.randint(*children)
        for c in range(n_children):
            mp, md, mr, mt = marks[min(len(marks) - 1, int(len(marks) * rng.uniform(0.35, 1.0)))]
            n1, n2 = _frame(md)
            az = rng.uniform(0, 2 * math.pi)
            ang = rng.uniform(*spread)
            cd = (md * math.cos(ang) + (n1 * math.cos(az) + n2 * math.sin(az)) * math.sin(ang)).normalized()
            branch(mp - cd * mr * 0.5, cd, L * child_length * rng.uniform(0.75, 1.2), mr * child_radius / max(taper, 0.3) * 0.62, level + 1)

    branch(Vector(start), Vector(direction), length_mm * MM, radius_mm * MM, 0)
    return skin


def finish(skin, name, material, collection=None, subdivide=1):
    me = bpy.data.meshes.new(name)
    me.from_pydata(skin.verts, [], skin.faces)
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(material)
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    if subdivide:
        m = obj.modifiers.new("Smooth", "SUBSURF")
        m.levels = m.render_levels = subdivide
    # knots and swellings, so it is not a clean pipe
    tex = bpy.data.textures.new(f"{name}_knots", "CLOUDS")
    tex.noise_scale = 0.012
    d = obj.modifiers.new("Knots", "DISPLACE")
    d.texture = tex
    d.strength = 0.0028
    d.mid_level = 0.5
    d.texture_coords = "GLOBAL"
    return obj
