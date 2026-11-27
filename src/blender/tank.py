"""The glass tank and its water.

Glass: one closed shell (outer box minus inner box, open at the top), 5 mm
thick, with real absorption so the panes are clear face-on and green edge-on,
as in the footage.

Water: a closed body whose sides and bottom reach 1 mm into the glass. Cycles
has no nested-dielectric priority; overlapping the liquid into the glass makes
the chain of refractions air > glass > water come out with the right final
direction (Snell's law composes across parallel interfaces regardless of the
order the interfaces are met in). The top is a finely divided, gently rippling
surface that casts real refractive caustics from the lamp.
"""
import bpy

from bl_common import MM, new_material, new_mesh_object

TANK = 300.0 * MM
GLASS = 5.0 * MM
OVERLAP = 1.0 * MM
EDGE_GLOW = 3.0


def glass_material():
    mat, nt, out = new_material("TankGlass")
    bsdf = nt.nodes.new("ShaderNodeBsdfGlass")
    bsdf.distribution = "MULTI_GGX"
    bsdf.inputs["Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.52
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    # Soda-lime glass: clear through 5 mm, teal through 300 mm.
    absorb = nt.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.70, 0.96, 0.92, 1.0)
    absorb.inputs["Density"].default_value = 12.0
    # Lamp light that enters the panes is piped along them and makes their edges
    # glow teal in the footage. A path tracer will not find that light, so the
    # glass is given a faint glow of its own: invisible across 5 mm, bright along 300.
    glow = nt.nodes.new("ShaderNodeEmission")
    glow.inputs["Color"].default_value = (0.20, 0.95, 0.80, 1.0)
    glow.inputs["Strength"].default_value = EDGE_GLOW
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(absorb.outputs[0], add.inputs[0])
    nt.links.new(glow.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Volume"])
    return mat


def build_glass(collection=None):
    o, i, t = 0.0, GLASS, TANK
    # 0-3 outer bottom, 4-7 outer top, 8-11 inner floor, 12-15 inner top
    v = [(o, o, o), (t, o, o), (t, t, o), (o, t, o),
         (o, o, t), (t, o, t), (t, t, t), (o, t, t),
         (i, i, i), (t - i, i, i), (t - i, t - i, i), (i, t - i, i),
         (i, i, t), (t - i, i, t), (t - i, t - i, t), (i, t - i, t)]
    f = [(0, 3, 2, 1),                                               # outer bottom
         (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),     # outer walls
         (8, 9, 10, 11),                                             # inner floor (faces up)
         (8, 12, 13, 9), (9, 13, 14, 10), (10, 14, 15, 11), (11, 15, 12, 8),   # inner walls (face inward)
         (4, 5, 13, 12), (5, 6, 14, 13), (6, 7, 15, 14), (7, 4, 12, 15)]       # rim
    obj = new_mesh_object("TankGlass", v, f, collection)
    obj.data.materials.append(glass_material())
    obj.visible_shadow = False     # the lamp shines in through the open top; glass must not block spill light
    return obj


def water_material(tint=(0.88, 0.975, 0.955), density=1.5, scatter=0.0, scatter_colour=(0.9, 0.95, 0.9)):
    """Clear water by default. `tint` is the colour left after about a metre;
    lower `tint` and raise `density` for blackwater. `scatter` adds haze."""
    mat, nt, out = new_material("Water")
    bsdf = nt.nodes.new("ShaderNodeBsdfGlass")
    bsdf.distribution = "MULTI_GGX"
    bsdf.inputs["Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.333
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    absorb = nt.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (*tint, 1.0)
    absorb.inputs["Density"].default_value = density
    if scatter > 0:
        sc = nt.nodes.new("ShaderNodeVolumeScatter")
        sc.inputs["Color"].default_value = (*scatter_colour, 1.0)
        sc.inputs["Density"].default_value = scatter
        sc.inputs["Anisotropy"].default_value = 0.6
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(absorb.outputs[0], add.inputs[0])
        nt.links.new(sc.outputs[0], add.inputs[1])
        nt.links.new(add.outputs[0], out.inputs["Volume"])
    else:
        nt.links.new(absorb.outputs[0], out.inputs["Volume"])
    return mat


def build_water(level_mm=268.0, collection=None, ripple_mm=0.35, ripple_scale=0.085, fine_mm=0.22, fine_scale=0.020, grid=200, material=None):
    """Water body up to `level_mm` above the tank's outer base."""
    lo = GLASS - OVERLAP
    hi = TANK - GLASS + OVERLAP
    z0 = GLASS - OVERLAP
    z1 = level_mm * MM
    n = grid
    verts, faces = [], []
    # top surface grid
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((lo + (hi - lo) * i / n, lo + (hi - lo) * j / n, z1))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    top_count = len(verts)
    # bottom corners
    b = len(verts)
    verts += [(lo, lo, z0), (hi, lo, z0), (hi, hi, z0), (lo, hi, z0)]
    faces.append((b, b + 3, b + 2, b + 1))
    # walls: fan each side from the top edge down to the two bottom corners
    front = [i for i in range(n + 1)]
    back = [n * (n + 1) + i for i in range(n + 1)]
    left = [j * (n + 1) for j in range(n + 1)]
    right = [j * (n + 1) + n for j in range(n + 1)]
    faces.append(tuple([b, b + 1] + front[::-1]))
    faces.append(tuple([b + 1, b + 2] + right[::-1]))
    faces.append(tuple([b + 2, b + 3] + back))
    faces.append(tuple([b + 3, b] + left))
    obj = new_mesh_object("Water", verts, faces, collection)
    for p in obj.data.polygons:
        p.use_smooth = p.index < n * n          # only the rippling surface is smooth
    obj.data.materials.append(material or water_material())

    # Ripples: displace only the surface, fading to nothing at the glass so the
    # body stays closed and the waterline stays straight.
    vg = obj.vertex_groups.new(name="Surface")
    for idx in range(top_count):
        x, y, _ = verts[idx]
        edge = min(x - lo, hi - x, y - lo, hi - y)
        w = min(1.0, edge / (12.0 * MM))
        if w > 0:
            vg.add([idx], w, "REPLACE")
    tex = bpy.data.textures.new("RippleNoise", "CLOUDS")
    tex.noise_scale = ripple_scale
    tex.noise_depth = 1
    drift = bpy.data.objects.new("RippleDrift", None)
    (collection or bpy.context.scene.collection).objects.link(drift)
    mod = obj.modifiers.new("Ripples", "DISPLACE")
    mod.texture = tex
    mod.texture_coords = "OBJECT"
    mod.texture_coords_object = drift
    mod.direction = "Z"
    mod.mid_level = 0.5
    mod.strength = ripple_mm * MM * 2.0
    mod.vertex_group = "Surface"
    # A second, finer set of ripples. The broad swell moves the reflection on
    # the underside; these short ones are what focus the LEDs into caustic lines
    # at the depth of the scape (focal length is about wavelength^2 / (13 x height)).
    fine = bpy.data.textures.new("RippleFine", "CLOUDS")
    fine.noise_scale = fine_scale
    fine.noise_depth = 0
    mod2 = obj.modifiers.new("RipplesFine", "DISPLACE")
    mod2.texture = fine
    mod2.texture_coords = "OBJECT"
    mod2.texture_coords_object = drift
    mod2.direction = "Z"
    mod2.mid_level = 0.5
    mod2.strength = fine_mm * MM * 2.0
    mod2.vertex_group = "Surface"
    obj.cycles.is_caustics_caster = True
    obj.visible_shadow = True
    obj["top_count"] = top_count
    obj["ripple_strengths"] = [mod.strength, mod2.strength]
    return obj, drift


def set_water_level(obj, level_mm, turbulence=1.0):
    """Move the surface to a new height and scale its ripples. Called per frame
    while the tank fills; the body's sides and bottom stay where they are."""
    n = obj["top_count"]
    me = obj.data
    z = level_mm * MM
    for i in range(n):
        me.vertices[i].co.z = z
    me.update()
    base = obj["ripple_strengths"]
    obj.modifiers["Ripples"].strength = base[0] * turbulence
    obj.modifiers["RipplesFine"].strength = base[1] * (0.6 + 0.4 * turbulence)


# --- reveal: the scape exists only below the water --------------------------------

REVEAL_GROUP = "RevealLevel"


def _reveal_group():
    g = bpy.data.node_groups.get(REVEAL_GROUP)
    if g is None:
        g = bpy.data.node_groups.new(REVEAL_GROUP, "ShaderNodeTree")
        g.interface.new_socket("Level", in_out="OUTPUT", socket_type="NodeSocketFloat")
        val = g.nodes.new("ShaderNodeValue")
        val.name = "level"
        val.outputs[0].default_value = 100.0
        out = g.nodes.new("NodeGroupOutput")
        g.links.new(val.outputs[0], out.inputs[0])
    return g


def add_reveal(materials):
    """Make each material invisible above a shared height, so that as the tank
    fills, the scape appears beneath the rising surface and nowhere else."""
    g = _reveal_group()
    for mat in materials:
        if mat is None or not mat.use_nodes or mat.get("reveal"):
            continue
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        if out is None or not out.inputs["Surface"].links:
            continue
        src = out.inputs["Surface"].links[0].from_socket
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], sep.inputs[0])
        grp = nt.nodes.new("ShaderNodeGroup")
        grp.node_tree = g
        above = nt.nodes.new("ShaderNodeMath")
        above.operation = "GREATER_THAN"
        nt.links.new(sep.outputs["Z"], above.inputs[0])
        nt.links.new(grp.outputs[0], above.inputs[1])
        clear = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(above.outputs[0], mix.inputs["Fac"])
        nt.links.new(src, mix.inputs[1])
        nt.links.new(clear.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        mat["reveal"] = True


def set_reveal_level(level_mm):
    """Everything wrapped by add_reveal is hidden above this height (give a huge value to show all)."""
    _reveal_group().nodes["level"].outputs[0].default_value = level_mm * MM


def animate_ripples(drift, frame_start, frame_end, speed_mm_per_s=14.0, fps=30):
    """Slide the noise field sideways and through itself so the ripples travel and evolve."""
    secs = (frame_end - frame_start) / fps
    drift.location = (0.0, 0.0, 0.0)
    drift.keyframe_insert("location", frame=frame_start)
    drift.location = (speed_mm_per_s * secs * MM, 0.35 * speed_mm_per_s * secs * MM, 0.6 * speed_mm_per_s * secs * MM)
    drift.keyframe_insert("location", frame=frame_end)
    for fc in drift.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.extrapolation = "LINEAR"
