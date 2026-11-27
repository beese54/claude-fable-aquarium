"""Scatter instances over a surface with Geometry Nodes.

One helper covers soil granules, carpet plants, pebbles, leaf litter and moss:
points are distributed on the target's faces (optionally weighted by a point
attribute painted from Python), and a random object from a collection is
instanced on each with random spin, tilt and size.
"""
import math

import bpy


def set_point_attribute(obj, name, values):
    """Write a per-vertex float attribute (used to weight scatter density)."""
    me = obj.data
    attr = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", list(values))
    me.update()


def collection_of(name, objects, hide=True):
    """Put source objects in their own collection, kept out of the render."""
    col = bpy.data.collections.new(name)
    for o in objects:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    if hide:
        # Not linked to the scene: instanced by Geometry Nodes only.
        pass
    return col


def scatter(target, source_collection, name="Scatter", density=None, min_distance=None, density_max=None,
            weight_attr=None, seed=0, scale=(0.8, 1.2), align="NORMAL", spin=True, tilt=0.0,
            keep_surface=True, sink=0.0, sway=None):
    """Add a Geometry Nodes modifier to `target` that scatters `source_collection`.

    density        points per square metre (random mode)
    min_distance   metres; switches to Poisson-disk mode (needs density_max)
    weight_attr    name of a 0..1 point attribute that scales density
    align          "NORMAL" (sit on the surface) or "UP" (grow straight up)
    tilt           radians of random lean
    sink           metres to push each instance into the surface along its up axis
    sway           (amplitude in radians, spatial frequency per metre, speed) to
                   make instances lean with a drifting current; None for none
    """
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = tree.nodes, tree.links
    gin = N.new("NodeGroupInput")
    gout = N.new("NodeGroupOutput")

    # Scatter on the object's own mesh only. Geometry nodes also act on the
    # geometry inside instances, so without this a second scatter stacked on the
    # same object would sprout from every instance the first one made.
    own = N.new("GeometryNodeSeparateComponents")
    L.new(gin.outputs[0], own.inputs[0])
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.inputs["Seed"].default_value = seed
    L.new(own.outputs["Mesh"], dist.inputs["Mesh"])
    if min_distance:
        dist.distribute_method = "POISSON"
        dist.inputs["Distance Min"].default_value = min_distance
        dist.inputs["Density Max"].default_value = density_max or 1.0 / (min_distance * min_distance)
        factor_socket = dist.inputs["Density Factor"]
    else:
        dist.distribute_method = "RANDOM"
        dist.inputs["Density"].default_value = density
        factor_socket = None

    if weight_attr:
        attr = N.new("GeometryNodeInputNamedAttribute")
        attr.data_type = "FLOAT"
        attr.inputs["Name"].default_value = weight_attr
        if factor_socket is not None:
            L.new(attr.outputs["Attribute"], factor_socket)
        else:
            mul = N.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = density
            L.new(attr.outputs["Attribute"], mul.inputs[0])
            L.new(mul.outputs[0], dist.inputs["Density"])

    info = N.new("GeometryNodeCollectionInfo")
    info.inputs["Collection"].default_value = source_collection
    info.inputs["Separate Children"].default_value = True
    info.inputs["Reset Children"].default_value = True
    info.transform_space = "ORIGINAL"

    inst = N.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    L.new(dist.outputs["Points"], inst.inputs["Points"])
    L.new(info.outputs["Instances"], inst.inputs["Instance"])

    # Random spin about the instance's own up axis, plus a random lean.
    rnd = N.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT_VECTOR"
    rnd.inputs["Min"].default_value = (-tilt, -tilt, 0.0)
    rnd.inputs["Max"].default_value = (tilt, tilt, 2 * math.pi if spin else 0.0)
    rnd.inputs["Seed"].default_value = seed + 11
    to_rot = N.new("FunctionNodeEulerToRotation")
    euler = rnd.outputs["Value"]
    if sway:
        # Plants lean with a slow current: a noise field that drifts through
        # time tilts neighbours together, so the bed moves in waves, not jitter.
        amp, per_metre, speed = sway
        pos = N.new("GeometryNodeInputPosition")
        clock = N.new("GeometryNodeInputSceneTime")
        w = N.new("ShaderNodeMath")
        w.operation = "MULTIPLY"
        w.inputs[1].default_value = speed
        L.new(clock.outputs["Seconds"], w.inputs[0])
        noise = N.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "4D"
        noise.inputs["Scale"].default_value = per_metre
        noise.inputs["Detail"].default_value = 1.0
        L.new(pos.outputs["Position"], noise.inputs["Vector"])
        L.new(w.outputs[0], noise.inputs["W"])
        centre = N.new("ShaderNodeVectorMath")
        centre.operation = "SUBTRACT"
        centre.inputs[1].default_value = (0.5, 0.5, 0.5)
        L.new(noise.outputs["Color"], centre.inputs[0])
        scale_v = N.new("ShaderNodeVectorMath")
        scale_v.operation = "MULTIPLY"
        scale_v.inputs[1].default_value = (2 * amp, 2 * amp, 0.0)
        L.new(centre.outputs["Vector"], scale_v.inputs[0])
        add = N.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        L.new(rnd.outputs["Value"], add.inputs[0])
        L.new(scale_v.outputs["Vector"], add.inputs[1])
        euler = add.outputs["Vector"]
    L.new(euler, to_rot.inputs["Euler"])
    if align == "NORMAL":
        rot = N.new("FunctionNodeRotateRotation")
        rot.rotation_space = "LOCAL"
        L.new(dist.outputs["Rotation"], rot.inputs["Rotation"])
        L.new(to_rot.outputs["Rotation"], rot.inputs["Rotate By"])
        L.new(rot.outputs["Rotation"], inst.inputs["Rotation"])
    else:
        L.new(to_rot.outputs["Rotation"], inst.inputs["Rotation"])

    srnd = N.new("FunctionNodeRandomValue")
    srnd.data_type = "FLOAT"
    srnd.inputs["Min"].default_value = scale[0]
    srnd.inputs["Max"].default_value = scale[1]
    srnd.inputs["Seed"].default_value = seed + 23
    L.new(srnd.outputs["Value"], inst.inputs["Scale"])

    out_geo = inst.outputs["Instances"]
    if sink:
        tr = N.new("GeometryNodeTranslateInstances")
        tr.inputs["Translation"].default_value = (0.0, 0.0, -sink)
        tr.inputs["Local Space"].default_value = True
        L.new(out_geo, tr.inputs["Instances"])
        out_geo = tr.outputs["Instances"]

    if keep_surface:
        join = N.new("GeometryNodeJoinGeometry")
        L.new(gin.outputs[0], join.inputs[0])
        L.new(out_geo, join.inputs[0])
        out_geo = join.outputs[0]
    L.new(out_geo, gout.inputs[0])

    mod = target.modifiers.new(name, "NODES")
    mod.node_group = tree
    return mod
