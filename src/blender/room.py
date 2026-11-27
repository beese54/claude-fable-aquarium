"""The filmed room, rebuilt: floor, wall and tray carrying the plate by camera
projection, plus the clip-on lamp and its cable as real geometry. Because the
plate is projected in the shader from the solved camera's position, the room
looks exactly as filmed from that camera and shows correct parallax from any
nearby one. The plate is emissive, so it is also what the tank's glass and
water reflect and refract.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

from bl_common import MM, box_object, cfg, emission_material, new_material, new_mesh_object
import paths


def projection_uv(nt, cam):
    """Nodes that turn world position into the plate's UV for the solved camera."""
    N = nt.nodes
    L = nt.links
    geo = N.new("ShaderNodeNewGeometry")
    R = _rodrigues(cam["opencv"]["rvec"])
    t = [v * MM for v in cam["opencv"]["tvec_mm"]]
    W, H = cam["image_size"]
    f = cam["focal_px"]
    cx, cy = cam["principal_point"]

    comps = []
    for i in range(3):
        dot = N.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = tuple(R[i])
        L.new(geo.outputs["Position"], dot.inputs[0])
        add = N.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = t[i]
        L.new(dot.outputs["Value"], add.inputs[0])
        comps.append(add.outputs[0])

    def scaled(numer, k, offset, flip):
        div = N.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        L.new(numer, div.inputs[0])
        L.new(comps[2], div.inputs[1])
        mad = N.new("ShaderNodeMath")
        mad.operation = "MULTIPLY_ADD"
        mad.inputs[1].default_value = -k if flip else k
        mad.inputs[2].default_value = (1.0 - offset) if flip else offset
        L.new(div.outputs[0], mad.inputs[0])
        return mad.outputs[0]

    u = scaled(comps[0], f / W, cx / W, flip=False)
    v = scaled(comps[1], f / H, cy / H, flip=True)     # image rows run downward
    comb = N.new("ShaderNodeCombineXYZ")
    L.new(u, comb.inputs[0])
    L.new(v, comb.inputs[1])
    return comb.outputs[0]


def _rodrigues(rvec):
    r = np.array(rvec, dtype=np.float64)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def plate_material(name="RoomPlate", image_name="room_plate_linear.exr", strength=1.0):
    mat, nt, out = new_material(name)
    cam = cfg("camera.json")
    img = bpy.data.images.load(str(paths.PLATE / image_name), check_existing=True)
    img.colorspace_settings.name = "Linear Rec.709"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    nt.links.new(projection_uv(nt, cam), tex.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def build_room(collection=None):
    room = cfg("room.json")
    col = collection or bpy.context.scene.collection
    plate = plate_material()
    fz = room["floor_z"] * MM

    # Wall line on the floor, from the two measured junction points.
    wl = Vector(room["wall"]["floor_junction_left"]) * MM
    wr = Vector(room["wall"]["floor_junction_right"]) * MM
    d = (wr - wl).normalized()
    a = wl - d * 4.0
    b = wr + d * 4.0

    # Floor: from behind the camera to the wall. Wall: from the floor up.
    floor = new_mesh_object("Floor", [(a.x, -2.0, fz), (b.x, -2.0, fz), (b.x, b.y, fz), (a.x, a.y, fz)], [(0, 1, 2, 3)], col)
    wall = new_mesh_object("Wall", [(a.x, a.y, fz), (b.x, b.y, fz), (b.x, b.y, fz + 3.0), (a.x, a.y, fz + 3.0)], [(0, 1, 2, 3)], col)
    for o in (floor, wall):
        o.data.materials.append(plate)

    # Tray: a card box lid. Its front face is in the plate; the rest is plain card.
    t = room["tray"]
    wall_t = 3.0 * MM
    x0, x1 = t["x_min"] * MM, t["x_max"] * MM
    y0, y1 = t["front_y"] * MM, 0.312
    z1 = fz + t["height"] * MM
    card = emission_material("TrayCard", (0.82, 0.82, 0.84), 1.0)
    parts = {
        "TrayFront": ((x0, y0, fz), (x1, y0 + wall_t, z1)),
        "TrayBack": ((x0, y1 - wall_t, fz), (x1, y1, z1)),
        "TrayLeft": ((x0, y0 + wall_t, fz), (x0 + wall_t, y1 - wall_t, z1)),
        "TrayRight": ((x1 - wall_t, y0 + wall_t, fz), (x1, y1 - wall_t, z1)),
        "TrayBase": ((x0, y0, fz), (x1, y1, fz + 1.0 * MM)),
    }
    tray = []
    for name, (lo, hi) in parts.items():
        o = box_object(name, lo, hi, col)
        o.data.materials.append(plate if name == "TrayFront" else card)
        tray.append(o)
    # The projected plate is only right for the face that looked at the camera.
    front = tray[0]
    front.data.materials.append(card)
    for p in front.data.polygons:
        p.material_index = 0 if p.normal.y < -0.5 else 1

    return {"floor": floor, "wall": wall, "tray": tray, "room": room}


LAMP_TILT_DEG = 20.0     # the LED panel throws its beam forward over the tank


def build_lamp(collection=None, led_strength=40.0):
    """Clip-on LED lamp: head, stem, clamp, cable. Returns the area light too."""
    room = cfg("room.json")
    cam_json = cfg("camera.json")
    col = collection or bpy.context.scene.collection
    lamp = room["lamp"]
    h = lamp["head"]
    # The lamp's own parts sit right beside its light source, where an idealised
    # area light would burn them out. They are given the tones they have in the
    # footage instead (emission), and still block light like any solid.
    # They carry the filmed frame itself by projection, so from the filmed
    # position they are the real lamp, and they keep its look as the camera moves.
    black = plate_material("LampPlate", "lit_plate_linear.exr")
    grey = black

    # Head as a box in its measured position (it sits a few degrees off square).
    rl, rr, fl, fr = (Vector(h[k]) * MM for k in ("rear_left", "rear_right", "front_left", "front_right"))
    zu = h["underside_z"] * MM
    zt = zu + h["thickness"] * MM
    v = [(fl.x, fl.y, zu), (fr.x, fr.y, zu), (rr.x, rr.y, zu), (rl.x, rl.y, zu),
         (fl.x, fl.y, zt), (fr.x, fr.y, zt), (rr.x, rr.y, zt), (rl.x, rl.y, zt)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    head = new_mesh_object("LampHead", v, f, col)
    head.data.materials.append(black)
    bev = head.modifiers.new("Bevel", "BEVEL")
    bev.width = 2.0 * MM
    bev.segments = 3

    # LED panel: what the camera sees glowing on the underside. Lighting itself
    # comes from the area light below, so this is hidden from diffuse rays.
    inset = 0.10
    c = (fl + fr + rr + rl) / 4

    def toward(p):
        return (p.x + (c.x - p.x) * inset, p.y + (c.y - p.y) * inset, zu - 0.3 * MM)
    led = new_mesh_object("LampLED", [toward(fl), toward(fr), toward(rr), toward(rl)], [(0, 3, 2, 1)], col)
    led.data.materials.append(emission_material("LED", (0.86, 0.92, 1.0), led_strength))
    led.visible_diffuse = False
    led.visible_volume_scatter = False
    led.visible_shadow = False
    # Seen directly, the head shows the filmed LEDs (projected); this bright panel
    # is only for what the water and glass reflect and refract of the lamp.
    led.visible_camera = False

    s = lamp["stem"]
    sx, sy = s["x"] * MM, s["y"] * MM
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=s["width"] * MM / 2, depth=(s["z_top"] - 300.0) * MM,
                                        location=(sx, sy, (s["z_top"] + 300.0) / 2 * MM))
    stem = bpy.context.active_object
    stem.name = "LampStem"
    stem.data.materials.append(black)
    _relink(stem, col)

    cl = lamp["clamp"]
    # The clamp's jaw reaches a little further down inside the rear glass than the block measured outside.
    clamp = box_object("LampClamp", (cl["x_min"] * MM, 0.288, (cl["z_min"] - 9.0) * MM), (cl["x_max"] * MM, 0.316, cl["z_max"] * MM), col)
    clamp.data.materials.append(grey)
    b2 = clamp.modifiers.new("Bevel", "BEVEL")
    b2.width = 1.5 * MM
    b2.segments = 2

    cable = _cable(cam_json, col, emission_material("CableBlack", (0.020, 0.020, 0.026), 1.0))

    # The light that actually illuminates the tank. The real fixture is a bar of
    # separate LED chips, and that matters: each chip is nearly a point, so each
    # throws its own crisp caustic net through the ripples and its own hard
    # shadow. One broad panel would smear all of that into mush.
    lights = []
    rows, cols = 2, 5
    for r in range(rows):
        for q in range(cols):
            u = (q + 0.5) / cols
            v = (r + 0.5) / rows
            front = fl.lerp(fr, 0.12 + 0.76 * u)
            rear = rl.lerp(rr, 0.12 + 0.76 * u)
            p = front.lerp(rear, 0.22 + 0.56 * v)
            ld = bpy.data.lights.new(f"LED_{r}{q}", "SPOT")
            ld.spot_size = math.radians(150)
            ld.spot_blend = 0.6
            ld.shadow_soft_size = 0.0022
            ld.color = (0.90, 0.95, 1.0)
            ld.cycles.is_caustics_light = True
            ob = bpy.data.objects.new(f"LED_{r}{q}", ld)
            col.objects.link(ob)
            ob.location = (p.x, p.y, zu - 1.0 * MM)
            ob.rotation_euler = (math.radians(-LAMP_TILT_DEG), 0.0, 0.0)   # spots shine along -Z; the tilt throws the beam forward
            ob.visible_camera = False
            lights.append(ob)
    return {"head": head, "led": led, "stem": stem, "clamp": clamp, "cable": cable, "lights": lights}


def _relink(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


# Cable as seen in the lit plate, from the clamp down and out to the left.
CABLE_PX = [(538, 655), (520, 690), (470, 728), (400, 752), (330, 772), (270, 800),
            (225, 850), (185, 915), (140, 975), (95, 1005), (60, 1022)]
CABLE_Y_MM = 312.0


def _cable(cam_json, col, mat):
    R = _rodrigues(cam_json["opencv"]["rvec"])
    t = np.array(cam_json["opencv"]["tvec_mm"])
    C = -R.T @ t
    f = cam_json["focal_px"]
    cx, cy = cam_json["principal_point"]
    pts = []
    for (px, py) in CABLE_PX:
        d = R.T @ np.array([(px - cx) / f, (py - cy) / f, 1.0])
        s = (CABLE_Y_MM - C[1]) / d[1]
        pts.append((C + s * d) * MM)
    cu = bpy.data.curves.new("LampCable", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    for p, co in zip(sp.points, pts):
        p.co = (co[0], co[1], co[2], 1.0)
    sp.use_endpoint_u = True
    sp.order_u = 4
    cu.bevel_depth = 1.8 * MM
    cu.bevel_resolution = 4
    obj = bpy.data.objects.new("LampCable", cu)
    col.objects.link(obj)
    obj.data.materials.append(mat)
    return obj

def build_glass_dirt(collection=None):
    """The dry front glass's limescale and smears, as a sheet of added light just
    in front of the pane. Only the camera sees it."""
    col = collection or bpy.context.scene.collection
    mat, nt, out = new_material("GlassDirt")
    img = bpy.data.images.load(str(paths.PLATE / "glass_dirt_linear.exr"), check_existing=True)
    img.colorspace_settings.name = "Linear Rec.709"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    nt.links.new(projection_uv(nt, cfg("camera.json")), tex.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(clear.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    y = -0.3 * MM
    obj = new_mesh_object("GlassDirt", [(0.004, y, 0.258), (0.296, y, 0.258), (0.296, y, 0.302), (0.004, y, 0.302)], [(0, 1, 2, 3)], col)
    obj.data.materials.append(mat)
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    obj.visible_shadow = False
    return obj
