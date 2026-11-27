"""Scene entry point.

  blender -b --python src/blender/build.py -- --scape empty --frame 1 --quality preview --out <file.exr>

--scape   empty | water | <scape id from config/scapes>
--frame   frame number on the film timeline (drives all animation)
--quality draft | preview | final
--out     output EXR path (scene-linear); develop it with src/comp/develop.py
--save    optional .blend path to save the built scene for inspection
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bl_common as bc  # noqa: E402
import room as room_mod  # noqa: E402
import tank as tank_mod  # noqa: E402


def build(scape="empty", frame=1, quality="preview", out=None, lamp_watts=None):
    scene = bc.reset_scene()
    device = bc.setup_render(scene, quality, out)
    bc.add_camera(scene)
    scene.frame_set(frame)

    world = bpy.data.worlds.new("Room")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.20, 0.20, 0.19, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

    room = room_mod.build_room()
    lamp = room_mod.build_lamp()
    glass = tank_mod.build_glass()
    lamp["light"].data.energy = lamp_watts if lamp_watts is not None else 120.0

    # Everything a scape puts under water goes in this collection, so lights can
    # be linked to the tank's contents alone.
    underwater = bpy.data.collections.new("Underwater")
    scene.collection.children.link(underwater)
    ctx = {"scene": scene, "room": room, "lamp": lamp, "glass": glass, "frame": frame, "water": None,
           "underwater": underwater, "fill": 0.05}
    if scape != "empty":
        water, drift = tank_mod.build_water()
        tank_mod.animate_ripples(drift, 0, 2400)
        ctx["water"] = water
    if scape not in ("empty", "water"):
        mod = importlib.import_module(f"scapes.{scape}")
        mod.build(ctx)
        add_fill(ctx, lamp["light"].data.energy)

    scene.frame_set(frame)
    return scene, device, ctx


def add_fill(ctx, key_watts):
    """Soft frontal fill for the tank's contents only.

    A path tracer finds the lamp's light after it refracts through the surface,
    but not the further share that a real tank traps and bounces between its
    mirror-like walls and the underside of the surface. That lost light is what
    keeps the camera-facing side of a real scape from going black. It is put
    back as one broad, soft source above and in front, as a fraction of the
    lamp's power, lighting only what is under water and shadowed only by it.
    """
    frac = ctx.get("fill", 0.0)
    if frac <= 0:
        return None
    ld = bpy.data.lights.new("TankFill", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = 0.50, 0.40
    ld.energy = key_watts * frac
    ld.color = (0.92, 0.97, 1.0)
    ob = bpy.data.objects.new("TankFill", ld)
    ctx["scene"].collection.objects.link(ob)
    ob.location = (0.15, -0.30, 0.46)
    aim = Vector((0.15, 0.15, 0.10)) - ob.location
    ob.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    ob.visible_camera = False
    ob.visible_glossy = False
    ob.visible_transmission = False
    ob.light_linking.receiver_collection = ctx["underwater"]
    ob.light_linking.blocker_collection = ctx["underwater"]
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scape", default="empty")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--quality", default="preview", choices=list(bc.QUALITY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--save")
    ap.add_argument("--lamp-watts", type=float)
    a = ap.parse_args(bc.script_args())

    t0 = time.time()
    scene, device, _ = build(a.scape, a.frame, a.quality, a.out, a.lamp_watts)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=a.save)
    print(f"[build] scape={a.scape} frame={a.frame} quality={a.quality} device={device} built in {time.time() - t0:.1f}s")
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] {time.time() - t1:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
