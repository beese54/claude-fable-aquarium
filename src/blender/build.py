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

    # A scape module may tune the shared scene through two optional dicts:
    #   LIGHT = dict(watts=50, colour=(r, g, b), fill=0.015, bounce=0.05)
    #   WATER = dict(tint=(r, g, b), density=1.5, scatter=0.0, level_mm=268, ...)
    mod = None
    if scape not in ("empty", "water"):
        mod = importlib.import_module(f"scapes.{scape}")
    light_cfg = dict(watts=50.0, colour=(0.90, 0.95, 1.0), fill=0.015, bounce=0.05)
    light_cfg.update(getattr(mod, "LIGHT", {}))
    water_cfg = dict(getattr(mod, "WATER", {}))

    room = room_mod.build_room()
    lamp = room_mod.build_lamp()
    glass = tank_mod.build_glass()
    watts = lamp_watts if lamp_watts is not None else light_cfg["watts"]
    for led in lamp["lights"]:
        led.data.energy = watts / len(lamp["lights"])
        led.data.color = light_cfg["colour"]

    # Everything a scape puts under water goes in this collection, so lights can
    # be linked to the tank's contents alone.
    underwater = bpy.data.collections.new("Underwater")
    scene.collection.children.link(underwater)
    ctx = {"scene": scene, "room": room, "lamp": lamp, "glass": glass, "frame": frame, "water": None,
           "underwater": underwater, "fill": light_cfg["fill"], "bounce": light_cfg["bounce"],
           "water_level_mm": water_cfg.get("level_mm", 268.0)}
    if scape != "empty":
        level = water_cfg.pop("level_mm", 268.0)
        ripple = {k: water_cfg.pop(k) for k in ("ripple_mm", "ripple_scale", "fine_mm", "fine_scale") if k in water_cfg}
        material = tank_mod.water_material(**water_cfg) if water_cfg else None
        water, drift = tank_mod.build_water(level_mm=level, material=material, **ripple)
        tank_mod.animate_ripples(drift, 0, 2400)
        ctx["water"] = water
    if mod is not None:
        mod.build(ctx)
        # Behind a filled tank the cable only shows as confusing mirror images in
        # the side glass and the surface; it is left out once there is water.
        lamp["cable"].hide_render = True
        add_fill(ctx, watts)
    scene.frame_set(frame)
    return scene, device, ctx


def add_fill(ctx, key_watts):
    """Soft light for the tank's contents only, standing in for trapped light.

    A path tracer finds the lamp's light after it refracts through the surface,
    but not the further share that a real tank keeps: light scattered up off
    the bed is thrown back down by the mirror-like underside of the surface,
    and the glass walls return most of what reaches them. That is what stops
    the shaded sides of a real scape going black. It is put back as two broad
    sources, each a fraction of the lamp's power: `bounce`, lying just under
    the surface and shining down, and a weaker `fill` from the front. Both
    light only what is under water and are shadowed only by it.
    """
    made = []
    specs = (("TankBounce", ctx.get("bounce", 0.0), (0.15, 0.15, 0.262), (0.15, 0.15, 0.0), 0.27, 0.27),
             ("TankFill", ctx.get("fill", 0.0), (0.15, -0.30, 0.40), (0.15, 0.15, 0.10), 0.50, 0.40))
    for name, frac, loc, aim_at, sx, sy in specs:
        if frac <= 0:
            continue
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = sx, sy
        ld.energy = key_watts * frac
        ld.color = (0.92, 0.97, 1.0)
        ob = bpy.data.objects.new(name, ld)
        ctx["scene"].collection.objects.link(ob)
        ob.location = loc
        aim = Vector(aim_at) - ob.location
        ob.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        ob.visible_camera = False
        ob.visible_glossy = False
        ob.visible_transmission = False
        ob.light_linking.receiver_collection = ctx["underwater"]
        ob.light_linking.blocker_collection = ctx["underwater"]
        made.append(ob)
    return made

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scape", default="empty")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--frames", help="render a sequence: first-last[:step], inclusive, into --outdir")
    ap.add_argument("--outdir", help="folder for a sequence; files are <frame:05d>.exr and existing ones are skipped")
    ap.add_argument("--quality", default="preview", choices=list(bc.QUALITY))
    ap.add_argument("--out")
    ap.add_argument("--save")
    ap.add_argument("--lamp-watts", type=float)
    a = ap.parse_args(bc.script_args())

    t0 = time.time()
    if a.frames:
        span, _, step = a.frames.partition(":")
        first, last = (int(v) for v in span.split("-"))
        frames = list(range(first, last + 1, int(step or 1)))
        outdir = Path(a.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [f for f in frames if not (outdir / f"{f:05d}.exr").is_file()]
        print(f"[seq] {len(frames)} frames requested, {len(frames) - len(todo)} already done")
        if not todo:
            return
        scene, device, ctx = build(a.scape, todo[0], a.quality, None, a.lamp_watts)
        mod = sys.modules.get(f"scapes.{a.scape}")
        print(f"[build] scape={a.scape} quality={a.quality} device={device} built in {time.time() - t0:.1f}s")
        for f in todo:
            t1 = time.time()
            scene.frame_set(f)
            ctx["frame"] = f
            if mod is not None and hasattr(mod, "animate"):
                mod.animate(ctx, f)
            # write to a temporary name first so a killed render never leaves a half file that looks finished
            tmp = outdir / f"_{f:05d}.exr"
            scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            tmp.replace(outdir / f"{f:05d}.exr")
            print(f"[frame] {f} {time.time() - t1:.1f}s", flush=True)
        print(f"[seq] done in {time.time() - t0:.0f}s")
        return

    if not a.out:
        sys.exit("--out is required for a single frame")
    scene, device, _ = build(a.scape, a.frame, a.quality, a.out, a.lamp_watts)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=a.save)
    print(f"[build] scape={a.scape} frame={a.frame} quality={a.quality} device={device} built in {time.time() - t0:.1f}s")
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] {time.time() - t1:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()