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
import timeline  # noqa: E402


# Balance between the lamp and the soft stand-in for trapped light (add_fill).
# Measured on the round 3 stills: with the scapes' LIGHT values taken at face
# value, about half of the light on an upward-facing surface came from the soft
# sources, so nothing had a hot top, a dark underside or a hard shadow, and a
# reviewer called every still flat. The lamp now carries the picture: its
# output is raised and the soft sources lowered, for every scape alike, so the
# scapes' own LIGHT dicts keep their meaning relative to each other.
FILL_GLOSSY = False      # whether the soft sources may show as highlights on wet surfaces
KEY_GAIN = 2.3
AMBIENT_GAIN = 0.6


def build(scape="empty", frame=1, quality="preview", out=None, lamp_watts=None, tweak=None):
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
    # look-development overrides from the command line (--fill, --bounce, --ripple-mm, --fine-mm)
    for k, v in (tweak or {}).items():
        if v is not None:
            (light_cfg if k in ("fill", "bounce", "fill_glossy") else water_cfg)[k] = v
    if scape == "empty":
        water_cfg.pop("wave_gain", None)

    room = room_mod.build_room()
    lamp = room_mod.build_lamp()
    glass = tank_mod.build_glass()
    room_mod.build_glass_dirt()
    watts = lamp_watts if lamp_watts is not None else light_cfg["watts"]
    for led in lamp["lights"]:
        led.data.energy = watts * KEY_GAIN / len(lamp["lights"])
        led.data.color = light_cfg["colour"]

    # Everything a scape puts under water goes in this collection, so lights can
    # be linked to the tank's contents alone.
    underwater = bpy.data.collections.new("Underwater")
    scene.collection.children.link(underwater)
    ctx = {"scene": scene, "room": room, "lamp": lamp, "glass": glass, "frame": frame, "water": None,
           "underwater": underwater, "fill": light_cfg["fill"], "bounce": light_cfg["bounce"],
           "fill_glossy": light_cfg.get("fill_glossy", FILL_GLOSSY),
           "water_level_mm": water_cfg.get("level_mm", 268.0)}
    if scape != "empty":
        level = water_cfg.pop("level_mm", 268.0)
        ripple = {k: water_cfg.pop(k) for k in ("ripple_mm", "ripple_scale", "fine_mm", "fine_scale") if k in water_cfg}
        ctx["wave_gain"] = water_cfg.pop("wave_gain", 1.0)        # height of the wave trains (tank.wave_surface)
        material = tank_mod.water_material(**water_cfg) if water_cfg else None
        water, drift = tank_mod.build_water(level_mm=level, material=material, **ripple)
        tank_mod.animate_ripples(drift, 0, timeline._TL["total_frames"] + 90)
        ctx["water"] = water
    protected = set(bpy.data.materials)        # room, lamp, glass, water: never hidden by the reveal
    scene_objects = set(bpy.data.objects)
    ctx["scape"] = scape
    ctx["module"] = mod
    if mod is not None:
        mod.build(ctx)
        # Every object the scape made must take the lamp's light through the water
        # surface, including the source objects that scatters instance (which are
        # in no scene collection and so are easy to miss). An object left out is
        # lit only by the soft fill: flat, and without shadows.
        for ob in bpy.data.objects:
            if ob not in scene_objects and ob.type in {"MESH", "CURVE"}:
                ob.cycles.is_caustics_receiver = True
        if timeline.is_filling(scape):
            tank_mod.add_reveal([m for m in bpy.data.materials if m not in protected])
        # Behind a filled tank the cable only shows as confusing mirror images in
        # the side glass and the surface; it is left out once there is water.
        lamp["cable"].hide_render = True
        add_fill(ctx, watts)
        ctx["base_objects"] = scene_objects
    apply_frame(ctx, frame)
    return scene, device, ctx


def render_lock():
    """Wait for any other Blender render on this machine to finish, then hold the
    lock until this process exits. Two path-traced scenes at once do not fit in
    16 GB; with the lock, renders started side by side simply queue."""
    if sys.platform != "win32":
        return
    import ctypes
    k = ctypes.windll.kernel32
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    handle = k.CreateMutexW(None, False, "AquariumDreamRender")
    t0 = time.time()
    k.WaitForSingleObject(handle, 0xFFFFFFFF)
    if time.time() - t0 > 1.0:
        print(f"[lock] waited {time.time() - t0:.0f}s for another render to finish")
    globals()["_render_lock"] = handle


def check_swim(ctx, reach_mm=11.0, every=5):
    """Does the shoal have room in this scape? For the frames the scape is on
    screen, short rays are cast from each fish in six directions; a fish counts
    as touching when scape geometry (stone, wood, plants, bed) lies within
    `reach_mm` of its middle. Prints one [swim] line; the fix for a high number
    is the scape's SWIM literal (or moving what is in the way)."""
    school = ctx.get("school")
    scape = ctx.get("scape")
    vis = timeline.visible().get(scape)
    if school is None or vis is None:
        return
    scene = ctx["scene"]
    ignore = {ob.name for ob in ctx.get("base_objects", ())} | {ob.name for ob in school.objects}
    for ob in school.objects:
        ob.hide_viewport = True
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    dirs = [Vector(d) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    reach = reach_mm * bc.MM
    touched, worst, total = {}, {}, 0
    for f in range(vis[0], vis[1] + 1, every):
        k = min(max(f - school.frame_start, 0), school.frames - 1)
        for i in range(len(school.fish)):
            p = Vector(tuple(float(c) * bc.MM for c in school.data["pos"][k, i]))
            total += 1
            hit_name = None
            for d in dirs:
                hit, _, _, _, ob, _ = scene.ray_cast(dg, p, d, distance=reach)
                if hit and ob is not None and ob.name not in ignore:
                    hit_name = ob.name
                    break
            if hit_name:
                touched[hit_name] = touched.get(hit_name, 0) + 1
                worst.setdefault(hit_name, (f, i, tuple(round(c / bc.MM) for c in p)))
    for ob in school.objects:
        ob.hide_viewport = False
    n = sum(touched.values())
    detail = "; ".join(f"{name} x{c} (e.g. frame {worst[name][0]} fish {worst[name][1]} at {worst[name][2]} mm)"
                       for name, c in sorted(touched.items(), key=lambda kv: -kv[1])[:6])
    print(f"[swim] {scape}: {n}/{total} fish samples ({100.0 * n / max(total, 1):.1f}%) within {reach_mm:.0f} mm of the scape"
          + (f": {detail}" if n else ""))


def apply_frame(ctx, frame):
    """Put the scene in its state for this frame: clock, water level, and whatever the scape animates by code."""
    ctx["scene"].frame_set(frame)
    ctx["frame"] = frame
    bc.move_camera(ctx["scene"].camera, frame)
    scape = ctx.get("scape")
    if ctx.get("water") is not None and timeline.is_filling(scape):
        level = timeline.water_level_mm(frame, scape)
        tank_mod.set_water_level(ctx["water"], level, timeline.turbulence(frame, scape))
        tank_mod.set_reveal_level(level - 0.6 if level < timeline.FULL_MM else 1.0e5)
        ctx["water_level_mm"] = level
    if ctx.get("water") is not None:
        turb = timeline.turbulence(frame, scape) if timeline.is_filling(scape) else 1.0
        tank_mod.wave_surface(ctx["water"], frame, ctx["water_level_mm"], ctx.get("wave_gain", 1.0) * turb)
    mod = ctx.get("module")
    if mod is not None and hasattr(mod, "animate"):
        mod.animate(ctx, frame)


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
        ld.energy = key_watts * frac * AMBIENT_GAIN
        ld.color = (0.92, 0.97, 1.0)
        ob = bpy.data.objects.new(name, ld)
        ctx["scene"].collection.objects.link(ob)
        ob.location = loc
        aim = Vector(aim_at) - ob.location
        ob.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        ob.visible_camera = False
        ob.visible_glossy = bool(ctx.get("fill_glossy", FILL_GLOSSY))
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
    ap.add_argument("--no-render", action="store_true", help="build the scene and run the checks only")
    ap.add_argument("--fill", type=float, help="look development: override the scape's fill fraction")
    ap.add_argument("--bounce", type=float, help="look development: override the scape's bounce fraction")
    ap.add_argument("--ripple-mm", type=float, help="look development: override the broad ripple height")
    ap.add_argument("--fine-mm", type=float, help="look development: override the fine ripple height")
    ap.add_argument("--wave-gain", type=float, help="look development: override the height of the wave trains")
    ap.add_argument("--fill-glossy", action="store_true", help="look development: let the soft sources make highlights")
    a = ap.parse_args(bc.script_args())

    render_lock()
    t0 = time.time()
    tweak = dict(fill=a.fill, bounce=a.bounce, ripple_mm=a.ripple_mm, fine_mm=a.fine_mm,
                 fill_glossy=True if a.fill_glossy else None, wave_gain=a.wave_gain)
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
        scene, device, ctx = build(a.scape, todo[0], a.quality, None, a.lamp_watts, tweak)
        print(f"[build] scape={a.scape} quality={a.quality} device={device} built in {time.time() - t0:.1f}s")
        for f in todo:
            t1 = time.time()
            apply_frame(ctx, f)
            # write to a temporary name first so a killed render never leaves a half file that looks finished
            tmp = outdir / f"_{f:05d}.exr"
            scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            tmp.replace(outdir / f"{f:05d}.exr")
            print(f"[frame] {f} {time.time() - t1:.1f}s", flush=True)
        print(f"[seq] done in {time.time() - t0:.0f}s")
        return

    if not a.out and not a.no_render:
        sys.exit("--out is required for a single frame")
    scene, device, ctx = build(a.scape, a.frame, a.quality, a.out, a.lamp_watts, tweak)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=a.save)
    print(f"[build] scape={a.scape} frame={a.frame} quality={a.quality} device={device} built in {time.time() - t0:.1f}s")
    check_swim(ctx)
    apply_frame(ctx, a.frame)
    if a.no_render:
        return
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] {time.time() - t1:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()