"""Close-up turntable stills of a single asset, for judging it on its own.

  blender -b --python src/blender/tests/closeup.py -- --what tetra --out <file.png>

Renders three views (side, three-quarter, front-ish) side by side under a soft
top light on a mid-grey ground, with a tail-beat pose so the bend can be judged.
The tetra gets six views in two rows (side, three-quarter, head-on; the other
side, going away, tail-on) under the tank's kind of light: a hard lamp above
and behind, broad soft light from overhead and a weak fill from the front.
It waits for the machine-wide render lock like every other render.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bl_common as bc  # noqa: E402


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="tetra")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--shrimp-t", default="0,0.3,0.6", help="shrimp: comma-separated t per shrimp (s)")
    ap.add_argument("--walk", type=float, default=1.0, help="shrimp: 0 grazing .. 1 walking")
    ap.add_argument("--snail-t", default="0,0.4,0.8", help="snail: comma-separated t per snail (s)")
    ap.add_argument("--snail-view", default="side", choices=("side", "glass"),
                    help="snail: on a floor seen from the side, or on the front glass seen through it")
    a = ap.parse_args(bc.script_args())

    scene = bc.reset_scene()
    bc.setup_render(scene, "preview")
    scene.cycles.samples = a.samples
    scene.render.resolution_x, scene.render.resolution_y = 1800, 520
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = a.out

    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.10, 0.13, 0.14, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

    span = 0.0
    if a.what == "tetra":
        import fish
        # position is the fish's snout, so each is shifted to sit in its cell
        poses = [((-0.028, 0, 0.018), (1, 0, 0), 0.6, 0.6, 0.0),
                 ((0.012, -0.012, 0.018), (0.75, -0.66, 0.05), 2.2, 0.7, 0.25),
                 ((0.045, -0.016, 0.018), (0.04, -1.0, 0.0), 4.0, 0.4, 0.0),
                 ((-0.062, 0, -0.018), (-1, 0.08, 0), 1.4, 0.8, -0.35),
                 ((-0.012, 0.012, -0.018), (-0.70, 0.70, 0.08), 3.0, 0.6, 0.2),
                 ((0.045, 0.016, -0.018), (0.04, 1.0, 0.06), 5.0, 0.5, 0.0)]
        for i, (pos, fwd, ph, amp, bend) in enumerate(poses):
            t = fish.Tetra(f"Tetra{i}")
            t.pose(pos, fwd, phase=ph, amp=amp, bend=bend)
        me = t.obj.data
        xs = [v.co.x for v in me.vertices]
        zs = [v.co.z for v in me.vertices]
        print(f"[closeup] tetra at size 1.0: {1000 * (max(xs) - min(xs)):.1f} mm long, "
              f"{1000 * (max(zs) - min(zs)):.1f} mm tall with fins, {len(me.vertices)} vertices, {len(me.polygons)} faces")
        span = 0.135
        scene.render.resolution_y = 1000
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.20, 0.23, 0.24, 1)
    elif a.what == "shrimp":
        import shrimp
        ts = [float(v) for v in a.shrimp_t.split(",")]
        for i, (pos, yaw) in enumerate([((-0.035, 0, 0), 0.0), ((0.0, 0, 0), -0.8), ((0.032, 0, 0), -1.45)]):
            s = shrimp.Shrimp(f"Shrimp{i}")
            s.pose(pos, yaw=yaw, t=ts[i % len(ts)], walk=a.walk)
        span = 0.105
    elif a.what == "snail":
        import snail
        ts = [float(v) for v in a.snail_t.split(",")]
        if a.snail_view == "glass":
            # on the far face of a front pane (y = 0), seen through it from -Y:
            # the soles should show as pale ovals ringed by the shell edge
            pane = bc.box_object("Pane", (-0.05, -0.00405, -0.03), (0.05, -0.00005, 0.03))
            pane.data.materials.append(bc.principled_material("Glass", (1, 1, 1), roughness=0.0, **{
                "Transmission Weight": 1.0, "IOR": 1.5}))
            places = [((-0.025, 0, 0), 0.0), ((0.0, 0, 0.0), 0.7), ((0.025, 0, 0.0), -2.6)]
            up = (0, 1, 0)
        else:
            # lowered so the 10 mm-high animals sit in the middle of the frame
            places = [((-0.025, 0, -0.005), 0.0), ((0.0, 0, -0.005), -0.9), ((0.025, 0, -0.005), -1.9)]
            up = (0, 0, 1)
        for i, (pos, yaw) in enumerate(places):
            s = snail.Snail(f"Snail{i}")
            s.pose(pos, yaw=yaw, t=ts[i % len(ts)], up=up)
        span = 0.08
    else:
        sys.exit(f"unknown --what {a.what}")

    cam_data = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", cam_data)
    scene.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = span
    cam.location = (0.0, -0.5, 0.004)
    look_at(cam, (0, 0, 0.0))
    scene.camera = cam

    lights = (((0.0, -0.10, 0.25), 1.6, 0.25), ((-0.2, -0.3, 0.05), 0.4, 0.3))
    if a.what == "tetra":
        lights = (((0.0, 0.13, 0.26), 2.2, 0.03), ((0.0, 0.0, 0.30), 2.4, 0.60), ((0.0, -0.40, 0.10), 0.35, 0.40))
    for loc, energy, size in lights:
        ld = bpy.data.lights.new("L", "AREA")
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new("L", ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        look_at(lo, (0, 0, 0))

    import build
    build.render_lock()
    bpy.ops.render.render(write_still=True)
    print(f"[closeup] wrote {a.out}")


if __name__ == "__main__":
    main()
