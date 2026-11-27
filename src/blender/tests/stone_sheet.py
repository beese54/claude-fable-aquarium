"""Every candidate stone in four orientations, graded, under a hard top light.
Used to pick which scan stands which way up for each role in a layout.

  blender -b --python src/blender/tests/stone_sheet.py -- --out <file.png> [--ids a,b,c] [--style seiryu|dragon|raw]

Columns are stones; rows are the pre-rotations listed in ROTS.
"""
import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import assets  # noqa: E402
import bl_common as bc  # noqa: E402

ROTS = [(0, 0, 0), (90, 0, 0), (0, 90, 0), (180, 0, 0)]
DEFAULT = "moon_rock_01,moon_rock_02,moon_rock_03,moon_rock_04,moon_rock_05,moon_rock_06,moon_rock_07"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=DEFAULT)
    ap.add_argument("--style", default="seiryu")
    a = ap.parse_args(bc.script_args())
    ids = a.ids.split(",")

    scene = bc.reset_scene()
    bc.setup_render(scene, "preview")
    scene.cycles.samples = 64
    cell = 0.16
    scene.render.resolution_x = 260 * len(ids)
    scene.render.resolution_y = 260 * len(ROTS)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = a.out
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.20, 0.22, 0.23, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world

    for c, aid in enumerate(ids):
        for r, rot in enumerate(ROTS):
            objs = assets.import_model(aid)
            obj = objs[0]
            for extra in objs[1:]:
                bpy.data.objects.remove(extra, do_unlink=True)
            x = (c - (len(ids) - 1) / 2) * cell
            z = ((len(ROTS) - 1) / 2 - r) * cell - 0.055
            assets.place(obj, (x, 0.0, z), longest_mm=120, pre_rot_deg=rot)
            if a.style == "seiryu":
                assets.grade_stone(obj, seed=c * 1.3)
            elif a.style == "dragon":
                assets.grade_dragon(obj, seed=c * 1.3)

    cam_data = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", cam_data)
    scene.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = cell * max(len(ids), len(ROTS) * len(ids) / len(ROTS))
    cam.location = (0.0, -2.0, 0.12)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    sun = bpy.data.lights.new("S", "SUN")
    sun.energy = 4.0
    sun.angle = 0.15
    so = bpy.data.objects.new("S", sun)
    scene.collection.objects.link(so)
    so.rotation_euler = (0.5, 0.15, 0.0)       # from above and a little behind, like the tank lamp
    bpy.ops.render.render(write_still=True)
    print(f"[stones] wrote {a.out}")


if __name__ == "__main__":
    main()
