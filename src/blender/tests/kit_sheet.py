"""Contact sheet of the plant and wood kit, for judging pieces on their own.

  blender -b --python src/blender/tests/kit_sheet.py -- --out <file.png> [--what plants|wood]
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
import plants  # noqa: E402
import wood  # noqa: E402
from scatter import collection_of, scatter  # noqa: E402


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--what", default="plants")
    a = ap.parse_args(bc.script_args())

    scene = bc.reset_scene()
    bc.setup_render(scene, "preview")
    scene.cycles.samples = 96
    scene.render.resolution_x, scene.render.resolution_y = 1800, 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = a.out
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.16, 0.19, 0.20, 1)
    scene.world = world
    col = scene.collection

    if a.what == "plants":
        x = -0.40
        for sp in plants.STEM_SPECIES:
            for o in plants.stem_plants(sp, variants=2):
                col.objects.link(o)
                o.location = (x, 0, 0)
                x += 0.035
            x += 0.012
        x = -0.40
        for group in (plants.hairgrass_tufts(variants=3), plants.carpet_sprigs(variants=4), plants.moss_fronds(variants=4),
                      plants.dead_leaves(variants=5), plants.floaters(variants=3)):
            for o in group:
                col.objects.link(o)
                o.location = (x, -0.02, -0.11)
                x += 0.045
            x += 0.02
        span, centre = 0.86, (0.0, 0.0, 0.01)
    else:
        bark = wood.bark_material()
        tree = wood.finish(wood.grow((-0.16, 0, -0.10), (0.1, 0, 1), 150, 7.5, seed=3, depth=3, gravity=-0.25, wander=0.5), "Tree", bark)
        moss = collection_of("MossSrc", plants.moss_fronds())
        scatter(tree, moss, name="Moss", density=260000.0, seed=4, scale=(0.7, 1.3), align="NORMAL", tilt=0.6)
        wood.finish(wood.grow((0.02, 0, 0.10), (0.3, 0, -1), 170, 5.0, seed=8, depth=3, gravity=0.45, wander=0.6, children=(2, 3)), "Roots", bark)
        wood.finish(wood.grow((0.20, 0, -0.10), (0.5, 0.1, 0.6), 190, 9.0, seed=21, depth=2, gravity=-0.05, wander=0.7, children=(2, 3)), "Spider", bark)
        span, centre = 0.62, (0.02, 0.0, 0.0)

    cam_data = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", cam_data)
    col.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = span
    cam.location = (centre[0], -1.0, centre[2] + 0.12)
    look_at(cam, centre)
    scene.camera = cam
    for loc, energy, size in (((0.0, 0.15, 0.6), 40.0, 0.8), ((-0.3, -0.8, 0.2), 12.0, 0.8)):
        ld = bpy.data.lights.new("L", "AREA")
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new("L", ld)
        col.objects.link(lo)
        lo.location = loc
        look_at(lo, centre)
    bpy.ops.render.render(write_still=True)
    print(f"[kit] wrote {a.out}")


if __name__ == "__main__":
    main()
