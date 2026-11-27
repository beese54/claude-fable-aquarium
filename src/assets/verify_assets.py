"""Import every glTF model from config/assets.json into an empty scene and
write paths.ASSETS/index.json with, per model: object count, triangle count,
bounding-box size in metres, and material names. Triangles and bbox exclude
*_LOD1.._LOD9 meshes (listed separately); per_object gives each mesh's own
size, which matters for sets laid out in a row. Texture assets are listed
with the map files found on disk; alpha/translucency maps that the glTF does
not reference are checked for presence.

Run:  blender -b --python src/assets/verify_assets.py
Exits 1 (via sys.exit after writing the index) if any model fails to import
or any texture asset has a missing map.
"""
import json
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402


def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def object_stats(ob, deps):
    """Triangles and world-space (min, max) corners of one mesh object."""
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    tris = len(me.loop_triangles)
    n = len(me.vertices)
    lo = hi = None
    if n:
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        mw = np.array(ob.matrix_world, dtype=np.float64)
        w = co @ mw[:3, :3].T + mw[:3, 3]
        lo, hi = w.min(axis=0), w.max(axis=0)
    ev.to_mesh_clear()
    return tris, lo, hi


def is_lower_lod(name):
    """True for *_LOD1.._LOD9 meshes (Poly Haven ships some rocks with LOD0..3 stacked)."""
    i = name.rfind("_LOD")
    return i >= 0 and name[i + 4:i + 5].isdigit() and name[i + 4:i + 5] != "0"


def mesh_stats(meshes):
    """Stats over the given mesh objects: total tris, overall bbox, per-object list, materials."""
    deps = bpy.context.evaluated_depsgraph_get()
    tris = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    per = []
    mats = set()
    for ob in meshes:
        t, l, h = object_stats(ob, deps)
        tris += t
        if l is not None:
            lo, hi = np.minimum(lo, l), np.maximum(hi, h)
            per.append({"name": ob.name, "triangles": t,
                        "bbox_m": [round(float(x), 4) for x in (h - l)]})
        for slot in ob.material_slots:
            if slot.material:
                mats.add(slot.material.name)
    size = [round(float(x), 4) for x in (hi - lo)] if tris else [0, 0, 0]
    return tris, size, sorted(per, key=lambda d: d["name"]), sorted(mats)


def main():
    cfg = json.loads((paths.CONFIG / "assets.json").read_text(encoding="utf-8"))
    index = {"models": {}, "textures": {}}
    failed = []
    for a in cfg["assets"]:
        root = paths.ASSETS / a["source"] / a["id"]
        if a["type"] == "texture":
            names = [f["path"] for f in a["files"] if not f.get("extract")]
            have = sorted(p.name for p in root.glob("*") if p.is_file() and not p.name.endswith(".part"))
            miss = [n for n in names if n not in have]
            if any(f.get("extract") for f in a["files"]):  # ambientCG zip: core maps must be extracted
                for suf in ("_Color.jpg", "_NormalGL.jpg", "_Roughness.jpg"):
                    if not any(h.endswith(suf) for h in have):
                        miss.append("*" + suf)
            index["textures"][a["id"]] = {"category": a["category"], "files": have, "missing": miss}
            if miss:
                failed.append(f"{a['id']}: missing {miss}")
            continue
        gltf = root / a["main_file"]
        empty_scene()
        try:
            res = bpy.ops.import_scene.gltf(filepath=str(gltf))
            if "FINISHED" not in res:
                raise RuntimeError(f"import returned {res}")
            objs = list(bpy.context.scene.objects)
            meshes = [o for o in objs if o.type == "MESH"]
            if not meshes:
                raise RuntimeError("no mesh objects")
            lods = [o for o in meshes if is_lower_lod(o.name)]
            main = [o for o in meshes if o not in lods]
            tris, size, per, mats = mesh_stats(main)
            missing_img = sorted(
                img.name for img in bpy.data.images
                if img.source == "FILE" and not img.packed_file
                and not Path(bpy.path.abspath(img.filepath)).is_file()
            )
            index["models"][a["id"]] = {
                "category": a["category"],
                "file": str(gltf.relative_to(paths.ASSETS)).replace("\\", "/"),
                "objects": len(objs),
                "mesh_objects": len(meshes),
                "lower_lod_objects": sorted(o.name for o in lods),
                "triangles": tris,
                "bbox_m": size,
                "per_object": per,
                "materials": mats,
                "images": len(bpy.data.images),
                "missing_images": missing_img,
            }
            for key in ("alpha_map", "translucency_map"):  # not referenced by the glTF itself
                if key in a:
                    index["models"][a["id"]][key] = a[key]
                    if not (root / a[key]).is_file():
                        missing_img.append(a[key])
            if missing_img:
                failed.append(f"{a['id']}: missing images {missing_img}")
            print(f"OK   {a['id']:28s} obj={len(objs):3d} lod1+={len(lods)} tris={tris:9d} bbox={size} mats={len(mats)}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            index["models"][a["id"]] = {"category": a["category"], "error": str(e)}
            failed.append(f"{a['id']}: {e}")
            print(f"FAIL {a['id']}: {e}")
    index["failed"] = failed
    out = paths.ASSETS / "index.json"
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {out}; {len(index['models'])} models, {len(index['textures'])} textures, {len(failed)} failures")
    sys.exit(1 if failed else 0)


main()
