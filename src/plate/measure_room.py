"""Room measurement (part of task T1.3).

Uses the solved camera to turn features read off the plates into 3D positions
by casting pixel rays onto known planes. Writes config/room.json, which the
Blender room builder reads. All output is millimetres in the tank frame
(X right, Y away from camera, Z up, origin at the tank's outer front-left-bottom).

Run:  python src/plate/measure_room.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from camera import Camera  # noqa: E402

hit = Camera().hit


def rnd(v, n=1):
    return [round(float(x), n) for x in np.atleast_1d(v)]


# --- picks (pixels on the unlit or lit plate) --------------------------------
FLOOR_Z = -2.0                        # tank stands on about 2 mm of card
TRAY_BOTTOM_L, TRAY_BOTTOM_R = (44, 1333), (990, 1360)   # tray front face, bottom corners
TRAY_TOP_L, TRAY_TOP_R = (45, 1206), (996, 1240)         # tray front face, top corners
WALL_FLOOR_L, WALL_FLOOR_R = (35, 992), (1040, 1016)     # where wall meets floor, either side of the tank
SKIRT_TOP_L, SKIRT_TOP_R = (35, 933), (1040, 948)        # top of the pale skirting band
STEM_TOP = (540, 348)                 # stem meets the lamp head (rear lower edge of the head)
STEM_HALF_WIDTH_PX = 8
HEAD_REAR_L, HEAD_REAR_R = (420, 348), (668, 348)        # lamp head underside, rear corners
HEAD_FRONT_L, HEAD_FRONT_R = (414, 302), (674, 302)      # lamp head underside, front corners
HEAD_FRONT_TOP_Y = 275                # top of the head's front face
CLAMP = {"left": 483, "right": 588, "top": 577, "bottom": 625}   # clamp block on the rear glass rim
STEM_Y = 303.0                        # stem sits just behind the rear glass (outer face at 300)


def main():
    out = {"units": "mm, tank frame", "floor_z": FLOOR_Z}

    # Tray: a card box lid the tank stands in. Front face from four corners.
    bl, br = hit(TRAY_BOTTOM_L, 2, FLOOR_Z), hit(TRAY_BOTTOM_R, 2, FLOOR_Z)
    y_front = float((bl[1] + br[1]) / 2)
    tl, tr = hit(TRAY_TOP_L, 1, bl[1]), hit(TRAY_TOP_R, 1, br[1])
    out["tray"] = {
        "front_bottom_left": rnd(bl), "front_bottom_right": rnd(br),
        "front_y": round(y_front, 1),
        "height": round(float((tl[2] + tr[2]) / 2 - FLOOR_Z), 1),
        "x_min": round(float(bl[0]), 1), "x_max": round(float(br[0]), 1),
    }

    # Wall: vertical plane through the two wall/floor junction points.
    wl, wr = hit(WALL_FLOOR_L, 2, FLOOR_Z), hit(WALL_FLOOR_R, 2, FLOOR_Z)
    y_wall = float((wl[1] + wr[1]) / 2)
    out["wall"] = {
        "floor_junction_left": rnd(wl), "floor_junction_right": rnd(wr),
        "y": round(y_wall, 1),
        "yaw_deg": round(float(np.degrees(np.arctan2(wr[1] - wl[1], wr[0] - wl[0]))), 2),
        "skirting_top_z": round(float((hit(SKIRT_TOP_L, 1, wl[1])[2] + hit(SKIRT_TOP_R, 1, wr[1])[2]) / 2), 1),
        "gap_behind_tank": round(y_wall - 300.0, 1),
    }

    # Lamp: stem rises behind the rear glass; head reaches forward over the water.
    stem_top = hit(STEM_TOP, 1, STEM_Y)
    z_under = float(stem_top[2])
    rear_l, rear_r = hit(HEAD_REAR_L, 2, z_under), hit(HEAD_REAR_R, 2, z_under)
    front_l, front_r = hit(HEAD_FRONT_L, 2, z_under), hit(HEAD_FRONT_R, 2, z_under)
    y_head_front = float((front_l[1] + front_r[1]) / 2)
    top_front = hit(((HEAD_FRONT_L[0] + HEAD_FRONT_R[0]) / 2, HEAD_FRONT_TOP_Y), 1, y_head_front)
    stem_w = np.linalg.norm(hit((STEM_TOP[0] + STEM_HALF_WIDTH_PX, 400), 1, STEM_Y) -
                            hit((STEM_TOP[0] - STEM_HALF_WIDTH_PX, 400), 1, STEM_Y))
    cl_tl = hit((CLAMP["left"], CLAMP["top"]), 1, STEM_Y)
    cl_br = hit((CLAMP["right"], CLAMP["bottom"]), 1, STEM_Y)
    out["lamp"] = {
        "stem": {"x": round(float(stem_top[0]), 1), "y": STEM_Y, "z_top": round(z_under, 1), "width": round(float(stem_w), 1)},
        "head": {
            "underside_z": round(z_under, 1),
            "rear_left": rnd(rear_l), "rear_right": rnd(rear_r),
            "front_left": rnd(front_l), "front_right": rnd(front_r),
            "width": round(float((np.linalg.norm(rear_r - rear_l) + np.linalg.norm(front_r - front_l)) / 2), 1),
            "depth": round(float(((rear_l[1] + rear_r[1]) - (front_l[1] + front_r[1])) / 2), 1),
            "thickness": round(float(top_front[2] - z_under), 1),
            "centre_xy": rnd(((rear_l + rear_r + front_l + front_r) / 4)[:2]),
        },
        "clamp": {"x_min": round(float(cl_tl[0]), 1), "x_max": round(float(cl_br[0]), 1),
                  "z_min": round(float(cl_br[2]), 1), "z_max": round(float(cl_tl[2]), 1)},
    }

    (paths.CONFIG / "room.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
