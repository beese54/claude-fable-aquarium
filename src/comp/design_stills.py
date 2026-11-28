"""Composite each scape's design still into the filmed frame for review.

  python src/comp/design_stills.py [scape ids...]

Reads <work>/renders/tests/gate2_<scape>.exr (rendered at timeline.still_frame)
and writes review/03_design_<scape>.jpg, then a contact sheet of all of them in
film order as review/03_design_all.jpg.
"""
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "comp")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import camera_move  # noqa: E402
import composite as comp  # noqa: E402
import paths  # noqa: E402
import timeline  # noqa: E402

LABELS = {"01_iwagumi": "Iwagumi", "02_dutch": "Dutch garden", "03_forest": "Forest", "04_blackwater": "Blackwater",
          "05_floating": "Floating islands", "06_mountain": "Mountain", "07_jungle": "Jungle", "08_cove": "Cove",
          "09_riverbed": "Riverbed", "10_lotus": "Lotus pond"}


def main():
    order = list(timeline.visible())
    only = sys.argv[1:] or order
    for scape in only:
        frame = timeline.still_frame(scape)
        p = paths.RENDERS / "tests" / f"gate2_{scape}.exr"
        if not p.is_file():
            print("missing", scape)
            continue
        lin = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., :3]
        img = comp.composite(lin, level_mm=timeline.water_level_mm(frame, scape), seed=frame,
                             plate_weight=1.0 - camera_move.render_weight(frame), frame=frame)
        out = paths.REVIEW / f"03_design_{scape}.jpg"
        cv2.imwrite(str(out), np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
        print("wrote", out.name)

    # contact sheet: the tank only, five to a row, numbered in film order
    tiles = []
    for k, scape in enumerate(order):
        p = paths.REVIEW / f"03_design_{scape}.jpg"
        if not p.is_file():
            continue
        t = cv2.resize(cv2.imread(str(p))[330:1400], (432, 428), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (432, 34), (245, 245, 245), -1)
        cv2.putText(t, f"{k + 1}. {LABELS.get(scape, scape)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(t)
    if tiles:
        while len(tiles) % 5:
            tiles.append(np.full_like(tiles[0], 245))
        rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
        sheet = np.vstack(rows)
        cv2.imwrite(str(paths.REVIEW / "03_design_all.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote 03_design_all.jpg", sheet.shape)


if __name__ == "__main__":
    main()
