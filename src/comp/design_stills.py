import sys, os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path[:0] = ["src", "src/comp"]
import cv2, numpy as np, composite as comp, camera_move, timeline, paths
shots = {"01_iwagumi": 760, "02_dutch": 1110, "03_forest": 1420, "04_blackwater": 1730, "05_floating": 2050}
for scape, frame in shots.items():
    p = paths.RENDERS / "tests" / f"gate2_{scape}.exr"
    if not p.is_file():
        print("missing", scape); continue
    lin = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., :3]
    img = comp.composite(lin, level_mm=timeline.water_level_mm(frame, scape), seed=frame,
                         plate_weight=1.0 - camera_move.render_weight(frame), frame=frame)
    out = paths.REVIEW / f"03_design_{scape}.jpg"
    cv2.imwrite(str(out), np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("wrote", out.name)
