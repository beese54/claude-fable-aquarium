"""Composite a rendered frame into the filmed plate.

While the camera is parked at the solved position, everything the water does
not touch is taken straight from the filmed plate: the room, the lamp, the rim
of the tank and the dry glass above the waterline, with their real smudges and
bloom. Only the wet part of the tank comes from the render. The seam between
the two is the waterline itself. The render is developed with the project's
display transform, given the phone's sharpening, and the whole frame then gets
grain measured from the footage (the plate is a temporal median, so it has none
of its own).

  python src/comp/composite.py <render.exr> <out.jpg|png> [--seed N] [--level-mm 268] [--no-plate]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "plate"))
import colour  # noqa: E402
import paths  # noqa: E402
from camera import Camera  # noqa: E402

W, H = 1080, 1920
_cache = {}


def lit_plate():
    if "plate" not in _cache:
        p = cv2.imread(str(paths.PLATE / "lit_plate.png"), cv2.IMREAD_UNCHANGED)
        _cache["plate"] = p.astype(np.float32) / 65535.0
    return _cache["plate"]


def wet_matte(level_mm=268.0, feather=1.2):
    """1 where the camera looks into water: the front glass from the tray's rim
    up to the waterline, and, while the surface is still below the camera, the
    surface itself seen from above."""
    key = ("wet", round(level_mm, 2))
    if key not in _cache:
        if len(_cache) > 40:                                  # a rising level makes many mattes; keep the cache small
            for k in [k for k in _cache if isinstance(k, tuple) and k[0] == "wet"]:
                del _cache[k]
        cam = Camera()
        room = json.loads((paths.CONFIG / "room.json").read_text(encoding="utf-8"))
        z0 = room["floor_z"] + room["tray"]["height"] + 1.0
        # The side panes seen edge-on (the outer 5 mm each side) stay as filmed:
        # water behind them barely changes how the glass edge looks.
        pts = [(5.5, 0, z0), (294.5, 0, z0), (294.5, 0, level_mm), (5.5, 0, level_mm),
               (5.5, 295, level_mm), (294.5, 295, level_mm)]
        px = cam.project(pts)
        S = 4
        m = np.zeros((H * S, W * S), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(px * S).astype(np.int32)), 255)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        _cache[key] = cv2.GaussianBlur(m, (0, 0), feather) if feather else m
    return _cache[key]

def grain(shape, luma, rng, stats=None):
    """Luma-weighted noise shaped like the footage's: slightly soft, strongest in the mid-tones."""
    if stats is None:
        stats = json.loads((paths.PLATE / "plate_stats.json").read_text(encoding="utf-8"))["grain_lit"]
    prof = stats["luma_sigma_by_brightness"]
    xs = np.array([(p["lo"] + p["hi"]) / 2 for p in prof])
    # The darkest bins in the footage are dominated by moving edges, not grain; cap them.
    ys = np.minimum(np.array([p["sigma"] for p in prof]), 0.014)
    sigma = np.interp(luma, xs, ys).astype(np.float32)
    n = rng.standard_normal(shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.65)
    n /= n.std()
    chroma = rng.standard_normal(shape).astype(np.float32)
    chroma = cv2.GaussianBlur(chroma, (0, 0), 0.9)
    chroma /= chroma.std()
    return sigma[..., None] * (0.85 * n[..., None] + 0.35 * chroma)


def unsharp(img, radius=1.1, amount=0.45):
    return img + amount * (img - cv2.GaussianBlur(img, (0, 0), radius))


def composite(render_linear, level_mm=268.0, seed=0, use_plate=True, add_grain=True):
    """render_linear: HxWx3 BGR scene-linear float. Returns HxWx3 BGR display float in 0..1."""
    disp = colour.linear_to_display(render_linear.astype(np.float64)).astype(np.float32)
    if disp.shape[:2] != (H, W):
        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_CUBIC)
    sharp = unsharp(disp)
    if use_plate:
        m = wet_matte(level_mm)[..., None]
        out = lit_plate() * (1 - m) + sharp * m
    else:
        out = sharp
    out = np.clip(out, 0, 1)
    if add_grain:
        luma = 0.0722 * out[..., 0] + 0.7152 * out[..., 1] + 0.2126 * out[..., 2]
        out = np.clip(out + grain(out.shape, luma, np.random.default_rng(seed)), 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exr")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level-mm", type=float, default=268.0)
    ap.add_argument("--no-plate", action="store_true")
    ap.add_argument("--beside-plate", action="store_true", help="put the filmed lit frame on the left for comparison")
    a = ap.parse_args()
    lin = cv2.imread(a.exr, cv2.IMREAD_UNCHANGED)
    if lin is None:
        sys.exit(f"cannot read {a.exr}")
    out = composite(lin[..., :3], a.level_mm, a.seed, use_plate=not a.no_plate)
    if a.beside_plate:
        plate = lit_plate()
        luma = 0.0722 * plate[..., 0] + 0.7152 * plate[..., 1] + 0.2126 * plate[..., 2]
        plate = np.clip(plate + grain(plate.shape, luma, np.random.default_rng(a.seed + 1)), 0, 1)
        out = np.hstack([plate, out])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    if a.out.lower().endswith((".jpg", ".jpeg")):
        cv2.imwrite(a.out, img8, [cv2.IMWRITE_JPEG_QUALITY, 94])
    else:
        cv2.imwrite(a.out, img8)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
