"""Develop a scene-linear render into display sRGB with the project's display
transform, optionally beside the plate for comparison.

  python src/comp/develop.py <render.exr> <out.jpg> [--beside lit|unlit] [--diff]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colour  # noqa: E402
import paths  # noqa: E402


def read_linear(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"cannot read {path}")
    return img[..., :3].astype(np.float32)


def develop(linear):
    return colour.linear_to_display(linear.astype(np.float64)).astype(np.float32)


def read_plate(which):
    p = cv2.imread(str(paths.PLATE / f"{which}_plate.png"), cv2.IMREAD_UNCHANGED)
    return p.astype(np.float32) / 65535.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exr")
    ap.add_argument("out")
    ap.add_argument("--beside", choices=["lit", "unlit"])
    ap.add_argument("--diff", action="store_true", help="print mean absolute difference against the plate, outside and inside the tank")
    a = ap.parse_args()

    disp = develop(read_linear(a.exr))
    h, w = disp.shape[:2]
    canvas = disp
    if a.beside or a.diff:
        plate = read_plate(a.beside or "lit")
        if plate.shape[:2] != (h, w):
            plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA)
        if a.diff:
            matte = cv2.imread(str(paths.PLATE / "tank_matte.png"), cv2.IMREAD_GRAYSCALE)
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            d = np.abs(disp - plate).mean(axis=2)
            print(f"mean abs difference vs plate: outside tank {d[~matte].mean():.4f}, inside tank {d[matte].mean():.4f}")
        if a.beside:
            canvas = np.hstack([plate, disp])
    cv2.imwrite(str(a.out), np.clip(canvas * 255 + 0.5, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
