"""The solved camera (config/camera.json) as a small projection helper.

Units are millimetres in the tank frame: X right, Y away from the camera, Z up,
origin at the tank's outer front-left-bottom corner. Pixels are x right, y down.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


class Camera:
    def __init__(self, path=None):
        cam = json.loads(Path(path or paths.CONFIG / "camera.json").read_text(encoding="utf-8"))
        self.W, self.H = cam["image_size"]
        self.f = cam["focal_px"]
        self.cx, self.cy = cam["principal_point"]
        self.R, _ = cv2.Rodrigues(np.array(cam["opencv"]["rvec"], dtype=np.float64))
        self.t = np.array(cam["opencv"]["tvec_mm"], dtype=np.float64)
        self.C = -self.R.T @ self.t

    def project(self, pts_mm):
        cam = (self.R @ np.atleast_2d(np.asarray(pts_mm, dtype=np.float64)).T).T + self.t
        return np.stack([self.f * cam[:, 0] / cam[:, 2] + self.cx,
                         self.f * cam[:, 1] / cam[:, 2] + self.cy], axis=1)

    def ray(self, px):
        d = self.R.T @ np.array([(px[0] - self.cx) / self.f, (px[1] - self.cy) / self.f, 1.0])
        return d / np.linalg.norm(d)

    def hit(self, px, axis, value):
        """Intersect a pixel's ray with the plane {axis = value}; axis is 0, 1 or 2."""
        d = self.ray(px)
        s = (value - self.C[axis]) / d[axis]
        if s <= 0:
            raise ValueError(f"plane is behind the camera for pixel {px}")
        return self.C + s * d

    def hull_mask(self, pts_mm, dilate=0):
        """Filled convex hull of projected 3D points, as a uint8 mask."""
        px = np.round(self.project(pts_mm)).astype(np.int32)
        m = np.zeros((self.H, self.W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(px), 255)
        if dilate:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
        return m


def box_corners(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
