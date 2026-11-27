"""The camera's slow move through the scape section of the film.

The filmed clip is locked off, and the film stays locked off through the
awakening so the rendered tank sits pixel-true in the filmed frame. Once the
first scape is established the camera eases into a gentle dolly: a slow push in
with a slight drift sideways and up and down, always turning to keep the centre
of the tank where it was in frame. Because the room is rebuilt in 3D, that gives
real parallax between glass, plants, fish and the wall. It eases back to the
filmed position before the end, so the last frames are the filmed framing again.

Pure numpy, shared by the Blender scene and the compositor. Units: millimetres
in the tank frame; rotations are OpenCV-style world-to-camera matrices.
"""
import json
import math
from pathlib import Path

import numpy as np

_TL = json.loads((Path(__file__).resolve().parent.parent / "config" / "timeline.json").read_text(encoding="utf-8"))
_CFG = _TL.get("camera_move", {})
START = _CFG.get("start", 700)                       # first moving frame
END = _CFG.get("end", _TL["total_frames"] - 20)      # back at rest by this frame
PUSH_MM = _CFG.get("push_mm", 35.0)
DRIFT_MM = _CFG.get("drift_mm", 24.0)
RISE_MM = _CFG.get("rise_mm", 9.0)
PIVOT = np.array(_CFG.get("pivot_mm", [150.0, 150.0, 150.0]))


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def envelope(frame):
    """0 at rest .. 1 fully into the move. Starts and ends very gently."""
    return (_smooth((frame - START) / 170.0) ** 1.5) * (_smooth((END - frame) / 120.0) ** 1.5)


def offset(frame):
    """Camera displacement from the filmed position at this frame, in mm."""
    e = envelope(frame)
    if e <= 0:
        return np.zeros(3)
    u = frame - START
    return np.array([DRIFT_MM * math.sin(2 * math.pi * u / 1500.0) * e,
                     PUSH_MM * e,
                     RISE_MM * math.sin(2 * math.pi * u / 1100.0 + 1.0) * e])


def render_weight(frame):
    """How much of the frame comes from the render rather than the filmed plate:
    0 while the camera is parked (plate pixels are used outside the water), 1
    while it moves. The hand-over happens while the move is still imperceptible."""
    return _smooth((frame - START + 2) / 12.0) * _smooth((END + 4 - frame) / 12.0)


def moved(R, C, off):
    """New world-to-camera rotation and centre after displacing by `off`,
    turning so the pivot keeps its place in frame."""
    C = np.asarray(C, dtype=np.float64)
    C2 = C + np.asarray(off, dtype=np.float64)
    a = PIVOT - C
    b = PIVOT - C2
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        return np.asarray(R, dtype=np.float64), C2
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Q = np.eye(3) + K + K @ K * ((1 - c) / (s * s))      # rotates a onto b
    return np.asarray(R, dtype=np.float64) @ Q.T, C2
