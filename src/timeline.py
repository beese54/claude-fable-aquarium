"""Things that depend on where we are in the film, shared by the Blender scene
and the compositor so the two can never disagree. Pure Python.

The awakening: after the light comes on, water rises through the empty tank
and the first scape is revealed beneath the surface as it climbs. The rise is
described in config/timeline.json on the `awakening` segment:
  "rise": [first_frame, last_frame], "level_from_mm": height the water starts at
"""
import json
from pathlib import Path

_TL = json.loads((Path(__file__).resolve().parent.parent / "config" / "timeline.json").read_text(encoding="utf-8"))
FULL_MM = 268.0


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _awakening():
    return next(s for s in _TL["segments"] if s["id"] == "awakening")


def is_filling(scape_id):
    return _awakening().get("to") == scape_id


def water_level_mm(frame, scape_id):
    """Height of the water surface above the tank's outer base at this frame."""
    a = _awakening()
    if a.get("to") != scape_id or "rise" not in a:
        return FULL_MM
    f0, f1 = a["rise"]
    lo = a.get("level_from_mm", 39.0)
    # ease in, then a long gentle settle: fast through the middle, slowing as it nears the top
    p = (frame - f0) / (f1 - f0)
    p = _smooth(p) ** 0.85
    return lo + (FULL_MM - lo) * p


def turbulence(frame, scape_id):
    """Multiplier on the surface ripples: lively while the tank fills, calming afterwards."""
    a = _awakening()
    if a.get("to") != scape_id or "rise" not in a:
        return 1.0
    f0, f1 = a["rise"]
    if frame <= f0:
        return 3.2
    if frame <= f1:
        return 3.2
    return 1.0 + 2.2 * (1.0 - _smooth((frame - f1) / 75.0))


def arrival(frame, scape_id, index, count):
    """0..1 size of animal `index` of `count`: they appear one after another once the water is nearly in."""
    a = _awakening()
    if a.get("to") != scape_id or "rise" not in a:
        return 1.0
    f1 = a["rise"][1]
    start = f1 - 26 + (index * 37 % count) * (44.0 / max(count, 1))
    return _smooth((frame - start) / 16.0)


def lamp_pulse(frame):
    """0..1 extra glow from the lamp just before the water arrives."""
    a = _awakening()
    if "rise" not in a:
        return 0.0
    f0 = a["rise"][0]
    x = (frame - (f0 - 14)) / 30.0
    if x <= 0 or x >= 1:
        return 0.0
    return _smooth(x * 2) if x < 0.5 else _smooth((1 - x) * 2)
