"""Scape 1: Iwagumi.

A Japanese stone arrangement: an odd number of stones of one kind, all leaning
the same way as if shaped by one current. The main stone (oyaishi) stands about
two fifths of the way across; the second stone (fukuishi) answers it; smaller
stones (soeishi, suteishi) settle the group. Dark soil banks up toward the back
and a hairgrass carpet covers everything else. Coordinates are millimetres:
X left to right, Y front to back.
"""
import math

import assets
import fish
import plants
import substrate
from bl_common import MM
from scatter import collection_of, scatter

# name, model, x, y, height above soil, how the scan is stood up, (lean, lean, heading), fraction buried
STONES = [
    ("oyaishi",  "moon_rock_05", 108, 176, 150, (90, 0, 0), (0, -8, 200),  0.24),
    ("fukuishi", "moon_rock_06", 204, 132, 84,  (90, 0, 0), (0, -10, 160), 0.26),
    ("soeishi",  "moon_rock_03", 52,  120, 62,  (0, 90, 0), (0, -9, 30),   0.28),
    ("sute_1",   "moon_rock_01", 252, 205, 30,  (0, 0, 0),  (0, -5, 140),  0.35),
    ("sute_2",   "moon_rock_04", 150, 72,  20,  (0, 0, 0),  (0, -4, 25),   0.40),
]


def ground(x, y):
    """Soil height in mm. Deep enough at the front glass to show as a dark band
    above the tray, banked high at the back, mounded under the main group."""
    slope = 60 + 60 * (y / 300) ** 1.2
    mound = 16 * math.exp(-(((x - 118) / 85) ** 2 + ((y - 185) / 75) ** 2))
    swell = 5 * math.exp(-(((x - 212) / 55) ** 2 + ((y - 140) / 60) ** 2))
    ripple = 2.2 * math.sin(x * 0.045 + 1.3) * math.cos(y * 0.038) + 1.0 * math.sin(x * 0.19) * math.sin(y * 0.11 + 0.5)
    return slope + mound + swell + ripple


def _clumps(x, y):
    """Patchiness for the carpet: 0.5 in thin spots to 1 in thick ones."""
    n = math.sin(x * 0.083 + 1.7 * math.sin(y * 0.061)) * math.cos(y * 0.079 + 1.3 * math.sin(x * 0.047))
    n2 = math.sin(x * 0.31 + 0.5) * math.sin(y * 0.27 + 1.1)
    v = 0.5 + 0.38 * n + 0.18 * n2
    return max(0.5, min(1.0, (v - 0.05) / 0.4))


def build(ctx):
    made = []

    soil = substrate.soil_material()
    bed = substrate.build_bed(ground, "Soil")
    bed.obj.data.materials.append(soil)
    substrate.add_granules(bed, soil, size_mm=2.6, seed=5)
    made.append(bed.obj)

    stones = []
    for k, (name, model, x, y, h, pre, rot, buried) in enumerate(STONES):
        obj = assets.import_model(model)[0]
        obj.name = name
        full = h / (1 - buried)
        assets.place(obj, bed.at(x, y), height_mm=full, pre_rot_deg=pre, rot_deg=rot, sink_mm=full * buried)
        assets.grade_stone(obj, seed=k * 1.7)
        stones.append((obj, x, y, assets.footprint_radius_mm(obj), ground(x, y) + h))
        made.append(obj)

    def carpet(x, y):
        w = _clumps(x, y)
        for _, sx, sy, r, _ in stones:
            d = math.hypot(x - sx, y - sy)
            w = min(w, max(0.0, min(1.0, (d - r * 0.70) / (r * 0.25))))
        return w * (0.55 + 0.45 * min(1.0, y / 60))      # thinner against the front glass
    bed.set_weight("carpet", carpet)
    tufts = collection_of("HairgrassSrc", plants.hairgrass_tufts(variants=6, blades=(10, 18), height_mm=(14, 34)))
    scatter(bed.obj, tufts, name="Hairgrass", min_distance=3.3 * MM, weight_attr="carpet", seed=21,
            scale=(0.45, 1.15), align="UP", tilt=0.30, sway=(0.07, 9.0, 0.35))

    keep = [(sx, sy, r * 0.8, top) for _, sx, sy, r, top in stones]
    school = fish.shoal(12, centre_mm=(196, 120, 172), spread_mm=(42, 50, 24), seed=4, heading_deg=175, keep_out=keep, floor=ground)
    made += [t.obj for t in school]

    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    return ctx
