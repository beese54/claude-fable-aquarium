"""Scape 1: Iwagumi.

A Japanese stone arrangement: an odd number of stones of one kind, all leaning
the same way as if shaped by one current. The main stone (oyaishi) stands about
two fifths of the way across; the second stone (fukuishi) answers it; smaller
stones (soeishi, suteishi) settle the group. Dark soil banks up toward the back
and a hairgrass carpet covers everything else. Coordinates are millimetres:
X left to right, Y front to back.
"""
import math
import random

import assets
import fish
import plants
import substrate
from bl_common import MM
from scatter import collection_of, scatter

# id, model, x, y, height, pre-rotation, final rotation (lean, lean, heading), sink
STONES = [
    ("oyaishi",  "moon_rock_05", 106, 178, 166, (0, 0, 0),   (0, -9, 18),   16),
    ("fukuishi", "moon_rock_06", 206, 128, 90,  (0, 0, 0),   (0, -11, -35), 11),
    ("soeishi",  "moon_rock_03", 50,  122, 58,  (0, 0, 0),   (0, -8, 70),   9),
    ("sute_1",   "moon_rock_07", 254, 196, 44,  (0, 0, 0),   (0, -6, 140),  7),
    ("sute_2",   "moon_rock_04", 148, 74,  26,  (0, 0, 0),   (0, -4, 25),   6),
]

def ground(x, y):
    """Soil height in mm: low at the front glass, banked at the back, mounded under the main group."""
    slope = 44 + 58 * (y / 300) ** 1.25
    mound = 20 * math.exp(-(((x - 120) / 85) ** 2 + ((y - 185) / 75) ** 2))
    swell = 6 * math.exp(-(((x - 215) / 55) ** 2 + ((y - 150) / 60) ** 2))
    ripple = 1.6 * math.sin(x * 0.045 + 1.3) * math.cos(y * 0.038)
    return slope + mound + swell + ripple


def build(ctx):
    rng = random.Random(11)
    made = []

    soil = substrate.soil_material()
    bed = substrate.build_bed(ground, "Soil")
    bed.obj.data.materials.append(soil)
    substrate.add_granules(bed, soil, size_mm=2.4, seed=5)
    made.append(bed.obj)

    stones = []
    for k, (name, model, x, y, h, pre, rot, sink) in enumerate(STONES):
        obj = assets.import_model(model)[0]
        obj.name = name
        assets.place(obj, bed.at(x, y), height_mm=h, pre_rot_deg=pre, rot_deg=rot, sink_mm=sink)
        assets.grade_stone(obj, seed=k * 1.7)
        stones.append((obj, x, y, assets.footprint_radius_mm(obj)))
        made.append(obj)

    def carpet(x, y):
        w = 1.0
        for _, sx, sy, r in stones:
            d = math.hypot(x - sx, y - sy)
            w = min(w, max(0.0, min(1.0, (d - r * 0.62) / (r * 0.30))))
        return w
    bed.set_weight("carpet", carpet)
    tufts = collection_of("HairgrassSrc", plants.hairgrass_tufts())
    scatter(bed.obj, tufts, name="Hairgrass", min_distance=4.2 * MM, weight_attr="carpet", seed=21,
            scale=(0.7, 1.2), align="UP", tilt=0.22)

    # A loose shoal in the open water to the right of the main stone, with a
    # few stragglers further back and to the left for depth.
    school = []
    spots = []
    while len(spots) < 12:
        if len(spots) < 9:
            p = (rng.uniform(150, 272), rng.uniform(28, 250), rng.uniform(95, 212))
        else:
            p = (rng.uniform(30, 120), rng.uniform(30, 95), rng.uniform(100, 190))
        if all(math.dist(p, q) > 30 for q in spots):
            spots.append(p)
    for i, (x, y, z) in enumerate(spots):
        heading = math.radians(180 + rng.uniform(-30, 30)) if rng.random() < 0.75 else math.radians(rng.uniform(-40, 40))
        pitch = rng.uniform(-0.12, 0.12)
        fwd = (math.cos(heading) * math.cos(pitch), math.sin(heading) * math.cos(pitch) * 0.6, math.sin(pitch))
        t = fish.Tetra(f"Tetra_{i:02d}", size=rng.uniform(0.86, 1.08))
        t.pose((x * MM, y * MM, z * MM), fwd, phase=rng.uniform(0, 6.283), amp=rng.uniform(0.25, 0.6), bend=rng.uniform(-0.25, 0.25))
        school.append(t)
        made.append(t.obj)
    assets.mark_underwater(made, ctx)
    ctx["bed"] = bed
    ctx["school"] = school
    return ctx
