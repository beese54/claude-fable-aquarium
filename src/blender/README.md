# Scene kit

Everything here runs inside Blender 4.5's Python. Scene units are metres in the
tank frame: X left to right, Y front to back (away from the camera), Z up, origin
at the tank's outer front-left-bottom corner. Layout code works in millimetres
and multiplies by `MM`. The tank's inside runs from 5 to 295 mm in X and Y; the
floor of the glass is at Z = 5; the water surface is at Z = 268; the tray hides
everything below Z = 38 from the camera. The camera is 555 mm in front of the
glass at X = 152, Z = 143. The lamp is at the back (Y 238 to 303), 422 mm up.

## Render a still

```powershell
.\render.ps1 -Scape 01_iwagumi -Quality draft   -Name mytest    # 540x960, about 3 s
.\render.ps1 -Scape 01_iwagumi -Quality preview -Name mytest    # 1080x1920, about 20 s
python src/comp/composite.py <work>\renders\tests\mytest.exr out.jpg   # into the filmed frame, with grain
```
`render.ps1` writes `<work>\renders\tests\<Name>.exr`, `.jpg` (developed, no plate) and `.log`.
`<work>` is `%LOCALAPPDATA%\aquarium_dream`. Errors from the scape show in the console and the log.

## A scape module (`scapes/<id>.py`)

```python
LIGHT = dict(watts=50, colour=(0.90, 0.95, 1.0), fill=0.015, bounce=0.05)   # optional
WATER = dict(tint=(0.88, 0.975, 0.955), density=1.5, scatter=0.0)             # optional

def build(ctx):
    ...                       # add objects
    assets.mark_underwater(list_of_objects, ctx)   # required for every object under water
    ctx["bed"] = bed; ctx["school"] = school
```
`scapes/01_iwagumi.py` is the worked example.

## The shoal and its swimming room

There is one shoal for the whole film (`src/sim/school.py`). Every scape calls
`fish.School()` and gets the same twelve tetras on the same path, so they carry
straight on while one scape dissolves into the next. A scape does not place the
fish. It declares the water they may use while it is on screen, as a plain
literal near the top of the module (it is read without importing the module):

```python
SWIM = {
    "box": (xmin, xmax, ymin, ymax, zmin, zmax),   # open water the shoal roams, mm
    "floor": (front_mm, back_mm),                  # bed height at the front and back glass
    "obstacles": [(x, y, radius, top_z), ...],     # cylinders standing on the bed; add a 5th
}                                                  # value bottom_z for a piece hanging in the water
```

Every scape must leave real open water: a box at least about 180 x 110 x 70 mm,
most of it in the front two thirds of the tank where the fish read best, with
nothing solid inside it that is not listed as an obstacle. Cover anything a
fish could swim into (stone, wood, dense plant groups) with cylinders.

Each single-frame build prints a `[swim]` line: the share of fish positions,
over all the frames the scape is on screen, that lie within 11 mm of scape
geometry, and what they touch. Keep it under about 2% by adjusting SWIM or the
layout. `--no-render` builds and checks without rendering:
`.
ender.ps1 -Scape <id> -Frame <n> -Name x -Extra "--no-render"`.
`python src/sim/school.py` prints where the shoal goes in every scape.

Blender renders take a machine-wide lock, so two renders started at once run
one after the other; a build may sit waiting for a few minutes.

## Pieces

| Module | What it gives |
|---|---|
| `substrate.build_bed(height_fn, name, gap_mm=1.2)` | Solid bed, stopping `gap_mm` short of the glass; `height_fn(x_mm, y_mm)` returns top height in mm. Returns a `Bed`: `.obj`, `.at(x, y, lift_mm=0)` (world point on the surface), `.z(x, y)`, `.set_weight(name, fn)` (paint a 0..1 attribute for scatter density). |
| `substrate.soil_material()`, `substrate.add_granules(bed, material, size_mm)` | Dark aquarium soil with real granules on the bed's top. |
| `substrate.add_cut_face(bed, material, size_mm, ...)`, `substrate.add_soil_face(bed, ...)` | The bed in section at the glass: instanced grains packed into the gap between the bed and the pane, with a ragged top line and a tone that falls with depth. Returns objects; do `made += ...`. Every scape calls it (see below). `substrate.sand_mix(name, pale=...)` gives a list of sand materials to pass as `material`. |
| `materials.white_sand()`, `beach_sand()`, `dark_gravel()`, `river_pebbles()`, `leaf_litter_floor()`, `driftwood()`, `dark_bogwood()`, `moss_surface()` | Scanned PBR materials by box projection: assign to any mesh, no UVs needed. `materials.pbr_box(...)` for others in `config/assets.json`. |
| `assets.import_model(id)` | Import a CC0 scan; returns its mesh objects (some ids hold several pieces). |
| `assets.place(obj, at, height_mm= or longest_mm=, pre_rot_deg=, rot_deg=, sink_mm=)` | Scale and stand an object on a point. `pre_rot_deg` chooses which face is down. |
| `assets.grade_stone(obj, ...)` | Seiryu look (blue-grey, veins, strata). Pass `dark=`, `light=`, `rust=`, `algae=`, `vein_amount=` to make other stone. |
| `assets.footprint_radius_mm(obj)`, `assets.world_bounds(obj)` | Sizes after placing. |
| `plants.hairgrass_tufts()`, `carpet_sprigs()`, `moss_fronds()`, `dead_leaves()`, `floaters()`, `stem_plants(species)` | Lists of small source objects for scattering. Species are the keys of `plants.STEM_SPECIES`. |
| `scatter.collection_of(name, objects)` then `scatter.scatter(target, collection, ...)` | Geometry Nodes scatter onto any mesh: `min_distance=` (Poisson) or `density=` (per m2), `weight_attr=`, `scale=(lo, hi)`, `align="UP"` or `"NORMAL"`, `tilt=`, `sway=(amp, per_metre, speed)`, `lift=` (metres along the surface normal). Several scatters can sit on one object. On a bed (or any object sharing its mesh) only the top surface is used, never the vertical faces at the glass; pass `top_only=False` to use every face. |
| `wood.grow(start, direction, length_mm, radius_mm, seed=, depth=, gravity=, wander=, children=)` then `wood.finish(skin, name, material)` | Branching driftwood: `gravity<0` grows up like a tree, `>0` hangs like roots. `skin.tips` lists branch ends (position, direction, radius). |
| `fish.shoal(n, centre_mm, spread_mm, seed=, heading_deg=, keep_out=, floor=)` | Places neon tetras for a still; returns `Tetra` objects (`.obj`). |
| `shrimp.Shrimp(name)`, `snail.Snail(name)` then `.pose(position, yaw=, t=, up=)` | Red cherry shrimp and nerite snail standing on any surface. |

See what the kit looks like with `tests/kit_sheet.py` (plants, wood), `tests/stone_sheet.py` (stones in four orientations) and `tests/closeup.py` (animals).

## The cut face at the glass

The camera looks through the front glass, so the bed's cross-section (tray
line at Z = 38 up to the bed's top) is in every shot. `build_bed` leaves it a
flat skirt; `add_cut_face` is what makes it substrate, and every scape calls it:

```python
made += substrate.add_soil_face(bed, seed=5)                                         # dark soil (a preset)
made += substrate.add_cut_face(bed, substrate.sand_mix("Face"), size_mm=1.2, tone=(1.3, 0.6))   # sand
# two kinds in one face: call once per kind with weight(x_mm, y_mm, depth_mm) -> 0..1
made += substrate.add_soil_face(bed, weight=lambda x, y, d: 1 - on_path(x, y))
made += substrate.add_cut_face(bed, sand, size_mm=1.2, weight=lambda x, y, d: on_path(x, y) * (d < 8))
```

- Grains are Geometry Nodes instances on a hidden sheet over the skirt, packed
  into the gap between the bed's solid and the pane: the front face to below
  the tray line, the left and right faces 10 mm deep. The side faces show only
  as a seam in the side panes' mirror; with the gap left empty that seam was a
  dark trough.
- `tone=(top, bottom)` multiplies the grain colour at the top of the face and
  `fade_mm` below it; patches and faint level bands are added, and each grain
  differs from the next by `vary`. It rides on the instances (`cut_dark`) and
  is read by the grain material (`shade_for_cut_face`, applied for you), so one
  material can serve the top of the bed and its face.
- `room=` is soft light from the room on the pane, as the grains' own glow
  (strongest where a grain is turned to the glass). It replaces the area
  lamps scapes used to aim at their cut face: it costs no samples and lights
  nothing else. Dark soil needs it (0.9 in the preset), bright white sand a
  little (0.3), most sand none. It is an absolute amount: if the lamp or the
  exposure changes a great deal, retune it. Do not add a lamp for the face.
- `size_mm`: at 1080 px across 300 mm a grain needs about 3 mm to read as a
  granule; sand at 1.2 mm reads as speckle, which is right for sand.
- Coarse stone: `build_bed(..., gap_mm=3.0)` so pebbles can stand out of the
  bed and look round, your own pebbles as `sources=` (with `thick_mm`, small
  `tilt`), a finer kind with `recess_mm=` behind them, and `backing=` (a dark
  colour) for the pores. `scapes/09_riverbed.py` is the example.
- Anything scattered on the bed stays on its top (see `scatter`), so carpets
  no longer hang down the face. Paint weights with `Bed.set_weight` as before.

## What makes a still read as real (learned on scape 1)

- The lamp is behind and above. Tops are bright, camera-facing sides are in shade. Do not fight that with more fill; use pale sand or open ground to bounce light instead.
- Bury hardscape by a quarter to a third. Nothing should sit on top of the bed.
- Keep the bed at least 60 mm high at the front glass so it shows as a band above the tray.
- Vary everything: plant height, density, colour, stone size. Even coverage reads as computer made.
- Fish go in loose groups at mixed angles, clear of the hardscape, in open water.
- Scale to the tank: it is only 30 cm. A main stone or tree above about two fifths of the water depth, or groups that fill the width, read as oversized. Leave more open water than feels necessary.
- Nothing in rows, blocks or mirror pairs. Vary spacing, lean, size and species mix; let groups interlock and overlap at their edges.
- Check a `preview` crop at 100% before calling anything finished. Stones and wood must not look like clay; leaves must not look like plastic.
