# Review record

Two rounds of independent review by a fresh-context reviewer (an Opus subagent
given only the stills and a description of what is real and what is rendered),
with what was done about each finding. Scores are the reviewer's, on the stills
as they stood before the fixes listed; the stills in this folder are the fixed
ones and have not been re-scored independently.

## Round 1: look test (scape 1), scored 3/10

| Finding | Disposition |
|---|---|
| Fish read as stickers: all side-on, opaque, flat colours | Fixed: translucent body, mixed angles, shoaling simulation |
| Stones read as grey clay | Fixed: scan detail kept, strata, calcite veins, staining; stood on their best face and part buried |
| No visible soil band | Fixed: bed raised so it shows above the tray |
| Lighting too soft, no caustics | Fixed: lamp is ten discrete LED sources; fine ripples added; fill cut back |
| Side panes and surface show the room and cable | Justified and changed: the mirror images are physically right, but the cable is left out behind water because its reflections read as errors |
| Grass like a lawn | Fixed: clumped, uneven, cooler green |
| Glass edges change colour at the waterline | Fixed: filmed edges kept in the composite; rendered edges later matched to them |
| Milky veil, lifted blacks | Fixed (found in round 2): the glass was mirroring the plate's repeated bottom edge; room dimmed for reflection rays |
| Render too clean beside phone footage | Fixed: grain measured from the footage, phone-like sharpening; filmed glass smears kept above the waterline |
| Floor light pool unchanged by tank contents | Fixed: eased down as the tank fills |

## Round 2: five design stills

| Scape | Scale | Light | Grounding | Optics | Animals | Overall |
|---|---|---|---|---|---|---|
| 1 Iwagumi | 4 | 3 | 3 | 3 | 2 | 5/10 |
| 2 Dutch | 4 | 3 | 2 | 3 | 2 | 4/10 |
| 3 Forest | 3 | 3 | 3 | 3 | 3 | 4/10 |
| 4 Blackwater | 4 | 2 | 3 | 2 | 2 | 4/10 |
| 5 Floating Islands | 3 | 3 | 1 | 3 | 2 | 3/10 |

Shared findings:

| Finding | Disposition |
|---|---|
| Flat lighting, no cast shadows, nothing under the floating rocks | Fixed. Root cause: objects instanced by scatters were not registered to receive the lamp's light through the water surface, so every plant was lit by soft fill alone. All scape objects are now registered centrally in `build.py`. Fill reduced in scape 5 so island shadows read |
| Fish: pale back, thin stripe, toy saturation, head-on "pills" | Fixed: dark matt olive back, broader stripe, far less self-glow; simulation keeps headings at least a third of the way to side-on |
| Ghost of the lamp arm with a white band when the camera moves | Fixed: stem is a plain dark rod again; the gap it left in the projected glass smears is bridged |
| Front substrate cross-section flat and textureless | Open. Granules are there but small and soft at this size; the forest's sand pocket still reads as a block |
| Underside-of-surface mirror inconsistent between scapes | Justified: it is computed, and shows whatever lies along the reflected ray in each layout |
| No meniscus line, no particles in the water, everything very clean | Open (low priority) |
| Colour streaks on the top rim of the back glass | Justified: refraction of the planting through the rim; left as computed |
| Shrimp and snail read as simple shapes | Partly fixed: snails moved onto sand side-on so the shell shows; shrimp are about 60 px long and keep their modelled legs and feelers |

Per scape:

| Scape | Finding | Disposition |
|---|---|---|
| 1 | Hairgrass pokes through the side glass | Fixed: tufts start in from the glass |
| 1 | Veins look painted on | Reduced |
| 2 | Plants look like plastic decor | Largely fixed by the lighting fix (leaves now lit, shaded and translucent); groups still a little hedge-like |
| 3 | Sand path ends in a geometric block at the glass | Open |
| 3 | Trunks smooth, canopies repeat | Open |
| 4 | Roots seem to hang from the surface with no support | Open: intended as roots reaching in from a bank; could be made to break the surface more visibly |
| 4 | Tea colour reads as a flat filter | Justified in part: absorption does deepen with distance; shafts of light would help |
| 5 | White rod under an island (the sand-fall) | Fixed: solid core hidden, the fall is moving grains only |
| 5 | Snail looks like a coin | Fixed: moved onto the sand, side-on |
| 5 | Rock undersides look like cones | Open |

## Coordinator's own view after the fixes

Not an independent score. Scapes 1, 2 and 5 improved most. The weakest remaining
points are the substrate cross-sections at the front glass, the forest's trunks
and sand pocket, and the plain lower background of the blackwater scape.
