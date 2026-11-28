# Lessons

Rules learned on this project. Newest first.

## Owner feedback on the rough cut (2026-11-28)
- What happened: in the rough cut some fish moved backward or sideways. The shoal model moved each fish as a point pushed by forces and then pointed the body roughly along its travel, with the heading clamped toward side-on for the camera's sake, so body and motion could disagree. The owner also found the captions' type unsuited to the theme, and the picture not sharp enough (the rough cut was half size, low samples, heavily compressed, and I had not shown what full quality looks like).
- Rule: animals move the way their bodies allow. A fish goes where it points: it turns its body to change course and never slides backward or sideways. Check it with a number (motion along the heading is never negative), not by eye alone.
- Rule: do not bend physical behaviour to flatter the camera (the side-on clamp). Bias what the animal wants to do instead.
- Rule: when showing a low-quality draft, show one full-quality frame beside it, so the draft is judged for what it is meant to show.
- Rule: typography is part of the design. Offer a considered treatment, not a default face in one size.

## Owner feedback on the first five design stills (2026-11-28)
- What happened: the owner found the Iwagumi stones too big for the tank, the Dutch plant groups cramped into blocks, the Forest trees identical and symmetrically placed, the Blackwater too dark, and the Floating Islands big and crowded. They also asked that every tank leave the fish room to swim, and that the fish keep one steady path while the designs change.
- Rule: scale everything to the real tank. In a 30 cm cube a main stone or tree above about two fifths of the water depth, or a group that fills the width, reads as oversized. Leave more open water than feels necessary.
- Rule: arrange as nature would. No blocks, rows, hedges, repeated shapes or mirror symmetry. Every tree, stone and plant group is an individual; spacing is uneven; groups overlap at their edges.
- Rule: a mood (dark water, dim light) must not cost legibility. Judge brightness against the other scapes side by side.
- Rule: what carries across a cut must be continuous. The shoal is simulated once for the whole film and every scape poses the same fish; a scape declares its open water (SWIM) and the build reports any fish touching scenery.
- Rule: when feedback changes the designs, do not spend render time on a draft of the old ones. Apply the feedback first, then render.

## One render at a time (2026-11-28)
- What happened: I started a second Blender render alongside one already running. The machine (16 GB RAM) ran critically low on memory and Claude Code stopped both background jobs.
- Rule: run one Blender render at a time. Chain jobs in sequence in a single command instead of starting a second in parallel. This includes subagents: do not have several of them rendering at once for long stretches.
- Rule: when a background job is stopped for low memory, report it and wait to be asked before restarting.

## Register every object for the lamp's light (2026-11-28)
- What happened: plants scattered as instances were not Cycles caustics receivers, so they got no lamp light through the water, only soft fill. Two review rounds flagged "flat lighting" before the cause was found.
- Rule: when a look problem shows up in every scene, suspect the shared setup before the individual scenes. `build.py` now registers every object a scape creates.

## Version anything that is cached (2026-11-28)
- What happened: the shoal simulation cache was keyed on its arguments only, so a changed model silently reused old results.
- Rule: a cache key must include a version of the code that produced it.
