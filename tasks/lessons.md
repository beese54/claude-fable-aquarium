# Lessons

Rules learned on this project. Newest first.

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
