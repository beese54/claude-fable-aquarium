# Aquarium Dream: project brief

## Goal
Produce `output/aquarium_dream.mp4`: a portrait 1080x1920, 30 fps, roughly 115 s
video. It opens with the owner's real footage of an empty 30 x 30 x 30 cm glass
tank having its light fitted and switched on. From the moment the light is on,
the same tank, in the same room, seen by the same camera, comes to life as ten
different aquascapes. A viewer should believe they are watching real filmed
aquariums. The message: whatever you can imagine, you can build.

## Why it matters
The video is a public showcase of what Claude Fable 5.2 can build from one rough
clip and an idea. Realism and craft are the point. A flat or obviously
computer-generated result fails the brief even if every feature is present.

## Fixed inputs
- `Aquarium_Blank.mp4` is the opening scene. Do not alter its content: only
  rotate, tonemap HDR to SDR, and strip its audio.
- The tank is a 30 cm cube. Use that to solve the camera exactly.

## Must have
1. Ten distinct aquascapes, each on screen at least 7 s: Iwagumi (Japanese
   stone), Dutch garden, Mountain, Forest, Riverbed, Jungle, Blackwater
   Amazon, Lotus pond, Cove, Floating Islands. Each is scaled to a 30 cm tank,
   arranged as nature would (no blocks, rows, repeats or mirror symmetry) and
   leaves the fish real open water to swim in.
2. Living animals in every scape: a school of neon tetras that swim through
   real depth, toward and away from the camera. It is one shoal for the whole
   film: the fish keep one steady, unbroken path while the designs change
   around them. Red neocaridina shrimp in at
   least two scapes. A snail in at least one.
3. True 3D. Every scape is a full 3D scene path-traced through the solved
   camera with water, glass refraction, internal side-glass reflections, depth
   of field, light shafts and moving caustics. The camera makes a small real
   move in every scape so parallax is visible. No flat cards facing the camera.
4. A caption during every transition, on the theme of expanding your
   imagination, plus an opening and a closing caption.
5. A seamless hand-off from real footage to rendered footage: no jump in
   framing, exposure, colour or grain.
6. Ambient sound design only (water, bubbles, light click). CC0 or synthesised.

## How to work
- Everything is built by code in this repo (Blender Python, ffmpeg, OpenCV) so
  any piece of feedback becomes a precise, repeatable edit.
- Third-party assets must be CC0. Record each one with source and licence in
  `assets/MANIFEST.md`.
- Claude Fable 5.2 is the coordinator and owns art direction, realism, camera
  solve, integration and every review. Bounded, well-specified tasks go to
  subagents on Opus 5.5, one task each, with a written brief and an acceptance
  check. The coordinator verifies their output before it is committed.
- The owner's only involvement is feedback on designs from screenshots. Post
  stills to `review/`, show them in the conversation, and keep working on
  everything that does not depend on the answer. Never start the full-quality
  final render of a scape whose design still has not been approved.
- Judge every still yourself before showing it: scale, light match with the
  room, contact shadows, water and glass optics, animal motion. Ask a
  fresh-context reviewer to list anything that gives it away as CG, then fix
  those before the owner sees it.
- State lives in `progress_tracking.json` and `tasks/todo.md`. One commit and
  push per finished task. Any new session must be able to resume from those
  files alone.
- Long renders run as detached background processes that write one file per
  frame and skip frames that already exist, so they survive interruption.
- Heavy intermediates (downloaded assets, frame sequences) live outside the
  OneDrive-synced project folder, in the work directory named in
  `config/paths.json`.
- Report honestly. If a look test misses the realism bar, say so, show it, and
  re-plan rather than patching.

## Done means
See `definition_of_done.md`. In short: the MP4 passes every automated check,
the ten scapes are approved by the owner from stills, and the repo is public,
reproducible and fully committed.
