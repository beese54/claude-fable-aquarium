# Aquarium Dream

A short film that starts with real phone footage of an empty 30 cm glass cube and a light being switched on, then lets the same tank come to life as five different aquascapes. Everything after the light comes on is a 3D scene built by code and path-traced in Blender through a camera solved from the real footage.

Built with Claude Fable 5.2 as a demonstration that whatever you can imagine, you can build.

**Status:** in progress. See `tasks/todo.md` for where things stand.

## What is here

| Path | What it is |
|---|---|
| `PROMPT.md` | The project brief |
| `specification.json`, `definition_of_done.md` | What is being built and how it is judged |
| `progress_tracking.json`, `tasks/todo.md` | Task ledger |
| `config/` | Timeline, captions, paths, one file per scape |
| `src/plate/` | Footage prep and camera solve |
| `src/blender/` | Scene, assets, scapes, animation, render |
| `src/comp/` | Compositing, captions, sound, encode |
| `src/qa/` | Automated checks |
| `review/` | Stills posted for feedback along the way |

## Setup

Windows with an NVIDIA GPU, Git Bash, ffmpeg, Python 3.12.

```bash
bash init.sh
```

Reproduction steps will be filled in as the pipeline lands.

## Licence

Third-party assets are CC0 and listed in `assets/MANIFEST.md`.
