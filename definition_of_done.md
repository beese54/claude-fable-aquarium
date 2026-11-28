# Definition of done

Every criterion names the check that can fail it. Automated checks live in
`src/qa/checks.py` and run with `python src/qa/checks.py`.

## Final video (`output/aquarium_dream.mp4`)

| # | Criterion | Check |
|---|---|---|
| V1 | H.264, yuv420p, 1080x1920, 30 fps, 105 to 125 s, AAC stereo 48 kHz, under 95 MB, moov atom at the front | `checks.py container` (ffprobe) |
| V2 | The first 16.6 s is the owner's clip, changed only by rotation and tonemapping: mean SSIM 0.95 or higher against the tonemapped source | `checks.py intro_ssim` |
| V3 | Hand-off from real to rendered footage has no jump: the frame difference at the seam is no more than 1.5x the median of the 30 frame differences either side | `checks.py seam` |
| V4 | No black frames and no frozen run longer than 3 frames after 18.5 s | `checks.py frames` |
| V5 | Ten scapes, each on screen 7 s or more, in the order Iwagumi, Dutch, Mountain, Forest, Riverbed, Jungle, Blackwater, Lotus pond, Cove, Floating Islands | `checks.py timeline` against `config/timeline.json` |
| V6 | Thirteen captions appear at their configured times and the caption text passes a spell check | `checks.py captions` |
| V7 | Audio peak at or below -1 dBFS, integrated loudness between -18 and -14 LUFS | `checks.py audio` (ffmpeg ebur128) |

## 3D and realism

| # | Criterion | Check |
|---|---|---|
| R1 | Camera solve reprojection error 2 px or less (RMS over the tank corners) | `src/plate/solve_camera.py` prints RMS; `review/01_camera_overlay.jpg` |
| R2 | Parallax in every scape: tracked near and far points move by different amounts over the shot (ratio 1.15 or more) | `checks.py parallax` |
| R3 | Every scape has 8 or more neon tetras whose distance to the camera changes by 5 cm or more while the scape is on screen | `checks.py livestock` reads baked animation data |
| R4 | Red shrimp in at least two scapes; a snail in at least one | `checks.py livestock` |
| R5 | Coordinator scores each scape still 4 of 5 or higher on: scale, light match, contact shadows, water and glass optics, animal motion | Scores recorded in `review/scores.md` |
| R6 | A fresh-context reviewer lists what reveals CG in each still; each item is fixed or justified | Recorded in `review/scores.md` |
| R8 | One shoal for the whole film: every scape poses the same simulated fish, and no fish moves more than 4.5 mm in one frame anywhere in the run, so the shoal's path is unbroken through every transition | `checks.py livestock` |
| R9 | Every scape leaves the shoal open water: fewer than 2% of fish positions within 11 mm of scape geometry while it is on screen | `[swim]` line printed by every single-frame build of `src/blender/build.py` |
| R7 | Owner approves the ten design stills (Gate 2) and the final video (Gate 4) | Recorded in `progress_tracking.json` |

## Repo

| # | Criterion | Check |
|---|---|---|
| P1 | `beese54/claude-fable-aquarium` is public and `git status` is clean with nothing unpushed | `gh repo view`, `git status -sb` |
| P2 | Every task in `progress_tracking.json` is `done` with a commit hash | `checks.py ledger` |
| P3 | README explains how to reproduce from a clean clone | Read-through |
| P4 | Every third-party asset is CC0 and listed with its source in `assets/MANIFEST.md` | `checks.py manifest` compares the manifest to the files on disk |
| P5 | The committed clip has no audio stream | `checks.py footage` (ffprobe) |
