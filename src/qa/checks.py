"""Automated acceptance checks for the film (see definition_of_done.md).

  python src/qa/checks.py                         # every check, table, exit 1 if any fails
  python src/qa/checks.py container seam          # named checks only
  python src/qa/checks.py --video other.mp4 ...   # a different file than output/aquarium_dream.mp4
  python src/qa/checks.py --json                  # machine-readable results

Every check returns pass/fail, a one-line measured value and, on failure, a
one-line reason. A check that cannot run (missing input, tool error) fails with
the reason; it never raises. All inputs come from a `Ctx`, so src/qa/selftest.py
can point any check at modified copies without touching the real repo files.
"""
import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "plate")]
import paths  # noqa: E402

# --- constants from definition_of_done.md and the brief ------------------------

W, H, FPS = 1080, 1920, 30
DURATION_S = (105.0, 125.0)
MAX_BYTES = 95_000_000             # "under 95 MB": decimal MB, the stricter reading
SSIM_MIN = 0.95
SEAM_FACTOR = 1.5
SEAM_WINDOW = 30
FRAMES_AFTER_S = 18.5
# assemble.py eases the last 18 frames to black on purpose; black is only allowed there.
END_FADE_FRAMES = 18
FREEZE_MAX_RUN = 3                 # frames showing one picture; 4 or more fails
# Two frames show the same picture when under 1% of pixels changed by 3+ luma levels
# and under 50 pixels changed by 20+. Encoded repeats are not exact: x264 keeps
# refining a held picture, touching 0.03-0.3% of pixels by at most ~13 levels, so an
# exact-match test misses them. The 20-level count keeps real local motion (a fish, a
# caption fading) from reading as frozen even when grain has been encoded away.
FREEZE_FRAC_LEVELS, FREEZE_FRAC = 3, 0.01
FREEZE_BIG_LEVELS, FREEZE_BIG_PIXELS = 20, 50
# blackdetect defaults: a pixel is black below 10% of the luma range, a frame is
# black when 98% of its pixels are. Y here is limited range (16..235).
BLACK_Y = 16 + 0.10 * 219
BLACK_FRAC = 0.98
# film order
SCAPES = ["01_iwagumi", "02_dutch", "06_mountain", "03_forest", "09_riverbed",
          "07_jungle", "04_blackwater", "10_lotus", "08_cove", "05_floating"]
SCAPE_MIN_FRAMES = 210
CAPTIONS_MIN = 7
CAPTION_BAND = (1490, 1650)        # rows; assemble.py sets the lead on a baseline at 1566 and the capitals at 1634
CAPTION_DARK = 0.6                 # a pixel is ink if darker than 0.6 x the clean band's median
CAPTION_MIN_GAIN = 0.01            # ink must cover 1% more of the band than at a clean frame...
CAPTION_MIN_RATIO = 2.0            # ...and at least twice as much
TRUE_PEAK_MAX = -1.0
LUFS_RANGE = (-18.0, -14.0)
PARALLAX_MIN_PX = 4.0
PARALLAX_MIN_RATIO = 1.15
NEAR_MM = (150.0, 0.0, 150.0)      # centre of the front glass
FAR_MM = (150.0, 1822.0, 150.0)    # wall straight behind it
FISH_MIN = 8
DEPTH_MEDIAN_MIN_MM = 20.0         # the middle fish of the shoal changes its distance to the camera by this much per scape
SIDE_ON_MEDIAN_MAX_DEG = 30.0      # the owner wants the fish seen mostly from the side
BACKWARD_MAX_MM_S = 0.5            # a fish never moves backward along its own heading
STEP_MAX_MM = 4.5                  # farthest a fish may move in one frame (135 mm/s): more is a jump
DEPTH_MIN_MM = 50.0

# Spell check: the words the captions are expected to use plus a little thematic
# slack. A word not listed fails, which is the point: "stilness" must not pass.
VOCAB = set("""
every empty tank is a blank page imagine stillness abundance forest beneath the
surface river world away impossible whatever you can imagine build created with
mountains under water stream that never stops jungle in glass pond bloom hidden cove
an and of in on to into from your our it its this that what if be is are was
one new deep quiet calm still wild green garden stone stones water light life
living dream dreams create make made grow grows anything everything something
imagination idea ideas float floating island islands sky clear far beyond
""".split())
PRODUCT_TOKENS = {"Claude", "Fable", "5.2"}


@dataclass
class Result:
    name: str
    dod: str
    ok: bool
    value: str
    reason: str = ""
    seconds: float = 0.0


def run_cmd(args, cwd=None, timeout=600):
    """subprocess.run that returns (returncode, stdout, stderr); raises only FileNotFoundError."""
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return p.returncode, p.stdout, p.stderr


@dataclass
class Ctx:
    """Every input a check reads. Defaults are the real project; selftest overrides."""
    video: Path = paths.OUTPUT / "aquarium_dream.mp4"
    intro: Path = paths.PLATE / "intro_sdr.mp4"
    timeline_path: Path = paths.CONFIG / "timeline.json"
    ledger_path: Path = paths.REPO / "progress_tracking.json"
    manifest_path: Path = paths.REPO / "assets" / "MANIFEST.md"
    assets_json: Path = paths.CONFIG / "assets.json"
    assets_dir: Path = paths.ASSETS
    scapes_dir: Path = SRC / "blender" / "scapes"
    cache_dir: Path = paths.CACHE
    footage: Path = paths.REPO / "footage" / "aquarium_blank_noaudio.mp4"
    repo_dir: Path = paths.REPO
    camera: Optional[Callable] = None      # () -> object with at_frame(f).project(pts); default plate.camera.Camera
    run: Callable = run_cmd                # command runner for the repo check
    _probe: dict = field(default_factory=dict)

    def timeline(self):
        return json.loads(Path(self.timeline_path).read_text(encoding="utf-8-sig"))

    def make_camera(self):
        if self.camera is not None:
            return self.camera()
        from camera import Camera
        return Camera()


class CheckFailed(Exception):
    """Raised inside a check to fail it with a reason (missing input etc.)."""


# --- ffmpeg helpers -------------------------------------------------------------

def need_file(p, what):
    p = Path(p)
    if not p.is_file():
        raise CheckFailed(f"{what} not found: {p}")
    return p


def ffprobe(ctx, path):
    key = str(path)
    if key not in ctx._probe:
        need_file(path, "video")
        rc, out, err = run_cmd(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=60)
        if rc != 0:
            raise CheckFailed(f"ffprobe failed: {err.strip().splitlines()[-1] if err.strip() else rc}")
        ctx._probe[key] = json.loads(out)
    return ctx._probe[key]


def frame_count(ctx, path):
    key = "count:" + str(path)
    if key not in ctx._probe:
        need_file(path, "video")
        rc, out, err = run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)], timeout=120)
        if rc != 0 or not out.strip():
            raise CheckFailed(f"could not count frames: {err.strip()[:120]}")
        ctx._probe[key] = int(out.strip().split(",")[0])
    return ctx._probe[key]


def video_size(ctx, path):
    v = [s for s in ffprobe(ctx, path)["streams"] if s.get("codec_type") == "video"]
    if not v:
        raise CheckFailed("no video stream")
    return int(v[0]["width"]), int(v[0]["height"])


def frames(path, select, w, h, pix_fmt="gray", planes_y=True):
    """Yield raw frames chosen by an ffmpeg `select` expression, in order.
    Luma comes straight from the Y plane (no range conversion), so thresholds are in Y levels."""
    vf = f"select={select}" + (",extractplanes=y" if planes_y else "")
    ch = 1 if pix_fmt == "gray" else 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-vf", vf,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * ch
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(h, w) if ch == 1 else a.reshape(h, w, ch)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def handoff_frame(tl):
    """First frame after the filmed intro (499 in the current timeline)."""
    foot = [s for s in tl["segments"] if s["kind"] == "footage"]
    if not foot:
        raise CheckFailed("timeline has no footage segment")
    return int(foot[0]["end"])


# --- V1 container ---------------------------------------------------------------

def top_level_atoms(path):
    """Types of the MP4's top-level boxes, in file order."""
    out = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        pos = 0
        while pos + 8 <= end and len(out) < 64:
            f.seek(pos)
            hdr = f.read(8)
            size = int.from_bytes(hdr[:4], "big")
            kind = hdr[4:8].decode("latin-1")
            if size == 1:
                size = int.from_bytes(f.read(8), "big")
            elif size == 0:
                size = end - pos
            if size < 8:
                break
            out.append(kind)
            pos += size
    return out


def check_container(ctx):
    path = need_file(ctx.video, "video")
    info = ffprobe(ctx, path)
    v = [s for s in info["streams"] if s.get("codec_type") == "video"]
    a = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    bad = []
    if not v:
        raise CheckFailed("no video stream")
    v = v[0]
    num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0
    dur = float(info["format"].get("duration", 0))
    size = path.stat().st_size
    atoms = top_level_atoms(path)
    faststart = "moov" in atoms and "mdat" in atoms and atoms.index("moov") < atoms.index("mdat")
    if v.get("codec_name") != "h264":
        bad.append(f"codec {v.get('codec_name')} not h264")
    if v.get("pix_fmt") != "yuv420p":
        bad.append(f"pix_fmt {v.get('pix_fmt')}")
    if (v.get("width"), v.get("height")) != (W, H):
        bad.append(f"size {v.get('width')}x{v.get('height')}")
    if abs(fps - FPS) > 0.01:
        bad.append(f"{fps:.3f} fps")
    if not DURATION_S[0] <= dur <= DURATION_S[1]:
        bad.append(f"duration {dur:.2f}s")
    if size >= MAX_BYTES:
        bad.append(f"{size / 1e6:.1f} MB")
    if not faststart:
        bad.append(f"moov not before mdat ({' '.join(atoms[:6])})")
    if not a:
        bad.append("no audio stream")
        audio = "no audio"
    else:
        a = a[0]
        audio = f"{a.get('codec_name')} {a.get('channels')}ch {a.get('sample_rate')}Hz"
        if a.get("codec_name") != "aac":
            bad.append(f"audio codec {a.get('codec_name')}")
        if a.get("channels") != 2:
            bad.append(f"{a.get('channels')} audio channels")
        if str(a.get("sample_rate")) != "48000":
            bad.append(f"audio {a.get('sample_rate')} Hz")
    value = (f"{v.get('codec_name')} {v.get('pix_fmt')} {v.get('width')}x{v.get('height')} {fps:.2f}fps "
             f"{dur:.2f}s {size / 1e6:.1f}MB; {audio}; faststart={'yes' if faststart else 'no'}")
    return not bad, value, "; ".join(bad)


# --- V2 intro SSIM --------------------------------------------------------------

def check_intro_ssim(ctx):
    path = need_file(ctx.video, "video")
    intro = need_file(ctx.intro, "tonemapped intro")
    n = handoff_frame(ctx.timeline())
    have = frame_count(ctx, path)
    if have < n:
        raise CheckFailed(f"video has {have} frames, fewer than the {n}-frame intro")
    iw, ih = video_size(ctx, intro)
    # scale the film to the intro's size so a wrong-size film is still compared on content
    graph = (f"[0:v]trim=end_frame={n},setpts=PTS-STARTPTS,scale={iw}:{ih}[a];"
             f"[1:v]trim=end_frame={n},setpts=PTS-STARTPTS[b];[a][b]ssim")
    rc, out, err = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-i", str(path), "-i", str(intro),
                            "-filter_complex", graph, "-an", "-f", "null", "-"])
    m = re.findall(r"SSIM Y:([\d.]+).*?All:([\d.]+)", err)
    if rc != 0 or not m:
        raise CheckFailed(f"ffmpeg ssim failed: {err.strip().splitlines()[-1] if err.strip() else rc}")
    y, allv = float(m[-1][0]), float(m[-1][1])
    ok = allv >= SSIM_MIN
    return ok, f"mean SSIM {allv:.4f} (Y {y:.4f}) over frames 0-{n - 1}", "" if ok else f"SSIM {allv:.4f} < {SSIM_MIN}"


# --- V3 seam --------------------------------------------------------------------

def check_seam(ctx):
    path = need_file(ctx.video, "video")
    h0 = handoff_frame(ctx.timeline())            # first rendered frame
    first, last = h0 - 1 - SEAM_WINDOW, h0 + SEAM_WINDOW
    if frame_count(ctx, path) <= last:
        raise CheckFailed(f"video too short: need frames {first}-{last}")
    w, h = video_size(ctx, path)
    diffs, prev = [], None
    for img in frames(path, f"between(n\\,{first}\\,{last})", w, h, "bgr24", planes_y=False):
        cur = img.astype(np.int16)
        if prev is not None:
            diffs.append(float(np.abs(cur - prev).mean()))
        prev = cur
    if len(diffs) != 2 * SEAM_WINDOW + 1:
        raise CheckFailed(f"read {len(diffs) + 1} frames around the seam, expected {2 * SEAM_WINDOW + 2}")
    # diffs[k] is frame first+k -> first+k+1; the seam is (h0-1) -> h0
    seam = diffs[SEAM_WINDOW]
    before, after = diffs[:SEAM_WINDOW], diffs[SEAM_WINDOW + 1:]
    med = float(np.median(before + after))
    ok = seam <= SEAM_FACTOR * med
    value = (f"seam diff {seam:.2f} vs median {med:.2f} (ratio {seam / max(med, 1e-9):.2f}; "
             f"median before {np.median(before):.2f}, after {np.median(after):.2f})")
    return ok, value, "" if ok else f"seam {seam:.2f} > {SEAM_FACTOR} x median {med:.2f}"


# --- V4 black and frozen frames ---------------------------------------------------

def check_frames(ctx):
    path = need_file(ctx.video, "video")
    total = frame_count(ctx, path)
    start = int(round(FRAMES_AFTER_S * FPS))
    if total <= start:
        raise CheckFailed(f"video has only {total} frames; nothing after {FRAMES_AFTER_S}s")
    fade_from = total - END_FADE_FRAMES
    w, h = video_size(ctx, path)
    span = FREEZE_MAX_RUN + 1                      # frames in the shortest run that fails

    def same(a, b):
        d = np.abs(a - b)
        return (np.count_nonzero(d >= FREEZE_FRAC_LEVELS) < FREEZE_FRAC * d.size
                and np.count_nonzero(d >= FREEZE_BIG_LEVELS) < FREEZE_BIG_PIXELS)

    black, win, pairs, frozen_ends, n = [], [], [], [], start - 1
    for y in frames(path, f"gte(n\\,{start})", w, h, "gray"):
        n += 1
        if np.count_nonzero(y <= BLACK_Y) >= BLACK_FRAC * y.size and n < fade_from:
            black.append(n)
        y = y.astype(np.int16)
        if win:
            pairs.append(same(y, win[-1]))
        win = (win + [y])[-span:]
        # frames n-3..n are one picture: every step is a repeat AND the ends match, so
        # slow real motion that is tiny per frame but adds up over 4 frames is not a freeze
        if len(win) == span and all(pairs[-(span - 1):]) and same(win[0], win[-1]):
            frozen_ends.append(n)
    runs = []                                       # (first, last) frame of each frozen run
    for e in frozen_ends:
        if runs and e == runs[-1][1] + 1:
            runs[-1][1] = e
        else:
            runs.append([e - span + 1, e])
    longest = max(runs, key=lambda r: r[1] - r[0], default=None)
    bad = []
    if black:
        bad.append(f"{len(black)} black frames, first {black[0]}")
    if longest:
        bad.append(f"{len(runs)} frozen runs, longest {longest[1] - longest[0] + 1} frames at {longest[0]}-{longest[1]}")
    value = (f"frames {start}-{n} checked (last {END_FADE_FRAMES} = end fade); black {len(black)}; "
             f"frozen runs >{FREEZE_MAX_RUN} frames {len(runs)}; repeated-frame steps {sum(pairs)}")
    return not bad, value, "; ".join(bad)


# --- V5 timeline ----------------------------------------------------------------

def check_timeline(ctx):
    tl = ctx.timeline()
    fps = tl.get("fps", FPS)
    scapes = [s for s in tl["segments"] if s["kind"] == "scape"]
    ids = [s["id"] for s in scapes]
    lens = {s["id"]: s["end"] - s["start"] for s in scapes}
    total = int(tl["total_frames"])
    bad = []
    if ids != SCAPES:
        bad.append(f"scapes {ids} != {SCAPES}")
    short = [f"{k} {v}f" for k, v in lens.items() if v < SCAPE_MIN_FRAMES]
    if short:
        bad.append(f"shorter than {SCAPE_MIN_FRAMES} frames: {', '.join(short)}")
    secs = total / fps
    if not DURATION_S[0] <= secs <= DURATION_S[1]:
        bad.append(f"total {secs:.2f}s outside {DURATION_S[0]:.0f}-{DURATION_S[1]:.0f}s")
    try:
        have = frame_count(ctx, ctx.video)
        vid = f"video {have} frames"
        if abs(have - total) > 1:
            bad.append(f"video has {have} frames, timeline {total}")
    except CheckFailed as e:
        vid = "video unreadable"
        bad.append(str(e))
    value = f"{len(ids)} scapes, min {min(lens.values(), default=0)}f, total {total}f = {secs:.2f}s; {vid}"
    return not bad, value, "; ".join(bad)


# --- V6 captions ----------------------------------------------------------------

def spelling_errors(text):
    bad = []
    for raw in text.split():
        tok = raw.strip(".,;:!?\"'()‘’“”-")
        if not tok:
            continue
        if tok in PRODUCT_TOKENS:
            continue
        if not tok.replace("'", "").replace("’", "").isalpha():
            bad.append(tok)
        elif tok.lower() not in VOCAB:
            bad.append(tok)
    return bad


def caption_frames(tl, total):
    """(caption, middle frame, nearest caption-free frame) for each caption."""
    caps = tl["captions"]

    def covered(f, margin=2):
        return any(c["start"] - margin <= f < c["end"] + margin for c in caps)

    out = []
    for c in caps:
        mid = (c["start"] + c["end"]) // 2
        ref = None
        for k in range(1, total):
            for f in (c["start"] - k, c["end"] - 1 + k):
                if 0 <= f < total - END_FADE_FRAMES and not covered(f):
                    ref = f
                    break
            if ref is not None:
                break
        out.append((c, mid, ref))
    return out


def check_captions(ctx):
    tl = ctx.timeline()
    caps = tl.get("captions", [])
    bad = []
    if len(caps) < CAPTIONS_MIN:
        bad.append(f"{len(caps)} captions < {CAPTIONS_MIN}")
    typos = [(c["text"], spelling_errors(c["text"])) for c in caps]
    typos = [(t, e) for t, e in typos if e]
    if typos:
        bad.append("spelling: " + "; ".join(f"{e} in '{t}'" for t, e in typos))
    # on screen?
    try:
        path = need_file(ctx.video, "video")
        total = frame_count(ctx, path)
        w, h = video_size(ctx, path)
        plan = caption_frames(tl, total)
        wanted = sorted({f for _, m, r in plan for f in (m, r) if f is not None and 0 <= f < total})
        if not wanted:
            raise CheckFailed("no caption frames inside the video")
        r0, r1 = CAPTION_BAND
        sel = "+".join(f"eq(n\\,{f})" for f in wanted)
        bands = {f: img[r0:r1].astype(np.float32) for f, img in zip(wanted, frames(path, sel, w, h, "gray"))}
        missing, worst = [], None
        for c, m, r in plan:
            if m not in bands or r not in bands:
                missing.append(f"'{c['text']}' (frame {m} not in video)")
                continue
            thr = CAPTION_DARK * float(np.median(bands[r]))
            ink_m = float((bands[m] < thr).mean())
            ink_r = float((bands[r] < thr).mean())
            gain = ink_m - ink_r
            if worst is None or gain < worst[0]:
                worst = (gain, c["text"], ink_m, ink_r)
            if not (gain >= CAPTION_MIN_GAIN and ink_m >= CAPTION_MIN_RATIO * ink_r):
                missing.append(f"'{c['text']}' at {m} (ink {ink_m:.3f} vs {ink_r:.3f} at {r})")
        if missing:
            bad.append("not on screen: " + "; ".join(missing))
        screen = (f"on screen {len(plan) - len(missing)}/{len(plan)}"
                  + (f", weakest '{worst[1][:24]}' ink {worst[2]:.3f} vs {worst[3]:.3f}" if worst else ""))
    except CheckFailed as e:
        bad.append(str(e))
        screen = "on-screen check not run"
    value = f"{len(caps)} captions, {'no typos' if not typos else f'{len(typos)} with typos'}, {screen}"
    return not bad, value, "; ".join(bad)


# --- V7 audio -------------------------------------------------------------------

def check_audio(ctx):
    path = need_file(ctx.video, "video")
    if not any(s.get("codec_type") == "audio" for s in ffprobe(ctx, path)["streams"]):
        raise CheckFailed("no audio stream")
    rc, out, err = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-i", str(path), "-map", "0:a:0",
                            "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"])
    li = re.findall(r"\bI:\s+(-?[\d.]+|-inf) LUFS", err)
    pk = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", err)
    if rc != 0 or not li or not pk:
        raise CheckFailed(f"ebur128 failed: {err.strip().splitlines()[-1] if err.strip() else rc}")
    lufs, peak = float(li[-1]), float(pk[-1])
    bad = []
    if peak > TRUE_PEAK_MAX:
        bad.append(f"true peak {peak:+.1f} dBFS > {TRUE_PEAK_MAX}")
    if not LUFS_RANGE[0] <= lufs <= LUFS_RANGE[1]:
        bad.append(f"loudness {lufs:.1f} LUFS outside {LUFS_RANGE[0]:.0f}..{LUFS_RANGE[1]:.0f}")
    return not bad, f"integrated {lufs:.1f} LUFS, true peak {peak:+.1f} dBFS", "; ".join(bad)


# --- R2 parallax ----------------------------------------------------------------

def check_parallax(ctx):
    tl = ctx.timeline()
    segs = {s["id"]: s for s in tl["segments"] if s["kind"] == "scape"}
    cam0 = ctx.make_camera()
    pts = np.array([NEAR_MM, FAR_MM])
    parts, bad = [], []
    for sid in SCAPES:
        s = segs.get(sid)
        if s is None:
            parts.append(f"{sid}: not in timeline")
            bad.append(f"{sid} missing from timeline")
            continue
        px = np.array([cam0.at_frame(f).project(pts) for f in range(s["start"], s["end"])])   # (F, 2, 2)
        near, far = px[:, 0], px[:, 1]
        travel_n = float(np.linalg.norm(np.ptp(near, axis=0)))
        travel_f = float(np.linalg.norm(np.ptp(far, axis=0)))
        rel = float(np.linalg.norm(np.ptp(far - near, axis=0)))
        lo, hi = sorted((travel_n, travel_f))
        ratio = hi / lo if lo > 1e-9 else (math.inf if hi > 1e-9 else 0.0)
        parts.append(f"{sid}: rel {rel:.1f}px near {travel_n:.1f} far {travel_f:.1f} ratio {ratio:.2f}")
        if rel < PARALLAX_MIN_PX or ratio < PARALLAX_MIN_RATIO:
            bad.append(f"{sid} rel {rel:.1f}px / ratio {ratio:.2f}")
    reason = ("below " + f"{PARALLAX_MIN_PX:.0f}px or ratio {PARALLAX_MIN_RATIO}: " + ", ".join(bad)) if bad else ""
    return not bad, "; ".join(parts), reason


# --- R3, R4 livestock -----------------------------------------------------------

def _calls(src, module, cls):
    """Number of calls to module.cls(...) in Python source (comments and strings do not count)."""
    tree = ast.parse(src)
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == cls and isinstance(n.func.value, ast.Name) and n.func.value.id == module)


def check_livestock(ctx):
    from sim import school
    tl = ctx.timeline()
    segs = {s["id"]: s for s in tl["segments"] if s["kind"] == "scape"}
    try:
        centre = np.asarray(ctx.make_camera().C, dtype=float)       # the camera at rest
    except Exception:
        centre = np.array([152.0, -555.0, 143.0])
    parts, bad, shrimp, snail = [], [], [], []
    # One shoal for the whole film: every scape poses the same simulated fish, so
    # their path cannot break at a transition. Check the run itself for a jump.
    import timeline as film_timeline
    on_screen = film_timeline.visible()
    data = school.film(ctx.cache_dir)
    pos, f0 = data["pos"], int(data["frame_start"])
    step = float(np.linalg.norm(np.diff(pos, axis=0), axis=2).max())
    parts.append(f"one shoal of {pos.shape[1]}, frames {f0}-{f0 + pos.shape[0] - 1}, largest step {step:.1f} mm/frame")
    if step > STEP_MAX_MM:
        bad.append(f"a fish moves {step:.1f} mm in one frame (limit {STEP_MAX_MM})")
    if f0 + pos.shape[0] < int(tl["total_frames"]):
        bad.append("the shoal's run ends before the film does")
    # A fish goes where it points, and is mostly seen from the side.
    fwd = data["fwd"]
    along = float((np.sum(np.diff(pos, axis=0) * fwd[:-1], axis=2) * FPS).min())
    off = np.abs(fwd[..., 1]) / (np.linalg.norm(fwd[..., :2], axis=2) + 1e-9)
    side = float(np.degrees(np.arcsin(np.median(off))))
    parts.append(f"slowest motion along a fish's own heading {along:.2f} mm/s; median angle from side-on {side:.0f} deg")
    if along < -BACKWARD_MAX_MM_S:
        bad.append(f"a fish moves backward at {-along:.1f} mm/s")
    if side > SIDE_ON_MEDIAN_MAX_DEG:
        bad.append(f"median heading is {side:.0f} deg from side-on (limit {SIDE_ON_MEDIAN_MAX_DEG:.0f})")
    for sid in SCAPES:
        mod = Path(ctx.scapes_dir) / f"{sid}.py"
        if not mod.is_file():
            parts.append(f"{sid}: no module")
            bad.append(f"scape module {mod.name} does not exist yet")
            continue
        try:
            src = mod.read_text(encoding="utf-8-sig")
            n_school = _calls(src, "fish", "School")
            if _calls(src, "shrimp", "Shrimp"):
                shrimp.append(sid)
            if _calls(src, "snail", "Snail"):
                snail.append(sid)
        except SyntaxError as e:
            parts.append(f"{sid}: unparseable")
            bad.append(f"{mod.name} does not parse (line {e.lineno})")
            continue
        if not n_school:
            parts.append(f"{sid}: no shoal")
            bad.append(f"{sid}: module does not call fish.School()")
            continue
        seg = segs.get(sid)
        if seg is None:
            parts.append(f"{sid}: not in timeline")
            bad.append(f"{sid} missing from timeline")
            continue
        first, last = on_screen[sid]                   # counting the dissolves either side
        a, b = max(first - f0, 0), last + 1 - f0
        rep = school.report({"pos": pos[a:b]}, camera_mm=centre)
        med = rep["depth_change_mm_min_median_max"][1]
        parts.append(f"{sid}: depth median {med:.0f} mm")
        if med < DEPTH_MEDIAN_MIN_MM:
            bad.append(f"{sid}: the middle fish changes depth by only {med:.0f} mm (need {DEPTH_MEDIAN_MIN_MM:.0f})")
    parts.append(f"shrimp in {','.join(shrimp) or 'none'}; snail in {','.join(snail) or 'none'}")
    if len(shrimp) < 2:
        bad.append(f"shrimp in {len(shrimp)} scape modules, need 2")
    if len(snail) < 1:
        bad.append("no scape module creates a snail")
    return not bad, "; ".join(parts), "; ".join(bad)


# --- P2 ledger ------------------------------------------------------------------

def check_ledger(ctx):
    tasks = json.loads(need_file(ctx.ledger_path, "ledger").read_text(encoding="utf-8-sig"))["tasks"]
    bad = []
    for t in tasks:
        commit = str(t.get("commit") or "").strip()
        if t.get("status") != "done":
            bad.append(f"{t.get('id')} {t.get('status')}")
        elif not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            # DoD P2 asks for a commit hash, so placeholders such as "see git log" do not count
            bad.append(f"{t.get('id')} commit {repr(commit) if commit else 'empty'}")
    ok_n = len(tasks) - len(bad)
    reason = f"{len(bad)} not done with a commit hash: " + ", ".join(bad[:8]) + (" ..." if len(bad) > 8 else "")
    return not bad, f"{ok_n}/{len(tasks)} tasks done with a commit hash", reason if bad else ""


# --- P4 manifest ----------------------------------------------------------------

def check_manifest(ctx):
    cfg = json.loads(need_file(ctx.assets_json, "assets config").read_text(encoding="utf-8-sig"))["assets"]
    text = need_file(ctx.manifest_path, "manifest").read_text(encoding="utf-8-sig")
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.strip().startswith("|") or len(cells) < 2 or cells[0] in ("id", "") or set(cells[0]) <= set("-: "):
            continue
        rows[cells[0]] = cells
    cfg_ids = {a["id"] for a in cfg}
    missing = sorted(cfg_ids - rows.keys())
    extra = sorted(rows.keys() - cfg_ids)
    no_dir = [a["id"] for a in cfg if not (Path(ctx.assets_dir) / a["source"] / a["id"]).is_dir()]
    not_cc0 = [a["id"] for a in cfg if str(a.get("license", "")).upper() != "CC0"]
    not_cc0 += [k for k, c in rows.items() if not any(x.upper() == "CC0" for x in c)]
    bad = []
    if missing:
        bad.append(f"not in manifest: {', '.join(missing[:5])}")
    if extra:
        bad.append(f"in manifest but not config: {', '.join(extra[:5])}")
    if no_dir:
        bad.append(f"no folder: {', '.join(no_dir[:5])}{' ...' if len(no_dir) > 5 else ''}")
    if not_cc0:
        bad.append(f"not CC0: {', '.join(sorted(set(not_cc0))[:5])}")
    value = f"{len(cfg_ids)} config assets, {len(rows)} manifest rows, {len(cfg) - len(no_dir)} folders on disk"
    return not bad, value, "; ".join(bad)


# --- P5 footage -----------------------------------------------------------------

def check_footage(ctx):
    streams = ffprobe(ctx, need_file(ctx.footage, "committed clip"))["streams"]
    kinds = [s.get("codec_type") for s in streams]
    ok = kinds == ["video"]
    return ok, f"streams: {', '.join(kinds) or 'none'}", "" if ok else f"expected exactly one video stream, found {kinds}"


# --- P1 repo --------------------------------------------------------------------

def check_repo(ctx):
    bad, parts = [], []

    def run(args):
        try:
            return ctx.run(args, cwd=str(ctx.repo_dir), timeout=60)
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return (124, "", "timed out")

    r = run(["git", "status", "--porcelain"])
    if r is None or r[0] != 0:
        bad.append("git status failed" if r else "git not found")
    else:
        dirty = [l for l in r[1].splitlines() if l.strip()]
        parts.append(f"{len(dirty)} uncommitted")
        if dirty:
            bad.append(f"{len(dirty)} uncommitted paths, e.g. {dirty[0].strip()}")
    r = run(["git", "rev-list", "--count", "@{u}..HEAD"])
    if r is None or r[0] != 0:
        bad.append("no upstream branch to compare with" if r else "git not found")
    else:
        ahead = int(r[1].strip() or 0)
        parts.append(f"{ahead} unpushed")
        if ahead:
            bad.append(f"{ahead} commits not pushed")
    r = run(["gh", "repo", "view", "--json", "visibility,nameWithOwner"])
    if r is None:
        bad.append("gh is not installed")
    elif r[0] != 0:
        bad.append(f"gh repo view failed (offline or not logged in?): {(r[2].strip().splitlines() or ['?'])[-1][:100]}")
    else:
        info = json.loads(r[1])
        parts.append(f"{info.get('nameWithOwner')} {info.get('visibility')}")
        if info.get("visibility") != "PUBLIC":
            bad.append(f"repo visibility {info.get('visibility')}")
    return not bad, "; ".join(parts) or "nothing measured", "; ".join(bad)


# --- runner ---------------------------------------------------------------------

CHECKS = {
    "container": ("V1", check_container),
    "intro_ssim": ("V2", check_intro_ssim),
    "seam": ("V3", check_seam),
    "frames": ("V4", check_frames),
    "timeline": ("V5", check_timeline),
    "captions": ("V6", check_captions),
    "audio": ("V7", check_audio),
    "parallax": ("R2", check_parallax),
    "livestock": ("R3,R4", check_livestock),
    "ledger": ("P2", check_ledger),
    "manifest": ("P4", check_manifest),
    "footage": ("P5", check_footage),
    "repo": ("P1", check_repo),
}


def run_check(name, ctx):
    dod, fn = CHECKS[name]
    t = time.perf_counter()
    try:
        ok, value, reason = fn(ctx)
    except CheckFailed as e:
        ok, value, reason = False, "not measured", str(e)
    except FileNotFoundError as e:
        ok, value, reason = False, "not measured", f"tool or file missing: {e}"
    except Exception as e:       # a broken input must fail the check, not stop the run
        ok, value, reason = False, "not measured", f"error: {type(e).__name__}: {e}"
    return Result(name, dod, bool(ok), value, "" if ok else (reason or "failed"), round(time.perf_counter() - t, 2))


def run_all(names, ctx):
    return [run_check(n, ctx) for n in names]


def print_table(results):
    w = max(len(r.name) for r in results)
    for r in results:
        line = f"{r.name:<{w}}  {r.dod:<5}  {'PASS' if r.ok else 'FAIL'}  {r.seconds:6.1f}s  {r.value}"
        print(line)
        if not r.ok:
            print(f"{'':<{w}}  {'':<5}        {'':>7}  reason: {r.reason}")
    n = sum(r.ok for r in results)
    print(f"\n{n}/{len(results)} checks pass, {sum(r.seconds for r in results):.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checks", nargs="*", help=f"any of: {' '.join(CHECKS)}")
    ap.add_argument("--video", help="film to check (default output/aquarium_dream.mp4)")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    a = ap.parse_args(argv)
    unknown = [c for c in a.checks if c not in CHECKS]
    if unknown:
        ap.error(f"unknown check(s): {', '.join(unknown)}; choose from {', '.join(CHECKS)}")
    ctx = Ctx()
    if a.video:
        ctx.video = Path(a.video)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("warning: ffmpeg/ffprobe not on PATH; video checks will fail", file=sys.stderr)
    results = run_all(a.checks or list(CHECKS), ctx)
    if a.json:
        print(json.dumps([r.__dict__ for r in results], indent=1))
    else:
        print_table(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
