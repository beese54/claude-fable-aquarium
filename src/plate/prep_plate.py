"""Plate prep (task T1.1).

From the owner's clip, produce:
  footage/aquarium_blank_noaudio.mp4   committed copy: video stream untouched, audio and metadata removed
  <work>/plate/intro_sdr.mp4           tonemapped SDR intro, near-lossless, 1080x1920
  <work>/plate/tail/%04d.png           16-bit SDR frames of the last TAIL frames
  <work>/plate/head/%04d.png           16-bit SDR frames of the first HEAD frames
  <work>/plate/lit_plate.png           temporal median of the settled lit frames (16-bit)
  <work>/plate/unlit_plate.png         temporal median of the opening frames (16-bit)
  <work>/plate/plate_stats.json        settled window, exposure drift, grain profile

Run:  python src/plate/prep_plate.py
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

# HLG bt2020 -> SDR bt709. Chosen by eye from side-by-side tests: keeps the room
# white as a phone would show it while holding detail in the lit tank.
TONEMAP = (
    "zscale=t=linear:npl=140,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=mobius:param=0.5,zscale=t=bt709:m=bt709:r=tv"
)
HEAD = 90   # frames before the hands enter
TAIL = 60   # frames at the end, after the light is on


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def probe_frames(path) -> int:
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    return int(r.stdout.strip().strip(","))


def strip_audio(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c:v", "copy",
         "-an", "-map_metadata", "-1", "-movflags", "+faststart", dst])
    r = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(dst)])
    kinds = [k.strip(",") for k in r.stdout.split()]
    assert kinds == ["video"], f"expected only a video stream, got {kinds}"


def tonemap_intro(src: Path, dst: Path):
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-an",
         "-vf", TONEMAP + ",format=yuv420p",
         "-c:v", "libx264", "-preset", "slow", "-crf", "6",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
         dst])


def extract_png16(src: Path, out_dir: Path, start: int, count: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-an",
         "-vf", f"select='between(n,{start},{start + count - 1})'," + TONEMAP.replace("r=tv", "r=pc") + ",format=rgb48le",
         "-fps_mode", "passthrough", "-start_number", str(start), out_dir / "%04d.png"])
    return sorted(out_dir.glob("*.png"))


def load_stack(files):
    return np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0 for f in files])


def luma(stack):
    # OpenCV loads BGR
    return 0.0722 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.2126 * stack[..., 2]


PLATE_FRAMES = 30   # frames at the very end used for the clean lit plate
LUT_BINS = 96


def exposure_matched(stack):
    """The phone's auto-exposure is still easing down when the clip ends, so the
    last frames differ by a global tone change (not a pure gain once tonemapped).
    The scene is static, so for each frame and channel learn a 1D curve that maps
    its values onto the final frames, then apply it.
    Returns the matched stack and each frame's mid-grey shift before matching."""
    ref = np.median(stack[-5:], axis=0)
    out = np.empty_like(stack)
    shifts = []
    edges = np.linspace(0.0, 1.0, LUT_BINS + 1)
    for t, frame in enumerate(stack):
        for c in range(3):
            src = frame[..., c].ravel()
            dst = ref[..., c].ravel()
            idx = np.clip(np.digitize(src, edges) - 1, 0, LUT_BINS - 1)
            n = np.bincount(idx, minlength=LUT_BINS)
            ok = n > 500
            xs = (np.bincount(idx, weights=src, minlength=LUT_BINS)[ok] / n[ok])
            ys = (np.bincount(idx, weights=dst, minlength=LUT_BINS)[ok] / n[ok])
            ys = np.maximum.accumulate(ys)          # keep the curve monotonic
            out[t, ..., c] = np.interp(frame[..., c], xs, ys)
            if c == 1:
                shifts.append(float(np.interp(0.5, xs, ys) - 0.5))
    return out, np.array(shifts, dtype=np.float32)


def grain_profile(stack, median):
    """Temporal noise of the luma as a function of brightness, plus spatial correlation."""
    y = luma(stack)
    ym = luma(median[None])[0]
    resid = y - ym[None]
    bins = np.linspace(0, 1, 11)
    prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ym >= lo) & (ym < hi)
        if m.sum() > 2000:
            prof.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                         "sigma": round(float(resid[:, m].std()), 5), "pixels": int(m.sum())})
    r = resid[0]
    r = r - r.mean()
    denom = float((r * r).mean()) or 1.0
    corr = {f"dx{d}": round(float((r[:, :-d] * r[:, d:]).mean() / denom), 3) for d in (1, 2, 3)}
    corr.update({f"dy{d}": round(float((r[:-d, :] * r[d:, :]).mean() / denom), 3) for d in (1, 2, 3)})
    chroma = {c: round(float((stack[..., i] - median[..., i][None]).std()), 5) for i, c in enumerate("bgr")}
    return {"luma_sigma_by_brightness": prof, "spatial_correlation": corr, "channel_sigma": chroma}


def save16(path, img):
    cv2.imwrite(str(path), np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16))


def main():
    paths.ensure_dirs()
    src = paths.SOURCE_CLIP
    if not src.is_file():
        sys.exit(f"source clip not found: {src}")
    n = probe_frames(src)
    print(f"source frames: {n}")

    committed = paths.REPO / "footage" / "aquarium_blank_noaudio.mp4"
    strip_audio(src, committed)
    print(f"wrote {committed.relative_to(paths.REPO)} ({committed.stat().st_size / 1e6:.1f} MB, video only)")

    intro = paths.PLATE / "intro_sdr.mp4"
    tonemap_intro(src, intro)
    assert probe_frames(intro) == n, "tonemapped intro lost frames"
    print(f"wrote {intro}")

    head_files = extract_png16(src, paths.PLATE / "head", 0, HEAD)
    tail_files = extract_png16(src, paths.PLATE / "tail", n - TAIL, TAIL)
    assert len(head_files) == HEAD and len(tail_files) == TAIL, (len(head_files), len(tail_files))

    tail = load_stack(tail_files)
    means = luma(tail).mean(axis=(1, 2))
    start = TAIL - PLATE_FRAMES
    settled, gains = exposure_matched(tail[start:])
    final = float(luma(settled).mean(axis=(1, 2))[-5:].mean())
    spread = float(np.ptp(luma(settled).mean(axis=(1, 2))))
    assert spread < 0.003, f"exposure matching left a spread of {spread:.4f}"
    lit = np.median(settled, axis=0)
    save16(paths.PLATE / "lit_plate.png", lit)

    head = load_stack(head_files)
    unlit = np.median(head, axis=0)
    save16(paths.PLATE / "unlit_plate.png", unlit)

    h, w = lit.shape[:2]
    assert (w, h) == (1080, 1920), (w, h)
    stats = {
        "frames": n,
        "size": [w, h],
        "tonemap_filter": TONEMAP,
        "tail_first_frame": n - TAIL,
        "tail_mean_luma": [round(float(m), 4) for m in means],
        "plate_first_frame": n - TAIL + start,
        "plate_frames": int(len(settled)),
        "plate_mean_luma": round(final, 4),
        "plate_midgrey_shift_before_matching": [round(float(g), 4) for g in gains],
        "plate_spread_after_matching": round(spread, 5),
        "exposure_note": "auto-exposure is still easing down at the end of the clip; the final frames define the plate exposure",
        "grain_lit": grain_profile(settled, lit),
        "grain_unlit": grain_profile(head, unlit),
    }
    (paths.PLATE / "plate_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"lit plate from frames {stats['plate_first_frame']}..{n - 1}, mid-grey shift {gains[0]:+.3f}..{gains[-1]:+.3f}, "
          f"spread after matching {spread:.4f}, mean luma {final:.4f}")
    print("grain lit:", stats["grain_lit"]["luma_sigma_by_brightness"])
    print("spatial corr:", stats["grain_lit"]["spatial_correlation"])

    for name, img in (("lit_plate", lit), ("unlit_plate", unlit)):
        cv2.imwrite(str(paths.PLATE / f"{name}_preview.jpg"), (np.clip(img, 0, 1) * 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("OK")


if __name__ == "__main__":
    main()
