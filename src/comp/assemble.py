"""Assemble the film from the timeline.

For every output frame this decides what is on screen (filmed intro, the held
plate, a scape, or a transition between two), composites it into the filmed
room, adds the caption, and pipes the frame to ffmpeg. Rendered frames are read
from <work>/renders/<scape id>/<frame:05d>.exr; if a frame is missing the
nearest rendered one is used, so a rough cut can be made from a sparse draft.

  python src/comp/assemble.py --out output/aquarium_dream.mp4            # full film
  python src/comp/assemble.py --out draft.mp4 --range 480-1000 --half    # a stretch, half size
  python src/comp/assemble.py --still 800 --out frame.jpg                # one frame
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "comp"))
import camera_move  # noqa: E402
import composite as comp  # noqa: E402
import paths  # noqa: E402
import timeline  # noqa: E402

W, H = 1080, 1920
FONTS = paths.REPO / "assets" / "fonts"
CAPTION_FADE = 14          # frames
CAPTION_Y = 1585           # baseline region: clean floor below the tray
CAPTION_INK = (44, 40, 38)  # warm charcoal, BGR


# --- sources -------------------------------------------------------------------

class Intro:
    """Sequential reader for the tonemapped filmed intro."""

    def __init__(self):
        self.cap = cv2.VideoCapture(str(paths.PLATE / "intro_sdr.mp4"))
        self.next = 0

    def frame(self, n):
        if n < self.next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            self.next = n
        while self.next <= n:
            ok, img = self.cap.read()
            if not ok:
                raise RuntimeError(f"intro frame {n} unreadable")
            self.next += 1
        return img.astype(np.float32) / 255.0


_index = {}


def render_linear(scape, n):
    """Scene-linear render for a scape at frame n, or the nearest one that exists. None if there are none."""
    folder = paths.RENDERS / scape
    if scape not in _index:
        _index[scape] = np.array(sorted(int(p.stem) for p in folder.glob("[0-9]*.exr"))) if folder.is_dir() else np.array([])
    have = _index[scape]
    if have.size == 0:
        return None
    k = int(have[np.argmin(np.abs(have - n))])
    img = cv2.imread(str(folder / f"{k:05d}.exr"), cv2.IMREAD_UNCHANGED)
    return None if img is None else img[..., :3]


def placeholder(label):
    img = comp.lit_plate().copy()
    m = comp.wet_matte()[..., None]
    img = img * (1 - m) + np.array([0.42, 0.40, 0.36], np.float32) * m
    cv2.putText(img, f"render pending: {label}", (150, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (1, 1, 1), 2, cv2.LINE_AA)
    return img


def scape_frame(scape, n, grain=True):
    lin = render_linear(scape, n)
    if lin is None:
        return placeholder(scape)
    return comp.composite(lin, level_mm=timeline.water_level_mm(n, scape), seed=n, add_grain=grain,
                          plate_weight=1.0 - camera_move.render_weight(n), frame=n)


_settle = {}
SETTLE_FRAMES = 45


def settle_offset(first_plate_frame):
    """The clean plate is a median of the clip's last frames, so it sits a hair
    off the very last filmed frame (the phone's exposure was still easing). The
    broad difference between the two is carried into the hold and faded out, so
    the cut from footage to plate is invisible; grain is left out of it."""
    if "d" not in _settle:
        cap = cv2.VideoCapture(str(paths.PLATE / "intro_sdr.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_plate_frame - 1)
        ok, last = cap.read()
        cap.release()
        if not ok:
            _settle["d"] = None
        else:
            d = last.astype(np.float32) / 255.0 - comp.lit_plate()
            _settle["d"] = cv2.GaussianBlur(d, (0, 0), 18)
    return _settle["d"]


def plate_frame(n, grain=True, hold_start=None):
    out = comp.lit_plate()
    if hold_start is not None and n - hold_start < SETTLE_FRAMES:
        d = settle_offset(hold_start)
        if d is not None:
            out = out + d * (1.0 - _smooth(0, SETTLE_FRAMES, n - hold_start + 1))
    if grain:
        luma = 0.0722 * out[..., 0] + 0.7152 * out[..., 1] + 0.2126 * out[..., 2]
        out = np.clip(out + comp.grain(out.shape, luma, np.random.default_rng(n)), 0, 1)
    return out


# --- transitions ----------------------------------------------------------------

VEIL = 0.20               # how far the water whitens at the middle of a transition

def _smooth(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


class Bubbles:
    """A curtain of fine bubbles rising through the tank during a transition."""

    def __init__(self, seed, frames, count=170):
        rng = np.random.default_rng(seed)
        self.frames = frames
        self.x = rng.uniform(110, 970, count)
        self.start = rng.uniform(-0.15, 0.75, count) * frames
        self.speed = rng.uniform(14, 30, count)             # px per frame
        self.size = rng.uniform(1.2, 4.2, count) ** 1.0
        self.wob = rng.uniform(0, 6.28, count)
        self.wamp = rng.uniform(2, 9, count)

    def layer(self, k, strength):
        """Additive light layer at frame k of the transition."""
        lay = np.zeros((H, W), np.float32)
        if strength <= 0:
            return lay
        age = k - self.start
        y = 1215 - age * self.speed
        x = self.x + self.wamp * np.sin(age * 0.33 + self.wob)
        for xi, yi, si, ai in zip(x, y, self.size, age):
            if ai < 0 or yi < 520 or yi > 1215:
                continue
            cv2.circle(lay, (int(round(xi * 4)), int(round(yi * 4))), int(round(si * 4)), 1.0, -1, cv2.LINE_AA, 2)
        lay = cv2.GaussianBlur(lay, (0, 0), 0.9)
        return lay * strength


def lamp_glow():
    if "glow" not in comp._cache:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.exp(-(((xx - 542) / 260) ** 2 + ((yy - 330) / 200) ** 2))
        comp._cache["glow"] = g
    return comp._cache["glow"]


def transition_mix(p):
    """Share of the incoming scape, 0..1, at position p (0..1) through a transition."""
    return _smooth(timeline.MIX[0], timeline.MIX[1], p)


def transition_frame(a_img, b_img, p, k, bubbles, n=None):
    """a_img, b_img: display frames (already composited; either may be None where
    it is not seen). p: 0..1 through the transition.

    One scape dissolves slowly into the next. The shoal is the same fish on the
    same path in both pictures, so the fish stay solid and keep their course
    while everything around them changes. A soft swell of light from the lamp
    and a thin curtain of bubbles carry the change; neither is strong enough to
    hide the fish."""
    veil = math.sin(math.pi * p) ** 1.5
    mix = transition_mix(p)
    img = b_img if a_img is None else a_img if b_img is None else a_img * (1 - mix) + b_img * mix
    m = comp.wet_matte(frame=n)[..., None]
    # light gathers in the water from the top down
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    white = np.array([1.0, 0.985, 0.96], np.float32) * (1.0 - 0.25 * yy)
    img = img * (1 - m * veil * VEIL) + white * (m * veil * VEIL)
    img = img + (bubbles.layer(k, veil * 0.38)[..., None] * m)
    # the lamp swells with it
    img = img + lamp_glow()[..., None] * (0.10 * veil) * np.array([1.0, 0.97, 0.93], np.float32)
    return np.clip(img, 0, 1)


# --- captions -------------------------------------------------------------------

_fonts = {}


def _font(italic, size, weight):
    key = (italic, size, weight)
    if key not in _fonts:
        f = ImageFont.truetype(str(FONTS / ("CormorantGaramond-Italic-VF.ttf" if italic else "CormorantGaramond-VF.ttf")), size)
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
        _fonts[key] = f
    return _fonts[key]


_caption_cache = {}


def caption_layer(text):
    """Alpha mask (float HxW) for a caption, drawn once."""
    if text not in _caption_cache:
        credit = text.lower().startswith("created with")
        font = _font(italic=not credit, size=50 if credit else 66, weight=500)
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        spacing = 3.0 if credit else 0.6
        widths = [d.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        if total > W - 120:                                   # never run off the frame
            font = _font(italic=not credit, size=int(font.size * (W - 120) / total), weight=500)
            widths = [d.textlength(ch, font=font) for ch in text]
            total = sum(widths) + spacing * (len(text) - 1)
        x = (W - total) / 2
        for ch, w in zip(text, widths):
            d.text((x, CAPTION_Y), ch, font=font, fill=255, anchor="ls")
            x += w + spacing
        _caption_cache[text] = np.asarray(img, np.float32) / 255.0
    return _caption_cache[text]


def add_captions(img, n, timeline):
    for c in timeline["captions"]:
        if not (c["start"] <= n < c["end"]):
            continue
        a = min(_smooth(0, CAPTION_FADE, n - c["start"]), _smooth(0, CAPTION_FADE, c["end"] - n))
        rise = int(round(6 * (1 - _smooth(0, CAPTION_FADE * 1.5, n - c["start"]))))
        lay = caption_layer(c["text"])
        if rise:
            lay = np.roll(lay, rise, axis=0)
        alpha = (lay * a * 0.90)[..., None]
        img = img * (1 - alpha) + (np.array(CAPTION_INK, np.float32) / 255.0) * alpha
    return img


# --- the film -------------------------------------------------------------------

class Film:
    def __init__(self):
        self.tl = paths.timeline()
        self.intro = Intro()
        self.bubbles = {}
        segs = self.tl["segments"]
        self.prev_scape = {}
        last = None
        for s in segs:
            if s["kind"] == "transition":
                self.prev_scape[s["id"]] = last
            if s["kind"] == "scape":
                last = s["id"]
        self.last_scape = last

    def segment(self, n):
        for s in self.tl["segments"]:
            if s["start"] <= n < s["end"]:
                return s
        raise ValueError(f"frame {n} is outside the timeline")

    def frame(self, n):
        s = self.segment(n)
        kind = s["kind"]
        if kind == "footage":
            img = self.intro.frame(n)
        elif kind == "plate":
            img = plate_frame(n, hold_start=s["start"])
        elif kind == "scape":
            img = scape_frame(s["id"], n)
        elif kind == "outro":
            img = scape_frame(self.last_scape, n)
        elif kind == "transition":
            length = s["end"] - s["start"]
            k = n - s["start"]
            p = (k + 0.5) / length
            a_id = self.prev_scape[s["id"]]
            if "rise" in s:
                # The awakening is not a blend: water rises in the scene itself and
                # the compositor follows the waterline. Before it starts, the lamp swells.
                img = plate_frame(n) if n < s["rise"][0] else scape_frame(s["to"], n)
                pulse = timeline.lamp_pulse(n)
                if pulse > 0:
                    img = np.clip(img + lamp_glow()[..., None] * (0.22 * pulse) * np.array([1.0, 0.97, 0.93], np.float32), 0, 1)
            else:
                mix = transition_mix(p)
                a = None if mix >= 1 else plate_frame(n) if a_id is None else scape_frame(a_id, n)
                b = None if mix <= 0 else scape_frame(s["to"], n)
                if s["id"] not in self.bubbles:
                    self.bubbles[s["id"]] = Bubbles(seed=s["start"], frames=length)
                img = transition_frame(a, b, p, k, self.bubbles[s["id"]], n)
        else:
            raise ValueError(kind)
        img = add_captions(img, n, self.tl)
        fade = _smooth(self.tl["total_frames"], self.tl["total_frames"] - 18, n)   # ease to black at the very end
        return np.clip(img * fade, 0, 1)

    def renders_needed(self):
        """Frame ranges each scape must be rendered for, including transition overlap."""
        return timeline.visible()


def encode(film, out, first, last, half=False, crf=17, audio=None, preset="slow"):
    w, h = (W // 2, H // 2) if half else (W, H)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(film.tl["fps"]), "-i", "-"]
    if audio and Path(audio).is_file():
        cmd += ["-ss", f"{first / film.tl['fps']:.4f}", "-i", str(audio)]
    # The finished file must stay under 95 MB to live in the repo: cap the video
    # rate so that even a film that sits at the cap all the way through fits.
    seconds = film.tl["total_frames"] / film.tl["fps"]
    rate = int(min(9000, 0.93 * 95 * 8192 / seconds - 200))
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-maxrate", f"{rate}k", "-bufsize", f"{2 * rate}k",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart"]
    if audio and Path(audio).is_file():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest"]
    cmd += [str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(first, last + 1):
        img = film.frame(n)
        if half:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        proc.stdin.write(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
        if (n - first) % 60 == 0:
            print(f"  frame {n} ({n - first + 1}/{last - first + 1})", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--range", help="first-last frames, inclusive; default the whole film")
    ap.add_argument("--half", action="store_true", help="half-size output for quick drafts")
    ap.add_argument("--still", type=int, help="write just this frame as an image to --out")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--needs", action="store_true", help="print the frame ranges each scape must be rendered for")
    a = ap.parse_args()
    film = Film()
    if a.needs:
        for k, (f0, f1) in film.renders_needed().items():
            print(f"{k}: {f0}-{f1} ({f1 - f0 + 1} frames)")
        return
    if not a.out:
        sys.exit("--out is required")
    if a.still is not None:
        img = film.frame(a.still)
        cv2.imwrite(a.out, np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"wrote {a.out}")
        return
    first, last = (0, film.tl["total_frames"] - 1)
    if a.range:
        first, last = (int(v) for v in a.range.split("-"))
    audio = None if a.no_audio else paths.WORK / "audio" / "soundtrack.wav"
    encode(film, a.out, first, last, half=a.half, crf=a.crf, audio=audio, preset=a.preset)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
