"""Prove every check in checks.py can both pass and fail.

  python src/qa/selftest.py            # build fixtures once (reused next time), run all cases
  python src/qa/selftest.py --rebuild  # rebuild the synthetic videos

Video and audio checks run on synthetic films built under <work>/qa_scratch: a
good one (the real tonemapped intro, then a held lit plate with fresh grain every
frame, the real captions at their frames, the 18-frame fade to black, and a
-16 LUFS stereo track) and variants that each break one thing. The other checks
run against modified copies of their inputs in the same folder, or with a stub
camera / command runner. Nothing in the repo is modified.
"""
import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

QA = Path(__file__).resolve().parent
sys.path[:0] = [str(QA), str(QA.parent), str(QA.parent / "comp"), str(QA.parent / "plate")]
import checks  # noqa: E402
import paths  # noqa: E402

SCRATCH = paths.WORK / "qa_scratch"
TL = paths.timeline()
HANDOFF = checks.handoff_frame(TL)          # 499
TOTAL = TL["total_frames"]
SR = 48000
GRAIN = 2.0           # per-channel noise sigma, 8-bit levels; similar frame-to-frame change to the filmed intro
XFADE = 45            # frames to ease from the last filmed frame to the clean plate (see build_film)
CAPTION_INK = np.array((44, 40, 38), np.float32)   # assemble.CAPTION_INK, BGR


# --- fixture building -----------------------------------------------------------

def ffmpeg(*args):
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", *map(str, args)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-400:])


class Sources:
    """Pictures the synthetic films are made of, loaded once."""

    def __init__(self):
        lp = cv2.imread(str(paths.PLATE / "lit_plate.png"), cv2.IMREAD_UNCHANGED)
        self.plate = np.clip(lp.astype(np.float32) / 257.0 + 0.5, 0, 255).astype(np.uint8) if lp.dtype == np.uint16 else lp
        rng = np.random.default_rng(1)
        # A bank of grain fields, split into positive and negative parts for saturating adds.
        noise = [np.round(rng.normal(0, GRAIN, self.plate.shape)) for _ in range(12)]
        self.grain_pos = [np.clip(z, 0, 255).astype(np.uint8) for z in noise]
        self.grain_neg = [np.clip(-z, 0, 255).astype(np.uint8) for z in noise]
        self._intro = None
        import assemble          # the film's own caption renderer, so the ink matches the real film
        self.caption_layer = assemble.caption_layer

    def intro(self):
        if self._intro is None:
            cap = cv2.VideoCapture(str(paths.PLATE / "intro_sdr.mp4"))
            out = []
            while len(out) < HANDOFF:
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"intro_sdr.mp4 ended at frame {len(out)}")
                out.append(img)
            self._intro = out
        return self._intro


def held(src, n, from_intro):
    """A frame after the intro: the lit plate with grain, drifting a fraction of a pixel
    per frame like the film's slow camera move. The drift matters: at the film's bitrate
    x264 throws most grain away, so a perfectly still grained plate decodes as a run of
    repeated frames, which is exactly what V4 must catch.
    Right after real footage it eases from the last filmed frame to the plate: the lit
    plate alone differs from frame 498 by about 2.2 levels, over twice the intro's own
    frame-to-frame change, so a hard cut to it would (rightly) fail V3."""
    k = n - HANDOFF
    if from_intro and 0 <= k < XFADE:
        s = (k + 1) / (XFADE + 1)
        s = s * s * (3 - 2 * s)
        base = np.clip(src.intro()[-1].astype(np.float32) * (1 - s) + src.plate.astype(np.float32) * s + 0.5, 0, 255).astype(np.uint8)
    else:
        base = src.plate
    ease = min(1.0, max(0.0, k / 60.0)) ** 2 if from_intro else 1.0
    # a slow circle: constant speed (0.25 px/frame), so there is no instant where it stops
    dx = 8.0 * math.sin(2 * math.pi * n / 200.0) * ease
    dy = 8.0 * math.cos(2 * math.pi * n / 200.0) * ease
    img = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (base.shape[1], base.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g = (n * 7) % len(src.grain_pos)
    return cv2.subtract(cv2.add(img, src.grain_pos[g]), src.grain_neg[g])      # saturating, fast


def draw_captions(img, n, caps, src):
    for c in caps:
        if not c["start"] <= n < c["end"]:
            continue
        a = min(1.0, (n - c["start"]) / 14.0, (c["end"] - n) / 14.0)      # assemble's 14-frame fade
        lay = src.caption_layer(c["text"])
        rows = np.nonzero(lay.max(axis=1) > 0)[0]
        y0, y1 = rows[0], rows[-1] + 1
        alpha = (lay[y0:y1] * a * 0.90)[..., None]
        img[y0:y1] = np.clip(img[y0:y1] * (1 - alpha) + CAPTION_INK * alpha + 0.5, 0, 255).astype(np.uint8)
    return img


def build_film(out, n_frames, src, intro="real", after_gain=1.0, captions=None, black=None, freeze=None,
               audio=None, fade_end=True):
    """Encode a synthetic film like assemble.encode does (libx264, yuv420p, faststart, AAC)."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1080x1920", "-r", "30", "-i", "-"]
    if audio:
        cmd += ["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(audio), "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "17", "-maxrate", "8M", "-bufsize", "16M",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2", "-shortest"]
    cmd += [str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frozen = None
    for n in range(n_frames):
        if freeze and freeze[0] < n < freeze[1] and frozen is not None:
            proc.stdin.write(frozen)
            continue
        if n < HANDOFF and intro:
            img = src.intro()[n]
            if intro == "flip":                          # the clip rotated 180 degrees: a wrong rotation
                img = img[::-1, ::-1]
        else:
            img = held(src, n, bool(intro))
        img = img.copy()
        if n >= HANDOFF and after_gain != 1.0:
            img = np.clip(img.astype(np.float32) * after_gain, 0, 255).astype(np.uint8)
        if captions:
            img = draw_captions(img, n, captions, src)
        if fade_end and n >= n_frames - 18:              # assemble.py eases the last 18 frames to black
            t = min(1.0, (n_frames - n) / 18.0)
            img = (img.astype(np.float32) * t * t * (3 - 2 * t)).astype(np.uint8)
        if black and black[0] <= n < black[1]:
            img[:] = 0
        buf = np.ascontiguousarray(img).tobytes()
        if freeze and n == freeze[0]:
            frozen = buf
        proc.stdin.write(buf)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"encode of {out} failed")


def loudness(raw):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw),
                        "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"], capture_output=True, text=True)
    import re
    return float(re.findall(r"\bI:\s+(-?[\d.]+) LUFS", p.stderr)[-1]), float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", p.stderr)[-1])


def build_audio(folder):
    """Sine-and-noise stereo tracks: good at -16 LUFS, one 4 dB too loud, one with peaks near 0 dBFS."""
    dur = TOTAL / 30 + 0.5
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (t.size, 2))
    k = np.ones(48) / 48                                   # soft low-pass so the noise sounds like water, not hiss
    noise = np.stack([np.convolve(noise[:, i], k, mode="same") for i in range(2)], axis=1)
    tone = np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t))
    sig = (np.stack([tone, np.roll(tone, 37)], axis=1) * 0.5 + noise * 2.0).astype(np.float32)
    probe = folder / "audio_probe.f32"
    sig.tofile(probe)
    lufs, _ = loudness(probe)
    good = (sig * 10 ** ((-16.0 - lufs) / 20)).astype(np.float32)
    good.tofile(folder / "audio_good.f32")
    (good * 10 ** (4 / 20)).astype(np.float32).tofile(folder / "audio_loud.f32")
    peaky = good.copy()
    for s in range(5, int(dur) - 5, 10):                   # 60 ms 1 kHz bursts near full scale
        i = s * SR
        peaky[i:i + 2880] = (0.97 * np.sin(2 * np.pi * 1000 * t[:2880]))[:, None]
    peaky.tofile(folder / "audio_peak.f32")
    probe.unlink()


def build_fixtures(rebuild=False):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    f = {k: SCRATCH / f"{k}.mp4" for k in ("good", "bad_caption", "bad_seam", "bad_intro", "bad_black", "bad_freeze",
                                         "edge_freeze3", "bad_size", "bad_nofaststart", "bad_loud", "bad_peak",
                                         "bad_framecount")}
    if not rebuild and all(p.is_file() for p in f.values()):
        return f
    t0 = time.perf_counter()
    build_audio(SCRATCH)
    src = Sources()
    caps = TL["captions"]
    a_good = SCRATCH / "audio_good.f32"
    print("building fixtures in", SCRATCH, flush=True)
    build_film(f["good"], TOTAL, src, captions=caps, audio=a_good)
    # one caption left out; intro replaced by plate frames since only captions are checked
    build_film(f["bad_caption"], TOTAL, src, intro=None,
               captions=[c for c in caps if not c["text"].startswith("Imagine a forest")])
    build_film(f["bad_seam"], HANDOFF + 45, src, after_gain=1.05, fade_end=False)        # 5% exposure jump at the hand-off
    build_film(f["bad_intro"], HANDOFF + 20, src, intro="flip", fade_end=False)
    build_film(f["bad_black"], 900, src, intro=None, black=(700, 710))
    build_film(f["bad_freeze"], 900, src, intro=None, freeze=(700, 710))                 # one picture for 10 frames
    build_film(f["edge_freeze3"], 900, src, intro=None, freeze=(700, 703))               # 3 frames: allowed
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=75", "-f", "f32le", "-ar", SR, "-ac", 2, "-i", a_good,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ar", SR, "-ac", 2, "-shortest", "-movflags", "+faststart", f["bad_size"])
    ffmpeg("-i", f["good"], "-c", "copy", f["bad_nofaststart"])                             # no +faststart
    for name, raw in (("bad_loud", "audio_loud.f32"), ("bad_peak", "audio_peak.f32")):
        ffmpeg("-i", f["good"], "-f", "f32le", "-ar", SR, "-ac", 2, "-i", SCRATCH / raw, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", SR, "-ac", 2, "-shortest", "-movflags", "+faststart",
               f[name])
    ffmpeg("-i", f["good"], "-c", "copy", "-frames:v", TOTAL - 60, "-movflags", "+faststart", f["bad_framecount"])
    print(f"fixtures built in {time.perf_counter() - t0:.0f}s", flush=True)
    return f


# --- non-video fixtures -----------------------------------------------------------

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def timeline_variant(name, edit):
    tl = copy.deepcopy(TL)
    edit(tl)
    return write_json(SCRATCH / "config" / f"timeline_{name}.json", tl)


def _seg(tl, sid):
    return next(s for s in tl["segments"] if s["id"] == sid)


def swap_23(tl):
    a, b = _seg(tl, "02_dutch"), _seg(tl, "03_forest")
    a["id"], b["id"] = "03_forest", "02_dutch"


def livestock_fixture(name, modules, fish_axis=1, drop=(), short=()):
    """Scape modules (id -> source) and school caches for every scape in the timeline."""
    root = SCRATCH / "livestock" / name
    (root / "scapes").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    for p in list((root / "scapes").glob("*.py")) + list((root / "cache").glob("*.npz")):
        p.unlink()
    for sid in checks.SCAPES:
        if sid in drop:
            continue
        (root / "scapes" / f"{sid}.py").write_text(modules.get(sid, ""), encoding="utf-8")
        s = _seg(TL, sid)
        f0, f1 = s["start"] - 30, s["end"] + 30
        if sid in short:
            f1 = s["end"] - 60
        t = np.arange(f1 - f0)[:, None]
        n = 10
        pos = np.zeros((f1 - f0, n, 3), np.float32) + np.array([150, 150, 150], np.float32)
        # each fish swims +-40 mm along one axis: along Y it changes depth by ~80 mm, along X barely at all
        pos[:, :, fish_axis] += 40 * np.sin(2 * np.pi * t / 200 + np.arange(n)[None, :])
        np.savez(root / "cache" / f"school_{sid}_000000000000.npz", pos=pos, frame_start=np.array(f0))
    return root


MOD = "import fish, shrimp, snail\nSCAPE = 'x'\nschool = fish.School(SCAPE, 10, 0, 1, box=None)\n"


class StubCamera:
    """The real solved camera with its move scaled by `k` (0 = locked off)."""

    def __init__(self, k):
        from camera import Camera
        self.cam, self.k = Camera(), k
        self.C = self.cam.C

    def at_frame(self, f):
        import camera_move
        c = copy.copy(self.cam)
        off = camera_move.offset(f) * self.k
        if off.any():
            c.R, c.C = camera_move.moved(self.cam.R, self.cam.C, off)
            c.t = -c.R @ c.C
        return c


def fake_runner(status="", ahead="0", vis="PUBLIC", gh=True, upstream=True):
    def run(args, cwd=None, timeout=None):
        if args[0] == "gh":
            if not gh:
                raise FileNotFoundError("gh")
            return 0, json.dumps({"visibility": vis, "nameWithOwner": "x/y"}), ""
        if args[:2] == ["git", "status"]:
            return 0, status, ""
        if args[:2] == ["git", "rev-list"]:
            return (0, ahead + "\n", "") if upstream else (128, "", "fatal: no upstream configured")
        raise AssertionError(args)
    return run


def manifest_fixture(name, rows, folders, cfg_ids=("rock_a", "moss_b")):
    root = SCRATCH / "manifest" / name
    for sub in ("assets",):
        (root / sub).mkdir(parents=True, exist_ok=True)
    src = {"rock_a": "polyhaven", "moss_b": "ambientcg"}
    write_json(root / "assets.json", {"assets": [{"id": i, "source": src[i], "license": "CC0"} for i in cfg_ids]})
    table = "| id | name | licence | source page |\n|---|---|---|---|\n" + "".join(
        f"| {i} | {i} | {lic} | https://example.invalid/{i} |\n" for i, lic in rows)
    (root / "MANIFEST.md").write_text("# Manifest\n\n" + table, encoding="utf-8")
    import shutil
    if (root / "files").exists():
        shutil.rmtree(root / "files")
    for i in folders:
        (root / "files" / src[i] / i).mkdir(parents=True, exist_ok=True)
    return dict(assets_json=root / "assets.json", manifest_path=root / "MANIFEST.md", assets_dir=root / "files")


def ledger_fixture(name, edit):
    tasks = [{"id": f"T{i}", "status": "done", "commit": f"{i:07x}abc"} for i in range(5)]
    edit(tasks)
    return write_json(SCRATCH / "ledger" / f"{name}.json", {"tasks": tasks})


# --- cases ----------------------------------------------------------------------

def cases(f):
    """(check, label, expect_pass, ctx overrides)."""
    tv = timeline_variant
    caps = TL["captions"]
    good_mods = {"01_iwagumi": MOD + "s = shrimp.Shrimp('a')\n", "02_dutch": MOD + "s = shrimp.Shrimp('b')\n",
                 "03_forest": MOD + "n = snail.Snail('n')\n", "04_blackwater": MOD, "05_floating": MOD}
    one_shrimp = dict(good_mods, **{"02_dutch": MOD + "# s = shrimp.Shrimp('b')  commented out\n"})
    no_snail = dict(good_mods, **{"03_forest": MOD})
    ls = lambda root: dict(scapes_dir=root / "scapes", cache_dir=root / "cache")   # noqa: E731
    return [
        ("container", "good film", True, dict(video=f["good"])),
        ("container", "720x1280", False, dict(video=f["bad_size"])),
        ("container", "moov after mdat", False, dict(video=f["bad_nofaststart"])),
        ("container", "file missing", False, dict(video=SCRATCH / "nope.mp4")),
        ("intro_ssim", "good film", True, dict(video=f["good"])),
        ("intro_ssim", "intro rotated 180", False, dict(video=f["bad_intro"])),
        ("seam", "good film", True, dict(video=f["good"])),
        ("seam", "5% exposure jump at 499", False, dict(video=f["bad_seam"])),
        ("frames", "good film (incl. end fade)", True, dict(video=f["good"])),
        ("frames", "3-frame repeat (allowed)", True, dict(video=f["edge_freeze3"])),
        ("frames", "10 black frames", False, dict(video=f["bad_black"])),
        ("frames", "10 frozen frames", False, dict(video=f["bad_freeze"])),
        ("timeline", "real timeline + good film", True, dict(video=f["good"])),
        ("timeline", "02/03 swapped", False, dict(video=f["good"], timeline_path=tv("swap", swap_23))),
        ("timeline", "02 only 200 frames", False, dict(video=f["good"], timeline_path=tv("short", lambda t:
            _seg(t, "02_dutch").__setitem__("end", 1190)))),
        ("timeline", "05 missing", False, dict(video=f["good"], timeline_path=tv("four", lambda t:
            t["segments"].remove(_seg(t, "05_floating"))))),
        ("timeline", "total 90 s", False, dict(video=f["good"], timeline_path=tv("long", lambda t:
            t.__setitem__("total_frames", 2700)))),
        ("timeline", "video 60 frames short", False, dict(video=f["bad_framecount"])),
        ("captions", "real captions + good film", True, dict(video=f["good"])),
        ("captions", "typo 'stilness'", False, dict(video=f["good"], timeline_path=tv("typo", lambda t:
            t["captions"][1].__setitem__("text", "Imagine stilness.")))),
        ("captions", "only 6 captions", False, dict(video=f["good"], timeline_path=tv("six", lambda t:
            t.__setitem__("captions", t["captions"][:6])))),
        ("captions", "one caption not drawn", False, dict(video=f["bad_caption"])),
        ("audio", "good film (-16 LUFS)", True, dict(video=f["good"])),
        ("audio", "4 dB too loud", False, dict(video=f["bad_loud"])),
        ("audio", "peaks near 0 dBFS", False, dict(video=f["bad_peak"])),
        ("audio", "no audio stream", False, dict(video=f["bad_black"])),
        ("parallax", "real camera move", True, {}),
        ("parallax", "locked-off camera", False, dict(camera=lambda: StubCamera(0.0))),
        ("parallax", "move scaled to 5%", False, dict(camera=lambda: StubCamera(0.05))),
        ("parallax", "05 missing from timeline", False, dict(timeline_path=tv("four", lambda t:
            t["segments"].remove(_seg(t, "05_floating"))))),
        ("livestock", "fish swim in depth, shrimp x2, snail", True, ls(livestock_fixture("good", good_mods))),
        ("livestock", "fish swim sideways only", False, ls(livestock_fixture("sideways", good_mods, fish_axis=0))),
        ("livestock", "05 module not written yet", False, ls(livestock_fixture("nomod", good_mods, drop=("05_floating",)))),
        ("livestock", "cache ends before scape", False, ls(livestock_fixture("short", good_mods, short=("02_dutch",)))),
        ("livestock", "shrimp in one module only", False, ls(livestock_fixture("oneshrimp", one_shrimp))),
        ("livestock", "no snail", False, ls(livestock_fixture("nosnail", no_snail))),
        ("ledger", "all done with hashes", True, dict(ledger_path=ledger_fixture("good", lambda t: None))),
        ("ledger", "one in progress", False, dict(ledger_path=ledger_fixture("wip", lambda t: t[2].update(status="in_progress")))),
        ("ledger", "commit empty", False, dict(ledger_path=ledger_fixture("empty", lambda t: t[3].update(commit=None)))),
        ("ledger", "commit 'see git log'", False, dict(ledger_path=ledger_fixture("placeholder", lambda t: t[0].update(commit="see git log")))),
        ("manifest", "config, manifest, folders agree", True,
         manifest_fixture("good", [("rock_a", "CC0"), ("moss_b", "CC0")], ["rock_a", "moss_b"])),
        ("manifest", "manifest lists an extra asset", False,
         manifest_fixture("extra", [("rock_a", "CC0"), ("moss_b", "CC0"), ("tree_c", "CC0")], ["rock_a", "moss_b"])),
        ("manifest", "asset not in manifest", False, manifest_fixture("missing", [("rock_a", "CC0")], ["rock_a", "moss_b"])),
        ("manifest", "asset folder missing", False,
         manifest_fixture("nodir", [("rock_a", "CC0"), ("moss_b", "CC0")], ["rock_a"])),
        ("manifest", "licence CC-BY", False,
         manifest_fixture("ccby", [("rock_a", "CC0"), ("moss_b", "CC-BY 4.0")], ["rock_a", "moss_b"])),
        ("footage", "committed clip (real)", True, {}),
        ("footage", "clip with an audio stream", False, dict(footage=f["good"])),
        ("repo", "clean, pushed, public", True, dict(run=fake_runner())),
        ("repo", "uncommitted file", False, dict(run=fake_runner(status="?? new.py\n"))),
        ("repo", "2 commits unpushed", False, dict(run=fake_runner(ahead="2"))),
        ("repo", "no upstream", False, dict(run=fake_runner(upstream=False))),
        ("repo", "private repo", False, dict(run=fake_runner(vis="PRIVATE"))),
        ("repo", "gh unavailable", False, dict(run=fake_runner(gh=False))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", nargs="*", help="limit to these checks")
    a = ap.parse_args()
    f = build_fixtures(a.rebuild)
    summary = {}
    t0 = time.perf_counter()
    for name, label, expect, over in cases(f):
        if a.only and name not in a.only:
            continue
        ctx = checks.Ctx(**over)
        r = checks.run_check(name, ctx)
        right = r.ok == expect
        s = summary.setdefault(name, {"good": [], "bad": []})
        s["good" if expect else "bad"].append(right)
        print(f"{'ok ' if right else 'XX '} {name:<10} {'good' if expect else 'bad '} {label:<36} -> "
              f"{'PASS' if r.ok else 'FAIL'} {r.seconds:5.1f}s  {r.value[:110]}")
        if not r.ok:
            print(f"{'':>53}reason: {r.reason[:150]}")
    print(f"\n{'check':<11}passes on good  fails on bad   ({time.perf_counter() - t0:.0f}s)")
    for name, s in summary.items():
        print(f"{name:<11}{'yes' if s['good'] and all(s['good']) else 'NO':<16}{'yes' if s['bad'] and all(s['bad']) else 'NO'}")
    ok = all(all(s["good"]) and s["good"] and all(s["bad"]) and s["bad"] for s in summary.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
