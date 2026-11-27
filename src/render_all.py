"""Render every frame the film needs, scape by scape. Safe to stop and restart:
finished frames are skipped.

  python src/render_all.py --quality draft --step 2      # rough cut (set AQUARIUM_RENDER_SET=draft first)
  python src/render_all.py --quality final               # the real thing
  python src/render_all.py --quality final --only 01_iwagumi,02_dutch

Progress is appended to <renders>/render_all.log; each scape's Blender output
goes to <renders>/<scape>.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "comp"))
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="final")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--only", help="comma-separated scape ids")
    a = ap.parse_args()
    import assemble
    needs = assemble.Film().renders_needed()
    only = set(a.only.split(",")) if a.only else None
    paths.RENDERS.mkdir(parents=True, exist_ok=True)
    log = paths.RENDERS / "render_all.log"

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    ok = True
    for scape, (first, last) in needs.items():
        if only and scape not in only:
            continue
        outdir = paths.RENDERS / scape
        frames = list(range(first, last + 1, a.step))
        left = [f for f in frames if not (outdir / f"{f:05d}.exr").is_file()]
        say(f"{scape}: {len(frames)} frames, {len(left)} to render ({a.quality}, step {a.step})")
        if not left:
            continue
        cmd = [paths.blender_exe(), "-b", "--python", str(paths.REPO / "src" / "blender" / "build.py"), "--",
               "--scape", scape, "--frames", f"{first}-{last}:{a.step}", "--outdir", str(outdir), "--quality", a.quality]
        t0 = time.time()
        with open(paths.RENDERS / f"{scape}.log", "a", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        left_after = [f for f in frames if not (outdir / f"{f:05d}.exr").is_file()]
        say(f"{scape}: blender exit {rc}, {len(left) - len(left_after)} rendered in {time.time() - t0:.0f}s, {len(left_after)} still missing")
        ok = ok and not left_after
    say("ALL DONE" if ok else "INCOMPLETE: rerun to continue")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
