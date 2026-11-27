"""Single source of truth for where things live.

Works both in system Python and inside Blender's bundled Python (stdlib only).
"""
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_cfg = json.loads((REPO / "config" / "paths.json").read_text(encoding="utf-8"))


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(p))


WORK = Path(os.environ.get("AQUARIUM_WORK_DIR") or _expand(_cfg["work_dir"]))
PLATE = WORK / "plate"          # tonemapped frames, clean plates, masks
ASSETS = WORK / "assets"        # downloaded CC0 assets
# Frame sequences per scape. AQUARIUM_RENDER_SET=draft keeps a rough cut's frames
# apart from the final ones (renders_draft/ beside renders/).
_set = os.environ.get("AQUARIUM_RENDER_SET")
RENDERS = WORK / (f"renders_{_set}" if _set else "renders")
CACHE = WORK / "cache"          # baked animation, misc
REVIEW = REPO / "review"        # stills shown to the owner (committed)
OUTPUT = REPO / "output"
CONFIG = REPO / "config"
SOURCE_CLIP = REPO / _cfg["source_clip"]


def ensure_dirs() -> None:
    for d in (WORK, PLATE, ASSETS, RENDERS, CACHE, REVIEW, OUTPUT):
        d.mkdir(parents=True, exist_ok=True)


def blender_exe() -> str:
    for cand in _cfg["blender_candidates"]:
        p = _expand(cand)
        if p.is_file():
            return str(p)
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError("Blender 4.5 not found; run init.sh")


def timeline() -> dict:
    return json.loads((CONFIG / "timeline.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    ensure_dirs()
    print("repo  ", REPO)
    print("work  ", WORK)
    try:
        print("blender", blender_exe())
    except FileNotFoundError as e:
        print("blender", e)
