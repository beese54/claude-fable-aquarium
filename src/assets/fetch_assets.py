"""Download the CC0 assets listed in config/assets.json into paths.ASSETS.

Layout: paths.ASSETS/<source>/<id>/<file path from the config>
  - Poly Haven models keep the relative paths their .gltf expects (textures/...).
  - ambientCG zips are kept (so re-runs can skip them) and the listed maps are
    extracted next to them.
  - Every asset gets thumb.png.

Re-runnable: a file that already exists with the expected size is skipped
(thumbnails, which have no size in the APIs, are skipped if non-empty).
Failed downloads are retried. Exits 1 if anything is missing at the end.

Usage:  python src/assets/fetch_assets.py [--only id1,id2] [--workers N]
Standard library plus paths only.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

UA = "aquarium-dream-fetch/1.0"
RETRIES = 4
CHUNK = 1 << 20


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def is_done(dest: Path, spec: dict) -> bool:
    if not dest.is_file():
        return False
    size = spec.get("size")
    if size is None:
        return dest.stat().st_size > 0
    return dest.stat().st_size == size


def download(url: str, dest: Path, spec: dict) -> str:
    """Download url to dest via a .part file. Returns 'skip' or 'ok'; raises on final failure."""
    if is_done(dest, spec):
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
            got = part.stat().st_size
            if spec.get("size") is not None and got != spec["size"]:
                raise IOError(f"size {got} != expected {spec['size']}")
            if got == 0:
                raise IOError("empty file")
            if spec.get("md5") and md5sum(part) != spec["md5"]:
                raise IOError("md5 mismatch")
            part.replace(dest)
            return "ok"
        except (urllib.error.URLError, OSError, IOError) as e:
            last = e
            if part.exists():
                part.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"{url}: {last}")


def extract(zip_path: Path, patterns: list) -> list:
    """Extract members whose names end with one of patterns. Returns extracted paths."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name  # flatten; never trust member dirs
            if not name or not any(name.endswith(p) for p in patterns):
                continue
            dest = zip_path.parent / name
            if not (dest.is_file() and dest.stat().st_size == info.file_size):
                with z.open(info) as src, dest.open("wb") as dst:
                    while True:
                        b = src.read(CHUNK)
                        if not b:
                            break
                        dst.write(b)
            out.append(dest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated asset ids")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = json.loads((paths.CONFIG / "assets.json").read_text(encoding="utf-8"))
    assets = cfg["assets"]
    if args.only:
        keep = set(args.only.split(","))
        assets = [a for a in assets if a["id"] in keep]

    jobs = []
    for a in assets:
        root = paths.ASSETS / a["source"] / a["id"]
        for spec in a["files"]:
            dest = (root / spec["path"]).resolve()
            if root.resolve() not in dest.parents:
                print(f"REFUSED path outside asset dir: {spec['path']}")
                return 1
            jobs.append((a, spec, dest))

    stats = {"ok": 0, "skip": 0}
    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, spec["url"], dest, spec): (a, spec, dest) for a, spec, dest in jobs}
        for fut in cf.as_completed(futs):
            a, spec, dest = futs[fut]
            try:
                res = fut.result()
                stats[res] += 1
                if res == "ok":
                    print(f"  got  {a['source']}/{a['id']}/{spec['path']}")
            except Exception as e:  # noqa: BLE001
                failures.append(str(e))
                print(f"  FAIL {a['id']}/{spec['path']}: {e}")

    # extract zips (ambientCG)
    for a, spec, dest in jobs:
        if spec.get("extract") and dest.is_file():
            try:
                got = extract(dest, spec["extract"])
                if not got:
                    failures.append(f"{a['id']}: nothing matched in {dest.name}")
            except zipfile.BadZipFile as e:
                failures.append(f"{a['id']}: bad zip {e}")

    # final check
    missing = [str(d) for a, s, d in jobs if not is_done(d, s)]
    total = sum(d.stat().st_size for a, s, d in jobs if d.is_file())
    extracted = 0
    for a, s, d in jobs:
        if s.get("extract") and d.is_file():
            for p in d.parent.iterdir():
                if p != d and p.name != "thumb.png" and p.is_file():
                    extracted += p.stat().st_size
    print(f"\n{len(assets)} assets, {len(jobs)} files: {stats['ok']} downloaded, {stats['skip']} already present")
    print(f"on disk: {total / 1e6:.1f} MB downloaded files + {extracted / 1e6:.1f} MB extracted maps")
    print(f"into {paths.ASSETS}")
    if missing or failures:
        for m in missing:
            print("MISSING", m)
        for f in failures:
            print("ERROR", f)
        return 1
    print("all files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
