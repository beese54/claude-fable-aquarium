#!/usr/bin/env bash
# Sets up and verifies the toolchain for Aquarium Dream on Windows (Git Bash).
# Safe to re-run. Exits non-zero if anything required is missing.
set -u
cd "$(dirname "$0")"
fail=0
ok()   { printf '  ok    %s\n' "$1"; }
miss() { printf '  MISS  %s\n' "$1"; fail=1; }

echo "Tools"
for t in ffmpeg ffprobe git gh python; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else miss "$t"; fi
done

echo "Blender 4.5 LTS"
BLENDER="${PROGRAMFILES:-/c/Program Files}/Blender Foundation/Blender 4.5/blender.exe"
if [ ! -f "$BLENDER" ]; then
  echo "  installing with winget..."
  winget install --id BlenderFoundation.Blender.LTS.4.5 --source winget \
    --accept-package-agreements --accept-source-agreements --silent --disable-interactivity
fi
if [ -f "$BLENDER" ]; then ok "$("$BLENDER" --version | head -n 1)"; else miss "blender.exe"; fi

echo "Python packages"
python - <<'PY' || fail=1
import importlib.util, subprocess, sys
need = {"cv2": "opencv-python", "numpy": "numpy", "PIL": "pillow", "scipy": "scipy"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
for mod in need:
    assert importlib.util.find_spec(mod), mod
    print(f"  ok    {mod}")
PY

echo "Folders"
python src/paths.py || fail=1

echo "Source clip"
if [ -f Aquarium_Blank.mp4 ] || [ -f footage/aquarium_blank_noaudio.mp4 ]; then ok "clip present"; else miss "Aquarium_Blank.mp4 or footage/aquarium_blank_noaudio.mp4"; fi

echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"; else echo "  note  no NVIDIA GPU found; Cycles will fall back to CPU"; fi

[ "$fail" -eq 0 ] && echo "READY" || { echo "NOT READY"; exit 1; }
