#!/usr/bin/env bash
#
# Regenerate action recognition for all demo streams using the current
# CHECKPOINT set in action_recognition/pipeline.py.
#
# Reuses the cached YOLO (yolo.npz) and CLIP (*_clip.npy) features — only the
# action stage + decision.npz are recomputed. Run this after changing the model
# path in pipeline.py.
#
#   1. edit action_recognition/pipeline.py  ->  CHECKPOINT = 'weights/<model>.pt'
#   2. ./regen_actions.sh
#   3. restart the app server (command printed at the end)
#
set -euo pipefail
cd "$(dirname "$0")"

echo "Active model:"
grep -E "^CHECKPOINT" action_recognition/pipeline.py
echo

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Clear each stream's action/decision outputs (keep yolo.npz + *_clip.npy) and
# stage its video into a temp folder for the pipeline to scan.
uv run python - "$TMP" <<'PY'
import json, os, sys, glob
tmp = sys.argv[1]
streams = json.load(open("app/streams.json"))
for s in streams:
    path = s["path"]
    stem = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join("outputs/pipeline", stem)
    for f in glob.glob(os.path.join(d, "*_actions.npy")):
        os.remove(f)
    dec = os.path.join(d, "decision.npz")
    if os.path.exists(dec):
        os.remove(dec)
    src = os.path.abspath(path)
    if os.path.exists(src):
        os.symlink(src, os.path.join(tmp, stem + os.path.splitext(path)[1]))
    else:
        print(f"  WARN missing video: {src}")
print(f"  staged {len(streams)} streams; cleared their actions + decision.npz")
PY

echo
echo "Recomputing actions (YOLO + CLIP reused from cache)..."
uv run python action_recognition/pipeline.py --video-folder "$TMP"

echo
echo "Done. Restart the app server to load the new decisions:"
echo "  pkill -f 'app/server.py'; uv run python app/server.py"
