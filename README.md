# Fauna

Species-agnostic behavioral sensing for animal health. A computer-vision pipeline that
*understands* animal behavior from a plain camera — no collars, nothing on the animal.

Three decoupled stages, communicating via `.npy` files on disk:

```
video.mp4
  ├─ clip/   CLIP ViT-B/32 per-frame encode      → [T, 512] features
  ├─ yolo/   YOLO11-pose track (markerless kpts)  → [Frames, 10, 74]
  └─ action_recognition/  Transformer over features → 140-way multi-label actions
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Everything installs into a project-local
`.venv/` — your system Python is never touched.

```bash
uv sync          # creates .venv (managed Python 3.12) + installs all deps
```

Run anything inside the env without activating:

```bash
uv run python action_recognition/train.py
```

…or activate a shell once:

```bash
source .venv/bin/activate
```

## Running the stages

```bash
# 1. Extract CLIP features from videos  (edit paths first — see Notes)
uv run python clip/scripts/extract_clip_feats.py

# 2. YOLO pose tracking → keypoint tensors
uv run python yolo/scripts/predict_pose_final.py

# 3a. Train the action recognizer
uv run python action_recognition/train.py

# 3b. Inference + temporal activation plot
uv run python action_recognition/inference.py \
    --checkpoint checkpoints/best.pt \
    --feature dataset/clip_features/test/rgb/<id>.npy \
    --visualize --gt dataset/gt.json
```

## Notes / known gotchas

- **Hardware:** all stages auto-select the best device — **CUDA → MPS (Apple GPU) → CPU**.
  On this Mac they run on the Apple GPU (MPS); on the original Linux/NVIDIA box they still
  use CUDA. Training AMP (mixed precision) only engages on CUDA; MPS runs full FP32.
- **Hardcoded paths:** the `yolo/` and `clip/` scripts contain absolute Linux paths
  (`/home/ulas/...`) that must be edited for this machine.
- **`extract_clip_feats.py`** runs hardcoded paths at module top level (bottom of file),
  so importing it crashes — guard that block under `if __name__ == "__main__":`.
- **Data & weights** (`dataset/`, YOLO `best.pt`) are not in the repo and are gitignored.
- `uv.lock` pins exact versions and **is committed** for reproducible installs.
