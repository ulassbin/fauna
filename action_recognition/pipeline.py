"""
Fauna inference pipeline.

Per-video output layout (under out_root/<video_name>/):
    yolo.npz                  — YOLO tracking output
    global_clip.npy           — CLIP features for full video  [T, 512]
    actor_<id>_clip.npy       — CLIP features per tracked actor [T, 512]
    global_actions.npy        — action scores for full video   [140]
    actor_<id>_actions.npy    — action scores per actor        [140]
    decision.npz              — combined output for the UI layer

Stages 1-2 (YOLO + CLIP extraction) are stubs.
Stages 3-4 (action recognition + decision) are real.
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from inference import ActionRecognizer
from utils import get_device

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_FOLDER    = 'data/vid_final'
OUT_ROOT        = 'outputs/pipeline'
CHECKPOINT      = 'weights/best_firstrun.pt'
ACTION_CSV      = 'actions_ak.csv'
YOLO_MODEL      = 'weights/best.pt'
DEVICE          = None   # None = auto (CUDA > MPS > CPU)
THRESHOLD       = 0.3
MAX_ACTORS      = 10
PAD             = 0.25   # bbox padding per side for actor crops
SMOOTH          = 0.3    # EMA alpha for crop-center stabilization
CROP_SIZE       = 256
# ──────────────────────────────────────────────────────────────────────────────

# ── Species knowledge ─────────────────────────────────────────────────────────
YOLO_CLASS_NAMES = {0: 'Amphibian', 1: 'Bird', 2: 'Fish', 3: 'Mammal', 4: 'Reptile'}

SPECIES_CANDIDATES = {
    0: [  # Amphibian
        'frog', 'toad', 'salamander', 'newt', 'axolotl', 'tree frog', 'bullfrog',
        'dart frog', 'fire salamander',
    ],
    1: [  # Bird
        'chicken', 'duck', 'goose', 'turkey', 'pigeon', 'parrot', 'canary',
        'budgerigar', 'cockatoo', 'cockatiel', 'lovebird', 'macaw', 'parakeet',
        'eagle', 'owl', 'sparrow', 'crow', 'penguin', 'flamingo', 'swan',
        'peacock', 'robin', 'hummingbird', 'pelican', 'stork', 'heron',
    ],
    2: [  # Fish
        'goldfish', 'koi', 'clownfish', 'guppy', 'betta fish', 'angelfish',
        'oscar fish', 'neon tetra', 'catfish', 'salmon', 'trout', 'carp',
        'tilapia', 'bass', 'tuna', 'shark', 'piranha', 'discus fish',
    ],
    3: [  # Mammal
        'cat', 'dog', 'rabbit', 'hamster', 'guinea pig', 'gerbil', 'rat', 'mouse',
        'cow', 'horse', 'sheep', 'goat', 'pig', 'donkey', 'llama', 'alpaca',
        'deer', 'fox', 'wolf', 'lion', 'tiger', 'leopard', 'cheetah',
        'bear', 'elephant', 'giraffe', 'zebra', 'monkey', 'chimpanzee',
        'panda', 'koala', 'kangaroo',
    ],
    4: [  # Reptile
        'gecko', 'iguana', 'chameleon', 'bearded dragon', 'blue tongue skink',
        'corn snake', 'ball python', 'boa constrictor', 'king snake',
        'tortoise', 'red-eared slider turtle', 'box turtle',
        'crocodile', 'alligator', 'komodo dragon', 'monitor lizard',
    ],
}
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Species refinement via CLIP text similarity
# ══════════════════════════════════════════════════════════════════════════════

def load_clip(device):
    """Load OpenAI CLIP (installed via pip/git). Returns (model, preprocess)."""
    import clip
    model, preprocess = clip.load("ViT-B/32", device=str(device))
    model.eval()
    return model, preprocess


def clip_encode_frames(frames_bgr, clip_model, preprocess, device, batch_size=64):
    """Encode BGR frames -> [N, 512] float32 (raw image features, no L2 norm)."""
    from PIL import Image
    out = []
    for i in range(0, len(frames_bgr), batch_size):
        batch = frames_bgr[i:i + batch_size]
        tens = torch.stack([
            preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in batch
        ]).to(device)
        with torch.no_grad():
            out.append(clip_model.encode_image(tens).float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.empty((0, 512), np.float32)


def _read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _square_pad_box(x1, y1, x2, y2, pad):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    return cx, cy, side


def _ema(values, alpha):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _clamp(v, hi):
    return int(max(0, min(v, hi)))


def build_species_embeddings(clip_model, device) -> dict[int, np.ndarray]:
    """
    Pre-compute CLIP text embeddings for every species candidate.
    Returns dict: yolo_class_id → np.ndarray [N_species, 512] (L2-normalised).
    Loaded once in main() and passed through to avoid repeated model loading.
    """
    import clip
    embeddings = {}
    with torch.no_grad():
        for class_id, candidates in SPECIES_CANDIDATES.items():
            prompts = [f"a photo of a {name}" for name in candidates]
            tokens  = clip.tokenize(prompts).to(device)
            embs    = clip_model.encode_text(tokens).float()
            embs    = embs / embs.norm(dim=-1, keepdim=True)
            embeddings[class_id] = embs.cpu().numpy()   # [N, 512]
    print(f"[Species] Built CLIP text embeddings for {len(embeddings)} classes")
    return embeddings


def refine_actor_id(yolo_class_id: int, actor_clip: np.ndarray,
                    species_embeddings: dict) -> tuple[str, str]:
    """
    actor_clip: [T, 512] visual CLIP features for one actor
    Returns (broad_class, specific_species) e.g. ('Mammal', 'sheep')
    """
    broad = YOLO_CLASS_NAMES.get(yolo_class_id, f'class_{yolo_class_id}')
    text_embs = species_embeddings.get(yolo_class_id)
    if text_embs is None or len(actor_clip) == 0:
        return broad, broad

    mean_feat = actor_clip.mean(0).astype(np.float32)          # [512]
    mean_feat /= np.linalg.norm(mean_feat) + 1e-8
    sims      = text_embs @ mean_feat                          # [N_species]
    best      = int(np.argmax(sims))
    specific  = SPECIES_CANDIDATES[yolo_class_id][best]
    return broad, specific


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — YOLO
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo(video_path: str, video_dir: str, yolo_model, device) -> str:
    """
    Run YOLO pose tracking and save yolo.npz.

    Arrays in yolo.npz:
        existence  [10, T, 1]    — 1.0 if animal present in frame
        ids        [10, T, 5]    — [track_id, class_id, conf, cx, cy]
        bboxes     [10, T, 4, 2] — 4 corners (tl, tr, br, bl) × (x, y)
        keypoints  [10, T, 23, 3]— x, y, confidence per keypoint
    """
    results = yolo_model.track(
        video_path, conf=0.1, stream=True, imgsz=640, persist=True,
        device=(0 if device.type == 'cuda' else device.type),
    )

    # Per-frame lists — we'll stack at the end
    frames_exist = []
    frames_ids   = []
    frames_bboxes = []
    frames_kpts  = []

    # Maps YOLO track_id → fixed slot (0–MAX_ACTORS-1) for the whole video
    track_to_slot: dict[int, int] = {}
    next_slot = 0

    for result in results:
        exist_f  = np.zeros((MAX_ACTORS, 1),       dtype=np.float32)
        ids_f    = np.zeros((MAX_ACTORS, 5),        dtype=np.float32)
        bboxes_f = np.zeros((MAX_ACTORS, 4, 2),    dtype=np.float32)
        kpts_f   = np.zeros((MAX_ACTORS, 23, 3),   dtype=np.float32)

        if result.boxes is not None and len(result.boxes) > 0:
            classes  = result.boxes.cls.cpu().numpy()       # [N]
            confs    = result.boxes.conf.cpu().numpy()      # [N]
            bboxes   = result.boxes.xyxy.cpu().numpy()      # [N, 4]
            kpts     = result.keypoints.data.cpu().numpy()  # [N, 23, 3]

            # Use YOLO track IDs when available; fall back to detection order
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(classes))

            for i in range(min(len(classes), MAX_ACTORS)):
                tid = int(track_ids[i])
                if tid not in track_to_slot:
                    if next_slot >= MAX_ACTORS:
                        continue
                    track_to_slot[tid] = next_slot
                    next_slot += 1
                slot = track_to_slot[tid]

                x1, y1, x2, y2 = bboxes[i]
                exist_f[slot, 0]  = 1.0
                ids_f[slot]       = [tid, classes[i], confs[i], (x1 + x2) / 2, (y1 + y2) / 2]
                bboxes_f[slot]    = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # tl tr br bl
                kpts_f[slot]      = kpts[i]

        frames_exist.append(exist_f)
        frames_ids.append(ids_f)
        frames_bboxes.append(bboxes_f)
        frames_kpts.append(kpts_f)

    # Stack [T, 10, ...] → transpose to [10, T, ...]
    existence = np.stack(frames_exist).transpose(1, 0, 2)      # [10, T, 1]
    ids       = np.stack(frames_ids).transpose(1, 0, 2)        # [10, T, 5]
    bboxes_np = np.stack(frames_bboxes).transpose(1, 0, 2, 3)  # [10, T, 4, 2]
    keypoints = np.stack(frames_kpts).transpose(1, 0, 2, 3)    # [10, T, 23, 3]

    out_path = yolo_path(video_dir)
    np.savez(out_path, existence=existence, ids=ids, bboxes=bboxes_np, keypoints=keypoints)
    print(f"  [yolo] Saved {out_path}  T={len(frames_exist)}, slots used={next_slot}")
    return out_path


def yolo_path(video_dir: str) -> str:
    return os.path.join(video_dir, 'yolo.npz')


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — CLIP feature extraction  (stub)
# ══════════════════════════════════════════════════════════════════════════════

def extract_global_clip(video_path: str, video_dir: str,
                        clip_model, preprocess, device) -> str:
    """Whole-frame CLIP features for the full video -> global_clip.npy [T, 512]."""
    frames = _read_frames(video_path)
    feats = clip_encode_frames(frames, clip_model, preprocess, device)
    out = global_clip_path(video_dir)
    np.save(out, feats.astype(np.float32))
    print(f"  [clip] global_clip.npy  {feats.shape}")
    return out


def extract_actor_clips(video_path: str, video_dir: str,
                        clip_model, preprocess, device,
                        pad=PAD, smooth=SMOOTH, crop_size=CROP_SIZE) -> list[str]:
    """Per-actor padded + stabilized crops from yolo.npz bboxes -> actor_<slot>_clip.npy [T, 512].

    Full length (frame-aligned with global_clip); frames where the actor is
    absent are hold-filled, and the `existence` mask in yolo.npz marks real ones.
    """
    yolo_file = yolo_path(video_dir)
    if not os.path.exists(yolo_file):
        print("  [clip] no yolo.npz - run YOLO first")
        return []
    data   = np.load(yolo_file)
    exist  = data['existence']   # [10, T, 1]
    bboxes = data['bboxes']      # [10, T, 4, 2]  corners tl,tr,br,bl
    frames = _read_frames(video_path)
    T = min(len(frames), exist.shape[1])
    H, W = frames[0].shape[:2]

    saved = []
    for slot in range(exist.shape[0]):
        present = np.where(exist[slot, :T, 0] > 0.5)[0]
        if len(present) == 0:
            continue
        cxs, cys, sides = [], [], []
        for t in present:
            x1, y1 = bboxes[slot, t, 0]
            x2, y2 = bboxes[slot, t, 2]
            cx, cy, side = _square_pad_box(x1, y1, x2, y2, pad)
            cxs.append(cx); cys.append(cy); sides.append(side)
        side = _clamp(np.percentile(sides, 95), min(W, H)) or 32
        sx, sy = _ema(cxs, smooth), _ema(cys, smooth)
        center_at = {int(present[k]): (sx[k], sy[k]) for k in range(len(present))}

        raw = [None] * T
        for t in range(T):
            if t in center_at:
                cx, cy = center_at[t]
                x0 = _clamp(cx - side / 2, W - side)
                y0 = _clamp(cy - side / 2, H - side)
                raw[t] = cv2.resize(frames[t][y0:y0 + side, x0:x0 + side], (crop_size, crop_size))
        # forward-fill then back-fill so every frame has a crop (absent = hold)
        last = None
        for t in range(T):
            if raw[t] is not None:
                last = raw[t]
            raw[t] = last
        first_real = next((c for c in raw if c is not None), None)
        crops = [c if c is not None else first_real for c in raw]

        feats = clip_encode_frames(crops, clip_model, preprocess, device)
        out = os.path.join(video_dir, f"actor_{slot}_clip.npy")
        np.save(out, feats.astype(np.float32))
        saved.append(out)
        print(f"  [clip] actor_{slot}_clip.npy  {feats.shape}  ({len(present)}/{T} present)")
    return saved


def global_clip_path(video_dir: str) -> str:
    return os.path.join(video_dir, 'global_clip.npy')


def actor_clip_paths(video_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(video_dir, 'actor_*_clip.npy')))


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — Action recognition  (real)
# ══════════════════════════════════════════════════════════════════════════════

def actions_path_for(clip_path: str) -> str:
    return clip_path.replace('_clip.npy', '_actions.npy')


def run_action_recognition(video_dir: str, recognizer: ActionRecognizer):
    """Run AR on global + all actor clip files, save *_actions.npy as [T, 140]."""
    clip_files = [global_clip_path(video_dir)] + actor_clip_paths(video_dir)
    clip_files = [p for p in clip_files if os.path.exists(p)]

    if not clip_files:
        print(f"  [actions] No clip files found in {video_dir}, skipping")
        return

    for clip_file in clip_files:
        out_path = actions_path_for(clip_file)
        if os.path.exists(out_path):
            print(f"  [actions] Already exists: {os.path.basename(out_path)}")
            continue
        frame_scores = recognizer.predict_frame_scores_full(clip_file)   # [T, 140]
        np.save(out_path, frame_scores)
        top = np.argsort(frame_scores.mean(0))[::-1][:3]
        print(f"  [actions] {os.path.basename(clip_file)} → shape {frame_scores.shape}, top3 {list(top)}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — Actor identification  (real once YOLO exists)
# ══════════════════════════════════════════════════════════════════════════════

def get_actor_species(video_dir: str, species_embeddings: dict) -> dict[str, tuple[str, str]]:
    """
    Read YOLO class IDs from yolo.npz, then refine to specific species via CLIP similarity.
    Returns dict: actor_key → (broad_class, specific_species)
    e.g. 'actor_0' → ('Mammal', 'sheep')
    """
    yolo_file = yolo_path(video_dir)
    if not os.path.exists(yolo_file):
        return {
            os.path.basename(p).replace('_clip.npy', ''): ('unknown', 'unknown')
            for p in actor_clip_paths(video_dir)
        }

    data  = np.load(yolo_file)
    ids   = data['ids']       # [10, T, 5]  col 0 = yolo class id
    exist = data['existence'] # [10, T, 1]

    result = {}
    for slot in range(ids.shape[0]):
        valid = exist[slot, :, 0].astype(bool)
        if not valid.any():
            continue
        class_ids   = ids[slot, valid, 1].astype(int)   # col 1 = class_id
        yolo_class  = int(np.bincount(class_ids).argmax())
        actor_key   = f"actor_{slot}"

        # Load actor clip features for CLIP refinement
        clip_path = os.path.join(video_dir, f"{actor_key}_clip.npy")
        actor_clip = np.load(clip_path).astype(np.float32) if os.path.exists(clip_path) else np.empty((0, 512))

        broad, specific = refine_actor_id(yolo_class, actor_clip, species_embeddings)
        result[actor_key] = (broad, specific)
        print(f"  [id] {actor_key}: {broad} → {specific}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — Decision output  (real)
# ══════════════════════════════════════════════════════════════════════════════

def _resample_yolo_to_clip_T(arr: np.ndarray, T_clip: int) -> np.ndarray:
    """Uniformly resample YOLO array [10, raw_T, ...] → [10, T_clip, ...]."""
    raw_T = arr.shape[1]
    if raw_T == T_clip:
        return arr
    idx = np.linspace(0, raw_T - 1, T_clip, dtype=int)
    return arr[:, idx]


def save_decision(video_name: str, video_dir: str, species_embeddings: dict):
    """
    Combine all stage outputs into decision.npz for the UI layer.

    Keys in decision.npz:
        existence         [10, T, 1]     — resampled to clip T
        ids               [10, T, 5]
        bboxes            [10, T, 4, 2]
        keypoints         [10, T, 23, 3]

        global_clip       [T, 512]
        global_actions    [T, 140]

        actor_<id>_clip      [T, 512]
        actor_<id>_actions   [T, 140]
        actor_<id>_broad     str — e.g. 'Mammal'
        actor_<id>_species   str — e.g. 'sheep'
    """
    decision = {}

    # Determine clip T from global clip (reference for resampling)
    g_clip = global_clip_path(video_dir)
    T_clip = np.load(g_clip).shape[0] if os.path.exists(g_clip) else None

    # YOLO — resample raw frame arrays to clip T
    yolo_file = yolo_path(video_dir)
    if os.path.exists(yolo_file):
        yolo = np.load(yolo_file)
        for key in yolo.files:
            arr = yolo[key]
            if T_clip is not None and arr.ndim >= 2 and arr.shape[1] != T_clip:
                arr = _resample_yolo_to_clip_T(arr, T_clip)
            decision[key] = arr

    # Global clip + actions
    if os.path.exists(g_clip):
        decision['global_clip'] = np.load(g_clip)
    g_actions = actions_path_for(g_clip)
    if os.path.exists(g_actions):
        decision['global_actions'] = np.load(g_actions)   # [T, 140]

    # Per-actor clip + actions + species
    species_map = get_actor_species(video_dir, species_embeddings)
    for clip_file in actor_clip_paths(video_dir):
        stem = os.path.basename(clip_file).replace('_clip.npy', '')
        decision[f'{stem}_clip'] = np.load(clip_file)
        act_file = actions_path_for(clip_file)
        if os.path.exists(act_file):
            decision[f'{stem}_actions'] = np.load(act_file)   # [T, 140]
        broad, specific = species_map.get(stem, ('unknown', 'unknown'))
        decision[f'{stem}_broad']   = np.array(broad)
        decision[f'{stem}_species'] = np.array(specific)

    out_path = os.path.join(video_dir, 'decision.npz')
    np.savez(out_path, **decision)
    print(f"  [decision] Saved {out_path}  ({len(decision)} arrays)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Fauna inference pipeline")
    p.add_argument('--video-folder', default=VIDEO_FOLDER)
    p.add_argument('--out-root',     default=OUT_ROOT)
    p.add_argument('--yolo-model',   default=YOLO_MODEL)
    p.add_argument('--checkpoint',   default=CHECKPOINT)
    p.add_argument('--action-csv',   default=ACTION_CSV)
    p.add_argument('--pad',          type=float, default=PAD)
    p.add_argument('--smooth',       type=float, default=SMOOTH)
    p.add_argument('--crop-size',    type=int,   default=CROP_SIZE)
    p.add_argument('--threshold',    type=float, default=THRESHOLD)
    return p.parse_args()


def main():
    from ultralytics import YOLO as YOLOModel
    a = parse_args()
    device = get_device()
    print(f"[pipeline] device = {device}")

    yolo_model         = YOLOModel(a.yolo_model)
    recognizer         = ActionRecognizer(a.checkpoint, a.action_csv, device=str(device), threshold=a.threshold)
    clip_model, preprocess = load_clip(device)
    species_embeddings = build_species_embeddings(clip_model, device)

    video_files = [f for f in os.listdir(a.video_folder) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos in {a.video_folder}\n")

    for v_file in video_files:
        video_path = os.path.join(a.video_folder, v_file)
        video_name = os.path.splitext(v_file)[0]
        video_dir  = os.path.join(a.out_root, video_name)
        os.makedirs(video_dir, exist_ok=True)

        print(f"── {video_name}")

        # Stage 1: YOLO
        if not os.path.exists(yolo_path(video_dir)):
            run_yolo(video_path, video_dir, yolo_model, device)
        print(f"Ran until yolo!")
        # Stage 2: CLIP features
        if not os.path.exists(global_clip_path(video_dir)):
            try:
                extract_global_clip(video_path, video_dir, clip_model, preprocess, device)
            except NotImplementedError:
                print("  [clip] global stub — skipping")
        else:
            print(f"Clip already exists moving on!")
        
        if not actor_clip_paths(video_dir):
            try:
                extract_actor_clips(video_path, video_dir, clip_model, preprocess, device,
                                    pad=a.pad, smooth=a.smooth, crop_size=a.crop_size)
            except NotImplementedError:
                print("  [clip] actor stub — skipping")
        else:
            print(f"Actor clips already exists moving on!")
        # Stage 3: Action recognition  (real)
        run_action_recognition(video_dir, recognizer)
        
        # Stage 4+5: Decision output  (real)
        save_decision(video_name, video_dir, species_embeddings)

        print(f"Done for a single video {video_name}")


if __name__ == '__main__':
    main()