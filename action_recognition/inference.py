"""
Inference pipeline for animal action recognition.

Usage (CLI):
    python inference.py --checkpoint checkpoints/best.pt --feature path/to/video.npy
    python inference.py --checkpoint checkpoints/best.pt --feature path/to/video.npy --keypoints path/to/video_keypoints.npy

Usage (API):
    recognizer = ActionRecognizer('checkpoints/best.pt', 'dataset/actions_ak.csv')
    preds, display = recognizer.predict('dataset/clip_features/test/rgb/AAACXZTV.npy', animal_group='Mammal')
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import ActionTransformer
from utils import load_action_info, get_top_predictions, build_display_string


# ── Animal group mapping ───────────────────────────────────────────────────────
# Maps YOLO Animal Kingdom class ID → broad display group.
# Populated by load_yolo_animal_groups() at runtime from the model weights,
# or you can hardcode known classes here.
_ANIMAL_GROUPS: dict[int, str] = {}


def load_yolo_animal_groups(model_path: str) -> dict[int, str]:
    """
    Reads class names from the YOLO checkpoint and assigns each to a broad group.
    Call once at startup if you want automatic animal-group labelling.
    """
    try:
        from ultralytics import YOLO
        yolo = YOLO(model_path)
        names = yolo.names  # dict {int: str}
        # Simple heuristic grouping — refine as needed
        mammal_keywords  = {'mammal', 'dog', 'cat', 'horse', 'lion', 'tiger', 'bear',
                             'elephant', 'deer', 'wolf', 'fox', 'rabbit', 'monkey', 'ape',
                             'cow', 'pig', 'sheep', 'goat', 'giraffe', 'zebra', 'rhino'}
        bird_keywords    = {'bird', 'eagle', 'hawk', 'owl', 'parrot', 'penguin', 'duck',
                             'goose', 'swan', 'flamingo', 'crow', 'pigeon', 'sparrow'}
        reptile_keywords = {'reptile', 'snake', 'lizard', 'crocodile', 'turtle', 'gecko',
                             'iguana', 'chameleon', 'alligator'}
        fish_keywords    = {'fish', 'shark', 'ray', 'tuna', 'salmon', 'clown', 'eel'}
        insect_keywords  = {'insect', 'butterfly', 'bee', 'ant', 'beetle', 'dragonfly',
                             'grasshopper', 'moth', 'spider'}

        groups = {}
        for cid, name in names.items():
            nl = name.lower()
            if any(k in nl for k in mammal_keywords):
                groups[cid] = 'Mammal'
            elif any(k in nl for k in bird_keywords):
                groups[cid] = 'Bird'
            elif any(k in nl for k in reptile_keywords):
                groups[cid] = 'Reptile'
            elif any(k in nl for k in fish_keywords):
                groups[cid] = 'Fish'
            elif any(k in nl for k in insect_keywords):
                groups[cid] = 'Insect'
            else:
                groups[cid] = name.capitalize()
        return groups
    except Exception as e:
        print(f"[Warning] Could not load YOLO animal groups: {e}")
        return {}


def get_animal_group_from_keypoints(keypoint_path: str) -> str | None:
    """
    keypoint_path: .npy of shape [Frames, 10, 74], column 0 = YOLO class ID
    Returns the most frequently detected animal group across all frames.
    """
    if not _ANIMAL_GROUPS:
        return None
    data = np.load(keypoint_path)          # [F, 10, 74]
    class_ids = data[:, :, 0].flatten()
    class_ids = class_ids[class_ids > 0]   # 0 = empty slot
    if len(class_ids) == 0:
        return None
    most_common = int(np.bincount(class_ids.astype(int)).argmax())
    return _ANIMAL_GROUPS.get(most_common)


# ──────────────────────────────────────────────────────────────────────────────

class ActionRecognizer:
    def __init__(self, checkpoint_path: str, action_csv: str,
                 device: str | None = None, threshold: float = 0.3):
        self.device      = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold   = threshold
        self.action_info = load_action_info(action_csv)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt['cfg']
        self.max_len = cfg['max_len']

        self.model = ActionTransformer(
            d_model    = cfg['d_model'],
            nhead      = cfg['nhead'],
            num_layers = cfg['num_layers'],
            dropout    = 0.0,
            max_len    = cfg['max_len'],
        ).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f"[Inference] Loaded {checkpoint_path}  (epoch {ckpt['epoch']}, best mAP={ckpt['best_map']:.4f})")

    def _prepare(self, feat: np.ndarray):
        T = feat.shape[0]
        if T >= self.max_len:
            idx  = np.linspace(0, T - 1, self.max_len, dtype=int)
            feat = feat[idx]
            mask = np.ones(self.max_len, dtype=bool)
        else:
            pad  = np.zeros((self.max_len - T, feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
            mask = np.array([True] * T + [False] * (self.max_len - T))
        return feat, mask

    def predict(self, feat, animal_group: str | None = None):
        """
        feat: path to .npy file  OR  numpy array [T, 512]
        Returns (predictions, display_string)
          predictions: list of (label_int, action_name, category, score)
        """
        if isinstance(feat, str):
            feat = np.load(feat).astype(np.float32)

        feat, mask = self._prepare(feat)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(feat_t, mask_t)[0]

        preds   = get_top_predictions(logits, self.action_info, threshold=self.threshold)
        display = build_display_string(preds, animal_group)
        return preds, display

    def predict_frame_scores_full(self, feat) -> np.ndarray:
        """
        Returns per-frame sigmoid scores [T, 140] matching the input clip length exactly.
          T <= max_len : pad + mask (single forward pass)
          T >  max_len : sliding window (stride = max_len//2), average overlapping frames
        """
        if isinstance(feat, str):
            feat = np.load(feat).astype(np.float32)

        T, D = feat.shape
        W    = self.max_len

        if T <= W:
            pad    = np.zeros((W - T, D), dtype=np.float32)
            feat_p = np.concatenate([feat, pad], axis=0)
            mask   = np.array([True] * T + [False] * (W - T))
            feat_t = torch.from_numpy(feat_p).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model.forward_temporal(feat_t, mask_t)[0, :T].cpu().numpy()
            return 1.0 / (1.0 + np.exp(-logits))   # [T, 140]

        # Sliding window
        stride  = W // 2
        scores  = np.zeros((T, 140), dtype=np.float32)
        counts  = np.zeros(T,        dtype=np.float32)

        starts = list(range(0, T - W + 1, stride))
        if not starts or starts[-1] + W < T:
            starts.append(T - W)   # always cover the tail

        mask_t = torch.ones(1, W, dtype=torch.bool).to(self.device)
        for start in starts:
            chunk  = feat[start : start + W]
            feat_t = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model.forward_temporal(feat_t, mask_t)[0].cpu().numpy()
            scores[start : start + W] += 1.0 / (1.0 + np.exp(-logits))
            counts[start : start + W] += 1.0

        return scores / counts[:, None]   # [T, 140]

    def predict_scores(self, feat) -> np.ndarray:
        """Returns full sigmoid score vector [140] for a clip feature file or array."""
        if isinstance(feat, str):
            feat = np.load(feat).astype(np.float32)
        feat_p, mask = self._prepare(feat)
        feat_t = torch.from_numpy(feat_p).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(feat_t, mask_t)[0].cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))  # [140]

    def predict_video_id(self, video_id: str, feature_root: str,
                         subset: str = 'test', animal_group: str | None = None):
        path = os.path.join(feature_root, subset, 'rgb', f"{video_id}.npy")
        return self.predict(path, animal_group)

    def predict_temporal(self, feat, topk: int = 5):
        """
        Returns per-clip sigmoid activations for the top-k classes.

        feat: path to .npy  OR  numpy array [T, 512]
        Returns:
            activations:  np.ndarray [T_valid, topk]  — sigmoid scores over time
            top_labels:   list of (label_int, action_name, category, video_score)
            video_logits: np.ndarray [140]
        """
        if isinstance(feat, str):
            feat = np.load(feat).astype(np.float32)

        feat_p, mask = self._prepare(feat)
        T_valid = int(mask.sum())  # number of non-padded clips

        feat_t = torch.from_numpy(feat_p).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # video-level prediction for ranking top-k
            video_logits  = self.model(feat_t, mask_t)[0].cpu().numpy()
            # per-clip logits: [1, T, 140] → [T_valid, 140]
            frame_logits  = self.model.forward_temporal(feat_t, mask_t)[0].cpu().numpy()

        frame_logits  = frame_logits[:T_valid]                          # strip padding
        frame_scores  = 1.0 / (1.0 + np.exp(-frame_logits))            # sigmoid [T_valid, 140]

        # rank top-k by video-level score (most meaningful ordering)
        video_scores  = 1.0 / (1.0 + np.exp(-video_logits))
        top_indices   = np.argsort(video_scores)[::-1][:topk]
        top_labels    = [
            (int(i), self.action_info[i]['action'], self.action_info[i]['category'], float(video_scores[i]))
            for i in top_indices if i in self.action_info
        ]
        top_activations = frame_scores[:, top_indices]                  # [T_valid, topk]

        return top_activations, top_labels, video_logits, frame_scores  # frame_scores: [T_valid, 140]

    def save_activation_plot(self, feat, video_name: str, out_dir: str,
                             topk: int = 5, animal_group: str | None = None,
                             gt_annotations: list | None = None):
        """
        Runs temporal inference and saves:
          {out_dir}/{video_name}_activations.npy  — [T, topk] sigmoid scores
          {out_dir}/{video_name}_activations.png  — two subplots:
            top:    predicted top-k activation curves (✓/✗ vs GT)
            bottom: GT labels as horizontal bars over clip time

        gt_annotations: list of {'label': int, 'segment': [start, end]} from gt.json
        """
        os.makedirs(out_dir, exist_ok=True)
        activations, top_labels, _, all_scores = self.predict_temporal(feat, topk=topk)

        npy_path = os.path.join(out_dir, f"{video_name}_activations.npy")
        np.save(npy_path, all_scores)
        print(f"[Saved] {npy_path}  shape={all_scores.shape}  (all 140 classes)")

        T      = activations.shape[0]
        x      = np.arange(T)
        colors = plt.cm.tab10.colors

        has_gt = bool(gt_annotations)
        gt_set = {int(a['label']) for a in gt_annotations} if has_gt else set()

        # ── Figure layout ──────────────────────────────────────────────────────
        n_gt_rows = len(gt_annotations) if has_gt else 0
        fig_h     = 4 + max(n_gt_rows * 0.35, 1.5) if has_gt else 4
        fig, axes = plt.subplots(
            2 if has_gt else 1, 1,
            figsize=(12, fig_h),
            sharex=True,
            gridspec_kw={'height_ratios': [3, max(n_gt_rows, 1)]} if has_gt else {},
        )
        ax_pred = axes[0] if has_gt else axes
        ax_gt   = axes[1] if has_gt else None

        # ── Top subplot: predictions ───────────────────────────────────────────
        top_label_ids = {lid for lid, _, _, _ in top_labels}
        for i, (label_int, action, category, vid_score) in enumerate(top_labels):
            in_gt  = label_int in gt_set
            tick   = " ✓" if in_gt else " ✗"
            ax_pred.plot(x, activations[:, i],
                         color=colors[i % 10],
                         linewidth=2.0 if in_gt else 1.5,
                         alpha=1.0 if in_gt else 0.65,
                         label=f"{tick} {action} ({category})  {vid_score:.0%}")

        title = f"{video_name}" + (f"  —  {animal_group}" if animal_group else "")
        ax_pred.set_title(title, fontsize=11)
        ax_pred.set_ylabel("Activation")
        ax_pred.set_ylim(0, 1)
        ax_pred.axhline(self.threshold, color='gray', linestyle='--', linewidth=0.8,
                        label=f"threshold={self.threshold}")
        ax_pred.legend(loc='upper right', fontsize=8, framealpha=0.7)
        ax_pred.grid(axis='y', alpha=0.3)

        # ── Bottom subplot: ground truth bars ──────────────────────────────────
        if has_gt and ax_gt is not None:
            # Normalise segment frame indices → clip indices
            total_frames = max(int(a['segment'][1]) for a in gt_annotations)
            sorted_anns  = sorted(gt_annotations, key=lambda a: int(a['label']))

            for row, ann in enumerate(sorted_anns):
                label_int  = int(ann['label'])
                seg        = ann['segment']
                start_clip = seg[0] / total_frames * T
                dur_clip   = (seg[1] - seg[0]) / total_frames * T
                action     = self.action_info.get(label_int, {}).get('action', str(label_int))
                # Reuse the same color as the prediction curve if it's in top-k
                try:
                    col_idx = [lid for lid, _, _, _ in top_labels].index(label_int)
                    color   = colors[col_idx % 10]
                except ValueError:
                    color   = 'steelblue'
                ax_gt.broken_barh([(start_clip, dur_clip)], (row - 0.4, 0.8),
                                  facecolors=color, alpha=0.75)
                ax_gt.text(start_clip + dur_clip + 0.2, row, action,
                           va='center', fontsize=7.5)

            ax_gt.set_yticks(range(len(sorted_anns)))
            ax_gt.set_yticklabels(
                [self.action_info.get(int(a['label']), {}).get('action', str(a['label']))
                 for a in sorted_anns],
                fontsize=7.5,
            )
            ax_gt.set_ylim(-0.8, len(sorted_anns) - 0.2)
            ax_gt.set_xlabel("Clip index")
            ax_gt.set_ylabel("Ground Truth")
            ax_gt.grid(axis='x', alpha=0.2)
        else:
            ax_pred.set_xlabel("Clip index")

        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{video_name}_activations.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {png_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Animal action recognition inference')
    parser.add_argument('--checkpoint',  required=True, help='Path to best.pt checkpoint')
    parser.add_argument('--feature',     required=True, help='Path to clip features .npy [T, 512]')
    parser.add_argument('--csv',         default='dataset/actions_ak.csv')
    parser.add_argument('--keypoints',   default=None,  help='Optional keypoint .npy for animal group detection')
    parser.add_argument('--yolo_model',  default=None,  help='Optional YOLO .pt to load animal group names')
    parser.add_argument('--threshold',   type=float, default=0.3)
    parser.add_argument('--topk',        type=int,   default=5)
    parser.add_argument('--device',      default=None)
    parser.add_argument('--visualize',   action='store_true', help='Save activation plot + .npy')
    parser.add_argument('--out_dir',     default='outputs', help='Where to save visualizations')
    parser.add_argument('--gt',          default=None, help='Path to gt.json to overlay ground truth labels')
    args = parser.parse_args()

    if args.yolo_model:
        _ANIMAL_GROUPS.update(load_yolo_animal_groups(args.yolo_model))

    recognizer = ActionRecognizer(args.checkpoint, args.csv,
                                  device=args.device, threshold=args.threshold)

    animal_group = None
    if args.keypoints:
        animal_group = get_animal_group_from_keypoints(args.keypoints)
        if animal_group:
            print(f"[Keypoints] Detected animal group: {animal_group}")

    preds, display = recognizer.predict(args.feature, animal_group)

    print("\nTop predictions:")
    if preds:
        for label, action, category, score in preds[:args.topk]:
            print(f"  [{label:3d}] {score:.0%}  {action:<35s} ({category})")
    else:
        print("  (none above threshold)")

    print(f"\nDemo display:\n  {display}")

    if args.visualize:
        video_name    = os.path.splitext(os.path.basename(args.feature))[0]
        gt_annotations = None
        if args.gt:
            import json
            with open(args.gt) as f:
                db = json.load(f)
            db = db.get('database', db)
            if video_name in db:
                gt_annotations = [
                    {'label': int(a['label']), 'segment': a['segment']}
                    for a in db[video_name]['annotations']
                ]
                print(f"[GT] {len(gt_annotations)} annotations: "
                      f"{[a['label'] for a in gt_annotations]}")
            else:
                print(f"[GT] {video_name} not found in {args.gt}")
        recognizer.save_activation_plot(
            args.feature, video_name, args.out_dir,
            topk=args.topk, animal_group=animal_group, gt_annotations=gt_annotations,
        )


if __name__ == '__main__':
    main()