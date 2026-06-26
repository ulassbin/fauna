import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class AnimalKingdomDataset(Dataset):
    def __init__(self, gt_path, feature_root, split='train',
                 max_len=256, val_ratio=0.15, seed=42):
        """
        split: 'train' | 'val'  (both carved from gt subset='train')
               'test'           (gt subset='test')
        """
        assert split in ('train', 'val', 'test')

        with open(gt_path) as f:
            gt = json.load(f)
        database = gt.get('database', gt)

        if split in ('train', 'val'):
            candidates = [(vid, info) for vid, info in database.items()
                          if info['subset'] == 'train']
            rng = random.Random(seed)
            rng.shuffle(candidates)
            n_val = int(len(candidates) * val_ratio)
            candidates = candidates[:n_val] if split == 'val' else candidates[n_val:]
            rgb_dir = os.path.join(feature_root, 'train', 'rgb')
        else:
            candidates = [(vid, info) for vid, info in database.items()
                          if info['subset'] == 'test']
            rgb_dir = os.path.join(feature_root, 'test', 'rgb')

        self.samples = []
        self.labels = []
        for vid, info in candidates:
            feat_path = os.path.join(rgb_dir, f"{vid}.npy")
            if not os.path.exists(feat_path):
                continue
            label_vec = np.zeros(140, dtype=np.float32)
            for ann in info['annotations']:
                label_vec[int(ann['label'])] = 1.0
            self.samples.append(feat_path)
            self.labels.append(label_vec)

        self.max_len = max_len
        self.num_classes = 140
        print(f"[Dataset] {split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = np.load(self.samples[idx]).astype(np.float32)  # [T, 512]
        T = feat.shape[0]

        if T >= self.max_len:
            # Uniform temporal sampling — preserves full-video coverage
            indices = np.linspace(0, T - 1, self.max_len, dtype=int)
            feat = feat[indices]
            mask = np.ones(self.max_len, dtype=bool)
        else:
            pad = np.zeros((self.max_len - T, feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
            mask = np.array([True] * T + [False] * (self.max_len - T))

        return (
            torch.from_numpy(feat),              # [max_len, 512]
            torch.from_numpy(mask),              # [max_len]  True = valid
            torch.from_numpy(self.labels[idx]),  # [140]
        )

    def compute_pos_weights(self):
        """Per-class positive weights for BCEWithLogitsLoss to handle class imbalance."""
        labels = np.stack(self.labels)           # [N, 140]
        pos = labels.sum(0).clip(min=1)
        neg = len(labels) - pos
        return torch.from_numpy(neg / pos).float()