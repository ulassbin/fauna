"""Fauna demo — self-contained CLIP zero-shot search engine.

Encodes per-frame CLIP image features for a video, then ranks frames by
cosine similarity against a free-text query. CLIP cosine sims live in a
narrow band (~0.18-0.32), so found/not-found is decided by a RELATIVE
threshold (mean + 1.5*std) with a small absolute floor — never an
absolute 0.5 cutoff.
"""

import os
import sys

import numpy as np

# ── Make the repo root and action_recognition importable for get_device ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_ACTION_DIR = os.path.join(_REPO_ROOT, "action_recognition")
for _p in (_REPO_ROOT, _ACTION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import get_device  # noqa: E402  (path-dependent import)

CACHE_DIR = os.path.join(_THIS_DIR, "cache")

# Sampling / search tuning constants.
TARGET_FPS = 3            # subsample to roughly this many frames per second
MAX_SAMPLES = 600         # hard cap on encoded frames per video
THR_FLOOR = 0.25          # absolute floor for the relative threshold
THR_STD_MULT = 1.0        # threshold = max(floor, mean + THR_STD_MULT * std)


class ClipSearcher:
    """Loads CLIP once; indexes videos into L2-normalized frame features."""

    def __init__(self, device=None):
        import clip
        import torch

        self.torch = torch
        self.device = device if device is not None else get_device()
        # clip.load expects a string device spec.
        self.model, self.preprocess = clip.load("ViT-B/32", device=str(self.device))
        self.model.eval()
        self._clip = clip
        # In-memory index cache: video_id -> dict of arrays / scalars.
        self.indexes = {}
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ── indexing ────────────────────────────────────────────────────────
    def _cache_path(self, video_id):
        return os.path.join(CACHE_DIR, f"{video_id}.npz")

    def _store_index(self, video_id, feats, times, fps, num_frames, duration):
        entry = {
            "feats": feats.astype(np.float32),
            "times": times.astype(np.float32),
            "fps": float(fps),
            "num_frames": int(num_frames),
            "duration": float(duration),
        }
        self.indexes[video_id] = entry
        return entry

    def index_video(self, video_id, video_path):
        """Decode, subsample (~3 fps, capped at 600), CLIP-encode, cache.

        Returns {"video_id","fps","num_frames","duration","n_indexed"}.
        Uses app/cache/{video_id}.npz when present (skips re-encode).
        """
        import cv2

        # Fast path: already in memory.
        if video_id in self.indexes:
            e = self.indexes[video_id]
            return {
                "video_id": video_id,
                "fps": e["fps"],
                "num_frames": e["num_frames"],
                "duration": e["duration"],
                "n_indexed": int(e["feats"].shape[0]),
            }

        # Cache file path: load and reuse if present.
        cache_path = self._cache_path(video_id)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            e = self._store_index(
                video_id,
                data["feats"],
                data["times"],
                float(data["fps"]),
                int(data["num_frames"]),
                float(data["duration"]),
            )
            return {
                "video_id": video_id,
                "fps": e["fps"],
                "num_frames": e["num_frames"],
                "duration": e["duration"],
                "n_indexed": int(e["feats"].shape[0]),
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not fps or fps != fps or fps <= 0:  # 0 / NaN guard
            fps = 25.0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        step = max(1, round(fps / TARGET_FPS))

        # First pass: collect indices we would sample at ~TARGET_FPS.
        sampled_indices = list(range(0, num_frames, step)) if num_frames > 0 else []

        # If the frame count is unknown/zero, fall back to walking the stream.
        if not sampled_indices:
            frames, times = self._decode_walk(cap, step, fps)
            cap.release()
        else:
            # If too many, sample uniformly down to MAX_SAMPLES.
            if len(sampled_indices) > MAX_SAMPLES:
                sel = np.linspace(0, len(sampled_indices) - 1, MAX_SAMPLES)
                sel = np.unique(np.round(sel).astype(int))
                sampled_indices = [sampled_indices[i] for i in sel]
            frames, times = self._decode_by_index(cap, sampled_indices, fps)
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        feats = self._encode_frames(frames)
        times_arr = np.asarray(times, dtype=np.float32)

        # duration: prefer frame_count / fps, else last sampled time.
        if num_frames > 0:
            duration = num_frames / fps
        else:
            duration = float(times_arr[-1]) + (1.0 / max(fps, 1.0))

        e = self._store_index(video_id, feats, times_arr, fps, num_frames, duration)

        # Persist cache.
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(
            cache_path,
            feats=e["feats"],
            times=e["times"],
            fps=np.float32(e["fps"]),
            num_frames=np.int64(e["num_frames"]),
            duration=np.float32(e["duration"]),
        )

        return {
            "video_id": video_id,
            "fps": e["fps"],
            "num_frames": e["num_frames"],
            "duration": e["duration"],
            "n_indexed": int(e["feats"].shape[0]),
        }

    def _decode_by_index(self, cap, indices, fps):
        """Grab specific frame indices via sequential reads (seek is flaky)."""
        import cv2

        wanted = set(indices)
        max_idx = max(indices)
        frames, times = [], []
        idx = 0
        while idx <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames.append(self._to_pil(frame))
                times.append(idx / fps)
            idx += 1
        return frames, times

    def _decode_walk(self, cap, step, fps):
        """Fallback when frame count is unknown: take every `step`th frame."""
        frames, times = [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(self._to_pil(frame))
                times.append(idx / fps)
                if len(frames) >= MAX_SAMPLES:
                    break
            idx += 1
        return frames, times

    @staticmethod
    def _to_pil(frame_bgr):
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _encode_frames(self, pil_frames, batch_size=64):
        """CLIP-encode PIL frames -> [N,512] float32, L2-normalized."""
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(pil_frames), batch_size):
                batch = pil_frames[i : i + batch_size]
                tensors = torch.stack(
                    [self.preprocess(img) for img in batch]
                ).to(self.device)
                out = self.model.encode_image(tensors)
                out = out / out.norm(dim=-1, keepdim=True)
                feats.append(out.float().cpu().numpy())
        arr = np.concatenate(feats, axis=0).astype(np.float32)
        # Defensive re-normalize on CPU.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    # ── text encoding ──────────────────────────────────────────────────
    def encode_text(self, text):
        """Encode free text -> [512] float32, L2-normalized."""
        torch = self.torch
        tokens = self._clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            out = self.model.encode_text(tokens)
            out = out / out.norm(dim=-1, keepdim=True)
        vec = out.float().cpu().numpy()[0].astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec.astype(np.float32)

    # ── search internals ───────────────────────────────────────────────
    def _ensure_index(self, video_id):
        if video_id not in self.indexes:
            cache_path = self._cache_path(video_id)
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                self._store_index(
                    video_id,
                    data["feats"],
                    data["times"],
                    float(data["fps"]),
                    int(data["num_frames"]),
                    float(data["duration"]),
                )
            else:
                raise KeyError(f"Video not indexed: {video_id}")
        return self.indexes[video_id]

    def _sims_and_thr(self, video_id, text):
        e = self._ensure_index(video_id)
        feats = e["feats"]
        text_vec = self.encode_text(text)
        sims = (feats @ text_vec).astype(np.float32)
        mean = float(sims.mean())
        std = float(sims.std())
        thr = max(THR_FLOOR, mean + THR_STD_MULT * std)
        return e, sims, thr

    @staticmethod
    def _merge_ranges(sims, thr, times):
        """Merge consecutive indices with sims>=thr; keep peak per range.

        Returns list of {"time","score","start_idx","peak_idx"} dicts.
        """
        ranges = []
        n = len(sims)
        i = 0
        while i < n:
            if sims[i] >= thr:
                j = i
                while j + 1 < n and sims[j + 1] >= thr:
                    j += 1
                # peak within [i, j]
                seg = sims[i : j + 1]
                peak_off = int(np.argmax(seg))
                peak_idx = i + peak_off
                ranges.append(
                    {
                        "time": float(times[peak_idx]),
                        "score": float(sims[peak_idx]),
                        "peak_idx": peak_idx,
                    }
                )
                i = j + 1
            else:
                i += 1
        return ranges

    # ── public search ──────────────────────────────────────────────────
    def search(self, video_id, text):
        """Rank frames against `text`.

        Returns {"found","best","matches","threshold"}.
        """
        e, sims, thr = self._sims_and_thr(video_id, text)
        times = e["times"]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        found = bool(best_score >= thr)
        best = {"time": float(times[best_idx]), "score": best_score} if found else None

        ranges = self._merge_ranges(sims, thr, times)
        ranges.sort(key=lambda r: -r["score"])
        matches = [{"time": r["time"], "score": r["score"]} for r in ranges]

        return {
            "found": found,
            "best": best,
            "matches": matches,
            "threshold": float(thr),
        }

    def matching_times(self, video_id, text):
        """Representative time (sec) per merged range where sims>=thr."""
        e, sims, thr = self._sims_and_thr(video_id, text)
        times = e["times"]
        ranges = self._merge_ranges(sims, thr, times)
        return [r["time"] for r in ranges]
