"""
Live action-recognition engine for the iPhone WebRTC demo.

Processing model (1 Hz):
    - The WebRTC recv loop pushes every decoded frame via `add_frame()` (sub-ms).
    - Once per second the engine takes that ~1 s batch of frames, DOWNSAMPLES it to
      `frames_per_batch`, CLIP-encodes those few frames, appends them to a short rolling
      feature window (`context_seconds`), runs the ActionTransformer over the window, and
      EMA-smooths the 140-class score vector before broadcasting the top-k.

CRITICAL: live CLIP features must be encoded with `pipeline.clip_encode_frames`
(RAW image features, NO L2-normalisation) — that is exactly how the offline features
the ActionTransformer was trained on were produced. `app/clipsearch.py` L2-normalises,
which would silently corrupt predictions, so it is deliberately NOT used here.

All heavy work (CLIP encode + transformer forward) runs in a single-worker
ThreadPoolExecutor so the aiortc event loop never stalls.
"""

import os
import sys
import time
import struct
import pickle
import asyncio
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# ── Make the parent repo + action_recognition importable ────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
_ACTION_DIR = os.path.join(_REPO, "action_recognition")
for _p in (_ACTION_DIR, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import load_clip, clip_encode_frames   # noqa: E402  RAW encoder (no L2 norm)
from inference import ActionRecognizer               # noqa: E402
from utils import get_device                         # noqa: E402

# Absolute defaults (server runs with cwd=iphoneStream/, so weights/csv must be absolute).
CHECKPOINT = os.path.join(_REPO, "weights", "best_firstrun.pt")
ACTION_CSV = os.path.join(_REPO, "actions_ak.csv")
YOLO_MODEL = os.path.join(_REPO, "weights", "best.pt")
WORKER_PY = os.path.join(_HERE, "yolo_worker.py")


# ── length-prefixed pickle over the worker's pipes (parent side) ─────────────────
def _send(f, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack(">I", len(b))); f.write(b); f.flush()


def _readn(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv(f):
    hdr = _readn(f, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack(">I", hdr)
    data = _readn(f, n)
    return None if data is None else pickle.loads(data)


class LiveActionEngine:
    def __init__(self, checkpoint=CHECKPOINT, action_csv=ACTION_CSV, *,
                 frames_per_batch=4, context_seconds=6, ema_alpha=0.5, topk=6,
                 yolo_model=YOLO_MODEL, det_conf=0.25, det_imgsz=640):
        self.device = get_device()
        print(f"[live] device = {self.device}")
        self.clip_model, self.preprocess = load_clip(self.device)
        self.recognizer = ActionRecognizer(checkpoint, action_csv, device=str(self.device))
        self.action_info = self.recognizer.action_info     # label -> {action, category}
        self.det_conf = det_conf
        self.det_imgsz = det_imgsz
        self._last_dets = []
        # YOLO runs in a SEPARATE process (yolo_worker.py). ultralytics pulls in cv2's
        # bundled ffmpeg (libswscale), which collides with aiortc's av in-process and
        # SIGBUSes frame.to_ndarray — so we isolate it and talk over a pipe.
        self._yolo = subprocess.Popen(
            [sys.executable, WORKER_PY, str(det_conf), str(det_imgsz),
             self.device.type, yolo_model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=_REPO, bufsize=0,
        )
        try:
            ready = _recv(self._yolo.stdout)               # blocks until the worker loads YOLO
            cls = ready.get("classes") if isinstance(ready, dict) else None
            print(f"[live] YOLO worker ready (pid {self._yolo.pid}); classes: {cls}")
        except Exception as e:
            print(f"[live] YOLO worker failed to start: {e}")
            self._yolo = None

        self.frames_per_batch = frames_per_batch
        self.context_len = max(1, int(context_seconds) * frames_per_batch)
        self.ema_alpha = ema_alpha
        self.topk = topk

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._sec_frames = []                              # raw BGR frames for the current second
        self._feat_buf = deque(maxlen=self.context_len)    # rolling downsampled CLIP features
        self._ema = None                                   # smoothed [140] sigmoid scores
        self.latest = None                                 # last broadcast payload
        self.subscribers = set()                           # result websockets
        self._frames_in = 0                                # ingest counter (for observed fps)
        self._running = False

    # ── called from the WebRTC recv loop (event loop, sub-ms) ───────────────────
    def add_frame(self, bgr):
        # Cap defensively so a backed-up second can't grow unbounded.
        if len(self._sec_frames) < 240:
            self._sec_frames.append(bgr)
        self._frames_in += 1

    def _downsample(self, frames):
        if len(frames) <= self.frames_per_batch:
            return frames
        idx = np.linspace(0, len(frames) - 1, self.frames_per_batch).astype(int)
        return [frames[i] for i in idx]

    # ── runs on the worker thread (serialized) ──────────────────────────────────
    def _shrink(self, frames):
        """Downscale to det_imgsz longest side to keep the pipe + YOLO cheap."""
        out = []
        for f in frames:
            h, w = f.shape[:2]
            s = self.det_imgsz / float(max(h, w))
            out.append(cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1.0 else f)
        return out

    def _detect(self, frames):
        """Send the batch to the YOLO worker process; return [{label,count,conf}]."""
        if not self._yolo or self._yolo.poll() is not None:
            return self._last_dets
        try:
            _send(self._yolo.stdin, self._shrink(frames))
            res = _recv(self._yolo.stdout)
            if isinstance(res, list):
                self._last_dets = res
            return self._last_dets
        except Exception as e:
            print("[live] yolo worker comm error:", e)
            return self._last_dets

    def _process(self, frames):
        dets = self._detect(frames)                                 # YOLO animal-group labels
        feats = clip_encode_frames(frames, self.clip_model, self.preprocess, self.device)  # [k,512] RAW
        for f in feats:
            self._feat_buf.append(f)
        window = np.stack(list(self._feat_buf)).astype(np.float32)   # [T,512], T <= context_len
        scores = self.recognizer.predict_scores(window)             # [140] sigmoid
        return scores, int(window.shape[0]), dets

    def _top(self, scores):
        idx = np.argsort(scores)[::-1][: self.topk]
        out = []
        for i in idx:
            i = int(i)
            info = self.action_info.get(i, {})
            out.append({
                "label": i,
                "action": info.get("action", f"#{i}"),
                "category": info.get("category", ""),
                "score": round(float(scores[i]), 4),
            })
        return out

    # ── 1 Hz loop ───────────────────────────────────────────────────────────────
    async def run(self, broadcast):
        """broadcast: async fn(payload_dict) -> None"""
        self._running = True
        loop = asyncio.get_event_loop()
        while self._running:
            t0 = time.monotonic()
            frames, self._sec_frames = self._sec_frames, []
            fps = len(frames)
            if frames:
                ds = self._downsample(frames)
                try:
                    scores, nframes, dets = await loop.run_in_executor(self.executor, self._process, ds)
                except Exception as e:                       # keep the loop alive on a bad tick
                    print("[live] inference error:", e)
                    scores = None
                if scores is not None:
                    a = self.ema_alpha
                    self._ema = scores if self._ema is None else a * scores + (1 - a) * self._ema
                    self.latest = {
                        "t": time.time(),
                        "actions": self._top(self._ema),
                        "all_scores": [round(float(s), 4) for s in self._ema],
                        "detections": dets,
                        "n_frames": nframes,
                        "fps": fps,
                    }
                    await broadcast(self.latest)
            dt = time.monotonic() - t0
            await asyncio.sleep(max(0.0, 1.0 - dt))

    def stop(self):
        self._running = False
        if getattr(self, "_yolo", None):
            try:
                self._yolo.stdin.close()
            except Exception:
                pass
            self._yolo.terminate()
        self.executor.shutdown(wait=False)
