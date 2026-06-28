"""Standalone YOLO detector worker — runs in its OWN process.

WHY a separate process: ultralytics pulls in cv2's bundled ffmpeg (libswscale),
which collides with aiortc's `av` (its own libswscale) in one process and SIGBUSes
`frame.to_ndarray()`. This worker imports ONLY ultralytics/cv2 (never aiortc/av),
so the two never share an interpreter.

Protocol: length-prefixed pickle. Parent -> stdin: a list of BGR numpy frames.
Worker -> stdout: a list of {label,count,conf} (or {"ready":...}/{"error":...}).
The real stdout fd is dup'd before any library import so ultralytics' own prints
can't corrupt the binary channel.
"""

import os
import sys
import struct
import pickle

# ── isolate the binary channel: dup fd 1, then send library stdout to stderr ──
_REAL_OUT = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)                       # anything printing to stdout now goes to stderr
sys.stdout = sys.stderr


def _send(obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _REAL_OUT.write(struct.pack(">I", len(b)))
    _REAL_OUT.write(b)
    _REAL_OUT.flush()


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


def main():
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = sys.argv[4] if len(sys.argv) > 4 else os.path.join(repo, "weights", "best.pt")

    from ultralytics import YOLO
    y = YOLO(model)
    names = y.names
    yolo_dev = 0 if dev == "cuda" else dev
    _send({"ready": True, "classes": list(names.values())})

    stdin = sys.stdin.buffer
    while True:
        frames = _recv(stdin)
        if frames is None:
            break
        try:
            results = y.predict(frames, imgsz=imgsz, conf=conf, device=yolo_dev, verbose=False)
            maxcount, maxconf = {}, {}
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                cls = boxes.cls.cpu().numpy().astype(int)
                cf = boxes.conf.cpu().numpy()
                fcount = {}
                for c, p in zip(cls, cf):
                    nm = names.get(int(c), str(int(c)))
                    fcount[nm] = fcount.get(nm, 0) + 1
                    maxconf[nm] = max(maxconf.get(nm, 0.0), float(p))
                for nm, ct in fcount.items():
                    maxcount[nm] = max(maxcount.get(nm, 0), ct)
            dets = [{"label": nm, "count": maxcount[nm], "conf": round(maxconf[nm], 3)} for nm in maxcount]
            dets.sort(key=lambda d: -d["conf"])
            _send(dets)
        except Exception as e:
            _send({"error": str(e)})


if __name__ == "__main__":
    main()
