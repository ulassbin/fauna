"""Fauna demo — FastAPI backend.

Serves the single-page frontend, exposes CLIP search + server-side alerts,
streams videos with HTTP Range support, and runs a background asyncio alert
engine that fires notifications as virtual playback loops past matched
moments — even with no browser connected.

Run: `uv run python app/server.py`  →  http://127.0.0.1:8000
"""

import asyncio
import csv
import json
import mimetypes
import os
import re
import time
import uuid

import numpy as np

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from clipsearch import ClipSearcher

# ── Paths ───────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
STATIC_DIR = os.path.join(THIS_DIR, "static")
UPLOADS_DIR = os.path.join(THIS_DIR, "uploads")
STREAMS_JSON = os.path.join(THIS_DIR, "streams.json")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
PIPELINE_OUT = os.path.join(REPO_ROOT, "outputs", "pipeline")  # where decision.npz live
ACTIONS_CSV = os.path.join(REPO_ROOT, "actions_ak.csv")

# ── Alert-engine tuning ─────────────────────────────────────────────────
ALERT_TICK_SEC = 1.0       # how often the engine evaluates alerts
ALERT_WINDOW_SEC = 0.6     # virtual time must be within this of a match time
ALERT_COOLDOWN_SEC = 8.0   # don't refire the same (alert,time) within this
NOTIF_CAP = 200            # max notifications retained

# ── In-memory state (single implicit user) ──────────────────────────────
searcher: ClipSearcher = None         # created at startup
streams = []                          # [{id,name,path,fps,num_frames,duration}]
alerts = []                           # [{id,stream_id,query,created,times}]
notifications = []                    # newest-first, capped
uploads = {}                          # id -> {name,path,fps,duration,num_frames}
_action_names = None                  # label -> action name (lazy from actions_ak.csv)
_decision_cache = {}                  # stream_id -> parsed decision dict or {available:False}

_engine_start = None                  # monotonic-ish wall start for virtual time
_last_fired = {}                      # (alert_id, round(time)) -> last fire unix
_alert_task = None                    # asyncio task handle

app = FastAPI(title="Fauna")


# ── Helpers ─────────────────────────────────────────────────────────────
def _stream_by_id(sid):
    for s in streams:
        if s["id"] == sid:
            return s
    return None


def _source_record(sid):
    """Return ('stream'|'upload', record) or (None, None)."""
    s = _stream_by_id(sid)
    if s is not None:
        return "stream", s
    if sid in uploads:
        return "upload", uploads[sid]
    return None, None


def _path_for_source(sid):
    kind, rec = _source_record(sid)
    if rec is None:
        return None
    return rec["path"]


def _fmt_time(secs):
    secs = max(0, int(round(secs)))
    return f"{secs // 60}:{secs % 60:02d}"


def _push_notification(notif):
    notifications.insert(0, notif)
    del notifications[NOTIF_CAP:]


def _load_streams():
    """Read streams.json, resolve paths relative to repo root."""
    if not os.path.exists(STREAMS_JSON):
        print(f"[fauna] WARNING: {STREAMS_JSON} not found; no seed streams.")
        return []
    with open(STREAMS_JSON, "r") as f:
        raw = json.load(f)
    out = []
    for entry in raw:
        path = entry["path"]
        if not os.path.isabs(path):
            path = os.path.join(REPO_ROOT, path)
        out.append({"id": entry["id"], "name": entry["name"], "path": path})
    return out


# ── decision.npz (action timeline + species) ─────────────────────────────
def _load_action_names():
    global _action_names
    if _action_names is None:
        names = {}
        try:
            with open(ACTIONS_CSV) as f:
                for row in csv.DictReader(f):
                    names[int(row["Label"])] = row["Action"]
        except Exception as exc:
            print(f"[fauna] could not load actions csv: {exc}")
        _action_names = names
    return _action_names


def _decision_path_for(s):
    stem = os.path.splitext(os.path.basename(s["path"]))[0]
    return os.path.join(PIPELINE_OUT, stem, "decision.npz")


def _parse_decision(path):
    """Parse a pipeline decision.npz into per-frame top-6 actions + species."""
    d = np.load(path, allow_pickle=True)
    ga = d["global_actions"]                 # [T, 140] per-frame action scores
    T = int(ga.shape[0])
    names = _load_action_names()
    order = np.argsort(ga, axis=1)[:, ::-1][:, :6]   # top-6 indices per frame
    frames = []
    for t in range(T):
        frames.append([
            {"action": names.get(int(i), str(int(i))),
             "confidence": round(float(ga[t, int(i)]), 3)}
            for i in order[t]
        ])
    # dominant actor's species (the one present in the most frames)
    species, broad, best = None, None, -1.0
    exist = d["existence"] if "existence" in d.files else None
    for i in range(10):
        if f"actor_{i}_species" in d.files:
            pc = float(exist[i, :, 0].sum()) if exist is not None else 0.0
            if pc > best:
                best = pc
                species = str(d[f"actor_{i}_species"])
                broad = str(d[f"actor_{i}_broad"]) if f"actor_{i}_broad" in d.files else None
    return {"available": True, "n_frames": T, "species": species,
            "broad": broad, "frames": frames}


def _get_decision(sid):
    if sid in _decision_cache:
        return _decision_cache[sid]
    res = {"available": False}
    s = _stream_by_id(sid)
    if s is not None:
        path = _decision_path_for(s)
        if os.path.exists(path):
            try:
                res = _parse_decision(path)
            except Exception as exc:
                print(f"[fauna] decision parse error for {sid}: {exc}")
    _decision_cache[sid] = res
    return res


# ── Startup ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    global searcher, streams, _engine_start, _alert_task
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    print("[fauna] Loading CLIP model...")
    searcher = ClipSearcher()
    print(f"[fauna] CLIP ready on device: {searcher.device}")

    streams = _load_streams()
    for s in streams:
        try:
            if not os.path.exists(s["path"]):
                print(f"[fauna] WARNING: missing video for {s['id']}: {s['path']}")
                s["fps"] = 0.0
                s["num_frames"] = 0
                s["duration"] = 0.0
                continue
            print(f"[fauna] Indexing {s['id']} ({s['name']})...")
            meta = searcher.index_video(s["id"], s["path"])
            s["fps"] = meta["fps"]
            s["num_frames"] = meta["num_frames"]
            s["duration"] = meta["duration"]
            print(
                f"[fauna]   {s['id']}: {meta['n_indexed']} frames indexed, "
                f"duration={meta['duration']:.1f}s"
            )
        except Exception as exc:  # never let one bad video kill startup
            print(f"[fauna] ERROR indexing {s['id']}: {exc}")
            s["fps"] = 0.0
            s["num_frames"] = 0
            s["duration"] = 0.0

    _engine_start = time.time()
    _alert_task = asyncio.create_task(_alert_engine())
    print("[fauna] Alert engine started. Ready.")


@app.on_event("shutdown")
async def _shutdown():
    if _alert_task is not None:
        _alert_task.cancel()


# ── Alert engine ────────────────────────────────────────────────────────
async def _alert_engine():
    """Loop forever; fire notifications as virtual playback passes matches."""
    while True:
        try:
            now = time.time()
            for alert in list(alerts):
                s = _stream_by_id(alert["stream_id"])
                if s is None:
                    continue
                duration = s.get("duration") or 0.0
                if duration <= 0:
                    continue
                virtual_t = (now - _engine_start) % duration
                for mt in alert.get("times", []):
                    # circular distance between virtual time and match time
                    d = abs(virtual_t - mt)
                    d = min(d, duration - d)
                    if d <= ALERT_WINDOW_SEC:
                        key = (alert["id"], round(mt, 1))
                        last = _last_fired.get(key, 0.0)
                        if now - last >= ALERT_COOLDOWN_SEC:
                            _last_fired[key] = now
                            _push_notification(
                                {
                                    "id": uuid.uuid4().hex,
                                    "alert_id": alert["id"],
                                    "stream_id": s["id"],
                                    "stream_name": s["name"],
                                    "query": alert["query"],
                                    "time": float(mt),
                                    "score": float(alert.get("scores", {}).get(round(mt, 1), 0.0)),
                                    "created": now,
                                    "seen": False,
                                }
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[fauna] alert engine error: {exc}")
        await asyncio.sleep(ALERT_TICK_SEC)


# ── Range-aware video serving ───────────────────────────────────────────
def _send_video(path, request: Request):
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="video not found")

    file_size = os.path.getsize(path)
    content_type = "video/mp4"
    range_header = request.headers.get("range") or request.headers.get("Range")

    if range_header is None:
        # Full content; still advertise range support.
        def _full():
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _full(),
            media_type=content_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
        )

    # Parse "bytes=start-end"
    m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not m:
        raise HTTPException(status_code=400, detail="invalid Range header")
    start_s, end_s = m.group(1), m.group(2)

    if start_s == "":
        # suffix range: last N bytes
        length = int(end_s)
        if length <= 0:
            raise HTTPException(status_code=416, detail="invalid range")
        start = max(0, file_size - length)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1

    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    length = end - start + 1

    def _ranged():
        remaining = length
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(
        _ranged(), status_code=206, media_type=content_type, headers=headers
    )


# ── Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    if not os.path.exists(INDEX_HTML):
        return JSONResponse(
            {"error": "index.html not found", "expected": INDEX_HTML},
            status_code=404,
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/streams")
async def api_streams():
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "fps": s.get("fps", 0.0),
            "num_frames": s.get("num_frames", 0),
            "duration": s.get("duration", 0.0),
        }
        for s in streams
    ]


@app.get("/api/source/{sid}")
async def api_source(sid: str):
    kind, rec = _source_record(sid)
    if rec is None:
        raise HTTPException(status_code=404, detail="source not found")
    return {
        "id": sid,
        "name": rec["name"],
        "fps": rec.get("fps", 0.0),
        "duration": rec.get("duration", 0.0),
        "kind": kind,
    }


@app.get("/api/actions/{sid}")
async def api_actions(sid: str):
    """Per-frame top-6 actions + species for a stream (from its decision.npz).
    Returns {available:false} for uploads or streams without a decision file."""
    return await asyncio.to_thread(_get_decision, sid)


@app.get("/api/video/{sid}")
async def api_video(sid: str, request: Request):
    path = _path_for_source(sid)
    if path is None:
        raise HTTPException(status_code=404, detail="source not found")
    return _send_video(path, request)


@app.post("/api/search")
async def api_search(payload: dict):
    source_id = payload.get("source_id")
    query = (payload.get("query") or "").strip()
    if not source_id or not query:
        raise HTTPException(status_code=400, detail="source_id and query required")
    kind, rec = _source_record(source_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="source not found")

    try:
        result = await asyncio.to_thread(searcher.search, source_id, query)
    except KeyError:
        raise HTTPException(status_code=404, detail="source not indexed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {exc}")

    if result["found"] and result["best"]:
        b = result["best"]
        message = f"Found at {_fmt_time(b['time'])} (score {b['score']:.2f})"
    else:
        message = "No matching moment found."
    result["message"] = message
    return result


@app.get("/api/alerts")
async def api_alerts(stream_id: str = None):
    out = []
    for a in alerts:
        if stream_id is not None and a["stream_id"] != stream_id:
            continue
        out.append(
            {
                "id": a["id"],
                "stream_id": a["stream_id"],
                "query": a["query"],
                "created": a["created"],
            }
        )
    return out


@app.post("/api/alerts")
async def api_create_alert(payload: dict):
    stream_id = payload.get("stream_id")
    query = (payload.get("query") or "").strip()
    if not stream_id or not query:
        raise HTTPException(status_code=400, detail="stream_id and query required")

    # Alerts are stream-only.
    if stream_id in uploads:
        raise HTTPException(
            status_code=400, detail="alerts are not supported for uploads"
        )
    s = _stream_by_id(stream_id)
    if s is None:
        raise HTTPException(status_code=404, detail="stream not found")

    try:
        result = await asyncio.to_thread(searcher.search, stream_id, query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"alert precompute failed: {exc}")

    times = [m["time"] for m in result.get("matches", [])]
    scores = {round(m["time"], 1): m["score"] for m in result.get("matches", [])}

    alert = {
        "id": uuid.uuid4().hex,
        "stream_id": stream_id,
        "query": query,
        "created": time.time(),
        "times": times,
        "scores": scores,
    }
    alerts.append(alert)
    return {
        "id": alert["id"],
        "stream_id": alert["stream_id"],
        "query": alert["query"],
        "created": alert["created"],
        "n_matches": len(times),
    }


@app.delete("/api/alerts/{alert_id}")
async def api_delete_alert(alert_id: str):
    global alerts
    before = len(alerts)
    alerts = [a for a in alerts if a["id"] != alert_id]
    if len(alerts) == before:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"ok": True, "id": alert_id}


@app.get("/api/notifications")
async def api_notifications():
    # Already maintained newest-first.
    return notifications


@app.post("/api/notifications/seen")
async def api_notifications_seen(payload: dict = None):
    ids = None
    if payload:
        ids = payload.get("ids")
    if not ids:  # None or empty -> mark all
        for n in notifications:
            n["seen"] = True
    else:
        idset = set(ids)
        for n in notifications:
            if n["id"] in idset:
                n["seen"] = True
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    uid = "up_" + uuid.uuid4().hex[:10]

    orig = file.filename or "upload.mp4"
    base, ext = os.path.splitext(orig)
    if not ext:
        ext = ".mp4"
    safe_name = orig
    dest = os.path.join(UPLOADS_DIR, uid + ext)

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    try:
        meta = await asyncio.to_thread(searcher.index_video, uid, dest)
    except Exception as exc:
        # Clean up the unusable file.
        try:
            os.remove(dest)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"could not index upload: {exc}")

    uploads[uid] = {
        "name": safe_name,
        "path": dest,
        "fps": meta["fps"],
        "duration": meta["duration"],
        "num_frames": meta["num_frames"],
    }
    return {
        "id": uid,
        "name": safe_name,
        "fps": meta["fps"],
        "duration": meta["duration"],
    }


if __name__ == "__main__":
    import uvicorn

    # Ensure mp4 maps correctly for any non-range fallbacks.
    mimetypes.add_type("video/mp4", ".mp4")
    uvicorn.run(app, host="127.0.0.1", port=8000)
