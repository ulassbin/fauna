# Fauna demo web app — shared build spec

REPO: `/Users/michaeladewole/Others/hack/fauna` (branch: demo-app). Write files at the EXACT paths below.

## Environment
- Python 3.12 in a uv venv. Run anything with `uv run <cmd>` (e.g. `uv run python -m py_compile app/server.py`).
- Installed: torch, numpy, opencv-python (cv2), Pillow, fastapi, uvicorn[standard], python-multipart, and OpenAI CLIP (`import clip; model, preprocess = clip.load("ViT-B/32", device=str(dev))`).
- Device: the repo file `action_recognition/utils.py` defines `get_device()` returning a torch.device (CUDA > MPS > CPU). On this Mac it returns `mps`. Add the repo root and the `action_recognition` dir to `sys.path`, then `from utils import get_device`. Always pass `str(device)` to `clip.load`.
- Demo videos in `custom/`: cat.mp4, dog_eating.mp4, hawk_attack.mp4, wildcat.mp4, wolf.mp4 (~4961 frames, LONG), wolf2.mp4. Farm/wildlife footage.
- HACKATHON DEMO: must RUN reliably, look clean/modern (dark monitoring-dashboard aesthetic), demo well. No auth; single implicit user; all state in-memory (module-level) is fine. Do NOT overengineer.

## Product flow
- Home: grid of camera streams (from streams.json) + an "Upload video" entry. A notifications bell (unread count) is always visible in the header.
- Stream view: a looping `<video>` (autoplay, muted, loop). Two tools: (1) SEARCH free-text → jump to matched timepoint or "no match"; (2) ALERTS free-text standing query evaluated SERVER-SIDE.
- Upload view: SEARCH only (no alerts).
- Notifications: poll `GET /api/notifications` every 3s; unread badge on the bell; dropdown list newest-first; new ones pop a toast; clicking opens the stream seeked to the event time.

Engine = CLIP zero-shot image↔text similarity over per-frame features. No action-recognition model.

## FILE: app/clipsearch.py
Self-contained CLIP search engine. Keep names EXACT.
- `class ClipSearcher:`
  - `__init__(self, device=None)`: load CLIP ViT-B/32 once on `get_device()` (or given). Keep an in-memory dict of loaded indexes.
  - `index_video(self, video_id, video_path) -> dict`: decode the video; SUBSAMPLE frames — step = `max(1, round(fps/3))` (~3 fps), and if sampled count > 600 sample uniformly down to 600 (wolf.mp4 ~4961 frames, so this matters). CLIP-encode sampled frames → feats `[N,512]` float32 **L2-normalized**. Save `app/cache/{video_id}.npz` with arrays: feats, times (seconds, float32), fps, num_frames, duration. If the cache file exists, load it (skip re-encode). Return `{"video_id","fps","num_frames","duration","n_indexed"}`.
  - `encode_text(self, text) -> np.ndarray`: `[512]` float32, L2-normalized.
  - `search(self, video_id, text) -> dict`: `sims = feats @ text` → `[N]`; compute mean, std. `THR = max(0.21, mean + 1.5*std)`. `best = argmax(sims)`; `found = sims[best] >= THR`. matches = merge consecutive indices where `sims >= THR` into ranges; per range keep the peak `{time, score}`; sort by score desc. Return `{"found":bool, "best":{"time","score"}|null, "matches":[{"time","score"}...], "threshold":float}`.
  - `matching_times(self, video_id, text) -> list[float]`: representative time (sec) per merged range where `sims >= THR`. Used by the alert engine.
- IMPORTANT: CLIP cosine sims are ~0.18–0.32, so the RELATIVE threshold (mean + 1.5·std, floor 0.21) is what makes found/not-found sane. Do NOT use an absolute 0.5 threshold.

## FILE: app/server.py
FastAPI app. Runnable as `uv run python app/server.py` (uvicorn.run, host 127.0.0.1, port 8000). Also expose module-level `app`.
- Startup: load `app/streams.json` → `[{id,name,path}]` (resolve paths relative to repo root). Create one ClipSearcher; `index_video()` each stream (uses cache); start the asyncio alert engine.
- In-memory: streams (list), alerts (list of `{id,stream_id,query,created}`), notifications (list, capped ~200, newest-first), uploads (dict id→`{name,path,fps,duration}`).
- Endpoints (EXACT paths/shapes — the frontend depends on these):
  - `GET /` → serve app/static/index.html
  - `GET /api/streams` → `[{id,name,fps,num_frames,duration}]`
  - `GET /api/source/{id}` → `{id,name,fps,duration,kind:"stream"|"upload"}`
  - `GET /api/video/{id}` → serve the mp4 WITH HTTP RANGE support (206 partial) so `<video>` can seek. Works for stream AND upload ids. Implement Range header parsing yourself (FileResponse alone may not do range). Content-Type video/mp4.
  - `POST /api/search` body `{source_id, query}` → search(...) result + `{"message": human-readable}`
  - `GET /api/alerts` (optional `?stream_id=`) → list
  - `POST /api/alerts` body `{stream_id, query}` → create (precompute matching_times, store); return 400 if stream_id is an upload (alerts are stream-only)
  - `DELETE /api/alerts/{id}` → remove
  - `GET /api/notifications` → newest-first list
  - `POST /api/notifications/seen` body `{ids:[...]}` or `{}` (= all) → mark seen
  - `POST /api/upload` (multipart field name `file`) → save to app/uploads/, index it, register as an upload source, return `{id,name,fps,duration}`
- Alert engine (asyncio task, runs with NO client connected): record start time. A stream's virtual playback time = `(now - start) % duration`. Every ~1.0s: for each alert, if its stream's virtual time is within ~0.6s of any precomputed matching_time AND that alert+time has not fired within a COOLDOWN (~8s), append a notification `{id, alert_id, stream_id, stream_name, query, time (event time in the video), score, created (unix), seen:false}`; cap the list. So alerts fire as the loop passes a matching moment, even with no browser open.
- Robustness: wrap per-video indexing in try/except so one bad video does not crash startup; print logs to stdout.

## FILE: app/static/index.html
Single file. Tailwind via CDN: `<script src="https://cdn.tailwindcss.com"></script>`. Vanilla JS, no framework/build. Dark, modern security/monitoring dashboard look. JS view-switching (no router).
- Header: "Fauna" wordmark + tagline "behavioral sensing"; right side: a notifications BELL with an unread-count badge; clicking toggles a dropdown listing notifications (newest-first: stream name, query, event time; click → open that stream seeked to the time). Poll `GET /api/notifications` every 3s; on new ones bump the badge and show a toast.
- Home view: "Cameras" heading; responsive grid of stream cards (`GET /api/streams`). Each card: a small looping muted autoplay `<video src=/api/video/{id}>` thumbnail, the name, a green LIVE dot, and an active-alert count if any. Plus an "Upload a video" card → upload view.
- Stream view: back button; big looping muted autoplay `<video src=/api/video/{id}>`. Search panel: input + Search → `POST /api/search {source_id:id, query}`; on found, set `video.currentTime = best.time`, flash a highlight, show a green banner "Found at M:SS (score X.XX)", and show other matches as clickable chips that seek; on not found, an amber banner "No matching moment found." Alerts panel: input + Create alert → `POST /api/alerts {stream_id:id, query}`; list this stream's alerts (`GET /api/alerts?stream_id=id`) each with an x → DELETE; a small note "Alerts run on the server and notify you even when you are not watching."
- Upload view: file picker + "Upload & index" → `POST /api/upload` (multipart field `file`); disabled/progress state while indexing; on done show the uploaded video player + the SAME search panel (NO alerts panel).
- Helpers: format seconds as M:SS; seek via `video.currentTime`; inline error messages on fetch failure.
- Polish: rounded cards, subtle borders, refined dark palette, good spacing, clear active states. This is the demo's face — make it look like a real product.

## FILE: app/streams.json
JSON array, 3 seed entries (paths relative to repo root; the user will swap them later):
```
[ {"id":"cam1","name":"Barn Camera","path":"custom/cat.mp4"},
  {"id":"cam2","name":"Yard Camera","path":"custom/dog_eating.mp4"},
  {"id":"cam3","name":"Field Camera","path":"custom/wolf2.mp4"} ]
```

## FILE: app/README.md
Short: what it is; how to run (`uv run python app/server.py` → open http://127.0.0.1:8000); how to change streams (edit app/streams.json); notes (CLIP zero-shot search; server-side alerts; uploads are search-only). Also APPEND `app/cache/` and `app/uploads/` to the repo `.gitignore` (keep existing lines).
