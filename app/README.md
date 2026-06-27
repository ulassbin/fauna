# Fauna — demo web app

Behavioral sensing for animal health. This is a hackathon demo: a dark
security/monitoring-style dashboard that runs CLIP zero-shot search and
server-side standing alerts over camera footage.

The matching engine is CLIP (ViT-B/32) image↔text similarity computed over
per-frame features — no action-recognition model. You type free text and the
app finds the matching moment in a video, or fires a notification the moment a
standing query becomes true.

## Run

From the repo root:

```
uv run python app/server.py
```

Then open http://127.0.0.1:8000

On first launch each stream is indexed with CLIP (frames are subsampled to
~3 fps). Results are cached under `app/cache/`, so subsequent starts are fast.

## Change the streams

Edit `app/streams.json`. Each entry is `{ "id", "name", "path" }`, where `path`
is relative to the repo root:

```json
[
  { "id": "cam1", "name": "Barn Camera",  "path": "custom/cat.mp4" },
  { "id": "cam2", "name": "Yard Camera",  "path": "custom/dog_eating.mp4" },
  { "id": "cam3", "name": "Field Camera", "path": "custom/wolf2.mp4" }
]
```

Delete the matching `app/cache/{id}.npz` if you point an id at a new video and
want it re-indexed.

## Notes

- **Search** (streams and uploads): free-text query → jumps to the best-matching
  timepoint, or reports "no match". Threshold is relative (mean + 1.5·std,
  floored at 0.21) because CLIP cosine similarities are small (~0.18–0.32).
- **Alerts** (streams only): a standing free-text query evaluated on the server.
  The alert engine runs a virtual playback loop with no browser connected, so
  you get notified even when you are not watching. Notifications appear in the
  header bell (poll every 3s) and as toasts.
- **Uploads** are search-only — no alerts.
- All state is in-memory (single implicit user, no auth). Restarting the server
  clears alerts, notifications, and uploads; the per-video CLIP cache persists.
