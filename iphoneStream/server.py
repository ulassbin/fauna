import os
import ssl
import asyncio
from datetime import datetime

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from live_infer import LiveActionEngine

_HERE = os.path.dirname(os.path.abspath(__file__))

pcs = set()
relay = MediaRelay()
latest_video_track = None
ENGINE: LiveActionEngine | None = None


def _tpl(name):
    return web.FileResponse(os.path.join(_HERE, "templates", name))


#
# Pages
#
async def index_page(request):
    return _tpl("live.html")


async def live_page(request):
    return _tpl("live.html")


async def phone_page(request):
    return _tpl("phone.html")


async def viewer_page(request):
    return _tpl("viewer.html")


#
# Results WebSocket — pushes 1 Hz action predictions to viewers
#
async def broadcast(payload):
    if ENGINE is None:
        return
    dead = []
    for ws in list(ENGINE.subscribers):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ENGINE.subscribers.discard(ws)


async def results_ws(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    if ENGINE is not None:
        ENGINE.subscribers.add(ws)
        if ENGINE.latest:
            await ws.send_json(ENGINE.latest)
    try:
        async for _ in ws:        # we don't expect inbound messages; keep the socket open
            pass
    finally:
        if ENGINE is not None:
            ENGINE.subscribers.discard(ws)
    return ws


#
# Phone WebRTC endpoint (publish)
#
async def offer(request):
    global latest_video_track
    params = await request.json()
    desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("Phone connected")

    @pc.on("track")
    def on_track(track):
        global latest_video_track
        print(f"Received {track.kind} track")
        if track.kind == "video":
            latest_video_track = relay.subscribe(track)        # relay to viewers
            proc = relay.subscribe(track)                      # separate sub for processing

            async def consume():
                while True:
                    try:
                        frame = await proc.recv()
                        if ENGINE is not None:
                            ENGINE.add_frame(frame.to_ndarray(format="bgr24"))
                    except Exception as e:
                        print("Processor stopped:", e)
                        break

            asyncio.create_task(consume())

    @pc.on("connectionstatechange")
    async def on_state():
        print("Phone state:", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


#
# Viewer WebRTC endpoint (subscribe, recvonly)
#
async def viewer_offer(request):
    params = await request.json()
    desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("Viewer connected")

    @pc.on("connectionstatechange")
    async def on_state():
        print("Viewer state:", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(desc)
    if latest_video_track:
        print("Sending video to viewer")
        pc.addTrack(latest_video_track)
    else:
        print("No video track available yet")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


#
# Lifecycle
#
async def on_startup(app):
    global ENGINE
    print("Loading models (CLIP + action recognizer)…")
    ENGINE = LiveActionEngine()
    app["infer_task"] = asyncio.create_task(ENGINE.run(broadcast))
    print("Ready.")


async def on_shutdown(app):
    print("Shutting down")
    if ENGINE is not None:
        ENGINE.stop()
    task = app.get("infer_task")
    if task:
        task.cancel()
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)


#
# App
#
app = web.Application()
app.router.add_get("/", index_page)
app.router.add_get("/live", live_page)
app.router.add_get("/phone", phone_page)
app.router.add_get("/viewer", viewer_page)
app.router.add_get("/results", results_ws)
app.router.add_post("/offer", offer)
app.router.add_post("/viewer-offer", viewer_offer)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)


if __name__ == "__main__":
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        os.path.join(_HERE, "cert.pem"),
        os.path.join(_HERE, "key.pem"),
    )
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=ssl_context)
