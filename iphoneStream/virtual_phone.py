"""Virtual phone — publish a local video to the running iphoneStream server over
WebRTC, exactly like templates/phone.html does. Lets you test the live action
pipeline + /live viewer WITHOUT a real phone or cert trust.

Usage:
    uv run python iphoneStream/virtual_phone.py [video] [server_url]
    # defaults: custom/dog_eating.mp4  https://127.0.0.1:8443

Then open the viewer in a browser:  https://<this-mac-ip>:8443/live
(Ctrl+C to stop.)
"""

import asyncio
import os
import ssl
import sys

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


async def main(video, base):
    if not os.path.isabs(video):
        video = os.path.join(REPO, video)
    if not os.path.exists(video):
        print(f"video not found: {video}")
        return

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # self-signed server cert

    pc = RTCPeerConnection()
    player = MediaPlayer(video, loop=True)   # loop so it keeps streaming
    pc.addTrack(player.video)
    await pc.setLocalDescription(await pc.createOffer())

    async with aiohttp.ClientSession() as s:
        async with s.post(
            base + "/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            ssl=ctx,
        ) as r:
            ans = await r.json()
        await pc.setRemoteDescription(RTCSessionDescription(**ans))

    print(f"📡 publishing {os.path.basename(video)} -> {base}  (looping)")
    print(f"   open the viewer:  {base.replace('127.0.0.1','<mac-lan-ip>')}/live")
    print("   Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()      # keep the connection alive
    finally:
        await pc.close()


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "custom/dog_eating.mp4"
    base = sys.argv[2] if len(sys.argv) > 2 else "https://127.0.0.1:8443"
    try:
        asyncio.run(main(video, base))
    except KeyboardInterrupt:
        pass
