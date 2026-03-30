"""Stream a video file to the WebSocket frame-processing endpoint.

Reads a local video with OpenCV, encodes each frame as a JPEG, and sends it
over the WebSocket.  Prints every server response to stdout.

Usage
-----
    python scripts/ws_stream_video.py --video path/to/video.mp4 \\
        --api-key YOUR_KEY \\
        [--host 127.0.0.1] [--port 8000] \\
        [--profile 480p] \\
        [--every-n-frames 5] \\
        [--jpeg-quality 80]

Arguments
---------
--video           Path to the input video file (required).
--api-key         API key used to authenticate with the server (required).
--host            Server hostname (default: 127.0.0.1).
--port            Server port (default: 8000).
--profile         Resolution profile sent to the server: 480p or 4k
                  (default: 480p).
--every-n-frames  Send one frame every N frames; useful for reducing load
                  without losing coverage (default: 1 — send every frame).
--jpeg-quality    JPEG encode quality 1-100 (default: 80).  Lower values
                  reduce bandwidth at the cost of detection accuracy.
--no-progress     Suppress the per-frame progress line.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import cv2
import websockets
from websockets.exceptions import ConnectionClosedError

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a video file to the blackjack-vision WebSocket endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--api-key", required=True, help="X-API-Key value")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument(
        "--profile",
        default="480p",
        choices=["480p", "4k"],
        help="Resolution profile",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        metavar="N",
        help="Send 1 frame every N frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        metavar="Q",
        help="JPEG encode quality (1-100)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-frame progress output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_frame(frame, jpeg_quality: int) -> bytes:
    """Encode an OpenCV BGR frame as a JPEG byte array."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _format_response(msg: dict) -> str:
    msg_type = msg.get("type", "unknown")
    if msg_type == "status":
        return f"[status] task_id={msg.get('task_id')} status={msg.get('status')}"
    if msg_type == "error":
        return f"[error] {msg.get('detail', msg)}"
    if msg_type == "result":
        parts = [f"session_id={msg.get('session_number')}"]
        data = msg.get("data", {})
        dealer = data.get("dealer_cards") or []
        if dealer:
            parts.append(f"dealer={dealer}")
        for i in range(1, 8):
            cards = data.get(f"player{i}_cards") or []
            if cards:
                parts.append(f"p{i}={cards}")
        return "[result] " + "  ".join(parts)
    return f"[{msg_type}] {msg}"


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def stream(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {video_path.name}  |  frames={total_frames}  fps={fps:.1f}")

    uri = f"ws://{args.host}:{args.port}/api/stream?profile={args.profile}&api_key={args.api_key}"
    print(f"Connecting to {uri} ...\n")

    results: list[dict] = []
    
    # We will use tasks to concurrently send frames and receive responses.
    # Shared state for progress tracking:
    state = {"sent": 0, "detections": 0, "done": False}

    async def _receive_loop(ws):
        try:
            while not state["done"]:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") == "result":
                    state["detections"] += 1
                    results.append(msg)
                
                # Print response unconditionally so user sees status/results immediately
                print(f"\n<<< {_format_response(msg)}")
        except ConnectionClosedError:
            pass # normal if we close it
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception as e:
            if not state["done"]:
                print(f"\nReceiver error: {e}", file=sys.stderr)

    try:
        async with websockets.connect(uri) as ws:
            # Start the background receiver task
            receiver_task = asyncio.create_task(_receive_loop(ws))
            
            # Wait briefly to let the "status" message arrive before spamming frames
            await asyncio.sleep(0.5)

            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1

                if (frame_idx - 1) % args.every_n_frames != 0:
                    continue

                # Encode to raw bytes and send
                buf = _encode_frame(frame, args.jpeg_quality)
                await ws.send(buf)
                state["sent"] += 1

                if not args.no_progress:
                    pct = frame_idx / total_frames * 100 if total_frames else 0
                    line = (
                        f"\rFrame {frame_idx:>6}/{total_frames}"
                        f"  sent={state['sent']}  detections={state['detections']}"
                        f"  [{pct:5.1f}%]"
                    )
                    print(line, end="", flush=True)

                # Small sleep to yield event loop (prevent completely starving the receiver)
                # and to roughly simulate streaming rather than dumping the file instantly.
                await asyncio.sleep(0.001)

            state["done"] = True
            
            # Wait a moment for trailing results to arrive
            await asyncio.sleep(1.0)
            receiver_task.cancel()

    except websockets.exceptions.InvalidStatus as exc:
        print(
            f"\nConnection refused (HTTP {exc.response.status_code}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ConnectionClosedError as exc:
        print(
            f"\nConnection closed by server (code {exc.rcvd.code if exc.rcvd else '?'}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.release()

    print(f"\n\nDone. Sent {state['sent']} frame(s), {state['detections']} detection(s).\n")

    if results:
        print("=== Detections ===")
        for r in results:
            print(_format_response(r))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(stream(args))
