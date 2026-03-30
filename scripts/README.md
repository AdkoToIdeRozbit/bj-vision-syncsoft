# Blackjack Vision Scripts

This directory contains standalone utility scripts for testing, evaluating, and interacting with the Blackjack Vision system from the command line.

## Available Scripts

### `ws_stream_video.py`

A WebSocket client that streams frames from a local video file to the live `/api/stream` endpoint on the server. It simulates a client application or an edge device capturing live game footage, validating real-time processing and pipeline throughput.

**Usage:**
```bash
uv run python scripts/ws_stream_video.py \
    --video /path/to/blackjack-sample-videos/480-blackjack.mp4 \
    --api-key your-api-key-here \
    --every-n-frames 3
```

**Key Arguments:**
*   `--video`: Path to the local video file to stream.
*   `--api-key`: API key required by the server (should match `API_KEY` in `.env`).
*   `--every-n-frames`: Adjusts the target framerate (e.g., `3` skips 2 out of 3 frames) to mimic a webcam's lower framerate over websockets.
*   `--profile`: Resolves server-side region-of-interest scaling (e.g., `480p`, `4k`). Defaults to `480p`.
*   `--host` / `--port`: Specify the destination server.

---

### `pattern-matching.py`

A standalone, offline script that processes an entire `.mp4` video locally and extracts game sessions directly via OpenCV template matching. It saves identified frames to an output folder (`game-sessions/` by default) and outputs a `results.json` mapping out detected cards.

This script runs entirely independently of the FastAPI server and is excellent for calibrating `.env` ROI parameters and template thresholds *before* using them in production.

**Usage:**
```bash
uv run python scripts/pattern-matching.py --video /path/to/video.mp4
```

*Note: Configuration parameters such as `REPLAY_ROI`, `REPLAY_THRESHOLD`, and `CARD_THRESHOLD` used by this script are pulled directly from your `.env` file.*

---

### `extract-session-ending-frames.py`

A focused utility that precisely scans a video for game-session boundaries (by detecting the replay button via template matching) and extracts only the final frame of each game (specifically, the 5th frame before the replay button appears).

This is ideal for rapidly generating clean, uncluttered frames of the card table right when all final cards are dealt, which are extremely useful for extracting new card templates or verifying model accuracy.

**Usage:**
```bash
uv run python scripts/extract-session-ending-frames.py --video input.mp4 \
    --template template-images/4k/replay.jpg \
    --roi 1000,1020,1150,1150 \
    --threshold 0.8
```

---

### `extract_frames.py`

A simple slice-and-dice script to convert an entire video into individual static image files at a specified frame interval.

**Usage:**
```bash
uv run python scripts/extract_frames.py \
    --video input.mp4 \
    --every-n-frames 10 \
    --output-dir frames
```

---

### `draw_bboxes.py`

A local visualization tool that takes an image and a raw list of absolute `x1,y1,x2,y2` pixel coordinates, drawing distinct colored rectangles over the image. This is highly useful for visually verifying if the ROI coordinates in your `.env` perfectly align with the table sections on a target frame length without needing to run the full pipeline.

**Usage:**
```bash
uv run python scripts/draw_bboxes.py test_frame.jpg 100,100,200,200 300,300,400,400 --show
```
