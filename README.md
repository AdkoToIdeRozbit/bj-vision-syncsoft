# Blackjack Vision

FastAPI service for blackjack video tracking tasks.

This repository provides:

- FastAPI app bootstrap and health check
- Upload endpoint for video processing tasks (with background vision processing)
- Task status endpoint
- SQLModel models for tracking tasks and game sessions
- OpenCV-based card detection via template matching (`app/vision/`)
- Alembic migration setup for MySQL
- Pytest API tests using in-memory SQLite

## Tech Stack

- Python `>=3.13`
- [FastAPI](https://fastapi.tiangolo.com)
- [SQLModel ORM](https://sqlmodel.tiangolo.com) (built on top of [SQLAlchemy](https://www.sqlalchemy.org))
- [Alembic](https://alembic.sqlalchemy.org/en/latest/) database migration tool
- [OpenCV](https://opencv.org) for video processing and template matching
- MySQL (runtime)
- [uv](https://docs.astral.sh/uv/) (dependency and environment management)

## Project Structure

```text
.
├── main.py                      # FastAPI app entry point
├── app/
│   ├── core/
│   │   ├── config.py            # Environment-based settings
│   │   └── db.py                # SQLAlchemy engine
│   ├── models.py                # SQLModel models and enums
│   ├── routes/
│   │   ├── deps.py              # DB session dependency
│   │   ├── game_sessions.py     # Game session list + CSV export endpoints
│   │   ├── stream.py            # WebSocket streaming endpoint
│   │   └── tasks.py             # Upload + status API + background worker
│   ├── vision/
│   │   ├── processor.py         # Card detection logic (OpenCV)
│   │   └── template-images/     # Card and replay-button templates
│   │       ├── 480p/            # Templates for 480p video sources
│   │       └── 4k/              # Templates for 4K video sources
│   └── alembic/                 # Migration environment + versions
├── frontend/
│   └── index.html               # Mini frontend app for testing the WebSocket stream
├── scripts/
│   └── pattern-matching.py      # Standalone CLI tool (same vision logic)
├── tests/                       # API + unit tests
├── alembic.ini
└── pyproject.toml
```

## Installation

Please follow this [guide](https://docs.astral.sh/uv/getting-started/installation) to install uv on your machine first, then:

```bash
uv sync
```

## Configuration

Create `.env` from `.env.example` and fill values:

```env
PROJECT_NAME="Blackjack Vision"
API_KEY=your-secret-api-key
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=username
MYSQL_PASSWORD=password
MYSQL_DB=blackjack_vision
```

Additional settings (with defaults) from `app/core/config.py`:

- `ENVIRONMENT=local`
- `LOCAL_UPLOAD_DIR=data/uploads`
- `MAX_UPLOAD_MB=512`
- `REPLAY_THRESHOLD=0.8` — confidence threshold for replay-button detection
- `CARD_THRESHOLD=0.8` — confidence threshold for card template matching
- `CUT_CARD_THRESHOLD=0.15` — minimum white-blob fraction for cut-card detection (see [Cut Card & Deck Tracking](#cut-card--deck-tracking) below)

## Database and Migrations

The app builds a MySQL DSN from environment variables:

`mysql+pymysql://<user>:<password>@<host>:<port>/<db>`

Apply migrations:

```bash
uv run alembic upgrade head
```

Create a new migration after model changes:

```bash
uv run alembic revision --autogenerate -m "describe_change"
uv run alembic upgrade head
```

## Run the API

```bash
uv run fastapi dev
```

Open Swagger UI at:

- `http://127.0.0.1:8000/docs`

All endpoints require an `X-API-Key` header matching the `API_KEY` environment variable.

## API Endpoints

### Health

- `GET /health`
  - Returns: `{"status": "ok"}`

### Tasks

- `POST /api/tasks/upload`
  - Multipart form fields:
    - `file` — video file (`video/*` content type required)
    - `profile` — resolution profile: `480p` (default) or `4k`
  - Enforces max upload size from `MAX_UPLOAD_MB`
  - Creates a `blackjack_tracking_task` row with `pending` status
  - Triggers background vision processing (status advances to `processing`, then `completed` or `failed`)
  - Returns: `{"task_id": <int>}`
  - Enforces a single active upload via a thread-safe in-memory lock

- `GET /api/tasks/{task_id}/status`
  - Returns task status payload:
    - `task_id`
    - `status` (`pending | processing | completed | failed`)
    - `csv_file_url`
    - `result`

### Stream

- `WS /api/stream`
  - Connect via WebSocket to stream individual image frames (as binary bytes like JPEG or PNG) for real-time card detection.
  - Query parameters:
    - `profile` — resolution profile: `480p` (default) or `4k` (not recommend video resolution higher than 720p)
    - `lookback` — sets how many frames back to inspect when the replay button is detected (default: 2, optimized for ~10fps streams).
  - Emits JSON payloads over the socket in real-time as `blackjack_game_session` rows are detected and saved. 
  - Status updates are also provided (e.g. `{"type": "status", "status": "processing"}`).

### Game Sessions

- `GET /api/game-sessions`
  - Returns paginated list of detected game sessions
  - Query params: `page`, `page_size`, `created_at_from`, `created_at_to`

- `GET /api/game-sessions/export`
  - Returns a CSV download of all game sessions
  - Query params: `created_at_from`, `created_at_to`

## Vision Processing

When a video is uploaded, the background worker:

1. Updates the task status to `processing`
2. Loads the card and replay-button templates for the selected resolution profile from `app/vision/template-images/`
3. Scans the video frame-by-frame for the replay button (using OpenCV template matching) to detect session boundaries; simultaneously monitors the cut-card ROI to track physical shoe changes (see [Cut Card & Deck Tracking](#cut-card--deck-tracking))
4. For each detected session-ending frame, identifies dealer and player cards via template matching, with split-hand support (see below)
5. Persists each detected game as a `blackjack_game_session` row (JSON result with cards per player/dealer, plus `session_number` and `deck_num`)
6. Updates the task status to `completed` (or `failed` on error)

### Cut Card & Deck Tracking

The processor watches a configurable `cut_card_roi` for the white plastic cut card that signals the end of a shoe. Detection uses a connected-component (blob) analysis on HSV-filtered frames:

1. Pixels that are bright (`V > 200`) **and** near-achromatic (`S < 80`) are extracted into a binary mask.
2. The mask is lightly dilated to bridge gaps from motion blur or the card's tilted edge.
3. The largest white blob is measured. If it covers at least `CUT_CARD_THRESHOLD` (default 15 %) of the ROI area the cut card is considered present.
4. **Left-edge rejection**: blobs that touch only the left border (dealer's shirt/sleeve) are discarded; the real cut card always reaches the right border of the ROI (inside the shoe).

When the cut card is detected, a `pending_deck_swap` flag is set. At the next session boundary the `deck_num` counter increments and the per-shoe `session_number` resets to 1. Emitted results carry both fields so every hand can be correlated with its physical shoe.

**Consecutive-swap cancellation**: if two back-to-back sessions both trigger a deck swap (possible when the cut card lingers across a session boundary), the *first* swap is treated as a false positive. The processor maintains an undo snapshot of the pre-swap counters and automatically restores them before applying the second (real) swap, ensuring `deck_num` and `session_number` are stamped correctly on both results.

### Split Hand Detection

Each player slot has up to three Region-of-Interest (ROI) areas configured via `PlayerROIConfig`:

| Field | Description |
|-------|-------------|
| `default` | Standard single-hand ROI |
| `split1` | First split hand ROI (optional) |
| `split2` | Second split hand ROI (optional) |

Detection follows a short-circuit cascade per player:

1. Check `default` ROI — if cards found, record as `{"hand1": [...]}` and stop.
2. Otherwise try `split1` — if cards found, also check `split2` and record as `{"hand1": [...], "hand2": [...]}` (omitting `hand2` if empty).
3. If nothing is found, the player field is `null`.

`split1` and `split2` ROI coordinates default to `null` in all profiles; set them in `_PROFILE_DEFAULTS` inside `app/vision/processor.py` once the exact pixel boundaries for each resolution are known.

As a result, the `player{1-7}_cards` field in `blackjack_game_session` is now a `dict` (or `null`), not a flat list:

```json
{
  "dealer_cards": ["ace_spades", "king_hearts"],
  "player1_cards": {"hand1": ["seven_clubs", "eight_diamonds"]},
  "player2_cards": {"hand1": ["five_spades"], "hand2": ["queen_hearts"]},
  "player3_cards": null
}
```

The CSV export (`GET /api/game-sessions/export`) serialises split hands as `hand1: card|card; hand2: card|card`.

### ROI Profile Defaults

Resolution profiles and their ROI defaults are defined in `app/vision/processor.py`.

Each player ROI is a `PlayerROIConfig` with three optional fields — `default`, `split1`, `split2` — encoded as `(x1, y1, x2, y2)` pixel rectangles. `None` means that seat/hand position is not yet calibrated for that profile.

#### Global ROIs

| Profile | Replay ROI | Dealer ROI |
|---------|------------|------------|
| `480p`  | `220,223,240,240` | `230,83,290,91` |
| `4k`    | `1000,1020,1150,1150` | `1070,560,1300,593` |

#### Player ROIs — `480p`

| Player | `default` | `split1` | `split2` |
|--------|-----------|----------|----------|
| player_1 | `null` | `null` | `null` |
| player_2 | `null` | `null` | `null` |
| player_3 | `null` | `null` | `null` |
| player_4 | `399,290,407,374` | `null` | `null` |
| player_5 | `null` | `null` | `null` |
| player_6 | `525,260,533,358` | `null` | `null` |
| player_7 | `null` | `null` | `null` |

#### Player ROIs — `4k`

| Player | `default` | `split1` | `split2` |
|--------|-----------|----------|----------|
| player_1 | `1045,1183,1077,1467` | `null` | `null` |
| player_2 | `1225,1260,1255,1534` | `null` | `null` |
| player_3 | `1440,1300,1475,1580` | `null` | `null` |
| player_4 | `1675,1240,1705,1594` | `null` | `null` |
| player_5 | `1910,1300,1938,1578` | `null` | `null` |
| player_6 | `2126,1270,2156,1536` | `null` | `null` |
| player_7 | `2305,1200,2335,1464` | `null` | `null` |

> **Adding split ROI coordinates**: set `split1` and optionally `split2` on the relevant `PlayerROIConfig` entries in `_PROFILE_DEFAULTS`. Use `scripts/pattern-matching.py` or `scripts/draw_bboxes.py` to calibrate the pixel rectangles for a given video source.

The standalone CLI script `scripts/pattern-matching.py` uses the same detection logic and can be run independently for debugging or batch processing.

## Frontend Stream Test App

A minimal, vanilla HTML/JS frontend application is included to easily test the real-time WebSocket `/api/stream` endpoint. It allows you to select a local video file, stream its frames to the backend, and displays the live card detection results in a clean UI.

To use it:

1. Guarantee your FastAPI backend is running (`uv run fastapi dev`).
2. Open `frontend/index.html` directly in your web browser, or serve it using Python's built-in HTTP server:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   Then navigate to `http://localhost:8080` in your browser.
3. Select a sample blackjack video file and hit **Connect & Start Stream**!

## Scripts & Utilities

See the [`scripts/README.md`](scripts/README.md) for offline testing flows, template matching calibration, and live stream simulation utilities.

*   `scripts/ws_stream_video.py`: Simulates live WebSocket streaming frames to test real-time card detection.
*   `scripts/pattern-matching.py`: Standalone CLI tool to debug ROIs and output results offline.

## Notes and Limitations

- The active-job lock (`_active_job_id`) is process-local, so it is not shared across multiple workers/processes. Use a distributed lock (e.g. Redis) for multi-process deployments.
- Uploaded files are stored under `LOCAL_UPLOAD_DIR` and are not automatically cleaned up.
- Session-ending frame images are not saved to disk — only DB records are created.

## Run Tests

```bash
uv run pytest
```

Tests use:

- In-memory SQLite
- FastAPI dependency override for DB session
- Temporary upload directory isolation
- Mocked vision processing (no real video files required)

