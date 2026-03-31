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
3. Scans the video frame-by-frame for the replay button (using OpenCV template matching) to detect session boundaries
4. For each detected session-ending frame, identifies dealer and player cards via template matching
5. Persists each detected game as a `blackjack_game_session` row (JSON result with cards per player/dealer)
6. Updates the task status to `completed` (or `failed` on error)

Resolution profiles and their ROI defaults are defined in `app/vision/processor.py`:

| Profile | Replay ROI | Dealer ROI |
|---------|-----------|------------|
| `480p`  | `220,223,240,240` | `230,83,290,91` |
| `4k`    | `1000,1020,1150,1150` | `1070,560,1300,593` |

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

