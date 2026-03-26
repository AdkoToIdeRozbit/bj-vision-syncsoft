import io
from unittest.mock import patch

from app.models import (
    BlackjackGameSession,
    BlackjackTrackingTask,
    BlackjackTrackingTaskStatus,
)


def test_get_task_status_returns_404_for_missing_task(client):
    response = client.get("/api/tasks/999999/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_get_task_status_returns_task_data(client, db_session):
    task = BlackjackTrackingTask(
        video_path="data/uploads/demo.mp4",
        status=BlackjackTrackingTaskStatus.COMPLETED,
        csv_output_path="data/outputs/demo.csv",
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    response = client.get(f"/api/tasks/{task.id}/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == str(task.id)
    assert payload["status"] == BlackjackTrackingTaskStatus.COMPLETED.value
    assert payload["csv_file_url"] == "data/outputs/demo.csv"
    assert payload["game_sessions"] == []


# ---------------------------------------------------------------------------
# Upload endpoint tests
# ---------------------------------------------------------------------------


def _make_video_file(filename: str = "test.mp4") -> tuple[str, tuple]:
    """Return a (field_name, (filename, bytes, content_type)) tuple for multipart."""
    return ("file", (filename, io.BytesIO(b"fake video content"), "video/mp4"))


def test_upload_returns_task_id_with_default_profile(client):
    """A valid upload with no profile specified should succeed with default '480p'."""
    with patch("app.routes.tasks._process_video_bg"):  # prevent actual processing
        response = client.post(
            "/api/tasks/upload",
            files=[_make_video_file()],
        )
    assert response.status_code == 200
    payload = response.json()
    assert "task_id" in payload
    assert isinstance(payload["task_id"], int)


def test_upload_accepts_explicit_profiles(client, monkeypatch):
    """Both '480p' and '4k' profile values should be accepted."""
    import app.routes.tasks as tasks_route

    for profile in ("480p", "4k"):
        # Reset the active job lock before each iteration so neither blocks the other
        monkeypatch.setattr(tasks_route, "_active_job_id", None)
        with patch("app.routes.tasks._process_video_bg"):
            response = client.post(
                "/api/tasks/upload",
                files=[_make_video_file()],
                data={"profile": profile},
            )
        assert response.status_code == 200, f"profile={profile} failed"


def test_upload_rejects_invalid_profile(client):
    """An unknown profile value should return HTTP 422 Unprocessable Entity."""
    with patch("app.routes.tasks._process_video_bg"):
        response = client.post(
            "/api/tasks/upload",
            files=[_make_video_file()],
            data={"profile": "720p"},
        )
    assert response.status_code == 422


def test_upload_rejects_non_video_content_type(client):
    response = client.post(
        "/api/tasks/upload",
        files=[("file", ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf"))],
    )
    assert response.status_code == 400


def test_upload_rejects_missing_filename(client):
    # An empty filename is rejected either by our validation (400) or
    # by FastAPI's multipart form parsing (422) depending on the framework version.
    response = client.post(
        "/api/tasks/upload",
        files=[("file", ("", io.BytesIO(b"data"), "video/mp4"))],
    )
    assert response.status_code in (400, 422)


def test_upload_conflict_when_job_active(client, monkeypatch):
    """A second upload while one is active should return HTTP 409."""
    import app.routes.tasks as tasks_route

    monkeypatch.setattr(tasks_route, "_active_job_id", 42)
    response = client.post(
        "/api/tasks/upload",
        files=[_make_video_file()],
    )
    assert response.status_code == 409


def test_upload_triggers_background_task_with_correct_args(client, db_session):
    """After upload, _process_video_bg should be called with (task_id, path, profile)."""
    captured: list = []

    def fake_bg(task_id: int, video_path: str, profile: str) -> None:
        captured.append((task_id, video_path, profile))

    with patch("app.routes.tasks._process_video_bg", side_effect=fake_bg):
        response = client.post(
            "/api/tasks/upload",
            files=[_make_video_file("game.mp4")],
            data={"profile": "4k"},
        )

    assert response.status_code == 200
    task_id = response.json()["task_id"]

    assert len(captured) == 1
    assert captured[0][0] == task_id
    assert "game.mp4" in captured[0][1]
    assert captured[0][2] == "4k"


def test_background_worker_creates_game_sessions(db_session, tmp_path, monkeypatch):
    """_process_video_bg should persist BlackjackGameSession rows and mark task COMPLETED."""
    import app.routes.tasks as tasks_route
    from app.routes.tasks import _process_video_bg

    monkeypatch.setattr(tasks_route, "_active_job_id", None)

    # Pre-create the task in the test DB
    task = BlackjackTrackingTask(
        video_path=str(tmp_path / "fake.mp4"),
        status=BlackjackTrackingTaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    # Mock process_video to return a synthetic result without touching actual video files
    fake_results = [
        {
            "session": 1,
            "frame": 100,
            "dealer": ["ace", "ten"],
            "player_1": [],
            "player_2": [],
            "player_3": [],
            "player_4": ["two"],
            "player_5": [],
            "player_6": ["seven"],
            "player_7": [],
        }
    ]

    with (
        patch("app.routes.tasks.process_video", return_value=fake_results),
        patch("app.routes.tasks.load_card_templates", return_value=[]),
        patch("app.routes.tasks.build_vision_config"),
        # Route the background worker's Session to use the test DB
        patch("app.routes.tasks.engine", db_session.get_bind()),
    ):
        _process_video_bg(task.id or 0, str(tmp_path / "fake.mp4"), "480p")

    db_session.expire_all()

    updated_task = db_session.get(BlackjackTrackingTask, task.id)
    assert updated_task is not None
    assert updated_task.status == BlackjackTrackingTaskStatus.COMPLETED

    sessions = db_session.exec(
        __import__("sqlmodel")
        .select(BlackjackGameSession)
        .where(BlackjackGameSession.task_id == task.id)
    ).all()
    assert len(sessions) == 1

    result = sessions[0].result
    assert "ace" in result["dealer_cards"]
    assert "ten" in result["dealer_cards"]
    assert "two" in result["player4_cards"]
    assert "seven" in result["player6_cards"]
