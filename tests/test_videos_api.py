from pathlib import Path

from app.models import BlackjackTrackingTask, BlackjackTrackingTaskStatus


def test_upload_video_success(client, db_session):
    response = client.post(
        "/api/tasks/upload",
        files={"file": ("round1.mp4", b"fake-video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "task_id" in payload

    task = db_session.get(BlackjackTrackingTask, payload["task_id"])
    assert task is not None
    assert task.status == BlackjackTrackingTaskStatus.PENDING
    assert task.video_path is not None
    assert Path(task.video_path).exists()


def test_upload_video_rejects_non_video_content_type(client):
    response = client.post(
        "/api/tasks/upload",
        files={"file": ("notes.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "content_type must be a video MIME type"


def test_upload_video_conflict_when_active_job_exists(client):
    first = client.post(
        "/api/tasks/upload",
        files={"file": ("first.mp4", b"video-1", "video/mp4")},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/tasks/upload",
        files={"file": ("second.mp4", b"video-2", "video/mp4")},
    )

    assert second.status_code == 409
    assert second.json()["detail"].startswith("A video is already being processed")


def test_upload_video_rejects_too_large_file(client):
    # MAX_UPLOAD_MB is patched to 1 in tests/conftest.py
    oversized = b"x" * (1024 * 1024 + 1)
    response = client.post(
        "/api/tasks/upload",
        files={"file": ("big.mp4", oversized, "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "file is larger than 1MB limit"
