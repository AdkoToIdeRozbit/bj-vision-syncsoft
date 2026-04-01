from datetime import datetime, timedelta, timezone

from app.models import BlackjackGameSession, CardClass


def _make_session(
    db_session, created_at: datetime | None = None, result: dict | None = None
):
    if result is None:
        result = {
            "player1_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player2_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player3_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player4_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player5_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player6_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "player7_cards": {"hand1": [CardClass.ACE, CardClass.KING]},
            "dealer_cards": [CardClass.TEN, CardClass.SEVEN],
        }
    gs = BlackjackGameSession(task_id=None, result=result)
    if created_at is not None:
        gs.created_at = created_at
        gs.updated_at = created_at
    db_session.add(gs)
    db_session.commit()
    db_session.refresh(gs)
    return gs


# ---------------------------------------------------------------------------
# List endpoint
# ---------------------------------------------------------------------------


def test_list_game_sessions_empty(client):
    response = client.get("/api/game-sessions")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["page"] == 1
    assert data["pages"] == 1


def test_list_game_sessions_returns_data(client, db_session):
    _make_session(db_session)

    response = client.get("/api/game-sessions")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert "id" in item
    assert "created_at" in item
    assert item["result"]["player1_cards"] == {"hand1": [CardClass.ACE, CardClass.KING]}
    assert item["result"]["dealer_cards"] == [CardClass.TEN, CardClass.SEVEN]


def test_list_game_sessions_pagination(client, db_session):
    for _ in range(5):
        _make_session(db_session)

    response = client.get("/api/game-sessions?page=1&page_size=2")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["pages"] == 3
    assert data["page"] == 1
    assert data["page_size"] == 2


def test_list_game_sessions_second_page(client, db_session):
    for _ in range(5):
        _make_session(db_session)

    response = client.get("/api/game-sessions?page=2&page_size=2")

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2


def test_list_game_sessions_last_partial_page(client, db_session):
    for _ in range(5):
        _make_session(db_session)

    response = client.get("/api/game-sessions?page=3&page_size=2")

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1


def test_list_game_sessions_filter_created_at_from(client, db_session):
    now = datetime.now(timezone.utc)
    _make_session(db_session, created_at=now - timedelta(days=10))
    recent = _make_session(db_session, created_at=now - timedelta(days=1))

    from_dt = (now - timedelta(days=5)).isoformat()
    response = client.get("/api/game-sessions", params={"created_at_from": from_dt})

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["id"] == recent.id


def test_list_game_sessions_filter_created_at_to(client, db_session):
    now = datetime.now(timezone.utc)
    old = _make_session(db_session, created_at=now - timedelta(days=10))
    _make_session(db_session, created_at=now - timedelta(days=1))

    to_dt = (now - timedelta(days=5)).isoformat()
    response = client.get("/api/game-sessions", params={"created_at_to": to_dt})

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["id"] == old.id


def test_list_game_sessions_filter_created_at_range(client, db_session):
    now = datetime.now(timezone.utc)
    _make_session(db_session, created_at=now - timedelta(days=20))
    middle = _make_session(db_session, created_at=now - timedelta(days=5))
    _make_session(db_session, created_at=now - timedelta(hours=1))

    from_dt = (now - timedelta(days=10)).isoformat()
    to_dt = (now - timedelta(days=2)).isoformat()
    response = client.get(
        "/api/game-sessions",
        params={"created_at_from": from_dt, "created_at_to": to_dt},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["id"] == middle.id


def test_list_game_sessions_requires_api_key(client):
    response = client.get("/api/game-sessions", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------


def test_export_game_sessions_csv_empty(client):
    response = client.get("/api/game-sessions/export")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    lines = response.text.strip().split("\n")
    assert len(lines) == 1  # header only
    assert lines[0].startswith("id,task_id,created_at,updated_at")


def test_export_game_sessions_csv_with_data(client, db_session):
    _make_session(db_session)

    response = client.get("/api/game-sessions/export")

    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 2  # header + 1 data row
    assert "ace|king" in lines[1]
    assert "ten|seven" in lines[1]


def test_export_game_sessions_csv_attachment_header(client):
    response = client.get("/api/game-sessions/export")

    assert "content-disposition" in response.headers
    assert "attachment" in response.headers["content-disposition"]
    assert "game_sessions.csv" in response.headers["content-disposition"]


def test_export_game_sessions_csv_filter_by_date(client, db_session):
    now = datetime.now(timezone.utc)
    _make_session(db_session, created_at=now - timedelta(days=10))
    _make_session(db_session, created_at=now - timedelta(days=1))

    from_dt = (now - timedelta(days=5)).isoformat()
    response = client.get(
        "/api/game-sessions/export", params={"created_at_from": from_dt}
    )

    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 2  # header + 1 row (only the recent session)


def test_export_game_sessions_csv_columns(client, db_session):
    _make_session(db_session)

    response = client.get("/api/game-sessions/export")

    header = response.text.split("\n")[0]
    for col in [
        "id",
        "task_id",
        "created_at",
        "updated_at",
        "player1_cards",
        "dealer_cards",
    ]:
        assert col in header
