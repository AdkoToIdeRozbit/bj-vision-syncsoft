import csv
import io
import math
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from app.models import BlackjackGameSession, GameResult
from app.routes.deps import APIKeyDep, SessionDep

router = APIRouter(
    prefix="/api/game-sessions",
    tags=["game-sessions"],
    dependencies=[APIKeyDep],
)


def _parse_result(raw: dict | GameResult | None) -> GameResult | None:
    if raw is None:
        return None
    if isinstance(raw, GameResult):
        return raw
    return GameResult.model_validate(raw)


class GameSessionResponse(BaseModel):
    id: int
    task_id: int | None = None
    result: GameResult | None = None
    created_at: datetime
    updated_at: datetime


class PaginatedGameSessionsResponse(BaseModel):
    items: list[GameSessionResponse]
    total: int
    page: int
    page_size: int
    pages: int


def _apply_date_filters(
    stmt, created_at_from: datetime | None, created_at_to: datetime | None
):
    if created_at_from is not None:
        stmt = stmt.where(BlackjackGameSession.created_at >= created_at_from)
    if created_at_to is not None:
        stmt = stmt.where(BlackjackGameSession.created_at <= created_at_to)
    return stmt


@router.get("", response_model=PaginatedGameSessionsResponse)
async def list_game_sessions(
    session: SessionDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    created_at_from: Annotated[datetime | None, Query()] = None,
    created_at_to: Annotated[datetime | None, Query()] = None,
) -> PaginatedGameSessionsResponse:
    count_stmt = _apply_date_filters(
        select(func.count()).select_from(BlackjackGameSession),
        created_at_from,
        created_at_to,
    )
    total: int = session.exec(count_stmt).one()

    data_stmt = (
        _apply_date_filters(
            select(BlackjackGameSession), created_at_from, created_at_to
        )
        .order_by(BlackjackGameSession.created_at.desc())  # type: ignore
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = session.exec(data_stmt).all()

    items = [
        GameSessionResponse(
            id=row.id or 0,
            task_id=row.task_id,
            result=_parse_result(row.result),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    return PaginatedGameSessionsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total > 0 else 1,
    )


_PLAYER_FIELDS = [f"player{i}_cards" for i in range(1, 8)]
_CSV_HEADERS = (
    ["id", "task_id", "created_at", "updated_at"] + _PLAYER_FIELDS + ["dealer_cards"]
)


def _cards_str(cards: list | None) -> str:
    return "|".join(cards) if cards else ""


@router.get("/export", response_class=StreamingResponse)
async def export_game_sessions_csv(
    session: SessionDep,
    created_at_from: Annotated[datetime | None, Query()] = None,
    created_at_to: Annotated[datetime | None, Query()] = None,
) -> StreamingResponse:
    stmt = _apply_date_filters(
        select(BlackjackGameSession), created_at_from, created_at_to
    ).order_by(BlackjackGameSession.created_at)  # type: ignore
    rows = session.exec(stmt).all()

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS, lineterminator="\n")
    writer.writeheader()

    for row in rows:
        result = _parse_result(row.result)
        writer.writerow(
            {
                "id": row.id,
                "task_id": row.task_id if row.task_id is not None else "",
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                **{
                    p: _cards_str(getattr(result, p, None) if result else None)
                    for p in _PLAYER_FIELDS
                },
                "dealer_cards": _cards_str(result.dealer_cards if result else None),
            }
        )

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=game_sessions.csv"},
    )
