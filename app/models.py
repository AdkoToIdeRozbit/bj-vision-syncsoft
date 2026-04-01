from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


class BlackjackTrackingTaskStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CardClass(StrEnum):
    UNKNOWN = "unknown"
    ACE = "ace"
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"
    SIX = "six"
    SEVEN = "seven"
    EIGHT = "eight"
    NINE = "nine"
    TEN = "ten"
    JACK = "jack"
    QUEEN = "queen"
    KING = "king"


class GameResult(SQLModel):
    session_number: int | None = None
    player1_cards: dict[str, list[CardClass]] | None = None
    player2_cards: dict[str, list[CardClass]] | None = None
    player3_cards: dict[str, list[CardClass]] | None = None
    player4_cards: dict[str, list[CardClass]] | None = None
    player5_cards: dict[str, list[CardClass]] | None = None
    player6_cards: dict[str, list[CardClass]] | None = None
    player7_cards: dict[str, list[CardClass]] | None = None
    dealer_cards: list[CardClass] | None = None


class BlackjackTrackingTask(SQLModel, table=True):
    __tablename__ = "blackjack_tracking_task"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    video_path: str
    status: BlackjackTrackingTaskStatus = Field(
        default=BlackjackTrackingTaskStatus.PENDING,
        index=True,
    )
    csv_output_path: Optional[str] = None

    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )

    game_sessions: list["BlackjackGameSession"] = Relationship(back_populates="task")


class BlackjackGameSession(SQLModel, table=True):
    __tablename__ = "blackjack_game_session"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    task_id: int | None = Field(
        foreign_key="blackjack_tracking_task.id", nullable=True, ondelete="SET NULL"
    )
    task: BlackjackTrackingTask | None = Relationship(back_populates="game_sessions")

    result: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))

    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
