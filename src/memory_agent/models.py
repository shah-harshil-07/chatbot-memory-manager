"""Domain models. A Memory is one atomic fact about a user plus its lifecycle metadata."""
from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Category(str, Enum):
    PROFILE = "profile"            # name, location, role
    PREFERENCE = "preference"      # likes, dislikes, how they want things done
    RELATIONSHIP = "relationship"  # people in their life
    WORK = "work"                  # team, projects, responsibilities
    EVENT = "event"                # plans, things that happened
    OTHER = "other"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    text: str
    category: Category = Category.OTHER
    importance: int = Field(5, ge=1, le=10)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime | None = None
    access_count: int = 0          # how often it was recalled
    confirmation_count: int = 1    # how often the user re-stated it

    # Temporal validity: invalidated memories are kept as history, never recalled.
    is_active: bool = True
    valid_until: datetime | None = None
    superseded_by: str | None = None

    # Provenance: which memories a merged/updated memory was derived from.
    source_ids: list[str] = Field(default_factory=list)
    origin: Literal["chat", "explicit", "consolidation"] = "chat"

    def to_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"id"})

    @classmethod
    def from_payload(cls, point_id: str, payload: dict) -> "Memory":
        return cls(id=str(point_id), **payload)


class ScoredMemory(BaseModel):
    memory: Memory
    score: float


class Fact(BaseModel):
    """A candidate fact extracted from conversation, before reconciliation."""
    text: str = Field(min_length=3)
    category: Category = Category.OTHER
    importance: int = Field(5, ge=1, le=10)


class Action(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class MemoryOperation(BaseModel):
    action: Action
    fact: str
    memory_id: str | None = None   # memory created (ADD/UPDATE) or confirmed (NOOP)
    target_id: str | None = None   # memory replaced or invalidated (UPDATE/DELETE)
    reason: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TurnResult(BaseModel):
    reply: str
    recalled: list[ScoredMemory]
    operations: list[MemoryOperation] = Field(default_factory=list)


class ClusterPlan(BaseModel):
    source_ids: list[str]
    source_texts: list[str]
    merged: list[str] = Field(default_factory=list)
    applied: bool = False
    skipped_reason: str | None = None


class MaintenanceReport(BaseModel):
    user_id: str
    dry_run: bool
    active_before: int
    active_after: int
    clusters: list[ClusterPlan] = Field(default_factory=list)
    purged: int = 0
