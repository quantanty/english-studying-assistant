from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import JSON, Field, Relationship

from src.models.base import BaseModel

if TYPE_CHECKING:
    from src.models.chat_sessions import ChatSession

class ChatMessage(BaseModel, table=True):
    __tablename__: str = "chat_messages"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(nullable=False, foreign_key="chat_sessions.id")
    role: str = Field(nullable=False)
    content: str = Field(nullable=False)
    status: str = Field(nullable=False, default="completed")
    extra_metadata: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column("metadata", JSONB)
    )
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    chat_session: Optional["ChatSession"] = Relationship(back_populates="messages")
