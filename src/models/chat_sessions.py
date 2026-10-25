from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlmodel import Field, Relationship
from uuid import UUID, uuid4

from src.models.base import BaseModel

if TYPE_CHECKING:
    from src.models.chat_messages import ChatMessage
    from src.models.users import User

class ChatSession(BaseModel, table=True):
    __tablename__: str = "chat_sessions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    title: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    user: Optional["User"] = Relationship(back_populates="chat_sessions")
    messages: List["ChatMessage"] = Relationship(back_populates="chat_session")

from src.models.chat_messages import ChatMessage