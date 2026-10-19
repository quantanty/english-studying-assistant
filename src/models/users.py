from typing import TYPE_CHECKING, List, Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship

from src.models.base import BaseModel

if TYPE_CHECKING:
    from src.models.chat_sessions import ChatSession

class User(BaseModel, table=True):
    __tablename__: str = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    google_sub: str = Field(unique=True, nullable=False)
    email: Optional[str] = Field(default=None)
    name: str = Field(default="")

    chat_sessions: List["ChatSession"] = Relationship(back_populates="user")
    
    
# Avoid circular imports
from src.models.chat_sessions import ChatSession
