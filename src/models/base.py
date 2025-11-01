from datetime import datetime, UTC
from typing import List, Optional
from sqlmodel import Field, SQLModel, Relationship


class BaseModel(SQLModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))