import re
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1
    )

class StreamResponse(BaseModel):
    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")

class ChatResponse(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")