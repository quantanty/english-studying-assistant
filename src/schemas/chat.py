import re
from typing import List, Literal, Union

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

class ChatRequzest(BaseModel):
    # messages: List[Message] = Field(
    #     ...,
    #     description="List of messages in the conversation",
    #     min_length=1
    # )
    message: Message

class MetadataPayload(BaseModel):
    type: Literal["metadata"] = "metadata"
    session_id: str
    session_name: str

class DeltaPayload(BaseModel):
    type: Literal["delta"] = "delta"
    content: str = Field(default="", description="The content of the current chunk")

class DonePayload(BaseModel):
    type: Literal["done"] = "done"

class ErrorPayload(BaseModel):
    type: Literal["error"] = "error"
    error: str

StreamResponse = Union[MetadataPayload, DeltaPayload, DonePayload]

class ChatResponse(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")