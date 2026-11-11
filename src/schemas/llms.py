from enum import StrEnum

from pydantic import BaseModel


class ModelSelection(BaseModel):
    model_type: str
    model_name: str