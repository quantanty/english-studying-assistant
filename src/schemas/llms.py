from enum import StrEnum

from pydantic import BaseModel


class GoogleModelName(StrEnum):
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"

class CustomGoogleModelName(StrEnum):
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"

class ModelType(StrEnum):
    GOOGLE = "google"
    CUSTOM_GOOGLE = "custom_google"
    
class ModelSelection(BaseModel):
    model_type: str
    model_name: str