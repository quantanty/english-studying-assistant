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

class LlamaCppModelName(StrEnum):
    XLAM_2_1B = "xLAM-2-1b-fc-r"
    GEMMA_3_1B = "gemma-3-1b"

class ModelType(StrEnum):
    GOOGLE = "google"
    CUSTOM_GOOGLE = "custom_google"
    LLAMA_CPP = "llama_cpp"
    
class ModelSelection(BaseModel):
    model_type: str
    model_name: str