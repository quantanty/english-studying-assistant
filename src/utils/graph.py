from typing import Literal

from src.schemas.chat import Message
from src.schemas.llms import (
    CustomGoogleModelName,
    GoogleModelName,
    LlamaCppModelName,
    ModelSelection,
    ModelType,
)
from src.schemas.state import SelectedModel


def dump_messages(messages: list[Message]) -> list[dict]:
    return [message.model_dump() for message in messages]

def select_model(option: Literal["fast", "medium", "pro"]) -> SelectedModel:
    if option == "pro":
        c_gemini_20_flash = ModelSelection(model_type=ModelType.CUSTOM_GOOGLE.value, model_name=CustomGoogleModelName.GEMINI_20_FLASH.value)
        gemini_20_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_20_FLASH.value)
        gemini_25_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_25_FLASH.value)
        return SelectedModel(
            intent_classifier=      c_gemini_20_flash,
            planner=                gemini_20_flash,
            orchestrator=           c_gemini_20_flash,
            word_entry_generator=   gemini_20_flash,
            quiz_generator=         gemini_20_flash,
            presenter=              gemini_25_flash
        )
    elif option == "medium":
        c_gemini_20_flash = ModelSelection(model_type=ModelType.CUSTOM_GOOGLE.value, model_name=CustomGoogleModelName.GEMINI_20_FLASH.value)
        gemini_20_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_20_FLASH.value)
        gemini_25_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_25_FLASH.value)
        return SelectedModel(
            intent_classifier=      c_gemini_20_flash,
            planner=                gemini_20_flash,
            orchestrator=           c_gemini_20_flash,
            word_entry_generator=   gemini_20_flash,
            quiz_generator=         gemini_20_flash,
            presenter=              gemini_25_flash
        )
    else:
        gemini_25_flash_lite = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_25_FLASH_LITE.value)
        gemma_3_1b = ModelSelection(model_type=ModelType.LLAMA_CPP.value, model_name=LlamaCppModelName.GEMMA_3_1B)
        return SelectedModel(
            intent_classifier=      gemma_3_1b,
            planner=                gemma_3_1b,
            orchestrator=           gemma_3_1b,
            word_entry_generator=   gemma_3_1b,
            quiz_generator=         gemini_25_flash_lite,
            presenter=              gemma_3_1b,
        )
