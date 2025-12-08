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
        c_gemini_20_flash = ModelSelection(model_type=ModelType.CUSTOM_GOOGLE.value, model_name=CustomGoogleModelName.GEMINI_20_FLASH.value)
        gemini_20_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_20_FLASH.value)
        gemini_25_flash = ModelSelection(model_type=ModelType.GOOGLE.value, model_name=GoogleModelName.GEMINI_25_FLASH.value)
        xLAM_2_1b = ModelSelection(model_type=ModelType.LLAMA_CPP.value, model_name=LlamaCppModelName.XLAM_2_1B)
        return SelectedModel(
            intent_classifier=      xLAM_2_1b,
            planner=                xLAM_2_1b,
            orchestrator=           xLAM_2_1b,
            word_entry_generator=   xLAM_2_1b,
            quiz_generator=         xLAM_2_1b,
            presenter=              xLAM_2_1b,
        )
