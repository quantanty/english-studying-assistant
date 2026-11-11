from typing import Literal

from src.config import settings
from src.schemas.chat import Message
from src.schemas.llms import  ModelSelection
from src.schemas.state import SelectedModel


def dump_messages(messages: list[Message]) -> list[dict]:
    return [message.model_dump() for message in messages]

def _get_model(model: dict):
    return ModelSelection(model_type=model[0], model_name=model[1])

def select_model(option: Literal["fast", "medium", "pro"]) -> SelectedModel:    
    models = settings.LLM_OPTIONS[option]
    default = models["default"]
    selected_model = SelectedModel(
        intent_classifier=_get_model(models.get("intent_classifier", default)),
        planner=_get_model(models.get("planner", default)),
        orchestrator=_get_model(models.get("orchestrator", default)),
        word_entry_generator=_get_model(models.get("word_entry_generator", default)),
        quiz_generator=_get_model(models.get("quiz_generator", default)),
        presenter=_get_model(models.get("presenter", default))
    )
    return selected_model