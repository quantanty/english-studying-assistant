import os
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    ChatGoogleGenerativeAIError,
)

from src.schemas.llms import CustomGoogleModelName, GoogleModelName, ModelType
from src.services.genai_chat_models import ChatGoogleGenAIJSON

load_dotenv()

LLM_POOL = {
    ModelType.GOOGLE.value:           [m.value for m in GoogleModelName],
    ModelType.CUSTOM_GOOGLE.value:    [m.value for m in CustomGoogleModelName],
}

LLM_CLASS = {
    ModelType.GOOGLE.value: ChatGoogleGenerativeAI,
    ModelType.CUSTOM_GOOGLE.value: ChatGoogleGenAIJSON
}


def create_llm(model_id: str | None = None, **kwargs):
    model_id = model_id or "google_genai:gemini-2.0-flash"
    return init_chat_model(
        model_id,
        **kwargs
    )

@cache
def get_model(model_type: str, model_name: str):
    if model_type not in ModelType:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_name not in LLM_POOL[model_type]:
        raise ValueError(f"Model type {model_type} does not support model name {model_name}")

    try:
        return LLM_CLASS[model_type](model=model_name)
    except ValueError as e:
        raise ValueError(f"Unsupported model name {model_name} with type {model_type}")