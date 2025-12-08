import os
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    ChatGoogleGenerativeAIError,
)
from langchain_openai.chat_models.base import ChatOpenAI

from src.config import settings
from src.schemas.llms import (
    CustomGoogleModelName,
    GoogleModelName,
    LlamaCppModelName,
    ModelType,
)
from src.services.genai_chat_models import ChatGoogleGenAIJSON

load_dotenv()

LLM_POOL = {
    ModelType.GOOGLE.value:             [m.value for m in GoogleModelName],
    ModelType.CUSTOM_GOOGLE.value:      [m.value for m in CustomGoogleModelName],
    ModelType.LLAMA_CPP.value:          [m.value for m in LlamaCppModelName]
}

LLM_CLASS = {
    ModelType.GOOGLE.value: ChatGoogleGenerativeAI,
    ModelType.CUSTOM_GOOGLE.value: ChatGoogleGenAIJSON
}


@cache
def get_model(model_type: str, model_name: str):
    if model_type not in ModelType:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_name not in LLM_POOL[model_type]:
        raise ValueError(f"Model type {model_type} does not support model name {model_name}")

    try:
        if model_type == "llama_cpp":
            if not (
                settings.LLAMACPP_URL[model_name]
            ):
                raise
            # return ChatLlamaCpp(
            #     model_path=str(settings.LLAMACPP_MODELS[model_name]),
            #     n_ctx=1<<15,
            #     n_gpu_layers=29,
            # )
            return ChatOpenAI(model=model_name, base_url=str(settings.LLAMACPP_URL[model_name]), api_key="dummy")
        return LLM_CLASS[model_type](model=model_name)
    except ValueError as e:
        raise ValueError(f"Unsupported model name {model_name} with type {model_type}")