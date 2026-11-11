import os
from functools import cache

from dotenv import load_dotenv
# from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models.base import ChatOpenAI

from src.config import settings
from src.services.genai_chat_models import ChatGoogleGenAIJSON

load_dotenv()

LLM_CLASS = {
    "lc-google": ChatGoogleGenerativeAI,
    "lc-google-custom": ChatGoogleGenAIJSON
}


@cache
def get_model(model_type: str, model_name: str):
    try:
        if model_type == "llama-cpp":
            if not (
                settings.LLM_URLS["llama-cpp"]
            ):
                raise
            return ChatOpenAI(model=model_name, base_url=str(settings.LLM_URLS["llama-cpp"].get(model_name, settings.LLM_URLS["llama-cpp"]["default"])), api_key="dummy")
        if model_type not in LLM_CLASS.keys():
            raise ValueError(f"Unsupported model type {model_type}")
        return LLM_CLASS[model_type](model=model_name)
    except ValueError as e:
        raise ValueError(f"Unsupported model name {model_name} with type {model_type}")