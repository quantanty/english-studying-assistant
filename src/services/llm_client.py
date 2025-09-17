from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

def create_llm(model_id:str = None, **kwargs):
    model_id = model_id or "google_genai:gemini-2.0-flash"
    return init_chat_model(
        model_id,
        **kwargs
    )