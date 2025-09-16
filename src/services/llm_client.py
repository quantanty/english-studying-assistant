from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

def create_llm(**kwargs):
    return init_chat_model(
        "google_genai:gemini-2.0-flash",
        **kwargs
    )

if __name__ == "__main__":
    llm = create_llm()
    print(llm)