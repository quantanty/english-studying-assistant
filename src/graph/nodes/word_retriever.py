from src.schemas.state import State

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import requests
import logging

logger = logging.getLogger(__name__)

class WordRetriever:
    def __init__(self, llm, tools):
        self.tools = tools
        self.llm = llm.bind_tools(tools)
    
    # TODO: retriever logic
    def __call__(self, state: State) -> State:
        words_to_define = state.get("words_to_define")
        for word in words_to_define:
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
            