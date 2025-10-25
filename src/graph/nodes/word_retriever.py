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
        words_to_define = state.get("words_to_define", [])
        collected_data = state.get("collected_data", {})
        for word in words_to_define:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url)
            if response.status_code != 200:
                collected_data[word] = [f"Failed to retrieve {word}: {response.status_code}"]
                continue

            data = response.json()
            collected_data[word] = [str(data[0])]

        return {"collected_data": collected_data}