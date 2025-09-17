from src.schemas.state import State, WordEntryState

from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class Presenter:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def __call__(self, state: State) -> State:
        node_results = state.get("node_results", "Results not found")
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            return state
        
        word_entries = state.get("word_entries", "")
        words_needed = state.get("words", [])
        words_info = "\n".join([
            str(word_entries.get(word, None)) or f"Failed to retrieve definition of '{word}'"
            for word in words_needed
        ])

        feed = [
            SystemMessage(self.prompt.format(node_results=node_results, word_entries=words_info)),
            *messages
        ]
        logger.debug(str(feed))
        message = self.llm.invoke(feed)
        return {**state, "messages": [message]}