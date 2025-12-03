from src.schemas.models import Meaning, WordEntry
from src.schemas.state import State, WordEntryState
from src.prompts import (
    WORD_ENTRY_GENERATOR_SYSTEM,
    WORD_ENTRY_GENERATOR_USER
)

from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class WordEntryGenerator:
    def __init__(self, llm):
        self.system = WORD_ENTRY_GENERATOR_SYSTEM
        self.user = WORD_ENTRY_GENERATOR_USER

    def __call__(self, state: State) -> State:
        words_to_define = state.get("words_to_define", [])
        collected_data = state.get("collected_data", {})

        if not words_to_define or not collected_data:
            return None
        
        word_entries = state.get("word_entries", {})

        for word in words_to_define:
            feed = self._construct_feed(word, collected_data)
            try:
                response = self.llm.invoke(feed)
                word_entries[word] = WordEntryState(
                    word=word,
                    result="success",
                    **response.model_dump()
                )
            except Exception as e:
                word_entries[word] = WordEntryState(result="error")

        return {
            "word_entries": word_entries
        }

    def _construct_feed(self, word: str, collected_data: dict):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(word=word, retrieved_information=collected_data))
        ]
        return feed