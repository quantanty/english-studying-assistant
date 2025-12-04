import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import WORD_ENTRY_GENERATOR_SYSTEM, WORD_ENTRY_GENERATOR_USER
from src.schemas.models import Meaning, WordEntry
from src.schemas.state import State, WordEntryState
from src.services.llm_client import get_model
from src.utils.context import get_words_to_define

logger = logging.getLogger(__name__)

class WordEntryGenerator:
    def __init__(self, default_model):
        self.default_model = default_model
        self.system = WORD_ENTRY_GENERATOR_SYSTEM
        self.user = WORD_ENTRY_GENERATOR_USER

    def __call__(self, state: State) -> State:
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("word_entry_generator")
        else:
            model_selection = None
        if model_selection:
            llm = get_model(model_type=model_selection.model_type, model_name=model_selection.model_name)
        else:
            llm = get_model(model_type=self.default_model.model_type, model_name=self.default_model.model_name)
        llm_structured_output = llm.with_structured_output(WordEntry)

        words_to_define = get_words_to_define(state)
        collected_data = state.get("collected_data", {})

        if not words_to_define or not collected_data:
            return {}
        
        word_entries = state.get("word_entries", {})

        for word in words_to_define:
            feed = self._construct_feed(word, collected_data)
            try:
                response = llm.invoke(feed)
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