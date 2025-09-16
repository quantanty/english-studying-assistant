from src.schemas.models import Meaning, WordEntry
from src.schemas.state import State, WordEntryState

from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class WordEntryGenerator:
    def __init__(self, llm, prompt):
        self.llm = llm.with_structured_output(WordEntry)#, retry=True, max_retries=3)
        self.prompt = prompt

    def __call__(self, state: State) -> State:
        words = state["words"]
        messages = [
            SystemMessage(content=self.prompt)
        ]
        word_entries = state.get("word_entries", {})
        node_results: str = state.get("node_results", "")
        node_results += "Results from WordEntryGenerator:\n"
        for word in words:
            if entry:=word_entries.get(word, None) and entry.get("result", "error") == "success":
                continue
            logger.debug(f"processing the word '{word}'")
            try:
                entry: WordEntry = self.llm.invoke(messages + [HumanMessage(content=word)])
                word_entries[word] = WordEntryState(word=word, result="success", **entry.model_dump())
                node_results += f"Successfully processed the word '{word}'.\n"
            except Exception as e:
                word_entries[word] = WordEntryState(result="error")
                node_results += f"Error while processing the word '{word}': {e}"

        return {**state, "word_entries": word_entries, "node_results": node_results}