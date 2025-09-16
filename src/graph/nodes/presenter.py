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
        query = state.get("query", "")
        word_entries = str(state.get("word_entries", ""))
        if not query:
            response = "Please provide a query"
            return {**state, "response": response}
        messages = [
            SystemMessage(self.prompt.format(node_results=node_results, word_entries=word_entries)),
            HumanMessage(query)
        ]
        response = self.llm.invoke(messages)
        return {**state, "response": response}