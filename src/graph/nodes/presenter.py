from src.schemas.state import State, WordEntryState
from src.utils.context import get_conversation_string
from src.utils.state_utils import get_word_defining_results

from langgraph.types import StreamWriter
from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class Presenter:
    def __init__(self, llm, system, user):
        self.llm = llm
        self.system = system
        self.user = user

    async def __call__(self, state: State,  writer: StreamWriter) -> State:
        # node_results = state.get("node_results", "Results not found")
        # messages = state.get("messages", [])
        # if not messages or not isinstance(messages[-1], HumanMessage):
        #     return state
        
        # word_entries = state.get("word_entries", {})
        # words_needed = state.get("words", [])
        # words_info = "\n".join([
        #     str(word_entries.get(word, None)) or f"Failed to retrieve definition of '{word}'"
        #     for word in words_needed
        # ])

        # feed = [
        #     SystemMessage(self.prompt.format(node_results=node_results, word_entries=words_info)),
        #     *messages
        # ]
        # logger.debug(str(feed))
        # tokens = []
        # for chunk in self.llm.stream(feed):
        #     if getattr(chunk, "content", None):
        #         tokens.append(chunk.content)
        #         writer({"presenter": chunk.content})
                
        # return {"messages": ["".join(tokens)]}

        conversation = get_conversation_string(state)
        prepared_data = get_word_defining_results(state.get("word_entries", {}))
        
        tokens = []
        feed = self._construct_feed(conversation, prepared_data)
        for chunk in self.llm.stream(feed):
            if getattr(chunk, "content", None):
                tokens.append(chunk.content)
                writer({"presenter": chunk.content})

        return {"messages": ["".join(tokens)]}

    def _construct_feed(self, conversation: str, prepared_data: str):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(conversation=conversation, prepared_data=prepared_data)),
        ]

        return feed