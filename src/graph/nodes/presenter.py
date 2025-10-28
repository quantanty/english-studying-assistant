from src.schemas.state import State, WordEntryState
from src.utils.context import get_conversation_string
from src.utils.state_utils import get_word_defining_results
from src.prompts import (
    PRESENTER_SYSTEM,
    PRESENTER_USER
)

from langgraph.types import StreamWriter
from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class Presenter:
    def __init__(self, llm):
        self.llm = llm
        self.system = PRESENTER_SYSTEM
        self.user = PRESENTER_USER

    async def __call__(self, state: State,  writer: StreamWriter) -> State:
        conversation = get_conversation_string(state)
        prepared_data = get_word_defining_results(state.get("word_entries", {}))
        
        tokens = []
        feed = self._construct_feed(conversation, prepared_data)
        async for chunk in self.llm.astream(feed):
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