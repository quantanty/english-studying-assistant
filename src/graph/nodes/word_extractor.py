from src.schemas.state import State
from src.schemas.models import WordExtractorOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class WordExtractor:
    def __init__(self, llm, system, user):
        self.llm = llm.with_structured_output(WordExtractorOutput)
        self.system = system
        self.user = user

    def __call__(self, state: State) -> State:
        query = state.get("messages")[-1]
        feed = self._construct_feed(query)
        response = self.llm.invoke(feed)


    def _construct_feed(self, query):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(query=query))
        ]
        return query