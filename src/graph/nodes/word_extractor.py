from src.schemas.state import State
from src.schemas.models import WordExtractorOutput
from src.prompts import (
    WORD_EXTRACTOR_SYSTEM,
    WORD_EXTRACTOR_USER
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class WordExtractor:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(WordExtractorOutput)
        self.system = WORD_EXTRACTOR_SYSTEM
        self.user = WORD_EXTRACTOR_USER

    def __call__(self, state: State) -> State:
        query = state.get("messages")[-1].content
        feed = self._construct_feed(query)
        response = self.llm.invoke(feed)

        return {
            "words_to_define": response.words
        }


    def _construct_feed(self, query):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(query=query))
        ]
        return feed