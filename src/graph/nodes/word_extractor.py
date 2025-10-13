from src.schemas.state import State

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class WordExtractor:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State) -> State:
        #TODO: agent logic
        pass