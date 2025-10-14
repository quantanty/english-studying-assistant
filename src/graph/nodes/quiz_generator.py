from src.schemas.state import State

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class QuizGenerator:
    def __init__(self, llm):
        # TODO: with_structured_output()
        self.llm = llm

    # TODO: quiz generator logic
    def __call__(self, state: State) -> State:
        pass