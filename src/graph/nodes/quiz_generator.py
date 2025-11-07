import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.schemas.state import State
from src.prompts import QUIZ_GENERATOR_SYSTEM, QUIZ_GENERATOR_USER

logger = logging.getLogger(__name__)


class QuizGenerator:
    def __init__(self, llm):
        # TODO: with_structured_output()
        self.llm = llm
        self.system = QUIZ_GENERATOR_SYSTEM
        self.user = QUIZ_GENERATOR_USER

    # TODO: quiz generator logic
    def __call__(self, state: State) -> State:
        pass
