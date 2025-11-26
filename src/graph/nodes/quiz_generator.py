import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.schemas.state import State, StepState
from src.schemas.models import GenerateQuizzesParams
from src.prompts import QUIZ_GENERATOR_SYSTEM, QUIZ_GENERATOR_USER

logger = logging.getLogger(__name__)


class QuizGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.system = QUIZ_GENERATOR_SYSTEM
        self.user = QUIZ_GENERATOR_USER

    async def __call__(self, state: State):
        qgen_messages = state.get("qgen_messages", [])

        if not qgen_messages:
            step_state = state.get("step_state")
            assert step_state
            params = step_state.get("params")
            assert isinstance(params, GenerateQuizzesParams)
            words = params.words
            instruction = params.instructions
            retrieved_information = state.get("collected_data")

            qgen_messages = self.construct_initial_feed(instruction, words, retrieved_information)

        response = await self.llm.ainvoke(qgen_messages)
        qgen_messages.append(response)

        return {"qgen_messages": qgen_messages}
        
        
    def construct_initial_feed(self, instruction, words, retrieved_information):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(words=words, instruction=instruction, retrieved_information=retrieved_information))
        ]
        return feed