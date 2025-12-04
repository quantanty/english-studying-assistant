import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.prompts import QUIZ_GENERATOR_SYSTEM, QUIZ_GENERATOR_USER
from src.schemas.models import GenerateQuizzesParams
from src.schemas.state import State, StepState
from src.services.llm_client import get_model
from src.tools.quiz import (
    create_fill_in_blank_question,
    create_multiple_choice_question,
    create_true_false_question,
)

logger = logging.getLogger(__name__)


class QuizGenerator:
    def __init__(self, default_model, tools):
        self.default_model = default_model
        self.tools = tools
        self.system = QUIZ_GENERATOR_SYSTEM
        self.user = QUIZ_GENERATOR_USER

    async def __call__(self, state: State):
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("quiz_generator")
        else:
            model_selection = None
        if model_selection:
            llm = get_model(model_type=model_selection.model_type, model_name=model_selection.model_name)
        else:
            llm = get_model(model_type=self.default_model.model_type, model_name=self.default_model.model_name)
        llm = llm.bind_tools(self.tools)
        
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

        response = await llm.ainvoke(qgen_messages)
        qgen_messages.append(response)

        return {"qgen_messages": qgen_messages}
        
        
    def construct_initial_feed(self, instruction, words, retrieved_information):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(words=words, instruction=instruction, retrieved_information=retrieved_information))
        ]
        return feed