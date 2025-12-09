import logging
from src.utils.context import write_report

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
        messages = state.get("messages")
        assert messages and isinstance(messages[-1], HumanMessage)
        user_message = messages[-1].content

        if not qgen_messages:
            step_state = state.get("step_state")
            assert step_state
            params = step_state.get("params")
            assert isinstance(params, GenerateQuizzesParams)
            words = params.words
            instruction = params.instructions_for_quiz_generator_agent
            retrieved_information = state.get("collected_data")

            qgen_messages = self.construct_initial_feed(instruction, words, retrieved_information, user_message)

        response = await llm.ainvoke(qgen_messages)

        if response.tool_calls:
            qgen_messages.append(response)
        else:
            report_string = response.content
            step_state = write_report(state, report_string)
            return {"step_state": step_state}

        return {"qgen_messages": qgen_messages}
        
        
    def construct_initial_feed(self, instruction, words, retrieved_information, user_message):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(words=words, instruction=instruction, retrieved_information=retrieved_information, user_message=user_message))
        ]
        return feed