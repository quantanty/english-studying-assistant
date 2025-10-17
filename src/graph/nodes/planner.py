from src.schemas.state import State
from src.schemas.models import PlannerOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, llm, system_prompt_1, system_prompt_2):
        self.llm = llm
        self.llm_structured_output = llm.with_structured_output(PlannerOutput)
        # reasoning prompt
        self.system_prompt_1 = system_prompt_1
        # structured output prompt
        self.system_prompt_2 = system_prompt_2

    def __call__(self, state: State) -> State:
        intent = "" # TODO: get intent from state
        chat_history = "" # TODO: get chat_history from state

        # PHASE 1
        reasoning_text = self.llm.invoke(
            messages=[
                SystemMessage(content=self.system_prompt_1),
                HumanMessage(content=f"Chat history: {chat_history}\nIntent: {intent}")
            ]
        )

        # PHASE 2
        plan = self.llm_structured_output.invoke(
            message=[
                SystemMessage(content=self.system_prompt_2),
                HumanMessage(content=f"Reasoning: {reasoning_text}")
            ]
        )