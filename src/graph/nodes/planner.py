from src.schemas.state import State
from src.schemas.models import PlannerOutput
from src.utils.context import get_conversation_string
from src.prompts import (
    PLANNER_SYSTEM_PHASE_1,
    PLANNER_SYSTEM_PHASE_2,
    PLANNER_USER_PHASE_1,
    PLANNER_USER_PHASE_2
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.llm_structured_output = llm.with_structured_output(PlannerOutput)
        # reasoning prompt
        self.system_1 = PLANNER_SYSTEM_PHASE_1
        self.user_1 = PLANNER_USER_PHASE_1
        # structured output prompt
        self.system_2 = PLANNER_SYSTEM_PHASE_2
        self.user_2 = PLANNER_USER_PHASE_2

    def __call__(self, state: State) -> State:
        intent = state.get("intent")
        conversation = get_conversation_string(state)

        # PHASE 1
        reasoning_response = self.llm.invoke(self._construct_reasoning_feed(conversation, intent))
        reasoning_text = reasoning_response.content

        # PHASE 2
        plan_response = self.llm_structured_output.invoke(self._construct_reasoning_feed(intent, reasoning_text))
        steps = plan_response.steps

        return {"plan": steps}
    
    def _construct_reasoning_feed(self, conversation, intent):
        feed = [
            SystemMessage(content=self.system_1),
            HumanMessage(content=self.user_1.format(conversation=conversation, intent=intent))
        ]
        return feed

    def _construct_structured_output_feed(self, intent, reasoning_text):
        feed = [
            SystemMessage(content=self.system_2),
            HumanMessage(content=self.user_2.format(intent=intent, reasoning=reasoning_text))
        ]
        return feed