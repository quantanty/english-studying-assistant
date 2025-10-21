from src.schemas.state import State
from src.schemas.models import PlannerOutput
from src.utils.context import get_conversation_string

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, llm, system_1, system_2, user_1, user_2):
        self.llm = llm
        self.llm_structured_output = llm.with_structured_output(PlannerOutput)
        # reasoning prompt
        self.system_1 = system_1
        self.user_1 = user_1
        # structured output prompt
        self.system_2 = system_2
        self.user_2 = user_2

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