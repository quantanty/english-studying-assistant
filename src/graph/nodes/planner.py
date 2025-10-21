from src.schemas.state import State
from src.schemas.models import PlannerOutput
from src.utils.context import get_conversation_string

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command, interrupt
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
        intent = state.get("intent")
        converstion = get_conversation_string(state)

        # PHASE 1
        reasoning_text = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt_1),
                HumanMessage(content=f"Chat history: {converstion}\nIntent: {intent}")
            ]
        )
        logger.debug(reasoning_text)

        # PHASE 2
        plan = self.llm_structured_output.invoke(
            [
                SystemMessage(content=self.system_prompt_2),
                HumanMessage(content=f"Reasoning: {reasoning_text}")
            ]
        )

        logger.debug(plan.steps)
        interrupt("interrupt from Planner")
        return {"plan": plan.steps}
    
    def _construct_feed_reasoning(self):
        pass

    def _construct_feed_structured_output(self):
        pass