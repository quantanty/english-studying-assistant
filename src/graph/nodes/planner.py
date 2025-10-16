from src.schemas.state import State
from src.schemas.models import PlannerOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, llm, system_prompt_1, system_prompt_2):
        self.llm = llm
        self.llm_structured_output = llm.with_structured_output(PlannerOutput)
        self.system_prompt_1 = system_prompt_1
        self.system_prompt_2 = system_prompt_2

    def __call__(self, state: State) -> State:
        # TODO: Planning logic
        pass