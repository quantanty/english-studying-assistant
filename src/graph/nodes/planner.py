import logging
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import (
    PLANNER_SYSTEM_PHASE_1,
    PLANNER_SYSTEM_PHASE_2,
    PLANNER_USER_PHASE_1,
    PLANNER_USER_PHASE_2,
)
from src.schemas.llms import ModelSelection, ModelType
from src.schemas.models import PlannerOutput
from src.schemas.state import State
from src.services.llm_client import get_model
from src.utils.context import get_conversation_string, get_intent_string

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, default_model: ModelSelection):
        self.default_model = default_model
        # reasoning prompt
        self.system_1 = PLANNER_SYSTEM_PHASE_1
        self.user_1 = PLANNER_USER_PHASE_1
        # structured output prompt
        self.system_2 = PLANNER_SYSTEM_PHASE_2
        self.user_2 = PLANNER_USER_PHASE_2

    def __call__(self, state: State) -> State:
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("planner")
        else:
            model_selection = None
        if model_selection:
            model_type=model_selection.model_type
            model_name=model_selection.model_name
            llm = get_model(model_type=model_type, model_name=model_name)
        else:
            model_type=self.default_model.model_type
            model_name=self.default_model.model_name
            llm = get_model(model_type=model_type, model_name=model_name)
        
        
        intent_string = get_intent_string(state)
        conversation = get_conversation_string(state)

        feed = self._construct_structured_output_feed(intent_string, conversation)

        if model_type == ModelType.LLAMA_CPP.value:
            plan_response = llm.invoke(
                feed,
                extra_body={
                    "id_slot": 1,
                    "json_schema": PlannerOutput.model_json_schema(),
                }
            )
            parsed = PlannerOutput.model_validate_json(cast(str, plan_response.content))
        else:
            llm_structured_output = llm.with_structured_output(PlannerOutput, include_raw=True)
            plan_response = llm_structured_output.invoke(feed)
            parsed = plan_response["raw"]
        steps = parsed.steps_in_the_plan

        return {"plan": steps}
    
    def _construct_reasoning_feed(self, conversation, intent):
        feed = [
            SystemMessage(content=self.system_1),
            HumanMessage(content=self.user_1.format(conversation=conversation, intent=intent))
        ]
        return feed

    def _construct_structured_output_feed(self, intent, conversation):
        feed = [
            SystemMessage(content=self.system_2),
            HumanMessage(content=self.user_2.format(intent=intent, conversation=conversation))
        ]
        return feed