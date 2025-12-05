import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import (
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_USER_NEXT,
    ORCHESTRATOR_USER_START,
)
from src.schemas.models import Action, OrchestratorOutput
from src.schemas.state import State, StepState
from src.services.llm_client import get_model
from src.utils.context import get_intent_string
from src.utils.human_in_the_loop import ask_permission
from src.utils.state_utils import get_word_defining_results

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, default_model):
        self.default_model = default_model
        
        self.system = ORCHESTRATOR_SYSTEM
        self.user_1 = ORCHESTRATOR_USER_START
        self.user_2 = ORCHESTRATOR_USER_NEXT

    async def __call__(self, state: State):
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("orchestrator")
        else:
            model_selection = None
        if model_selection:
            llm = get_model(model_type=model_selection.model_type, model_name=model_selection.model_name)
        else:
            llm = get_model(model_type=self.default_model.model_type, model_name=self.default_model.model_name)
        llm_structured_output = llm.with_structured_output(OrchestratorOutput, include_raw=True)
        
        orchestrator_messages = state.get("orchestrator_messages", [])
        if not orchestrator_messages:
            orchestrator_messages= self._init_feed(state)
        else:
            step_state = state.get("step_state")
            assert step_state
            self._append_feed(state, step_state, orchestrator_messages)
        
        response = await llm_structured_output.ainvoke(orchestrator_messages)
        orchestrator_messages.append(response["raw"])
        parsed = response["parsed"]
        step_state = StepState(
            action_type=parsed.result.action,
        )
        if hasattr(parsed.result, "params"):
            step_state["params"] = parsed.result.params

        return {
            "step_state": step_state,
            "orchestrator_messages": orchestrator_messages
        }


    def _init_feed(self, state: State):
        intent_string = get_intent_string(state)
        plan = state.get("plan")
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user_1.format(intent=intent_string, plan=plan))
        ]
        return feed
    
    def _append_feed(self, state: State, step_state: StepState, feed):
        type = step_state.get("action_type")
        # success = step_state.get("success")
        observation = step_state.get("observation")
        feed.append(HumanMessage(
            content=self.user_2.format(
                type=type,
                # success=success,
                observation=observation
            )
        ))