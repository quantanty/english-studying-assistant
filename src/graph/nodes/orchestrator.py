from src.schemas.state import State, StepState
from src.schemas.models import OrchestratorOutput, Action
from src.utils.state_utils import get_word_defining_results
from src.utils.human_in_the_loop import ask_permission
from src.prompts import (
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_USER_START,
    ORCHESTRATOR_USER_NEXT
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(OrchestratorOutput)
        
        self.system = ORCHESTRATOR_SYSTEM
        self.user_1 = ORCHESTRATOR_USER_START
        self.user_2 = ORCHESTRATOR_USER_NEXT

    async def __call__(self, state: State):
        orchestrator_messages = state.get("orchestrator_messages", [])
        if not orchestrator_messages:
            orchestrator_messages= self._init_feed(state)
        else:
            step_state = state.get("step_state", StepState())
            self._construct_observation_string(state, step_state)
            self._append_feed(state, step_state, orchestrator_messages)
        
        response = await self.llm.ainvoke(orchestrator_messages)
        orchestrator_messages.append(AIMessage(content=str(response.model_dump())))
        step_state = StepState(
            action_type=response.result.action,
        )
        if hasattr(response.result, "params"):
            step_state["params"] = response.result.params

        return {
            "step_state": step_state,
            "orchestrator_messages": orchestrator_messages
        }


    def _construct_observation_string(self, state: State, step_state: StepState):
        observation_string = ""

        action_type = step_state.get("action_type")
        
        if action_type == "extract_words":
            words_to_define = state.get("words_to_define", [])
            observation_string += f"Words to define: `{words_to_define}`."

        elif action_type == "define_words":
            words_to_define = state.get("words_to_define", [])
            word_entries = state.get("word_entries", {})
            if not words_to_define:
                observation_string += "There're no words to define."
            else:
                for word in words_to_define:
                    entry = word_entries.get(word)
                    if not entry:
                        observation_string += f"Error while defining \"{word}\".\n"
                    else:
                        result  = entry.get("result", "error")
                        if result == "success":
                            observation_string += f"Successfully defined \"{word}\".\n"
                        else:
                            observation_string += f"Error while defining \"{word}\".\n"
                
        elif action_type == "generate_quizzes":
            # This can be done by Quiz Generator llm
            ...

        step_state["observation"] = observation_string

    def _init_feed(self, state: State):
        intent = state.get("intent")
        plan = state.get("plan")
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user_1.format(intent=intent, plan=plan))
        ]
        return feed
    
    def _append_feed(self, state: State, step_state: StepState, feed):
        type = step_state.get("action_type")
        success = step_state.get("success")
        observation = step_state.get("observation")
        feed.append(HumanMessage(
            content=self.user_2.format(
                type=type,
                success=success,
                observation=observation
            )
        ))