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
        
        # prompt
        self.system = ORCHESTRATOR_SYSTEM
        self.user_1 = ORCHESTRATOR_USER_START
        self.user_2 = ORCHESTRATOR_USER_NEXT
        self.feed = []

    def __call__(self, state: State) -> State:
        if not self.feed:
            self._construct_feed_start(state)
        else:
            self._construct_observation_string(state)
            self._construct_feed_next(state)
        
        response = self.llm.invoke(self.feed)
        step_state = StepState(
            action_type=response.action
        )

        return {
            "step_state": step_state
        }

    def reset_feed(self):
        self.feed = []

    def _construct_observation_string(self, state: State):
        observation_string = ""

        step_state = state.get("step_state")
        action_type = step_state.get("action_type")
        
        if action_type == "extract_words":
            words_to_define = state.get("words_to_define", [])
            observation_string += f"Words to define: `{words_to_define}`."

        elif action_type == "define_words":
            words_to_define = state.get("words_to_define", [])
            word_entries = state.get("word_entries", {})
            if not words_to_define:
                observation_string += "There're no words to define."
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
            ...

        state["step_state"]["observation"] = observation_string

    def _construct_feed_start(self, state: State):
        intent = state.get("intent")
        plan = state.get("plan")
        self.feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user_1.format(intent=intent, plan=plan))
        ]
    
    def _construct_feed_next(self, state: State):
        step_state = state.get("step_state")
        type = step_state.get("action_type")
        success = step_state.get("success")
        observation = step_state.get("observation")
        self.feed.append(HumanMessage(
            content=self.user_2.format(
                type=type,
                success=success,
                observation=observation
            )
        ))