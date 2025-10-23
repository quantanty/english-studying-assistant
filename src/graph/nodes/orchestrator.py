from src.schemas.state import State, StepState
from src.schemas.models import OrchestratorOutput, Action
from src.utils.state_utils import get_word_defining_results
from src.utils.human_in_the_loop import ask_permission

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm, system, user_1, user_2):
        self.llm = llm.with_structured_output(OrchestratorOutput)
        
        # prompt
        self.system = system
        self.user_1 = user_1 # start
        self.user_2 = user_2 # next
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
            words_to_define = state.get("words_to_define")
            observation_string += f"words to define: {words_to_define}"
        elif action_type == "define_words":
            ...
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