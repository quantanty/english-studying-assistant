from src.schemas.state import State
from src.schemas.models import OrchestratorOutput, Action
from src.utils.state_utils import get_word_defining_results
from src.utils.human_in_the_loop import ask_permission

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm, observation_template):
        self.llm = llm.with_structured_output(OrchestratorOutput)
        self.observation_template = observation_template

    def __call__(self, state: State) -> State:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            return {**state, "action": Action.END}
        
        orchestrator_messages = state.get("orchestrator_messages")
        orchestrator_observation = state.get("orchestrator_observation", [])
        if orchestrator_observation:
            orchestrator_messages.append(HumanMessage(
                content=self.observation_template.format(observation=str(orchestrator_observation))
            ))

        logger.debug(orchestrator_messages)
        response = self.llm.invoke(orchestrator_messages)
        msg = AIMessage(
            content=str(response.model_dump_json())
        )
        orchestrator_messages.append(msg)

        logger.info(f"Orchestrator llm response: {response}")

        action = response.action
        node_results = state.get("node_results", "")
        node_results += "Result from Orchestrator:\n"
        node_results += f"Taking '{action}' action.\n"

        words = []
        match action:
            case Action.DEFINE:
                if response.words:
                    words = response.words
                    accept = ask_permission(f"The assistant want to retrieve {str(words)}, do you accept?")
                    if not accept:
                        action=Action.GENERAL
                        node_results += "Human didn't accept retrieving, change action to 'general_chat'"
                else:
                    action=Action.GENERAL
            case Action.GENERAL:
                pass
            case _:
                action = Action.END
        
        return {
            **state,
            "action": action,
            "words": words,
            "node_results": node_results,
            "orchestrator_messages": orchestrator_messages
        }