from src.schemas.state import State
from src.schemas.models import OrchestratorOutput, Action

from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm, prompt):
        self.llm = llm.with_structured_output(OrchestratorOutput)
        self.prompt = prompt

    def __call__(self, state: State) -> State:
        query = state.get("query", "")
        if not query:
            return state
        
        node_results = state.get("node_results", "Result from Orchestrator:\n")
        step_history = state.get("step_history", [])
        extracted_words = state.get("words", [])
        messages = [
            SystemMessage(content=self.prompt.format(step_history=str(step_history), extracted_words=str(extracted_words))),
            HumanMessage(content=query)
        ]
        response = self.llm.invoke(messages)
        logger.debug(f"Orchestrator llm response: {response}")

        action = response.action
        node_results += f"Taking '{action}' action."
        step_history.append(f"Orchestrator choose '{action}'")

        match action:
            case Action.DEFINE:
                if (words:=response.words):
                    extracted_words.extend(words)
                    step_history[-1] += f", extract words: {words}"
                else:
                    action=Action.GENERAL
                    step_history[-1] = f"Orchestrator choose '{action}'"
            case Action.GENERAL:
                pass
            case _:
                action = Action.GENERAL
                step_history[-1] = f"Orchestrator choose '{action}'"
        
        return {
            **state,
            "action": action,
            "step_history": step_history,
            "words": extracted_words,
            "node_results": node_results
        }