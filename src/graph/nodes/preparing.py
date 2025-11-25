from src.schemas import State
from src.graph.nodes.orchestrator import Orchestrator
from src.utils.context import get_conversation_string

from langchain_core.messages import SystemMessage, HumanMessage

def preparing(orchestrator: Orchestrator):
    def preparing_(state: State):
        try:
            messages = state["messages"]
            assert isinstance(messages[-1], HumanMessage)

            orchestrator_messages = []
            qgen_messages = []
            quizzes = []
        except Exception as e:
            return {"preparation_result": "error"}
        
        return {
            "preparation_result": "success",
            "words_to_define": [],
            "orchestrator_messages": orchestrator_messages,
            "qgen_messages": qgen_messages,
            "quizzes": quizzes
        }
    return preparing_