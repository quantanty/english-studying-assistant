from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.nodes.orchestrator import Orchestrator
from src.schemas import State
from src.utils.context import get_conversation_string


def preparing():
    def preparing_(state: State):
        try:
            messages = state.get("messages")
            assert messages
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