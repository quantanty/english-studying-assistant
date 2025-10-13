from src.schemas.state import State
from src.schemas.models import IntentClassifierOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(IntentClassifierOutput)

    def __call__(self, state: State) -> State:
        #TODO: get message from state
        #TODO: generate output
        pass