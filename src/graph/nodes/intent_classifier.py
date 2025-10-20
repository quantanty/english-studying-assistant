from src.schemas.state import State
from src.schemas.models import IntentClassifierOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, llm, system, user):
        self.llm = llm.with_structured_output(IntentClassifierOutput)
        
        # prompt
        self.system = system
        self.user = user

    def __call__(self, state: State) -> State:
        #TODO: get message from state
        #TODO: generate output
        messages = state.get("messages", [])
        # query = messages[-1]
        
        messages_ = [
            SystemMessage(content=self.system_prompt),
            *messages
        ]
        response = self.llm.invoke(messages_)
        logger.debug("Intent classifier reasoning: " + response.reasoning)
        return {"intent": response.intent}