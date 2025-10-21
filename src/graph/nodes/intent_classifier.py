from src.schemas.state import State
from src.schemas.models import IntentClassifierOutput
from src.utils.context import get_conversation_string

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
        conversation = get_conversation_string(state)
        
        feed = self._construct_feed(conversation)
        response = self.llm.invoke(feed)

        return {"intent": response.intent}
    
    def _construct_feed(self, messages):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(conversation=messages))
        ]
        return feed