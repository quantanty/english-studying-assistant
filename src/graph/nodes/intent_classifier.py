from src.schemas.state import State
from src.schemas.models import IntentClassifierOutput

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command, interrupt
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
        
        feed = self._construct_feed(messages)
        response = self.llm.invoke(feed)
        logger.debug("Intent classifier response:\n" + str(response))
        human_response = interrupt("interrupt from IntentClassifier")
        print(human_response["data"])
        return {"intent": response.intent}
    
    def _construct_feed(self, messages):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(conversation=messages))
        ]
        return feed