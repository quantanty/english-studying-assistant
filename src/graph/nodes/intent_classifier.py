from src.schemas.state import State
from src.schemas.models import IntentClassifierOutput
from src.utils.context import get_conversation_string
from src.prompts import (
    INTENT_CLASSIFIER_SYSTEM,
    INTENT_CLASSIFIER_USER
)
from google.genai import types
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, llm=None, client=None, model=None):
        if llm is None and client is None:
            raise ValueError("llm and client can't both be None")
        if llm:
            self.llm = llm.with_structured_output(IntentClassifierOutput.model_json_schema(), include_raw=True, method="json_schema")
        if client and model:
            self.client = client
            self.model=model
        
        self.system = INTENT_CLASSIFIER_SYSTEM
        self.user = INTENT_CLASSIFIER_USER

    def __call__(self, state: State) -> State:
        conversation = get_conversation_string(state)
        
        # use model provider's api directly
        if self.client:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=self.user.format(conversation=conversation))
                    ]
                )
            ]
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system,
                    response_mime_type="application/json",
                    response_json_schema=IntentClassifierOutput.model_json_schema()
                )
            )
            response = IntentClassifierOutput.model_validate_json(response.text)
        
        # use langchain integration
        else:
            feed = self._construct_feed(conversation)
            response = self.llm.invoke(feed)

        return {"intent": response.result.intent}
    
    def _construct_feed(self, messages):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(conversation=messages))
        ]
        return feed

        