import logging

from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import INTENT_CLASSIFIER_SYSTEM, INTENT_CLASSIFIER_USER
from src.schemas.llms import ModelSelection
from src.schemas.models import (
    GeneralChatIntent,
    IntentClassifierOutput,
    LearnWordsIntent,
)
from src.schemas.state import State
from src.services.llm_client import get_model
from src.utils.context import get_conversation_string

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, default_model: ModelSelection):
        self.default_model = default_model

        self.system = INTENT_CLASSIFIER_SYSTEM
        self.user = INTENT_CLASSIFIER_USER

    def __call__(self, state: State) -> State:
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("intent_classifier")
        else:
            model_selection = None
        if model_selection:
            llm = get_model(model_type=model_selection.model_type, model_name=model_selection.model_name)
        else:
            llm = get_model(model_type=self.default_model.model_type, model_name=self.default_model.model_name)
        llm_structured_output = llm.with_structured_output(IntentClassifierOutput, include_raw=True)
        
        conversation = get_conversation_string(state)
        
        feed = self._construct_feed(conversation)
        response = llm_structured_output.invoke(feed)
        parsed = response["parsed"]
        # TODO: retry mechanism

        result = parsed.result
        intent = result.intent
        if isinstance(result, LearnWordsIntent):
            words = result.words
            should_create_quizzes = result.should_create_quizzes
            return {
                "intent": intent,
                "words": words,
                "should_create_quizzes": should_create_quizzes,
            }
        
        return {"intent": intent}
    
    def _construct_feed(self, messages):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(conversation=messages))
        ]
        return feed

        