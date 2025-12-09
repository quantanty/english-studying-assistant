import json
import logging
from typing import cast

from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import INTENT_CLASSIFIER_SYSTEM, INTENT_CLASSIFIER_USER
from src.schemas.llms import ModelSelection, ModelType
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
            model_type=model_selection.model_type
            model_name=model_selection.model_name
            llm = get_model(model_type=model_type, model_name=model_name)
        else:
            model_type=self.default_model.model_type
            model_name=self.default_model.model_name
            llm = get_model(model_type=model_type, model_name=model_name)
        
        conversation = get_conversation_string(state)

        # Optimize for llama.cpp inference
        if model_type == ModelType.LLAMA_CPP.value:
            schema = "\n<output_schema>" + json.dumps(IntentClassifierOutput.model_json_schema(), indent=4) + "</output_schema>"
            feed = self._construct_feed(schema, conversation)
            response = llm.invoke(
                feed,
                extra_body={
                    "id_slot": 0,
                    "json_schema": IntentClassifierOutput.model_json_schema(),
                }
            )
            parsed = IntentClassifierOutput.model_validate_json(cast(str, response.content))
        else:
            schema = ""
            llm_structured_output = llm.with_structured_output(IntentClassifierOutput, include_raw=True)
            feed = self._construct_feed(schema, conversation)
            response = llm_structured_output.invoke(feed)#, extra_body={"id_slot": 1, "json_schema": IntentClassifierOutput.model_json_schema()})
            parsed = response["parsed"]
        # TODO: retry mechanism

        result = parsed.result
        intent = result.intent
        instruction = parsed.important_intent_and_instruction_summary
        if isinstance(result, LearnWordsIntent):
            words = result.words
            should_create_quizzes = result.should_create_quizzes
            return {
                "intent": intent,
                "words": words,
                "should_create_quizzes": should_create_quizzes,
                "instruction": instruction,
            }
        
        return {"intent": intent, "instruction": instruction}
    
    def _construct_feed(self, schema, messages):
        feed = [
            SystemMessage(content=self.system.format(schema=schema)),
            HumanMessage(content=self.user.format(conversation=messages))
        ]
        return feed

        