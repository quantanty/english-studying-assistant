import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import StreamWriter

from src.prompts import PRESENTER_SYSTEM, PRESENTER_USER
from src.schemas.state import State, WordEntryState
from src.services.llm_client import get_model
from src.utils.context import (
    get_conversation_string,
    get_intent_string,
    get_word_defining_results,
)

logger = logging.getLogger(__name__)

class Presenter:
    def __init__(self, default_model):
        self.default_model = default_model
        self.system = PRESENTER_SYSTEM
        self.user = PRESENTER_USER

    async def __call__(self, state: State,  writer: StreamWriter) -> State:
        selected_model = state.get("selected_model")
        if selected_model:
            model_selection = selected_model.get("orchestrator")
        else:
            model_selection = None
        if model_selection:
            llm = get_model(model_type=model_selection.model_type, model_name=model_selection.model_name)
        else:
            llm = get_model(model_type=self.default_model.model_type, model_name=self.default_model.model_name)
        
        intent = get_intent_string(state)
        conversation = get_conversation_string(state)
        prepared_data = get_word_defining_results(state.get("word_entries", {}))
        workflow_log = ""
        # TODO: assistant workflow log
        
        tokens = []
        feed = self._construct_feed(conversation, intent, workflow_log, prepared_data)
        async for chunk in llm.astream(feed):
            if getattr(chunk, "content", None):
                tokens.append(chunk.content)
                writer({"presenter": chunk.content})

        message = AIMessage(content="".join(tokens))
        return {"messages": [message]}

    def _construct_feed(self, conversation: str, intent: str, workflow_log: str, prepared_data: str):
        if workflow_log:
            _workflow_log = f"\n<workflow_log>\n{workflow_log}\n</workflow_log>"
        else:
            _workflow_log = ""
        if prepared_data:
            _prepared_data = f"\n<prepared_data>\n{workflow_log}\n</prepared_data>"
        else:
            _prepared_data = ""
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(
                content=self.user.format(
                    conversation=conversation,
                    intent=intent,
                    workflow_log=_workflow_log,
                    prepared_data=_prepared_data
                )
            ),
        ]

        return feed