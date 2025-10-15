from .models import Meaning, WordEntry, QuizQuestion

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated

class WordEntryState(TypedDict):
    word: str
    phonetic: Optional[str]
    meanings: List[Meaning]
    collocations: List[str]
    metadata: Optional[Dict]
    result: Literal["success", "error"]

class State(TypedDict):
    # ---- Conversation ----
    messages: Annotated[list[AnyMessage], add_messages]
    window_index: int
    summary: str

    # ---- IntentClassifier ____
    intent: Literal["learn_words", "general_chat"]

    # ---- Orchestrator ----
    preparation_result: str
    orchestrator_messages: List[AnyMessage]
    orchestrator_observation: List[str]
    action: Literal["define_words", "general_chat"]
    
    # ---- WordEntryGenerator ----
    words: List[str]
    collected_data: List[str]
    word_entries: Dict[str, WordEntryState]
    node_results: str

    # ---- Final output ----
    quiz: QuizQuestion

    # ---- Control / metadata ----
    metadata: Dict[str, Any]