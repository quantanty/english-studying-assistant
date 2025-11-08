from .models import Meaning, WordEntry, QuizQuestion, PlanStep

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

class StepState(TypedDict):
    action_type: Literal[
        "extract_words", "define_words", "generate_quizzes", "present", "end"
    ]
    success: bool
    observation: str

class State(TypedDict):
    # ---- Conversation ----
    messages: Annotated[list[AnyMessage], add_messages]
    window_index: int
    summary: str

    # ---- Intermediate steps ____
    intent: Literal["learn_words", "general_chat"]
    plan: List[PlanStep]
    orchestrator_messages: List[AnyMessage]
    orchestrator_observation: List[str]
    step_state: StepState
    words: List[str]
    words_to_define: List[str]
    collected_data: Dict[str, List[str]]
    word_entries: Dict[str, WordEntryState]
    qgen_messages: List[AnyMessage]
    

    # ---- Final output ----
    quizzes: List[QuizQuestion]

    # ---- Control / metadata ----
    metadata: Dict[str, Any]