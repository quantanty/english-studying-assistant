from .models import Meaning, WordEntry, QuizQuestion, PlanStep

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated
from .llms import ModelSelection

class WordEntryState(TypedDict, total=False):
    word: str
    phonetic: Optional[str]
    meanings: List[Meaning]
    collocations: List[str]
    metadata: Optional[Dict]
    result: Literal["success", "error"]

class StepState(TypedDict, total=False):
    action_type: Literal[
        "define_words", "generate_quizzes", "present", "end"
    ]
    params: Any
    success: Optional[bool]
    observation: Optional[str]

class SelectedModel(TypedDict):
    intent_classifier: ModelSelection
    planner: ModelSelection
    orchestrator: ModelSelection
    word_entry_generator: ModelSelection
    quiz_generator: ModelSelection
    presenter: ModelSelection

class State(TypedDict, total=False):
    # ---- Conversation ----
    messages: Annotated[list[AnyMessage], add_messages]
    window_index: int
    summary: str

    # ---- Model Selection ----
    selected_models: SelectedModel

    # ---- Intermediate steps ____
    preparation_result: Literal["success", "error"]
    intent: Literal["learn_words", "general_chat"]
    words: List[str]
    should_create_quizzes: bool
    instruction: Optional[str]
    plan: List[PlanStep]
    orchestrator_observation: List[str]
    step_state: StepState
    words_to_define: List[str]
    collected_data: Dict[str, List[str]]
    word_entries: Dict[str, WordEntryState]
    qgen_instruction: str
    error: str
    
    # ---- LLM messages ----
    orchestrator_messages: List[AnyMessage]
    qgen_messages: List[AnyMessage]
    

    # ---- Final output ----
    quizzes: List[QuizQuestion]

    # ---- Control / metadata ----
    metadata: Dict[str, Any]