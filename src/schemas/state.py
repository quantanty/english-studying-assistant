from .models import Meaning, WordEntry, QuizQuestion

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
    # ---- User input ----
    query: str

    # ---- Intermediate ----
    action: Literal["define_words", "general_chat"]
    step_history: List[str]
    
    words: List[str]
    collected_data: List[str]
    word_entries: Dict[str, WordEntryState]
    node_results: str

    # ---- Final output ----
    quiz: QuizQuestion
    response: Any

    # ---- Control / metadata ----
    metadata: Dict[str, Any]