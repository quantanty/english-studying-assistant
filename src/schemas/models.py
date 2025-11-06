from typing import List, Optional, Literal, Union, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

# ---------- Basic small models ----------
class Action(str, Enum):
    EXTRACT = "extract_words"
    DEFINE = "define_words"
    GENERAL = "general_chat"
    END = "end_session"

class Example(BaseModel):
    en: str = Field(..., description="A full English sentence illustrating this meaning. Must be grammatical and natural.")
    vi: Optional[str] = Field(None, description="Vietnamese translation of the English sentence. If not applicable, leave null.")

class SimilarWord(BaseModel):
    word: str = Field(..., description="Similar / confusable word")
    difference: str = Field(..., description="One-line explanation of how this word differs from the base word (e.g., 'affect is a verb, effect is a noun').")

class Meaning(BaseModel):
    definition: str = Field(..., description="Definition in English")
    part_of_speech: Optional[
        Literal[
            "noun", "verb", "adj", "adv", "prep",
            "conj", "pron", "det", "phrase", "idiom", "other"
        ]
    ] = Field(None, description="Part of speech (optional). Input should be 'noun', 'verb', 'adj', 'adv', 'prep', 'conj', 'pron', 'det', 'phrase', 'idiom' or 'other'")
    translation_vi: Optional[str] = Field(None, description="Vietnamese translation for this meaning")
    examples: List[Example] = Field(default_factory=list, min_length=1, description="Examples illustrating this meaning. Provide at least 1.")
    pro_tips: List[str] = Field(
        default_factory=list,
        min_length=1,
        description=(
            "Concise usage tips specifically about how to use THIS WORD/PHRASE in English. "
            "Focus on common mistakes, register (formal/informal), collocations, or subtle nuances. "
            "Do NOT provide general tips about the concept or domain (e.g., do not explain economics for 'economy'). "
            "Provide at least 1 tip."
        )
    )
    usage_note: Optional[str] = Field(None, description="Short usage note for this specific meaning")
    similar_words: List[SimilarWord] = Field(
        default_factory=list,
        min_length=1,
        description=(
            "Words or phrases with similar or related meanings to THIS WORD/PHRASE. "
            "Focus on items that learners might confuse in terms of meaning or usage (e.g., 'economy' vs 'finance', 'big' vs 'large'). "
            "Do NOT include word family forms (e.g., 'economy' vs 'economic'). "
            "Provide at least 1."
        )
    )

class ListOfMeanings(BaseModel):
    meanings: List[Meaning] = Field(default_factory=list, min_length=1, description="List of distinct meanings")

    

class IntentClassifierOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="A brief, internal, step-by-step thought process leading to the final 'intent' decision. This is for debugging and inspection, not for end-user display."
    )
    intent: Literal["learn_words", "general_chat"] = Field(
        ...,
        description=("The final classified intent of the user's message.")
    )
    
class PlanStep(BaseModel):
    action: str
    purpose: str

class PlannerOutput(BaseModel):
    steps: List[PlanStep]

class OrchestratorOutput(BaseModel):
    reasoning: str = Field(..., description="")
    action: Literal[
        "extract_words", "define_words", "generate_quizzes", "present", "end"
    ] = Field(..., description="action types")

class WordExtractorOutput(BaseModel):
    words: List[str] = Field(default_factory=list, description="English words to be defined.")

# ---------- Top-level model ----------
class WordEntry(BaseModel):
    phonetic: Optional[str] = Field(None, description="IPA transcription (e.g., /rʌn/), not arbitrary spelling. Always provide American version when possible.")
    meanings: List[Meaning] = Field(default_factory=list, description="List of distinct meanings")
    collocations: List[str] = Field(default_factory=list, description="Common 2–3 word combinations with the base word (e.g., 'run a business', 'fast runner').")

    @model_validator(mode="after")
    def ensure_meaning_exists(self):
        if not self.meanings:
            raise ValueError("At least one meaning should be provided")
        return self

class WordEntryFull(BaseModel):
    word: str = Field(..., description="Base word or phrase")
    phonetic: Optional[str] = Field(None, description="IPA transcription (e.g., /rʌn/), not arbitrary spelling.")
    meanings: List[Meaning] = Field(default_factory=list, description="List of distinct meanings")
    collocations: List[str] = Field(default_factory=list, description="Common 2–3 word combinations with the base word (e.g., 'run a business', 'fast runner').")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Technical metadata such as source dictionary name, timestamp, provider. Do not include language content.")

    @model_validator(mode="after")
    def ensure_meaning_exists(self):
        if not self.meanings:
            raise ValueError("At least one meaning should be provided")
        return self

# ---------- Quiz models (polymorphic) ----------
class MultipleChoiceQuestion(BaseModel):
    type: Literal["multiple_choice"] = "multiple_choice"
    question: str
    options: List[str] = Field(..., min_length=2)
    answer_index: int = Field(..., description="Index into options (0-based)")
    explanation: Optional[str] = Field(None, description="Explanation for the correct answer")

    @field_validator("answer_index")
    @classmethod
    def index_in_range(cls, v, values):
        opts = values.data.get("options")
        if opts is None:
            raise ValueError("options must be provided")
        if not (0 <= v < len(opts)):
            raise ValueError("answer_index out of range")
        return v


class FillInBlankQuestion(BaseModel):
    type: Literal["fill_in_blank"] = "fill_in_blank"
    question: str = Field(
        ..., description="Question text with blank, e.g. 'She ___ a small company.'"
    )
    answer: str = Field(..., description="Correct token(s) to fill the blank")
    hint: Optional[str] = None
    explanation: Optional[str] = None


class TrueFalseQuestion(BaseModel):
    type: Literal["true_false"] = "true_false"
    question: str
    answer: bool
    explanation: Optional[str] = None


QuizQuestion = Union[MultipleChoiceQuestion, FillInBlankQuestion, TrueFalseQuestion]