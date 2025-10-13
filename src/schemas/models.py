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
    difference: str = Field(None, description="One-line explanation of how this word differs from the base word (e.g., 'affect is a verb, effect is a noun').")

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

class OrchestratorOutput(BaseModel):
    action: Literal["define_words", "general_chat"] = Field(..., description="Next action. Input should be 'define_words', 'general_chat'")
    words: Optional[List[str]] = Field(None, description="English words that user want to learn, extracted from user's query.")
    # orchestrator_notes: Optional[str] = Field(None, "")

class IntentClassifierOutput(BaseModel):
    reasoning: str = Field(..., description="Thinking out loud")
    intent: Literal["learn_words", "general_chat"] = Field(..., description="User's intent")

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
    options: List[str] = Field(..., min_items=2)
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




# ---------- Example usage ----------
if __name__ == "__main__":
    example = WordEntry(
        word="run",
        phonetic="/rʌn/",
        meanings=[
            Meaning(
                definition="to move swiftly on foot",
                part_of_speech="verb",
                translation_vi="chạy",
                examples=[Example(en="I run every morning.", vi="Tôi chạy bộ mỗi sáng.")],
                pro_tips=[
                    "In sports contexts, pair with adverbs of manner: 'run quickly', 'run slowly'.",
                ],
                similar_words=[
                    SimilarWord(word="jog", difference="Jog = light running for exercise; slower than 'run'.")
                ],
            ),
            Meaning(
                definition="to manage or operate",
                part_of_speech="verb",
                translation_vi="điều hành",
                examples=[Example(en="She runs a small business.", vi="Cô ấy điều hành một doanh nghiệp nhỏ.")],
                pro_tips=[
                    "Use 'run a company' to mean 'operate' rather than 'control'.",
                ],
                usage_note="Common collocation: 'run a business', not 'make a business'",
            ),
        ],
        collocations=["run out of time", "run into trouble", "run a risk"],
        
        metadata={"source": "custom-dictionary", "generated_at": "2025-09-12T00:00:00+07:00"},
    )

    # Dump JSON
    print(example.model_dump_json(indent=2))

    quiz = MultipleChoiceQuestion(
        question="foobar question",
        options=['a', 'b'],
        answer_index=1
    )
    print(quiz.model_dump_json(indent=2))