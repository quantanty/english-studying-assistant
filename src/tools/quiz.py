from typing import List, Optional
from langchain.tools import tool
from src.schemas.models import (
    QuizQuestion,
    MultipleChoiceQuestion,
    FillInBlankQuestion,
    TrueFalseQuestion
)

@tool(args_schema=MultipleChoiceQuestion)
def create_multiple_choice_question(
    question: str,
    options: List[str],
    answer_index: int,
    explanation: Optional[str]
) -> MultipleChoiceQuestion:
    """
    Generate a multiple choice question for testing meaning comprehension.
    """ 
    return MultipleChoiceQuestion(
        question=question,
        options=options,
        answer_index=answer_index,
        explanation=explanation 
    )

@tool(args_schema=FillInBlankQuestion)
def create_fill_in_blank_question(
    question: str,
    answer: str,
    hint: Optional[str] = None,
    explanation: Optional[str] = None
):
    """
    Generate a fill in the blank question for testing meaning comprehension.
    """
    return FillInBlankQuestion(
        question=question,
        answer=answer,
        hint=hint,
        explanation=explanation
    )

@tool(args_schema=TrueFalseQuestion)
def create_true_false_question(
    question: str,
    answer: bool,
    explanation: Optional[str] = None
):
    """
    Generate a true false question for testing meaning comprehension.
    """
    return TrueFalseQuestion(question=question, answer=answer, explanation=explanation)