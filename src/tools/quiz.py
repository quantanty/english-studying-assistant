from typing import List, Optional, Union
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
) -> Union[MultipleChoiceQuestion, str]:
    """
    Generate a multiple choice question for testing meaning comprehension.
    """ 
    try:
        quiz = MultipleChoiceQuestion(
            question=question,
            options=options,
            answer_index=answer_index,
            explanation=explanation 
        )
        return quiz
    except Exception as e:
        return f"Error while creating multiple choice question: {str(e)}"

@tool(args_schema=FillInBlankQuestion)
def create_fill_in_blank_question(
    question: str,
    answer: str,
    hint: Optional[str] = None,
    explanation: Optional[str] = None
) -> Union[FillInBlankQuestion, str]:
    """
    Generate a fill in the blank question for testing meaning comprehension.
    """
    try:
        quiz =  FillInBlankQuestion(
            question=question,
            answer=answer,
            hint=hint,
            explanation=explanation
        )
        return quiz
    except Exception as e:
        return f"Error while creating fill in blank question: {str(e)}"

@tool(args_schema=TrueFalseQuestion)
def create_true_false_question(
    question: str,
    answer: bool,
    explanation: Optional[str] = None
) -> Union[TrueFalseQuestion, str]:
    """
    Generate a true false question for testing meaning comprehension.
    """
    try:
        quiz =  TrueFalseQuestion(question=question, answer=answer, explanation=explanation)
        return quiz
    except Exception as e:
        return f"Error while creating true false question: {str(e)}"