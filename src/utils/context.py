from src.schemas.state import State, WordEntryState
from src.schemas.models import DefineWordsParams

from langchain_core.messages import HumanMessage, AIMessage
import textwrap

def get_conversation_string(state: State):
    messages = state.get("messages", [])
    assert len(messages) > 0 and isinstance(messages[-1], HumanMessage)
    conversation_string = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_string += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_string += f"Assistant: {msg.content}\n"
    
    return conversation_string

def format_indent(content: str, indent: int) -> str:
    if not content:
        return ""
    
    prefix = " " * indent
    
    return textwrap.indent(content, prefix)

def get_intent_string(state: State):
    intent_string = ""

    intent = state.get("intent")
    intent_string += f"intent: {intent}"

    if intent == "learn_words":
        words = state.get("words", [])
        should_create_quizzes = state.get("should_create_quizzes", None)
        if words:
            intent_string += f"\nwords: {str(words)}"
        if should_create_quizzes is not None:
            intent_string += f"\nshould create quizzes: {str(should_create_quizzes)}"

    instruction = state.get("instruction")
    if instruction:
        intent_string += f"\ninstruction: {instruction}"

    return intent_string

def get_words_to_define(state: State):
    step_state = state.get("step_state")
    assert step_state
    params = step_state.get("params")
    assert isinstance(params, DefineWordsParams)
    return params.words

def write_report(state: State, report_string: str):
    step_state = state.get("step_state", {})
    step_state["observation"] = report_string
    return step_state

def get_word_defining_results(word_entries: dict[str, WordEntryState]) -> str:
    result_string = ""
    for word, entry in word_entries.items():
        result = entry.get("result", "error")
        result_string += f"'{word}': {result}\n"
    return result_string