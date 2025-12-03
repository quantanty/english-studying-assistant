from src.schemas.state import State

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

    return intent_string