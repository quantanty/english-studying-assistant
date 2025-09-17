from src.schemas.state import State

from langchain_core.messages import HumanMessage, AIMessage

def get_conversation_string(state: State):
    messages = state["messages"]
    conversation_string = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_string += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_string += f"Assistant: {msg.content}\n"
    
    return conversation_string