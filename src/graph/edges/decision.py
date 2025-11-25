from src.schemas.state import State

def after_orchestrator(state: State) -> str:
    step_state = state.get("step_state")
    action = step_state.get("action_type", "end")
    return action

def after_preparation(state: State) -> str:
    return state.get("preparation_result", "error")

def after_intent_classifier(state: State) -> str:
    intent = state.get("intent", "general_chat")
    if intent == "learn_words":
        return "learn_words"
    else:
        return "general_chat"

def after_quiz_generator(state: State) -> str:
    try:
        qgen_messages = state.get("qgen_messages", [])
        assert len(qgen_messages) > 0
        if hasattr(qgen_messages[-1], "tool_calls"):
            return "tool_call"
        else:
            return "finish"
    except Exception as e:
        return "finish"