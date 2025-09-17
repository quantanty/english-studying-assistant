from src.schemas.state import State

def after_orchestrator(state: State) -> str:
    action = state.get("action", "general_chat")
    return action

def after_preparation(state: State) -> str:
    return state.get("preparation_result", "error")