from src.schemas.state import State

def after_orchestrator(state: State) -> str:
    step_state = state.get("step_state")
    action = step_state.get("action_type")
    return action

def after_preparation(state: State) -> str:
    return state.get("preparation_result", "error")