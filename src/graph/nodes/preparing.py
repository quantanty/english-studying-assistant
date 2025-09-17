from src.schemas import State
from src.utils.context import get_conversation_string

from langchain_core.messages import SystemMessage, HumanMessage

def preparing(orchestrator_prompt, user_template):
    def preparing_(state: State) -> State:
        try:
            messages = state["messages"]
            assert isinstance(messages[-1], HumanMessage)

            conversation = get_conversation_string(state)

            words = state.get("words", [])
            content = user_template.format(conversation=conversation)
            if words:
                content += f"Processed words: {str(words)} (you don't need to retrieve information again)"
        except Exception as e:
            return {**state, "preparation_result": "error"}
        
        return {
            **state,
            "orchestrator_messages": [
                SystemMessage(orchestrator_prompt),
                HumanMessage(content)
            ],
            "preparation_result": "success"
        }
    return preparing_