from langchain_core.messages import AIMessage, ToolMessage

from src.schemas.models import QuizQuestion
from src.schemas.state import State


class QuizToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        try:
            qgen_messages = state.get("qgen_messages", [])
            assert len(qgen_messages) > 0
            message = qgen_messages[-1]
            assert isinstance(message, AIMessage) 
            
            tool_calls = message.tool_calls
            quizzes = state.get("quizzes", [])
            
            for tool_call in tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                if isinstance(tool_result, QuizQuestion):
                    content=f"Successfully created a {tool_result.type} quiz"
                else:
                    content = tool_result
                qgen_messages.append(
                    ToolMessage(
                        content=content,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
            return {"quizzes": quizzes, "qgen_message": qgen_messages} 
        except Exception as e:
            return {"error": str(e)}