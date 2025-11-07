from langchain_core.messages import ToolMessage
from src.schemas.state import State
from src.schemas.models import QuizQuestion

class QuizToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, tool_calls: list):
        for tool_call in tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            if isinstance(tool_result, QuizQuestion):
                ...
            else:
                ...