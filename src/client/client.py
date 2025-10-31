from typing import AsyncGenerator
from src.schemas.chat import ChatRequest, Message
import httpx
import json

class AssistantClient:
    def __init__(
        self,
        base_url: str = "http:0.0.0.0",
        timeout: float | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        
    def _parse_stream_line(self, line: str):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            
            if parsed["done"]:
                return None
            return parsed["content"]
        return None
        
    async def astream(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        chat_request = ChatRequest(messages=[
            Message(role="user", content=message)
        ])
        params = {
            "session_id": session_id
        }
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/chatbot/chat/stream",
                    json=chat_request.model_dump(),
                    params=params,
                    timeout=self.timeout
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise e