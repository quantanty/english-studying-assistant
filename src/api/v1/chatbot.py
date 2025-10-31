from src.graph.assistant import Assistant
from src.schemas.chat import (
    Message,
    ChatRequest,
    StreamResponse
)
from src.services.llm_client import create_llm

from fastapi.responses import StreamingResponse
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request
)

import json

router = APIRouter()
assistant = Assistant(llm=create_llm())

@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session_id: str
):
    try:
        async def event_generator():
            try:
                async for chunk in assistant.get_stream_response(
                    chat_request.messages, session_id
                ):
                    response = StreamResponse(content=chunk, done=False)
                    yield f"data: {json.dumps(response.model_dump())}\n\n"
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"
            except Exception as e:
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))