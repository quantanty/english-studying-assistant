import json
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.v1.auth import get_current_user
from src.graph.assistant import Assistant
from src.models.session import Session
from src.models.user import User
from src.schemas.chat import ChatRequest, ChatResponse, Message, StreamResponse
from src.services.llm_client import create_llm
from src.utils.graph import select_model
from langchain_core.messages import convert_to_openai_messages
from src.services.database import database_service
from src.utils.sanitization import sanitize_string

router = APIRouter()
fast_model = select_model("fast")
assistant = Assistant(default_model=fast_model)

@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session_id: str,
    user: User = Depends(get_current_user)
):
    sanitized_session_id = sanitize_string(session_id)
    if not await database_service.can_user_acccess_session(user.id, sanitized_session_id):
        raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

    try:
        async def event_generator():
            try:
                async for chunk in assistant.get_stream_response(
                    chat_request.messages, session_id, str(user.id)
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

@router.get("/messages/{session_id}", response_model=ChatResponse)
async def get_sesssion_messages(session_id: str, user: User = Depends(get_current_user)):
    try:
        sanitized_session_id = sanitize_string(session_id)
        if not await database_service.can_user_acccess_session(user.id, sanitized_session_id):
            raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")
        
        lc_messages = await assistant.get_chat_history(session_id)
        oai_messages = convert_to_openai_messages(
                cast(list, lc_messages)
            )
        return ChatResponse.model_validate({"messages": oai_messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@router.delete("/messages", response_model=ChatResponse)
async def clear_chat_history(session_id: str, user: User = Depends(get_current_user)):
    try:
        sanitized_session_id = sanitize_string(session_id)
        if not await database_service.can_user_acccess_session(user.id, sanitized_session_id):
            raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

        await assistant.clear_chat_history(session_id)
        return {"message": "Chat history cleared succesfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))