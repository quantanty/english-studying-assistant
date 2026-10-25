import json
import uuid
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import convert_to_openai_messages

from src.api.v1.auth import get_current_user
from src.graph.assistant import Assistant
from src.models.chat_sessions import ChatSession
from src.models.users import User
from src.schemas.chat import (ChatRequest, ChatResponse, DeltaPayload,
                              DonePayload, ErrorPayload, Message,
                              MetadataPayload, StreamResponse)
from src.services.database import database_service
from src.utils.graph import select_model
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

    if sanitized_session_id and not await database_service.can_user_acccess_session(user.id, sanitized_session_id):
        raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

    # if session_id is None:
    #     session_id = str(uuid.uuid4())
    #     session = await database_service.create_session(session_id, user.id)
    #     session_response = MetadataPayload(session_id=session.id, session_name=session.name)
    #     yield f"data: {json.dumps(session_response.model_dump())}\n\n"
    # elif not await database_service.can_user_acccess_session(user.id, sanitized_session_id):
    #     raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

    try:
        async def event_generator():
            try:
                session = await database_service.create_session(user.id, user.name)
                session_response = MetadataPayload(session_id=session.id, session_name=session.title)
                yield f"data: {json.dumps(session_response.model_dump())}\n\n"

                async for chunk in assistant.get_stream_response(
                    chat_request.message, session_id, str(user.id)
                ):
                    response = DeltaPayload(content=chunk)
                    yield f"data: {json.dumps(response.model_dump())}\n\n"
                final_response = DonePayload()
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"
            except Exception as e:
                error_response = ErrorPayload(error=str(e))
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