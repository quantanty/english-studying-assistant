import uuid
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings
from src.models.session import Session
from src.models.user import User
from src.schemas.auth import (SessionResponse, TokenResponse, UserCreate,
                              UserResponse)
from src.services.database import database_service
from src.utils.auth import create_access_token, verify_token
from src.utils.sanitization import (sanitize_email, sanitize_string,
                                    validate_password_strength)

router = APIRouter()
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    try:
        token = sanitize_string(credentials.credentials)
        
        user_id = verify_token(token)
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        user_id_int = int(user_id)
        user = await database_service.get_user(user_id_int)
        if user is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Session:
    try:
        token = sanitize_string(credentials.credentials)
        session_id = verify_token(token)
        if session_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )

        session_id = sanitize_string(session_id)
        session = await database_service.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return session
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    
@router.post("/register", response_model=UserResponse)
async def register_user(request: Request, user_data: UserCreate):
    try:
        sanitized_email = sanitize_email(user_data.email)
        
        password = user_data.password.get_secret_value()
        validate_password_strength(password)
        
        if await database_service.get_user_by_email(sanitized_email):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user = await database_service.create_user(email=sanitized_email, password=User.hash_password(password))
        
        token = create_access_token(str(user.id))
        
        return UserResponse(id=user.id, email=user.email, token=token)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    
@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request, username: str = Form(...), password: str = Form(...), grant_type: str = Form(default="password")
):
    try:
        username = sanitize_string(username)
        password = sanitize_string(password)
        grant_type = sanitize_string(grant_type)
        
        if grant_type != "password":
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant type. Must be 'password'"
            )
            
        user = await database_service.get_user_by_email(username)
        if not user or not user.verify_password(password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token = create_access_token(str(user.id))
        return TokenResponse(access_token=token.access_token, token_type="bearer", expires_at=token.expires_at)        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

@router.post("/session", response_model=SessionResponse)
async def create_session(user: User = Depends(get_current_user)):
    try:
        session_id = str(uuid.uuid4())

        session = await database_service.create_session(session_id, user.id)

        return SessionResponse(session_id=session_id, name=session.name)
    except ValueError as ve:
        raise HTTPException(
            status_code=422, detail=str(ve)
        )

@router.patch("/session/{session_id}/name", response_model=SessionResponse)
async def update_session_name(
    session_id: str,
    name: str = Form(...),
    user: User = Depends(get_current_user)
):
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_name = sanitize_string(name)

        chat_session = await database_service.get_session(session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if chat_session.user_id != user.id:
            raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

        session = await database_service.update_session_name(sanitized_session_id, sanitized_name)

        return SessionResponse(session_id=sanitized_session_id, name=session.name)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    try:
        sanitized_session_id = sanitize_string(session_id)
        chat_session = await database_service.get_session(sanitized_session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if chat_session.user_id != user.id:
            raise HTTPException(status_code=403, detail="You do not have permission to access this chat session.")

        await database_service.delete_session(sanitized_session_id)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))


@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(user: User = Depends(get_current_user)):
    try:
        sessions = await database_service.get_user_sessions(user.id)
        return [
            SessionResponse(
                session_id=sanitize_string(session.id),
                name=sanitize_string(session.name),
            )
            for session in sessions
        ]
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
