from typing import Any, List, Optional

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, create_engine, select

from src.config import Environment, settings
from src.models.chat_messages import ChatMessage
from src.models.chat_sessions import ChatSession
from src.models.users import User


class DatabaseService:
    def __init__(self):
        try:
            pool_size = settings.POSTGRES_POOL_SIZE
            max_overflow = settings.POSTGRES_MAX_OVERFLOW
            
            connection_url = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
            
            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=1800
            )
            
            SQLModel.metadata.create_all(self.engine)
        except SQLAlchemyError as e:
            if settings.ENVIRONMENT != Environment.PRODUCTION:
                raise
            
    async def create_user(self, email: str, password: str) -> User:
        with Session(self.engine) as session:
            user = User(email=email, hashed_password=password)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    async def create_google_user(self, google_sub: str, email: str, name: str) -> User:
        with Session(self.engine) as session:
            user = User(google_sub=google_sub, email=email, name=name)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        
    async def get_user(self, user_id: int) -> Optional[User]:
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return user
        
    async def get_user_by_email(self, email: str) -> Optional[User]:
        with Session(self.engine) as session:
            statement = select(User).where(User.email == email)
            user = session.exec(statement).first()
            return user

    async def get_user_by_google_sub(self, google_sub: str) -> Optional[User]:
        with Session(self.engine) as session:
            statement = select(User).where(User.google_sub == google_sub)
            user = session.exec(statement).first()
            return user


    async def delete_user_by_email(self, email: str) -> bool:
        with Session(self.engine) as session:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                return False
            
            session.delete(user)
            session.commit()
            return True

    async def create_session(self, user_id: int, name: str = "") -> ChatSession:
        with Session(self.engine) as session:
            chat_session = ChatSession(user_id=user_id, name=name)
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session

    async def delete_session(self, session_id: str) -> bool:
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                return False
            
            session.delete(chat_session)
            session.commit()
            return True

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            return chat_session

    async def get_user_sessions(self, user_id: int) -> List[ChatSession]:
        with Session(self.engine) as session:
            statement = select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at)
            sessions = session.exec(statement).all()
            return list(sessions)

    async def update_session_name(self, session_id: str, name: str) -> ChatSession:
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")

            chat_session.name = name
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session

    async def get_messages(self, session_id: str):
        with Session(self.engine) as session:
            statement = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).order_by(ChatMessage.id.asc())
            messages = session.exec(statement).all()
            return messages

    async def add_message(self, session_id: str, role: str, content: str, status: str, extra_metadata: dict[str, Any] | None = None):
        with Session(self.engine) as session:
            chat_message = ChatMessage(session_id=session_id, role=role, content=content, status=status, extra_metadata=extra_metadata)
            session.add(chat_message)
            session.commit()
            session.refresh(chat_message)
            return chat_message

    async def get_session_maker(self):
        return Session(self.engine)

    async def health_check(self) -> bool:
        try:
            with Session(self.engine) as session:
                session.exec(select(1)).first()
                return True
        except Exception as e:
            return False
        
    async def can_user_acccess_session(self, user_id: int, session_id: str):
        chat_session = await self.get_session(session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return chat_session.user_id == user_id
        

database_service = DatabaseService()
