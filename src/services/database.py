from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import (
    Session,
    SQLModel,
    create_engine,
    select
)

from src.config import Environment, settings
from src.models.session import Session as ChatSession
from src.models.user import User


class DatabaseService:
    def __init__(self):
        try:
            pool_size = settings.POSTGRES_POOL_SIZE
            max_overflow = settings.POSTGRES_MAX_OVERFLOW
            
            connection_url = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_HOST}/{settings.POSTGRES_DB}"
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
        
    async def get_user(self, user_id: int) -> Optional[User]:
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return user
        
    async def get_user_by_email(self, email: str) -> Optional[User]:
        with Session(self.engine) as session:
            statement = select(User).where(User.email == email)
            user = session.exec(statement).first()
            return user