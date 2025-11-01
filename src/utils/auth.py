import re
from datetime import (
    UTC,
    datetime,
    timedelta
)
from typing import Optional
from jose import (
    JWTError,
    jwt
)

from src.config import settings
from src.schemas.auth import Token
from src.utils.sanitization import sanitize_string

def create_access_token(thread_id: str, expires_delta: Optional[timedelta] = None) -> Token:
    """Create a new access token for a thread.

    Args:
        thread_id: The unique thread ID for the conversation.
        expires_delta: Optional expiration time delta.

    Returns:
        Token: The generated access token.
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": thread_id,
        "exp": expire,
        "iat": datetime.now(UTC),
        "jti": sanitize_string(f"{thread_id}-{datetime.now(UTC).timestamp()}"),  # Add unique token identifier
    }

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return Token(access_token=encoded_jwt, expires_at=expire)


def verify_token(token: str) -> Optional[str]:
    """Verify a JWT token and return the thread ID.

    Args:
        token: The JWT token to verify.

    Returns:
        Optional[str]: The thread ID if token is valid, None otherwise.

    Raises:
        ValueError: If the token format is invalid
    """
    if not token or not isinstance(token, str):
        raise ValueError("Token must be a non-empty string")

    # Basic format validation before attempting decode
    # JWT tokens consist of 3 base64url-encoded segments separated by dots
    if not re.match(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", token):
        raise ValueError("Token format is invalid - expected JWT format")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        thread_id: str = payload.get("sub")
        if thread_id is None:
            logger.warning("token_missing_thread_id")
            return None

        return thread_id

    except JWTError as e:
        return None