from src.schemas.chat import Message

def dump_messages(messages: list[Message]) -> list[dict]:
    return [message.model_dump() for message in messages]