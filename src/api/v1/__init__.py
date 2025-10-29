from fastapi import APIRouter

from src.api.v1.chatbot import router as chatbot_router


api_router = APIRouter()

api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])

@api_router.get("/health")
async def heath_check():
    return {"status": "healthy", "version": "1.0.0"}