from src.api.v1 import api_router
from src.config import settings

from fastapi import (
    FastAPI,
    Request,
    status
)

from fastapi.responses import JSONResponse

from typing import Dict, Any
from datetime import datetime
from src.services.database import database_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root(request: Request):
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
    }


@app.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    db_healthy = await database_service.health_check()
    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "component": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }
    status_code = status.HTTP_200_OK
    return JSONResponse(content=response, status_code=status_code)
