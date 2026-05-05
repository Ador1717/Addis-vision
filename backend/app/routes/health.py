from fastapi import APIRouter
from app.config import settings
from app.model_manager import model_manager
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        available_models=model_manager.available_models(),
    )
