from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import settings
from app.routes.detect import router as detect_router
from app.routes.health import router as health_router
from app.routes.annotate import router as annotate_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Industry-level Addis Traffic Object Detection & Segmentation API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router, prefix="/api")
app.include_router(detect_router, prefix="/api")
app.include_router(annotate_router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Addis Traffic Detection API",
        "version": settings.app_version,
        "docs": "/docs",
    }
