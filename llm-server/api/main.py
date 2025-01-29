from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import asyncio
from typing import Dict, Any
import json
import time

from services import TextModelService
from api.routers import text_router
from api.middleware.auth import AuthMiddleware
from api.middleware.monitoring import MonitoringMiddleware

# Configure logging
logging.basicConfig(filename="/mnt/data/llm-server/logs/api.log", level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("API")

# Initialize FastAPI app
app = FastAPI(title="DeepSeek-R1 API", description="API for DeepSeek-R1 large language model", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(MonitoringMiddleware)

# Initialize text service (global instance)
text_service = TextModelService(
    config_path="/mnt/data/llm-server/config/server_config.json", api_keys_path="/mnt/data/llm-server/config/text_api_keys.json", log_path="/mnt/data/llm-server/logs/text-model.log"
)

# Include text router
app.include_router(text_router.router, tags=["text"])


@app.on_event("startup")
async def startup_event():
    """Initialize text service on startup"""
    try:
        logger.info("Starting DeepSeek-R1 service initialization")

        # H100-specific CUDA optimizations
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)

        await text_service.initialize()
        logger.info("DeepSeek-R1 service initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Initiating shutdown")
        await text_service.cleanup()
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        text_health = await text_service.health_check()

        # System metrics
        disk_usage = Path("/mnt/data").statvfs()
        disk_space = {
            "total": disk_usage.f_blocks * disk_usage.f_frsize,
            "available": disk_usage.f_bavail * disk_usage.f_frsize,
            "used_percent": (1 - (disk_usage.f_bavail / disk_usage.f_blocks)) * 100,
        }

        return {"status": "healthy" if text_health["status"] == "healthy" else "degraded", "timestamp": time.time(), "services": {"text": text_health}, "system": {"disk_space": disk_space}}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {"error": {"code": exc.status_code, "message": exc.detail}}


# Store startup time for uptime tracking
startup_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=4,
        limit_concurrency=100,
        timeout_keep_alive=75,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "use_colors": None,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/mnt/data/llm-server/logs/uvicorn.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
            },
        },
    )

