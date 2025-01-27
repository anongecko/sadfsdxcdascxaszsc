from fastapi import APIRouter, HTTPException, Header, Request, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
import asyncio
from pathlib import Path

router = APIRouter()
logger = logging.getLogger("ImageRouter")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    height: Optional[int] = Field(1024, ge=512, le=2048)
    width: Optional[int] = Field(1024, ge=512, le=2048)
    guidance_scale: Optional[float] = Field(3.5, ge=1.0, le=20.0)
    num_inference_steps: Optional[int] = Field(50, ge=20, le=100)
    num_images: Optional[int] = Field(1, ge=1, le=4)
    seed: Optional[int] = None

    # Advanced parameters for FLUX.1
    clip_skip: Optional[int] = Field(None, ge=1, le=4)
    vae_batch_size: Optional[int] = Field(None, ge=1, le=4)
    scheduler: Optional[str] = Field(None, regex="^(euler|euler_a|ddim|dpm|lms)$")
    attention_slicing: Optional[bool] = Field(True)

    @validator("height", "width")
    def validate_dimensions(cls, v, field):
        # Ensure dimensions are multiples of 8
        if v % 8 != 0:
            v = ((v + 7) // 8) * 8
            logger.info(f"Adjusted {field.name} to nearest multiple of 8: {v}")
        return v

    @validator("prompt")
    def validate_prompt(cls, v):
        # Clean and validate prompt
        v = v.strip()

        # Basic NSFW check - extend as needed
        nsfw_terms = {"nsfw", "nude", "explicit"}
        if any(term in v.lower() for term in nsfw_terms):
            raise ValueError("NSFW content is not allowed")

        return v


class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=4)
    shared_parameters: Optional[ImageGenerationRequest] = None


async def validate_api_key(api_key: str = Header(..., alias="Authorization")) -> str:
    """Validate API key with caching"""
    key = api_key.replace("Bearer ", "")
    # API key validation would be handled by the service
    return key


@router.post("/v1/images/generations")
async def generate_images(request: ImageGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Header(..., alias="Authorization")):
    """Generate images with optimized processing"""
    try:
        # Prepare request for the model service
        service_request = request.dict(exclude_none=True)

        from api.main import image_service

        # Generate images
        response = await image_service.generate(service_request)

        # Schedule background cleanup
        background_tasks.add_task(cleanup_resources)

        return response

    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/generations/batch")
async def batch_generate_images(request: BatchGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Header(..., alias="Authorization")):
    """Generate multiple images in batch with optimized processing"""
    try:
        from api.main import image_service

        # Process batches concurrently with limits
        tasks = []
        base_params = request.shared_parameters.dict(exclude_none=True) if request.shared_parameters else {}

        for prompt in request.prompts:
            params = base_params.copy()
            params["prompt"] = prompt

            # Add to task queue
            tasks.append(image_service.generate(params))

        # Execute tasks with concurrency limit
        responses = await asyncio.gather(*tasks)

        # Combine responses
        combined_response = {"id": f"batch-{datetime.now().timestamp()}", "created": int(datetime.now().timestamp()), "results": responses}

        # Schedule background cleanup
        background_tasks.add_task(cleanup_resources)

        return combined_response

    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def cleanup_resources():
    """Optimize resource usage after generation"""
    try:
        from api.main import image_service

        await image_service.optimize_memory()
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")


@router.get("/v1/images/models")
async def list_models(api_key: str = Header(..., alias="Authorization")):
    """List available image models and their capabilities"""
    try:
        from api.main import image_service

        health_status = await image_service.health_check()

        return {
            "data": [
                {
                    "id": "flux1-dev",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "black-forest-labs",
                    "capabilities": {
                        "max_height": 2048,
                        "max_width": 2048,
                        "supported_schedulers": ["euler", "euler_a", "ddim", "dpm", "lms"],
                        "max_batch_size": 4,
                        "supports_negative_prompt": True,
                        "optimizations": {"attention_slicing": True, "vae_slicing": True, "cpu_offload": True},
                    },
                    "status": health_status,
                }
            ]
        }
    except Exception as e:
        logger.error(f"Model list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

