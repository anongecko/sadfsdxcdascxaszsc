from fastapi import APIRouter, HTTPException, Header, Request, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import json

router = APIRouter()
logger = logging.getLogger("TextRouter")


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1)
    temperature: Optional[float] = Field(0.6, ge=0.5, le=0.7)
    max_tokens: Optional[int] = Field(None, ge=1, le=163840)
    stream: Optional[bool] = Field(False)

    # Advanced parameters
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    stop_sequences: Optional[List[str]] = Field(None, max_items=4)

    @validator("messages")
    def validate_messages(cls, v):
        if not any(msg.role == "user" for msg in v):
            raise ValueError("At least one user message is required")
        return [msg for msg in v if msg.role != "system"]

    @validator("messages", each_item=True)
    def validate_message_content(cls, v):
        v.content = v.content.strip()
        if len(v.content) < 1:
            raise ValueError("Message content cannot be empty")
        return v


@router.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, background_tasks: BackgroundTasks, api_key: str = Header(..., alias="Authorization")):
    """Generate chat completion using DeepSeek-R1"""
    try:
        api_key = api_key.replace("Bearer ", "")

        service_request = {
            "messages": [msg.dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens if request.max_tokens else 163840,
        }

        # Add optional parameters
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop_sequences"]:
            if hasattr(request, param) and getattr(request, param) is not None:
                service_request[param] = getattr(request, param)

        from api.main import text_service

        if request.stream:
            return await stream_response(service_request, text_service)

        response = await text_service.generate(service_request)

        # Schedule cleanup if memory usage is high
        if await should_cleanup():
            background_tasks.add_task(cleanup_resources)

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: Dict[str, Any], text_service):
    """Stream response with H100-optimized chunking"""

    async def generate():
        try:
            response = await text_service.generate(request)
            content = response["choices"][0]["message"]["content"]

            # Optimize chunk size for H100 throughput
            chunk_size = min(max(len(content) // 20, 100), 1000)

            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                yield {
                    "id": response["id"],
                    "object": "chat.completion.chunk",
                    "created": response["created"],
                    "model": "deepseek-r1",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None if i + chunk_size < len(content) else "stop"}],
                }

                # Dynamic delay based on chunk size for H100 optimization
                await asyncio.sleep(0.01 * (len(chunk) / 100))

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return generate()


async def should_cleanup() -> bool:
    """Check if cleanup is needed based on H100 resource usage"""
    try:
        from api.main import text_service

        metrics = await text_service.get_gpu_metrics()
        return (
            metrics.get("memory_used_gb", 0) > 280  # 87.5% of 320GB
            or metrics.get("utilization", 0) > 95
        )
    except:
        return False


async def cleanup_resources():
    """Optimize H100 resource usage"""
    try:
        from api.main import text_service

        await text_service.optimize_memory()
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")


@router.get("/v1/models")
async def list_models(api_key: str = Header(..., alias="Authorization")):
    """List DeepSeek-R1 model and capabilities"""
    try:
        from api.main import text_service

        health_status = await text_service.health_check()

        return {
            "data": [
                {
                    "id": "deepseek-r1",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "deepseek",
                    "permission": [],
                    "root": "DeepSeek-R1-Q4_K_M",
                    "parent": None,
                    "capabilities": {"max_context_length": 163840, "streaming": True, "temperature_range": [0.5, 0.7], "recommended_temperature": 0.6},
                    "status": health_status,
                }
            ]
        }
    except Exception as e:
        logger.error(f"Model list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

