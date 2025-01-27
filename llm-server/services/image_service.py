import torch
from diffusers import FluxPipeline
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
from PIL import Image
import time
from .base_service import BaseModelService


class ImageModelService(BaseModelService):
    def __init__(self, config_path: str, api_keys_path: str, log_path: str):
        super().__init__(config_path, api_keys_path, log_path)
        self.model_lock = asyncio.Lock()
        self.pipe = None
        self.model_path = Path("/mnt/data/llm-server/models/image/flux1-dev")

    async def initialize(self) -> bool:
        """Initialize the FLUX.1 model with optimizations"""
        try:
            self.logger.info("Initializing FLUX.1 model")

            # Initialize pipeline with optimizations
            self.pipe = await asyncio.to_thread(
                FluxPipeline.from_pretrained,
                self.model_path,
                torch_dtype=torch.bfloat16,  # Better for H100
                device_map="auto",
            )

            # Enable memory optimizations
            self.pipe.enable_attention_slicing(slice_size="auto")

            if torch.cuda.is_available():
                # Move to GPU and optimize
                self.pipe.to("cuda")
                self.pipe.enable_model_cpu_offload()

                # Enable CUDA optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            self.is_loaded = True
            self.logger.info("FLUX.1 model initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            return False

    def _validate_dimensions(self, height: int, width: int) -> tuple:
        """Validate and adjust image dimensions"""
        # Ensure dimensions are multiples of 8
        height = (height // 8) * 8
        width = (width // 8) * 8

        # Apply limits
        max_size = self.config["models"]["image"]["limits"]["max_height"]
        height = min(height, max_size)
        width = min(width, max_size)

        return height, width

    async def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from prompt"""
        self.last_activity = datetime.now()
        start_time = time.time()

        async with self.model_lock:
            try:
                # Extract and validate parameters
                prompt = request["prompt"]
                height = request.get("height", 1024)
                width = request.get("width", 1024)
                height, width = self._validate_dimensions(height, width)

                guidance_scale = min(max(request.get("guidance_scale", 3.5), 1.0), 20.0)
                num_inference_steps = min(request.get("num_inference_steps", 50), 100)
                num_images = min(request.get("num_images", 1), self.config["models"]["image"]["limits"]["max_num_images"])

                negative_prompt = request.get("negative_prompt")
                seed = request.get("seed", int(torch.randint(0, 2**32 - 1, (1,)).item()))

                # Set generator for reproducibility
                generator = torch.Generator("cuda").manual_seed(seed)

                # Generate images with optimized parameters
                images = await asyncio.to_thread(
                    self.pipe,
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images,
                    negative_prompt=negative_prompt,
                    generator=generator,
                )

                # Convert images to base64
                image_data = []
                for img in images.images:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG", optimize=True)
                    image_data.append(base64.b64encode(buffered.getvalue()).decode())

                processing_time = time.time() - start_time
                self.update_performance_metrics(processing_time)

                # Get GPU metrics
                gpu_metrics = await self.get_gpu_metrics()

                return {
                    "id": f"flux-{str(uuid.uuid4())}",
                    "created": int(datetime.now().timestamp()),
                    "images": image_data,
                    "parameters": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                    },
                    "performance": {"processing_time": processing_time, "gpu_metrics": gpu_metrics},
                }

            except Exception as e:
                self.logger.error(f"Image generation error: {e}")
                raise
            finally:
                # Cleanup GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pipe is not None:
                self.pipe.to("cpu")
                del self.pipe
                self.pipe = None

            await super().cleanup()

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
