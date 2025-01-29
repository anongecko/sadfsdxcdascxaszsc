import subprocess
import asyncio
import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List
from .base_service import BaseModelService
from cachetools import TTLCache
import time
import logging


class TextModelService(BaseModelService):
    def __init__(self, config_path: str, api_keys_path: str, log_path: str):
        super().__init__(config_path, api_keys_path, log_path)
        self.model_lock = asyncio.Lock()
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour cache
        self.model_path = Path("/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf")

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for DeepSeek-R1"""
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":  # Skip system messages per model recommendations
                content = msg["content"].strip()
                if msg["role"] == "user":
                    formatted_messages.append(f"<｜User｜>{content}")
                elif msg["role"] == "assistant":
                    formatted_messages.append(f"<｜Assistant｜>{content}")

        return "".join(formatted_messages) + "<｜Assistant｜>"

    def _get_cache_key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Generate deterministic cache key"""
        message_str = json.dumps([{
            "role": m["role"],
            "content": m["content"]
        } for m in messages], sort_keys=True)
        return f"{hash(message_str)}:{temperature}"

    async def initialize(self) -> bool:
        """Initialize and verify DeepSeek model"""
        try:
            if not self.model_path.exists():
                self.logger.error(f"DeepSeek model not found at {self.model_path}")
                return False

            # Verify model can be loaded with H100 optimizations
            test_cmd = [
                "/home/azureuser/llama.cpp/main",
                "-m", str(self.model_path),
                "--n-gpu-layers", "1",
                "--ctx-size", "8",
                "--prompt", "<｜User｜>test<｜Assistant｜>",
                "-n", "1",
                "--numa",  # Enable NUMA optimization
                "--gpu-layers", "40"  # H100 optimization
            ]

            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(f"Model initialization failed: {stderr.decode()}")
                return False

            self.is_loaded = True
            self.logger.info("DeepSeek model initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False

    async def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text completion"""
        self.last_activity = datetime.now()
        start_time = time.time()

        try:
            # Validate and adjust temperature
            temperature = min(max(request.get("temperature", 0.6), 0.5), 0.7)
            cache_key = self._get_cache_key(request["messages"], temperature)

            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            # Format prompt
            prompt = self._format_prompt(request["messages"])

            async with self.model_lock:
                # H100-optimized command
                cmd = [
                    "/home/azureuser/llama.cpp/main",
                    "-m", str(self.model_path),
                    "--ctx-size", "163840",
                    "--batch-size", "4096",
                    "--threads", "40",
                    "--temp", str(temperature),
                    "--n-gpu-layers", "40",
                    "--mlock",
                    "--numa",
                    "--rope-scaling", "yarn",
                    "--rope-freq-base", "10000",
                    "--rope-freq-scale", "0.1",
                    "--gpu-memory-utilization", "98",  # Aggressive memory usage
                    "-p", prompt
                ]

                if request.get("max_tokens"):
                    cmd.extend(["-n", str(min(request["max_tokens"], 163840))])

                # Execute model
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    self.logger.error(f"Generation error: {stderr.decode()}")
                    raise RuntimeError(f"Model failed: {stderr.decode()}")

                response_text = stdout.decode().strip()
                processing_time = time.time() - start_time

                # Update metrics
                self.update_performance_metrics(processing_time)
                gpu_metrics = await self.get_gpu_metrics()

                # Format response
                response = {
                    "id": f"chatcmpl-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": "deepseek-r1",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.split()) + len(response_text.split())
                    },
                    "performance": {
                        "processing_time": processing_time,
                        "gpu_metrics": gpu_metrics
                    }
                }

                # Cache successful response
                self.response_cache[cache_key] = response
                return response

        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            raise