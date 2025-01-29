from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
from typing import Dict, Any
import asyncio
from datetime import datetime
import nvidia_smi
import torch
import psutil
import numpy as np


class BaseModelService(ABC):
    def __init__(self, config_path: str, api_keys_path: str, log_path: str):
        self.config_path = Path(config_path)
        self.api_keys_path = Path(api_keys_path)
        self.config = self._load_config()
        self.api_keys = self._load_api_keys()
        self.logger = self._setup_logging(log_path)
        self.last_activity = datetime.now()
        self.is_loaded = False
        self.performance_metrics = []
        nvidia_smi.nvmlInit()
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        # H100-specific CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)  # Ensure using first GPU

    def _setup_logging(self, log_path: str) -> logging.Logger:
        """Configure logging with file rotation"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with error handling"""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def _load_api_keys(self) -> Dict[str, Any]:
        """Load API keys with error handling"""
        try:
            with open(self.api_keys_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load API keys: {e}")

    async def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get detailed GPU metrics for H100"""
        try:
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = nvidia_smi.nvmlDeviceGetTemperature(self.gpu_handle, nvidia_smi.NVML_TEMPERATURE_GPU)
            power = nvidia_smi.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to Watts

            return {
                "utilization": info.gpu,
                "memory_used_gb": memory.used / 1024**3,
                "memory_free_gb": memory.free / 1024**3,
                "temperature": temp,
                "power_usage": power,
                "memory_percent": (memory.used / memory.total) * 100,
            }
        except Exception as e:
            self.logger.error(f"GPU metrics error: {e}")
            return {}

    async def optimize_memory(self):
        """Optimize memory usage for H100"""
        try:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Advanced memory defragmentation if available
                if hasattr(torch.cuda, "memory_stats"):
                    self.logger.info(f"GPU memory stats before optimization: {torch.cuda.memory_stats()}")

                if hasattr(torch.cuda, "memory_defrag"):
                    torch.cuda.memory_defrag()

                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()

            # Suggest Python garbage collection
            import gc

            gc.collect()

        except Exception as e:
            self.logger.error(f"Memory optimization error: {e}")

    def update_performance_metrics(self, latency: float):
        """Track performance metrics"""
        self.performance_metrics.append(latency)
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]

    async def get_performance_stats(self) -> Dict[str, float]:
        """Calculate detailed performance statistics"""
        if not self.performance_metrics:
            return {}

        metrics = np.array(self.performance_metrics)
        return {
            "avg_latency": float(np.mean(metrics)),
            "p50_latency": float(np.percentile(metrics, 50)),
            "p95_latency": float(np.percentile(metrics, 95)),
            "p99_latency": float(np.percentile(metrics, 99)),
            "min_latency": float(np.min(metrics)),
            "max_latency": float(np.max(metrics)),
            "std_latency": float(np.std(metrics)),
        }

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model"""
        pass

    @abstractmethod
    async def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output from input"""
        pass

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.api_keys.get("keys", [])

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check implementation"""
        try:
            gpu_metrics = await self.get_gpu_metrics()
            perf_stats = await self.get_performance_stats()

            gpu_health = (
                gpu_metrics.get("temperature", 100) < 85  # Below 85°C
                and gpu_metrics.get("memory_percent", 100) < 95  # Below 95% memory
                and gpu_metrics.get("utilization", 100) < 98  # Below 98% utilization
            )

            return {
                "status": "healthy" if self.is_loaded and gpu_health else "degraded",
                "last_activity": self.last_activity.isoformat(),
                "gpu_metrics": gpu_metrics,
                "performance_stats": perf_stats,
                "memory_usage": {
                    "ram": psutil.Process().memory_info().rss / 1024**3,  # GB
                    "gpu": gpu_metrics.get("memory_used_gb", 0),
                },
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.optimize_memory()
            self.is_loaded = False
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

