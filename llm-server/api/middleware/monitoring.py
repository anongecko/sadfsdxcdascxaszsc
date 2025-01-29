from fastapi import Request, HTTPException
import psutil
import time
import logging
import json
from typing import Dict, Any
import asyncio
from datetime import datetime
import nvidia_smi
from pathlib import Path


class MonitoringMiddleware:
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.last_check = time.time()
        self.check_interval = 5  # seconds
        self.metrics_history = {}
        self._lock = asyncio.Lock()
        self.active_requests = 0  # Track active requests

        # Initialize NVIDIA monitoring
        nvidia_smi.nvmlInit()
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        # Alert thresholds
        self.alert_thresholds = {"cpu": 90, "memory": 90, "gpu": 95, "gpu_memory": 90, "disk": 85}

    async def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get detailed GPU metrics for H100"""
        try:
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = nvidia_smi.nvmlDeviceGetTemperature(self.gpu_handle, nvidia_smi.NVML_TEMPERATURE_GPU)
            power = nvidia_smi.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to Watts

            return {
                "utilization": info.gpu,
                "memory_used": memory.used / 1024**3,
                "memory_total": memory.total / 1024**3,
                "memory_percent": (memory.used / memory.total) * 100,
                "temperature": temp,
                "power_usage": power,
            }
        except Exception as e:
            self.logger.error(f"GPU metrics error: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger("Monitoring")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("/mnt/data/llm-server/logs/monitoring.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open("/mnt/data/llm-server/config/server_config.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    async def check_services_health(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.config['server']['port']}/health", timeout=2) as response:
                    if response.status != 200:
                        return False
                    data = await response.json()
                    return data.get("services", {}).get("text", {}).get("status") == "healthy"
        except:
            return False

    async def get_system_metrics(self) -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/mnt/data")
            gpu_metrics = await self.get_gpu_metrics()
            net_connections = len([conn for conn in psutil.net_connections() if conn.status == "ESTABLISHED"])

            # Text model specific metrics
            model_metrics = {
                "gpu_memory_used": gpu_metrics.get("memory_used", 0),
                "gpu_memory_total": gpu_metrics.get("memory_total", 0),
                "requests_per_minute": self.active_requests,
                "gpu_utilization": gpu_metrics.get("utilization", 0),
            }

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {"percent": cpu_percent, "per_cpu": psutil.cpu_percent(percpu=True), "load_avg": psutil.getloadavg()},
                "memory": {"percent": memory.percent, "used_gb": memory.used / 1024**3, "available_gb": memory.available / 1024**3},
                "disk": {"percent": disk.percent, "used_gb": disk.used / 1024**3, "free_gb": disk.free / 1024**3},
                "gpu": gpu_metrics,
                "text_model": model_metrics,
                "network": {"connections": net_connections, "active_requests": self.active_requests},
            }

            # Update metrics history
            self.metrics_history[time.time()] = metrics

            return metrics

        except Exception as e:
            self.logger.error(f"System metrics error: {e}")
            return {}

    async def check_resources(self) -> bool:
        """Check if system resources are within acceptable limits"""
        try:
            metrics = await self.get_system_metrics()

            # Define thresholds
            thresholds = {"cpu_percent": 90, "memory_percent": 90, "gpu_memory_percent": 90, "disk_percent": 85, "gpu_temp": 80}

            # Check against thresholds
            if metrics.get("cpu", {}).get("percent", 0) > thresholds["cpu_percent"]:
                self.logger.warning(f"High CPU usage: {metrics['cpu']['percent']}%")
                return False

            if metrics.get("memory", {}).get("percent", 0) > thresholds["memory_percent"]:
                self.logger.warning(f"High memory usage: {metrics['memory']['percent']}%")
                return False

            if metrics.get("gpu", {}).get("temperature", 0) > thresholds["gpu_temp"]:
                self.logger.warning(f"High GPU temperature: {metrics['gpu']['temperature']}°C")
                return False

            gpu_memory_used = metrics.get("gpu", {}).get("memory_used_gb", 0)
            gpu_memory_total = gpu_memory_used + metrics.get("gpu", {}).get("memory_free_gb", 0)
            if gpu_memory_total > 0:
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                if gpu_memory_percent > thresholds["gpu_memory_percent"]:
                    self.logger.warning(f"High GPU memory usage: {gpu_memory_percent}%")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Resource check error: {e}")
            return True  # Default to allowing requests on error

    async def update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history with retention"""
        try:
            async with self._lock:
                current_time = time.time()
                self.metrics_history[current_time] = metrics

                # Keep last hour of metrics
                self.metrics_history = {k: v for k, v in self.metrics_history.items() if current_time - k <= 3600}

                # Save metrics to disk periodically
                if current_time - self.last_check > 300:  # Every 5 minutes
                    self._save_metrics_to_disk()
                    self.last_check = current_time

        except Exception as e:
            self.logger.error(f"Error updating metrics history: {e}")

    def _save_metrics_to_disk(self):
        """Save metrics history to disk"""
        try:
            metrics_file = Path("/mnt/data/llm-server/logs/metrics_history.json")
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f)
        except Exception as e:
            self.logger.error(f"Error saving metrics to disk: {e}")

    async def __call__(self, request: Request, call_next):
        """Middleware handler"""
        start_time = time.time()

        try:
            # Increment active requests counter
            self.active_requests += 1

            # Skip resource check for health endpoint
            if request.url.path != "/health":
                # Check resources periodically
                if time.time() - self.last_check >= self.check_interval:
                    metrics = await self.get_system_metrics()
                    if any(metrics.get(key, {}).get("percent", 0) > self.alert_thresholds[key] for key in ["cpu", "memory", "gpu"]):
                        raise HTTPException(status_code=503, detail="System under heavy load")
                    self.last_check = time.time()

            # Process request
            response = await call_next(request)

            # Update metrics
            request_time = time.time() - start_time
            metrics = await self.get_system_metrics()
            metrics["request"] = {"path": request.url.path, "method": request.method, "processing_time": request_time, "status_code": response.status_code}

            # Add monitoring headers
            response.headers["X-Processing-Time"] = str(request_time)
            response.headers["X-Resource-Usage"] = json.dumps(
                {"cpu": metrics.get("cpu", {}).get("percent", 0), "memory": metrics.get("memory", {}).get("percent", 0), "gpu": metrics.get("gpu", {}).get("utilization", 0)}
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise HTTPException(status_code=500, detail="Internal monitoring error")
        finally:
            # Decrement active requests counter
            self.active_requests -= 1
            # Log request metrics
            request_time = time.time() - start_time
            self.logger.info(f"Request: {request.method} {request.url.path} [{request_time:.3f}s]")

