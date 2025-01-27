#!/usr/bin/env python3
import asyncio
import psutil
import logging
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import nvidia_smi
import aiohttp
from typing import Dict, Any, Optional


class SystemMonitor:
    def __init__(self):
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.last_activity = time.time()
        self.metrics_history = {}
        self.alert_thresholds = {"cpu": 90, "memory": 90, "gpu": 95, "gpu_memory": 90, "disk": 85}
        nvidia_smi.nvmlInit()
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def _load_config(self) -> dict:
        with open("/mnt/data/llm-server/config/server_config.json", "r") as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        log_path = Path(self.config["storage"]["logs_path"]) / "monitor.log"
        logging.basicConfig(filename=str(log_path), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger("SystemMonitor")

    async def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get detailed GPU metrics"""
        try:
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = nvidia_smi.nvmlDeviceGetTemperature(self.gpu_handle, nvidia_smi.NVML_TEMPERATURE_GPU)
            power = nvidia_smi.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0

            return {
                "utilization": info.gpu,
                "memory_used": memory.used / 1024**3,  # Convert to GB
                "memory_total": memory.total / 1024**3,
                "memory_percent": (memory.used / memory.total) * 100,
                "temperature": temp,
                "power_usage": power,
            }
        except Exception as e:
            self.logger.error(f"GPU metrics error: {e}")
            return {}

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/mnt/data")
            gpu_metrics = await self.get_gpu_metrics()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {"percent": cpu_percent, "per_cpu": psutil.cpu_percent(percpu=True), "load_avg": psutil.getloadavg()},
                "memory": {"percent": memory.percent, "used_gb": memory.used / 1024**3, "available_gb": memory.available / 1024**3},
                "disk": {"percent": disk.percent, "used_gb": disk.used / 1024**3, "free_gb": disk.free / 1024**3},
                "gpu": gpu_metrics,
                "network": {"connections": len(psutil.net_connections())},
            }

            # Update metrics history (keep last 10 minutes)
            current_time = time.time()
            self.metrics_history[current_time] = metrics
            self.metrics_history = {k: v for k, v in self.metrics_history.items() if current_time - k <= 600}

            return metrics

        except Exception as e:
            self.logger.error(f"System metrics error: {e}")
            return {}

    async def check_services_health(self) -> bool:
        """Check if services are responding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.config['server']['port']}/health", timeout=2) as response:
                    return response.status == 200
        except:
            return False

    def should_optimize_resources(self, metrics: Dict[str, Any]) -> bool:
        """Determine if resource optimization is needed"""
        if not metrics:
            return False

        return metrics["memory"]["percent"] > self.alert_thresholds["memory"] or metrics["gpu"]["memory_percent"] > self.alert_thresholds["gpu_memory"]

    async def optimize_resources(self):
        """Perform resource optimization"""
        try:
            # Clear GPU cache
            subprocess.run(["/home/azureuser/llm-env/bin/python", "-c", "import torch; torch.cuda.empty_cache()"])

            # Sync filesystem
            subprocess.run(["sync"])

            # Clear page cache if memory pressure is high
            if psutil.virtual_memory().percent > 95:
                subprocess.run(["sudo", "sysctl", "vm.drop_caches=1"])

        except Exception as e:
            self.logger.error(f"Resource optimization error: {e}")

    async def graceful_shutdown(self):
        """Perform graceful shutdown sequence"""
        try:
            self.logger.info("Initiating graceful shutdown")

            # Stop services
            subprocess.run(["sudo", "systemctl", "stop", "llm-server"], check=True)
            await asyncio.sleep(5)

            # Clear GPU cache
            subprocess.run(["/home/azureuser/llm-env/bin/python", "-c", "import torch; torch.cuda.empty_cache()"])

            # Sync and shutdown
            subprocess.run(["sync"], check=True)
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    async def run(self):
        """Main monitoring loop"""
        self.logger.info("System monitor started")
        last_optimization = time.time()

        while True:
            try:
                # Get current metrics
                metrics = await self.get_system_metrics()
                service_healthy = await check_services_health()

                # Log significant changes
                if metrics:
                    significant_changes = []
                    last_metrics = self.metrics_history.get(sorted(self.metrics_history.keys())[-2]) if len(self.metrics_history) > 1 else None

                    if last_metrics:
                        for key in ["cpu", "memory", "gpu"]:
                            if abs(metrics[key]["percent"] - last_metrics[key]["percent"]) > 10:
                                significant_changes.append(f"{key}: {last_metrics[key]['percent']:.1f}% → {metrics[key]['percent']:.1f}%")

                    if significant_changes:
                        self.logger.info(f"Significant changes detected: {', '.join(significant_changes)}")

                # Check if optimization is needed
                current_time = time.time()
                if self.should_optimize_resources(metrics) and current_time - last_optimization > 300:  # 5 minutes between optimizations
                    await self.optimize_resources()
                    last_optimization = current_time

                # Check for shutdown condition
                if not service_healthy:
                    time_inactive = current_time - self.last_activity
                    if time_inactive >= self.config["monitoring"]["inactivity_threshold"]:
                        self.logger.info(f"System inactive for {time_inactive} seconds")
                        await self.graceful_shutdown()
                else:
                    self.last_activity = current_time

                await asyncio.sleep(self.config["monitoring"]["check_interval"])

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(self.config["monitoring"]["check_interval"])


if __name__ == "__main__":
    monitor = SystemMonitor()
    asyncio.run(monitor.run())
