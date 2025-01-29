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
import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
import socket
import requests


class SystemMonitor:
    def __init__(self):
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.last_activity = time.time()
        self.metrics_history = {}
        self.alert_thresholds = {
            "cpu": 90,
            "memory": 90,
            "gpu": 95,
            "gpu_memory": 90,
            "disk": 85
        }
        
        # Initialize NVIDIA monitoring for H100
        nvidia_smi.nvmlInit()
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        
        # Azure configuration
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.vm_name = os.getenv("AZURE_VM_NAME")
        self.api_endpoint = os.getenv("API_ENDPOINT")
        
        # Initialize Azure client
        if all([self.subscription_id, self.resource_group, self.vm_name]):
            self.credential = DefaultAzureCredential()
            self.compute_client = ComputeManagementClient(
                self.credential,
                self.subscription_id
            )
        else:
            self.logger.error("Azure credentials not properly configured")
            self.compute_client = None
        
        # Heartbeat and status tracking
        self.last_heartbeat = time.time()
        self.active_requests = 0
        self.status_file = Path("/mnt/data/llm-server/status/vm_status.json")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        with open("/mnt/data/llm-server/config/server_config.json", "r") as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        log_path = Path(self.config["storage"]["logs_path"]) / "monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger("SystemMonitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    async def get_gpu_metrics(self) -> Dict[str, Any]:
        try:
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = nvidia_smi.nvmlDeviceGetTemperature(self.gpu_handle, nvidia_smi.NVML_TEMPERATURE_GPU)
            power = nvidia_smi.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to Watts
            clock = nvidia_smi.nvmlDeviceGetClockInfo(self.gpu_handle, nvidia_smi.NVML_CLOCK_SM)

            return {
                "utilization": info.gpu,
                "memory_used": memory.used / 1024**3,  # Convert to GB
                "memory_total": memory.total / 1024**3,
                "memory_percent": (memory.used / memory.total) * 100,
                "temperature": temp,
                "power_usage": power,
                "sm_clock": clock
            }
        except Exception as e:
            self.logger.error(f"GPU metrics error: {e}")
            return {}

    async def get_system_metrics(self) -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/mnt/data")
            gpu_metrics = await self.get_gpu_metrics()
            net_connections = len([conn for conn in psutil.net_connections()
                                 if conn.status == 'ESTABLISHED'])

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "per_cpu": psutil.cpu_percent(percpu=True),
                    "load_avg": psutil.getloadavg()
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": memory.used / 1024**3,
                    "available_gb": memory.available / 1024**3
                },
                "disk": {
                    "percent": disk.percent,
                    "used_gb": disk.used / 1024**3,
                    "free_gb": disk.free / 1024**3
                },
                "gpu": gpu_metrics,
                "network": {
                    "connections": net_connections,
                    "active_requests": self.active_requests
                }
            }

            # Update metrics history
            self.metrics_history[time.time()] = metrics
            
            # Clean old metrics (keep last 10 minutes)
            current_time = time.time()
            self.metrics_history = {
                k: v for k, v in self.metrics_history.items()
                if current_time - k <= 600
            }

            return metrics

        except Exception as e:
            self.logger.error(f"System metrics error: {e}")
            return {}

    async def check_services_health(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self.config['server']['port']}/health",
                    timeout=2
                ) as response:
                    return response.status == 200
        except:
            return False

    async def update_status(self, status: str):
        """Update VM status file"""
        try:
            status_data = {
                "status": status,
                "last_updated": datetime.now().isoformat(),
                "vm_name": self.vm_name,
                "active_requests": self.active_requests
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f)
        except Exception as e:
            self.logger.error(f"Failed to update status: {e}")

    async def shutdown_vm(self):
        """Shutdown the VM using Azure API"""
        try:
            if self.compute_client:
                self.logger.info(f"Initiating shutdown for VM {self.vm_name}")
                
                # Update status before shutdown
                await self.update_status("shutting_down")
                
                # Deallocate VM
                self.compute_client.virtual_machines.begin_deallocate(
                    self.resource_group,
                    self.vm_name
                )
                self.logger.info("Shutdown command sent successfully")
            else:
                self.logger.error("Cannot shutdown VM: Azure client not configured")
        except Exception as e:
            self.logger.error(f"VM shutdown error: {e}")

    async def check_activity(self) -> bool:
        """Check if system should be considered active"""
        try:
            metrics = await self.get_system_metrics()
            current_time = time.time()
            
            # Consider active if:
            # 1. Recent network connections
            # 2. High GPU utilization
            # 3. Active API requests
            is_active = (
                metrics["network"]["connections"] > 0 or
                metrics["gpu"]["utilization"] > 10 or
                self.active_requests > 0 or
                current_time - self.last_activity < self.config["monitoring"]["inactivity_threshold"]
            )
            
            return is_active
            
        except Exception as e:
            self.logger.error(f"Activity check error: {e}")
            return True  # Assume active on error

    async def graceful_shutdown(self):
        """Perform graceful shutdown sequence"""
        try:
            self.logger.info("Initiating graceful shutdown")

            # Stop accepting new connections
            await self.update_status("shutting_down")

            # Wait for active requests to complete (max 5 minutes)
            shutdown_start = time.time()
            while self.active_requests > 0 and time.time() - shutdown_start < 300:
                self.logger.info(f"Waiting for {self.active_requests} active requests to complete")
                await asyncio.sleep(10)

            # Stop services
            subprocess.run(["sudo", "systemctl", "stop", "llm-server"], check=True)
            await asyncio.sleep(5)

            # Clear GPU cache
            subprocess.run(["/home/azureuser/llm-env/bin/python", "-c", "import torch; torch.cuda.empty_cache()"])

            # Sync filesystem
            subprocess.run(["sync"], check=True)

            # Shutdown VM via Azure API
            await self.shutdown_vm()

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    async def run(self):
        """Main monitoring loop"""
        self.logger.info("System monitor started")
        await self.update_status("running")
        
        last_optimization = time.time()
        check_interval = self.config["monitoring"]["check_interval"]

        while True:
            try:
                # Get current metrics
                metrics = await self.get_system_metrics()
                service_healthy = await self.check_services_health()

                # Check if system is active
                is_active = await self.check_activity()
                current_time = time.time()

                # Log significant changes
                if metrics and self.metrics_history:
                    last_metrics = self.metrics_history[min(self.metrics_history.keys())]
                    for key in ["cpu", "memory", "gpu"]:
                        if abs(metrics[key]["percent"] - last_metrics[key]["percent"]) > 10:
                            self.logger.info(
                                f"Significant {key} change: "
                                f"{last_metrics[key]['percent']:.1f}% → "
                                f"{metrics[key]['percent']:.1f}%"
                            )

                # Perform optimization if needed
                if (current_time - last_optimization > 300 and  # 5 minutes between optimizations
                    (metrics["memory"]["percent"] > 85 or
                     metrics["gpu"]["memory_percent"] > 85)):
                    self.logger.info("Running resource optimization")
                    await self.optimize_resources()
                    last_optimization = current_time

                # Check for shutdown condition
                if not is_active and service_healthy:
                    inactive_time = current_time - self.last_activity
                    self.logger.info(f"System inactive for {inactive_time} seconds")
                    
                    if inactive_time >= self.config["monitoring"]["inactivity_threshold"]:
                        await self.graceful_shutdown()
                        break

                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(check_interval)

    async def optimize_resources(self):
        """Optimize system resources"""
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


if __name__ == "__main__":
    monitor = SystemMonitor()
    asyncio.run(monitor.run())