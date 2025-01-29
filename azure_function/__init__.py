import azure.functions as func
import logging
import json
import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
import aiohttp
import asyncio
import time
from typing import Optional


class VMManager:
    def __init__(self, subscription_id: str, resource_group: str, vm_name: str):
        self.compute_client = ComputeManagementClient(DefaultAzureCredential(), subscription_id)
        self.resource_group = resource_group
        self.vm_name = vm_name

    async def get_power_state(self) -> str:
        """Get current VM power state"""
        vm = self.compute_client.virtual_machines.get(self.resource_group, self.vm_name, expand="instanceView")
        status = next((status.code.replace("PowerState/", "") for status in vm.instance_view.statuses if status.code.startswith("PowerState")), None)
        return status

    async def start_vm(self) -> bool:
        """Start VM and wait for it to be ready"""
        try:
            # Start the VM
            self.compute_client.virtual_machines.begin_start(self.resource_group, self.vm_name)

            # Wait for VM to be running
            start_time = time.time()
            while time.time() - start_time < 300:  # 5 minute timeout
                if await self.get_power_state() == "running":
                    return True
                await asyncio.sleep(10)
            return False
        except Exception as e:
            logging.error(f"VM start error: {e}")
            return False


async def check_model_health(api_endpoint: str) -> bool:
    """Check if DeepSeek model is healthy and ready"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_endpoint}/health", timeout=5) as response:
                if response.status != 200:
                    return False
                data = await response.json()
                text_service = data.get("services", {}).get("text", {})
                return text_service.get("status") == "healthy"
    except:
        return False


async def wait_for_model_ready(api_endpoint: str, timeout: int = 300) -> bool:
    """Wait for model to be ready with timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await check_model_health(api_endpoint):
            return True
        await asyncio.sleep(5)
    return False


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("DeepSeek-R1 API Gateway triggered")

    # Configuration
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    vm_name = os.environ["AZURE_VM_NAME"]
    api_endpoint = os.environ["API_ENDPOINT"]

    vm_manager = VMManager(subscription_id, resource_group, vm_name)

    try:
        # Check VM status
        power_state = await vm_manager.get_power_state()

        if power_state == "deallocated":
            logging.info(f"Starting VM {vm_name}")

            if not await vm_manager.start_vm():
                return func.HttpResponse(json.dumps({"error": "VM failed to start"}), mimetype="application/json", status_code=503)

            # Wait for DeepSeek model to be ready
            if not await wait_for_model_ready(api_endpoint):
                return func.HttpResponse(json.dumps({"error": "Model initialization timeout"}), mimetype="application/json", status_code=503)

        # Forward the request to the API
        async with aiohttp.ClientSession() as session:
            async with session.request(method=req.method, url=f"{api_endpoint}/{req.route_params['path']}", headers=dict(req.headers), data=await req.get_body(), timeout=300) as response:
                return func.HttpResponse(body=await response.text(), status_code=response.status, headers=dict(response.headers))

    except asyncio.TimeoutError:
        return func.HttpResponse(json.dumps({"error": "Request timeout"}), mimetype="application/json", status_code=504)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=500)

