import azure.functions as func
import logging
import json
import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
import aiohttp
import asyncio
import time

async def check_vm_status(compute_client, resource_group, vm_name):
    """Check VM power state"""
    vm = compute_client.virtual_machines.get(resource_group, vm_name, expand='instanceView')
    status = next((status.code.replace('PowerState/', '')
                  for status in vm.instance_view.statuses
                  if status.code.startswith('PowerState')), None)
    return status

async def wait_for_vm_running(compute_client, resource_group, vm_name, timeout=300):
    """Wait for VM to be running"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = await check_vm_status(compute_client, resource_group, vm_name)
        if status == 'running':
            return True
        await asyncio.sleep(10)
    return False

async def check_api_health(api_endpoint):
    """Check if API is healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_endpoint}/health", timeout=5) as response:
                return response.status == 200
    except:
        return False

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('API Gateway triggered')
    
    # Azure configuration
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    vm_name = os.environ["AZURE_VM_NAME"]
    api_endpoint = os.environ["API_ENDPOINT"]
    
    try:
        # Initialize Azure client
        credential = DefaultAzureCredential()
        compute_client = ComputeManagementClient(credential, subscription_id)
        
        # Check VM status
        status = await check_vm_status(compute_client, resource_group, vm_name)
        
        if status == 'deallocated':
            logging.info(f"Starting VM {vm_name}")
            # Start VM
            compute_client.virtual_machines.begin_start(resource_group, vm_name)
            
            # Wait for VM to be running
            if not await wait_for_vm_running(compute_client, resource_group, vm_name):
                return func.HttpResponse(
                    json.dumps({"error": "VM startup timeout"}),
                    mimetype="application/json",
                    status_code=503
                )
            
            # Wait for API to be healthy
            start_time = time.time()
            while time.time() - start_time < 300:  # 5 minute timeout
                if await check_api_health(api_endpoint):
                    break
                await asyncio.sleep(5)
            
            # Forward the request to the API
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=req.method,
                    url=f"{api_endpoint}/{req.route_params['path']}",
                    headers=dict(req.headers),
                    data=req.get_body()
                ) as response:
                    return func.HttpResponse(
                        body=await response.text(),
                        status_code=response.status,
                        headers=dict(response.headers)
                    )
        
        elif status == 'running':
            # VM is already running, forward request directly
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=req.method,
                    url=f"{api_endpoint}/{req.route_params['path']}",
                    headers=dict(req.headers),
                    data=req.get_body()
                ) as response:
                    return func.HttpResponse(
                        body=await response.text(),
                        status_code=response.status,
                        headers=dict(response.headers)
                    )
        
        else:
            return func.HttpResponse(
                json.dumps({"error": f"VM in invalid state: {status}"}),
                mimetype="application/json",
                status_code=503
            )
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )