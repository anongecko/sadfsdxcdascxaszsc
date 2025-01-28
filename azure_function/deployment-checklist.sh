#!/bin/bash

check_directories() {
    dirs=(
        "/mnt/data/llm-server/logs"
        "/mnt/data/llm-server/status"
        "/mnt/data/llm-server/models/text/deepseek-r1"
        "/mnt/data/llm-server/models/image/flux1-dev"
        "/mnt/data/cache"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "Creating directory: $dir"
            sudo mkdir -p "$dir"
            sudo chown -R azureuser:azureuser "$dir"
        fi
    done
}

check_environment() {
    required_vars=(
        "AZURE_SUBSCRIPTION_ID"
        "AZURE_RESOURCE_GROUP"
        "AZURE_VM_NAME"
        "API_ENDPOINT"
    )
    
    missing=0
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "Missing environment variable: $var"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

check_models() {
    if [ ! -f "/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf" ]; then
        echo "DeepSeek model not found!"
        exit 1
    fi
    
    if [ ! -d "/mnt/data/llm-server/models/image/flux1-dev" ]; then
        echo "FLUX model directory not found!"
        exit 1
    fi
}

main() {
    echo "Running pre-deployment checks..."
    check_directories
    check_environment
    check_models
    echo "Pre-deployment checks completed successfully"
}

main
