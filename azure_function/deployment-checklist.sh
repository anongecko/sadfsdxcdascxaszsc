#!/bin/bash

check_directories() {
    dirs=(
        "/mnt/data/llm-server/logs"
        "/mnt/data/llm-server/status"
        "/mnt/data/llm-server/models/text/deepseek-r1"
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

check_model() {
    MODEL_PATH="/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "DeepSeek model not found at: $MODEL_PATH"
        exit 1
    fi
    
    # Verify model file size
    EXPECTED_SIZE=17500000000  # Approximate size in bytes
    ACTUAL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH")
    
    if [ "$ACTUAL_SIZE" -lt "$EXPECTED_SIZE" ]; then
        echo "Warning: Model file size ($ACTUAL_SIZE bytes) is smaller than expected ($EXPECTED_SIZE bytes)"
        exit 1
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. GPU support is required."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        echo "ERROR: No NVIDIA GPU detected or driver issues."
        exit 1
    fi
}

main() {
    echo "Running pre-deployment checks..."
    check_directories
    check_environment
    check_model
    check_gpu
    echo "Pre-deployment checks completed successfully"
}

main