#!/bin/bash

LOG_FILE="/mnt/data/llm-server/logs/startup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

setup_azure_credentials() {
    log "Setting up Azure credentials"
    
    # Create Azure credentials directory
    AZURE_DIR="/home/azureuser/.azure"
    mkdir -p "$AZURE_DIR"
    
    # Set environment variables for Azure
    cat > /etc/environment <<EOF
AZURE_SUBSCRIPTION_ID="343fe172-46d2-417a-afef-f9c72fe53f3c"
AZURE_RESOURCE_GROUP="testing-m1-for-deepseek"
AZURE_VM_NAME="DeepseekR1"
API_ENDPOINT="https://api.hotshitai.com"
EOF

    source /etc/environment
}

check_model() {
    log "Checking Deepseek R1 model file"
    
    MODEL_PATH="/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf"
    if [ ! -f "$MODEL_PATH" ]; then
        log "ERROR: DeepSeek model not found at $MODEL_PATH"
        return 1
    fi
    
    # Check file size
    EXPECTED_SIZE=17500000000  # Approximate size in bytes
    ACTUAL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH")
    
    if [ "$ACTUAL_SIZE" -lt "$EXPECTED_SIZE" ]; then
        log "ERROR: Model file size ($ACTUAL_SIZE bytes) is smaller than expected ($EXPECTED_SIZE bytes)"
        return 1
    fi
    
    return 0
}

optimize_system() {
    log "Optimizing system for Deepseek R1"
    
    # Set environment variables for llama.cpp
    export OMP_NUM_THREADS=40
    export MKL_NUM_THREADS=40
    export NUMEXPR_NUM_THREADS=40
    
    # H100-specific CUDA settings
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TORCH_CUDA_ARCH_LIST=9.0
    export CUDA_LAUNCH_BLOCKING=1
    export OMP_NUM_THREADS=40
    export CUDA_VISIBLE_DEVICES=0
    export MKL_NUM_THREADS=40
    export NUMEXPR_MAX_THREADS=40
    export NVIDIA_TF32_OVERRIDE=1
    export CUDA_MODULE_LOADING=LAZY
}

preload_model() {
    log "Pre-loading Deepseek R1 model"
    
    MODEL_PATH="/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf"
    
    # Preload model into page cache
    log "Loading model into page cache"
    dd if=$MODEL_PATH of=/dev/null bs=1M &
    
    wait
    log "Model pre-loading completed"
}

setup_python_env() {
    log "Setting up Python environment"
    source /home/azureuser/llm-env/bin/activate
}

main() {
    log "Starting Deepseek R1 initialization"
    
    # Check model
    if ! check_model; then
        log "Model check failed"
        exit 1
    fi
    
    # Setup components
    setup_azure_credentials
    optimize_system
    setup_python_env
    preload_model
    
    log "Initialization completed successfully"
}

main