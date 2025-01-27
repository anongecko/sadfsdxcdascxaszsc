#!/bin/bash

LOG_FILE="/mnt/data/llm-server/logs/startup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_models() {
    log "Checking model files"
    
    # Check DeepSeek model
    if [ ! -f "/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf" ]; then
        log "ERROR: DeepSeek model not found"
        return 1
    fi
    
    # Check FLUX model files
    if [ ! -d "/mnt/data/llm-server/models/image/flux1-dev" ]; then
        log "ERROR: FLUX model directory not found"
        return 1
    fi
    
    return 0
}

optimize_llama_cpp() {
    log "Optimizing llama.cpp settings"
    
    # Set environment variables for llama.cpp
    export OMP_NUM_THREADS=40
    export BLAS_NUM_THREADS=40
    export MKL_NUM_THREADS=40
    export OPENBLAS_NUM_THREADS=40
    export VECLIB_MAXIMUM_THREADS=40
    export NUMEXPR_NUM_THREADS=40
}

preload_models() {
    log "Pre-loading models into memory"
    
    # Pre-load DeepSeek model
    log "Pre-loading DeepSeek model"
    dd if=/mnt/data/llm-server/models/text/deepseek-r1/DeepSeek-R1-Q4_K_M-merged.gguf of=/dev/null bs=1M &
    
    # Pre-load FLUX model components
    log "Pre-loading FLUX model"
    find /mnt/data/llm-server/models/image/flux1-dev -type f -name "*.safetensors" -exec dd if={} of=/dev/null bs=1M \; &
    
    wait
}

setup_python_env() {
    log "Setting up Python environment"
    
    # Activate virtual environment
    source /home/azureuser/llm-env/bin/activate
    
    # Set Python environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TORCH_CUDA_ARCH_LIST=9.0  # For H100
    export CUDA_LAUNCH_BLOCKING=1
    export CUDA_VISIBLE_DEVICES=0
}

main() {
    log "Starting initialization sequence"
    
    # Check models
    if ! check_models; then
        log "Model check failed"
        exit 1
    fi
    
    # Run optimizations
    optimize_llama_cpp
    setup_python_env
    
    # Pre-load models
    preload_models
    
    log "Initialization sequence completed"
}

main
