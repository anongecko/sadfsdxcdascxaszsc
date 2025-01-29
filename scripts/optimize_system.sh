#!/bin/bash

# Exit on error
set -e

# Create log directory
LOG_DIR="/mnt/data/llm-server/logs"
mkdir -p "$LOG_DIR" || { echo "Failed to create log directory"; exit 1; }
LOG_FILE="$LOG_DIR/optimize.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_requirements() {
    local required_cmds=("nvidia-smi" "bc" "awk" "free" "sysctl")
    
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "Error: Required command '$cmd' not found"
            exit 1
        fi
    done
}

optimize_cpu() {
    log "Optimizing CPU settings"
    
    # Check if CPU governors exist and set to performance
    if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -f "$cpu" ]; then
                echo "performance" | sudo tee "$cpu" || log "Failed to set CPU governor for $cpu"
            fi
        done
    else
        log "CPU frequency scaling not available"
    fi
    
    # Set CPU affinity and scheduling for optimal performance
    for pid in $(pgrep -f "llama.cpp"); do
        sudo taskset -pc 0-39 $pid || log "Failed to set CPU affinity for PID $pid"
    done
    
    # Set process scheduling for LLM server if service exists
    if systemctl list-units --full -all | grep -q "llm-server.service"; then
        sudo systemctl set-property llm-server.service CPUSchedulingPolicy=batch
        sudo systemctl set-property llm-server.service CPUSchedulingPriority=50
        sudo systemctl set-property llm-server.service CPUAffinity=0-39
    else
        log "llm-server.service not found, skipping process scheduling optimization"
    fi
}

optimize_memory() {
    log "Optimizing memory settings"
    
    # System memory optimizations
    local sysctl_params=(
        "vm.swappiness=0"
        "vm.vfs_cache_pressure=50"
        "vm.dirty_background_ratio=5"
        "vm.dirty_ratio=10"
        "vm.max_map_count=1000000"
        "vm.zone_reclaim_mode=0"
        "vm.min_free_kbytes=1048576"
        "vm.page-cluster=0"
        "vm.overcommit_memory=1"
    )
    
    for param in "${sysctl_params[@]}"; do
        sudo sysctl -w "$param" || log "Failed to set $param"
    done
    
    # Set huge pages for large model
    echo 16 | sudo tee /proc/sys/vm/nr_hugepages
    
    # Clear page cache if memory usage is high (use bc for floating point)
    local MEM_USAGE
    MEM_USAGE=$(free | awk '/^Mem/ {printf "%.2f", $3/$2 * 100}')
    if (( $(echo "$MEM_USAGE > 90" | bc -l) )); then
        log "High memory usage detected (${MEM_USAGE}%), clearing caches"
        sync
        echo 1 | sudo tee /proc/sys/vm/drop_caches || log "Failed to clear page cache"
    fi
}

optimize_gpu() {
    log "Optimizing GPU settings for H100"
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log "Error: nvidia-smi not found. GPU optimization skipped."
        return 1
    fi
    
    # Check if GPU is present
    if ! nvidia-smi &> /dev/null; then
        log "Error: No NVIDIA GPU detected"
        return 1
    fi
    
    # H100-specific optimizations
    local gpu_commands=(
        "nvidia-smi -pm 1"  # Enable persistent mode
        "nvidia-smi -pl 350"  # Set power limit to 350W for H100 PCIe
        "nvidia-smi --compute-mode=0"  # Change to DEFAULT mode for single model
        "nvidia-smi --persistence-mode=1"
        "nvidia-smi --gpu-name-as-index=0"
        "nvidia-smi --auto-boost-default=0"  # Disable auto boost for consistent performance
        "nvidia-smi -ac 1815,1410"  # Optimal memory,graphics clocks for H100
    )
    
    for cmd in "${gpu_commands[@]}"; do
        if ! sudo $cmd; then
            log "Failed to execute: $cmd"
        fi
    done
    
    # Set GPU memory growth and TF32
    export CUDA_VISIBLE_DEVICES=0
    export NVIDIA_TF32_OVERRIDE=1
    
    # Lock GPU clocks for consistent performance
    sudo nvidia-smi --lock-gpu-clocks=1410
    
    # Set fabric clock for H100
    sudo nvidia-smi --applications-clocks=1815,1410
    
    # Enable MIG mode if available
    if nvidia-smi mig -i 0 -lgip &> /dev/null; then
        sudo nvidia-smi -i 0 -mig 1 || log "Failed to enable MIG mode"
    fi
    
    # Monitor GPU settings
    log "Current GPU settings:"
    sudo nvidia-smi -q -d CLOCK,POWER,PERFORMANCE || log "Failed to query GPU settings"
}

optimize_io() {
    log "Optimizing I/O settings"
    
    # Optimize NVMe devices
    for device in /sys/block/nvme*; do
        if [ -d "$device" ]; then
            echo "none" | sudo tee "$device/queue/scheduler" 2>/dev/null || log "Failed to set scheduler for $device"
            echo "256" | sudo tee "$device/queue/read_ahead_kb" 2>/dev/null || log "Failed to set read_ahead_kb for $device"
            echo "2" | sudo tee "$device/queue/nomerges" 2>/dev/null || log "Failed to set nomerges for $device"
            echo "128" | sudo tee "$device/queue/nr_requests" 2>/dev/null || log "Failed to set nr_requests for $device"
        fi
    done
    
    # Optimize SATA SSDs
    for device in /sys/block/sd*; do
        if [ -d "$device" ]; then
            if grep -q "0" "$device/queue/rotational" 2>/dev/null; then
                echo "none" | sudo tee "$device/queue/scheduler" 2>/dev/null || log "Failed to set scheduler for $device"
                echo "256" | sudo tee "$device/queue/read_ahead_kb" 2>/dev/null || log "Failed to set read_ahead_kb for $device"
            fi
        fi
    done
    
    # Set I/O priority for LLM server
    for pid in $(pgrep -f "llama.cpp"); do
        sudo ionice -c 2 -n 0 -p $pid || log "Failed to set I/O priority for PID $pid"
    done
}

optimize_network() {
    log "Optimizing network settings"
    
    local network_params=(
        "net.core.somaxconn=65535"
        "net.core.netdev_max_backlog=65535"
        "net.ipv4.tcp_max_syn_backlog=65535"
        "net.ipv4.tcp_fastopen=3"
        "net.ipv4.tcp_tw_reuse=1"
        "net.core.rmem_max=16777216"
        "net.core.wmem_max=16777216"
        "net.ipv4.tcp_rmem=4096 87380 16777216"
        "net.ipv4.tcp_wmem=4096 87380 16777216"
        "net.ipv4.tcp_slow_start_after_idle=0"
        "net.ipv4.tcp_mtu_probing=1"
        "net.core.optmem_max=16777216"
        "net.ipv4.tcp_congestion_control=bbr"
    )
    
    for param in "${network_params[@]}"; do
        sudo sysctl -w "$param" || log "Failed to set $param"
    done
    
    # Enable BBR if available
    if [ -f "/proc/sys/net/ipv4/tcp_congestion_control" ]; then
        if grep -q "bbr" /proc/sys/net/ipv4/tcp_available_congestion_control; then
            echo "bbr" | sudo tee /proc/sys/net/ipv4/tcp_congestion_control
        fi
    fi
}

setup_numa() {
    log "Setting up NUMA configuration"
    
    # Check if numactl is available
    if ! command -v numactl &> /dev/null; then
        log "numactl not found, skipping NUMA optimization"
        return
    fi
    
    # Disable automatic NUMA balancing
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing
    
    # Set NUMA policy for LLM server
    for pid in $(pgrep -f "llama.cpp"); do
        sudo numactl --preferred=0 --pid $pid || log "Failed to set NUMA policy for PID $pid"
    done
}

main() {
    log "Starting system optimization for Deepseek R1 on H100"
    
    # Check requirements first
    check_requirements
    
    # Run optimizations
    optimize_cpu
    optimize_memory
    optimize_gpu
    optimize_io
    optimize_network
    setup_numa
    
    log "System optimization completed"
    
    # Final GPU status check
    nvidia-smi || log "Failed to get final GPU status"
}

main