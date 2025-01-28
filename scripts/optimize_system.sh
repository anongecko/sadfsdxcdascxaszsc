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
    )
    
    for param in "${sysctl_params[@]}"; do
        sudo sysctl -w "$param" || log "Failed to set $param"
    done
    
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
    }
    
    # Set GPU optimizations with error checking
    local gpu_commands=(
        "nvidia-smi -pm 1"
        "nvidia-smi -pl 350"  # 350W for H100 PCIe
        "nvidia-smi -c EXCLUSIVE_PROCESS"
        "nvidia-smi --persistence-mode=1"
        "nvidia-smi --compute-mode=EXCLUSIVE_PROCESS"
    )
    
    for cmd in "${gpu_commands[@]}"; do
        if ! sudo $cmd; then
            log "Failed to execute: $cmd"
        fi
    done
    
    # Monitor GPU settings
    log "Current GPU settings:"
    sudo nvidia-smi -q -d CLOCK || log "Failed to query GPU clock settings"
    sudo nvidia-smi -q -d POWER || log "Failed to query GPU power settings"
}

optimize_io() {
    log "Optimizing I/O settings"
    
    # Optimize NVMe devices
    for device in /sys/block/nvme*; do
        if [ -d "$device" ]; then
            echo "none" | sudo tee "$device/queue/scheduler" 2>/dev/null || log "Failed to set scheduler for $device"
            echo "256" | sudo tee "$device/queue/read_ahead_kb" 2>/dev/null || log "Failed to set read_ahead_kb for $device"
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
    )
    
    for param in "${network_params[@]}"; do
        sudo sysctl -w "$param" || log "Failed to set $param"
    done
}

main() {
    log "Starting system optimization"
    
    # Check requirements first
    check_requirements
    
    # Run optimizations
    optimize_cpu
    optimize_memory
    optimize_gpu
    optimize_io
    optimize_network
    
    log "System optimization completed"
}

main