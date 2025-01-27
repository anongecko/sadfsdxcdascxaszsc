#!/bin/bash

# Log file
LOG_FILE="/mnt/data/llm-server/logs/optimize.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

optimize_cpu() {
    log "Optimizing CPU settings"
    
    # Set CPU governor to performance
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" | sudo tee "$cpu"
    done
    
    # Disable CPU load balancing for model processes
    sudo systemctl set-property llm-server.service CPUAffinity=0-39
    
    # Set process scheduling priority
    sudo systemctl set-property llm-server.service CPUSchedulingPolicy=batch
    sudo systemctl set-property llm-server.service CPUSchedulingPriority=50
}

optimize_memory() {
    log "Optimizing memory settings"
    
    # System memory optimizations
    sudo sysctl -w vm.swappiness=0
    sudo sysctl -w vm.vfs_cache_pressure=50
    sudo sysctl -w vm.dirty_background_ratio=5
    sudo sysctl -w vm.dirty_ratio=10
    sudo sysctl -w vm.max_map_count=1000000
    sudo sysctl -w vm.zone_reclaim_mode=0
    sudo sysctl -w vm.min_free_kbytes=1048576
    
    # Clear page cache if memory usage is high
    if [ "$(free | awk '/^Mem/ {print $3/$2 * 100.0}')" -gt 90 ]; then
        log "High memory usage detected, clearing caches"
        sync
        echo 1 | sudo tee /proc/sys/vm/drop_caches
    fi
}

optimize_gpu() {
    log "Optimizing GPU settings for H100"
    
    # Set GPU persistence mode
    sudo nvidia-smi -pm 1
    
    # Set power limit (700W for H100)
    sudo nvidia-smi -pl 700
    
    # Enable compute mode
    sudo nvidia-smi -c EXCLUSIVE_PROCESS
    
    # Optimize clocks
    sudo nvidia-smi --auto-boost-default=0
    sudo nvidia-smi -ac 1815,1410
    
    # Set GPU memory settings
    sudo nvidia-smi --gpu-name=0 --application-clocks=1815,1410
}

optimize_io() {
    log "Optimizing I/O settings"
    
    # Set I/O scheduler for SSD
    echo "none" | sudo tee /sys/block/sdc/queue/scheduler
    
    # Increase read-ahead buffer
    echo "256" | sudo tee /sys/block/sdc/queue/read_ahead_kb
    
    # Set I/O priority for services
    sudo systemctl set-property llm-server.service IOSchedulingClass=best-effort
    sudo systemctl set-property llm-server.service IOSchedulingPriority=0
}

optimize_network() {
    log "Optimizing network settings"
    
    # Network optimizations
    sudo sysctl -w net.core.somaxconn=65535
    sudo sysctl -w net.core.netdev_max_backlog=65535
    sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535
    sudo sysctl -w net.ipv4.tcp_fastopen=3
    sudo sysctl -w net.ipv4.tcp_tw_reuse=1
}

main() {
    log "Starting system optimization"
    
    optimize_cpu
    optimize_memory
    optimize_gpu
    optimize_io
    optimize_network
    
    log "System optimization completed"
}

main
