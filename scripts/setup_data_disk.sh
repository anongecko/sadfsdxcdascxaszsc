#!/bin/bash

# Log file
LOG_FILE="/home/azureuser/disk_setup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

setup_data_disk() {
    log "Starting data disk setup"

    # Check if disk is already mounted
    if mountpoint -q /mnt/data; then
        log "Data disk is already mounted at /mnt/data"
        return 0
    }

    # Create mount point
    sudo mkdir -p /mnt/data
    
    # Find the data disk (usually sdc on Azure)
    DATA_DISK=$(lsblk -dpno NAME,SIZE | grep '512G' | cut -d' ' -f1)
    if [ -z "$DATA_DISK" ]; then
        log "ERROR: Could not find 512GB data disk"
        return 1
    }
    
    log "Found data disk: $DATA_DISK"

    # Create partition
    sudo parted $DATA_DISK mklabel gpt
    sudo parted $DATA_DISK mkpart primary ext4 0% 100%
    
    # Format partition
    PARTITION="${DATA_DISK}1"
    sudo mkfs.ext4 $PARTITION
    
    # Add to fstab
    PARTITION_UUID=$(sudo blkid -s UUID -o value $PARTITION)
    echo "UUID=$PARTITION_UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    
    # Mount disk
    sudo mount /mnt/data
    
    # Set permissions
    sudo chown -R azureuser:azureuser /mnt/data
    sudo chmod -R 755 /mnt/data
    
    log "Data disk setup completed successfully"
    return 0
}

optimize_disk() {
    log "Optimizing disk performance"
    
    # Set IO scheduler for SSD
    echo "none" | sudo tee /sys/block/sdc/queue/scheduler
    
    # Increase read-ahead buffer
    echo "256" | sudo tee /sys/block/sdc/queue/read_ahead_kb
    
    # Optimize mount options
    sudo mount -o remount,noatime,nodiratime /mnt/data
    
    log "Disk optimization completed"
}

main() {
    if setup_data_disk; then
        optimize_disk
        log "All disk setup steps completed successfully"
    else
        log "Disk setup failed"
        exit 1
    fi
}

main
