import shutil
import psutil
import torch
import wandb
import time

def log_system_stats(log_file="system_stats.log"):
    with open(log_file, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Timestamp: {timestamp}\n")
        
        # Log disk usage
        total, used, free = shutil.disk_usage("/")
        f.write(f"Disk Usage - Total: {total // (1024**3)} GB, Used: {used // (1024**3)} GB, Free: {free // (1024**3)} GB\n")

        # Log memory usage
        process = psutil.Process()
        mem_info = process.memory_info()
        f.write(f"Memory Usage - RSS: {mem_info.rss // (1024**2)} MB, VMS: {mem_info.vms // (1024**2)} MB\n")

        # Log GPU usage (if available)
        if torch.cuda.is_available():
            f.write(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() // (1024**2)} MB, Reserved: {torch.cuda.memory_reserved() // (1024**2)} MB\n")

        f.write("\n")



def log_system_stats_to_wandb():
    stats = {}
    total, used, free = shutil.disk_usage("/")
    stats["disk_total_gb"] = total // (1024**3)
    stats["disk_used_gb"] = used // (1024**3)
    stats["disk_free_gb"] = free // (1024**3)

    process = psutil.Process()
    mem_info = process.memory_info()
    stats["memory_rss_mb"] = mem_info.rss // (1024**2)
    stats["memory_vms_mb"] = mem_info.vms // (1024**2)

    if torch.cuda.is_available():
        stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() // (1024**2)
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() // (1024**2)

    wandb.log(stats)