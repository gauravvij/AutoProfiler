#!/usr/bin/env python3
"""
Hardware Profiler for AutoProfiler Framework
Detects CPU, RAM, GPU capabilities and outputs system constraints as JSON.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_cpu_info():
    """Get CPU information."""
    cpu_info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": os.cpu_count(),
        "logical_cores": os.cpu_count(),
    }
    
    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                if "model name" in content:
                    for line in content.split("\n"):
                        if "model name" in line and ":" in line:
                            cpu_info["model"] = line.split(":")[1].strip()
                            break
                        if "cpu cores" in line and ":" in line:
                            cpu_info["physical_cores"] = int(line.split(":")[1].strip())
        elif platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()
            result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["physical_cores"] = int(result.stdout.strip())
    except Exception as e:
        cpu_info["detection_error"] = str(e)
    
    return cpu_info


def get_memory_info():
    """Get RAM information."""
    mem_info = {}
    
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                content = f.read()
                for line in content.split("\n"):
                    if "MemTotal:" in line:
                        kb = int(line.split()[1])
                        mem_info["total_gb"] = round(kb / (1024 * 1024), 2)
                    if "MemAvailable:" in line:
                        kb = int(line.split()[1])
                        mem_info["available_gb"] = round(kb / (1024 * 1024), 2)
        elif platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                bytes_total = int(result.stdout.strip())
                mem_info["total_gb"] = round(bytes_total / (1024**3), 2)
        
        # Fallback using psutil if available
        if "total_gb" not in mem_info:
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_info["total_gb"] = round(mem.total / (1024**3), 2)
                mem_info["available_gb"] = round(mem.available / (1024**3), 2)
            except ImportError:
                pass
    except Exception as e:
        mem_info["detection_error"] = str(e)
    
    return mem_info


def get_gpu_info():
    """Get GPU information including VRAM."""
    gpu_info = {
        "available": False,
        "devices": []
    }
    
    # Try NVIDIA GPUs
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", 
                                "--format=csv,noheader,nounits"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_info["devices"].append({
                        "type": "cuda",
                        "name": parts[0],
                        "total_vram_gb": round(float(parts[1]) / 1024, 2),
                        "free_vram_gb": round(float(parts[2]) / 1024, 2)
                    })
                    gpu_info["available"] = True
    except FileNotFoundError:
        pass
    except Exception as e:
        gpu_info["nvidia_error"] = str(e)
    
    # Try AMD GPUs (rocm-smi)
    try:
        result = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info["rocm_detected"] = True
    except FileNotFoundError:
        pass
    except Exception as e:
        gpu_info["rocm_error"] = str(e)
    
    # Check for Apple Silicon
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["system_profiler", "SPDisplaysDataType"], 
                                   capture_output=True, text=True)
            if "Apple" in result.stdout:
                gpu_info["devices"].append({
                    "type": "metal",
                    "name": "Apple Silicon",
                    "total_vram_gb": "shared_with_ram"
                })
                gpu_info["available"] = True
        except Exception as e:
            gpu_info["metal_error"] = str(e)
    
    return gpu_info


def get_disk_info():
    """Get available disk space."""
    disk_info = {}
    try:
        import shutil
        stat = shutil.disk_usage(".")
        disk_info["total_gb"] = round(stat.total / (1024**3), 2)
        disk_info["free_gb"] = round(stat.free / (1024**3), 2)
        disk_info["used_gb"] = round(stat.used / (1024**3), 2)
    except Exception as e:
        disk_info["error"] = str(e)
    return disk_info


def generate_constraints(hardware):
    """Generate system constraints based on detected hardware."""
    constraints = {
        "max_context_length": 4096,
        "recommended_batch_size": 1,
        "max_quantization_bits": 8,
        "offload_strategy": "auto"
    }
    
    total_ram = hardware.get("memory", {}).get("total_gb", 8)
    gpu_available = hardware.get("gpu", {}).get("available", False)
    
    if gpu_available:
        for device in hardware.get("gpu", {}).get("devices", []):
            vram = device.get("total_vram_gb", 0)
            if isinstance(vram, (int, float)) and vram > 0:
                if vram >= 24:
                    constraints["max_context_length"] = 32768
                    constraints["max_quantization_bits"] = 16
                    constraints["recommended_batch_size"] = 4
                elif vram >= 16:
                    constraints["max_context_length"] = 16384
                    constraints["max_quantization_bits"] = 8
                    constraints["recommended_batch_size"] = 2
                elif vram >= 8:
                    constraints["max_context_length"] = 8192
                    constraints["max_quantization_bits"] = 4
                    constraints["recommended_batch_size"] = 1
                else:
                    constraints["max_context_length"] = 4096
                    constraints["max_quantization_bits"] = 4
                    constraints["offload_strategy"] = "hybrid"
    else:
        if total_ram >= 32:
            constraints["max_context_length"] = 8192
            constraints["max_quantization_bits"] = 4
        elif total_ram >= 16:
            constraints["max_context_length"] = 4096
            constraints["max_quantization_bits"] = 4
        else:
            constraints["max_context_length"] = 2048
            constraints["max_quantization_bits"] = 4
            constraints["offload_strategy"] = "disk"
    
    return constraints


def main():
    """Main entry point."""
    print("=" * 60)
    print("AutoProfiler Hardware Profiler")
    print("=" * 60)
    
    hardware = {
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "disk": get_disk_info(),
        "constraints": {}
    }
    
    hardware["constraints"] = generate_constraints(hardware)
    
    output_path = Path("hardware_profile.json")
    with open(output_path, "w") as f:
        json.dump(hardware, f, indent=2)
    
    # Print summary
    print("\n--- CPU ---")
    print(f"  Model: {hardware['cpu'].get('model', 'Unknown')}")
    print(f"  Cores: {hardware['cpu'].get('physical_cores', 'Unknown')}")
    
    print("\n--- Memory ---")
    print(f"  Total RAM: {hardware['memory'].get('total_gb', 'Unknown')} GB")
    print(f"  Available RAM: {hardware['memory'].get('available_gb', 'Unknown')} GB")
    
    print("\n--- GPU ---")
    if hardware['gpu']['available']:
        for i, device in enumerate(hardware['gpu']['devices']):
            print(f"  Device {i}: {device['name']}")
            print(f"    Type: {device['type']}")
            print(f"    VRAM: {device.get('total_vram_gb', 'Unknown')} GB")
    else:
        print("  No GPU detected (CPU-only mode)")
    
    print("\n--- Disk ---")
    print(f"  Free Space: {hardware['disk'].get('free_gb', 'Unknown')} GB")
    
    print("\n--- Constraints ---")
    for key, value in hardware['constraints'].items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Full profile written to: {output_path.absolute()}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
