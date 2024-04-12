import time
import psutil


def get_gpu_memory_usage():
    process = psutil.Process()
    return process.memory_info().vms / (1024 ** 3)  # Convert to GB

def measure_start():
    start_vram = get_gpu_memory_usage()
    start_time = time.time()
    return start_vram, start_time

def measure_end(start_vram, start_time):
    # Measure VRAM usage after running the pipeline
    end_vram = get_gpu_memory_usage()

    # Measure time taken
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.2f} seconds", flush=True)
    print(f"VRAM usage: {end_vram - start_vram:.2f} GB", flush=True)