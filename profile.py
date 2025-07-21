"""
Code to profile cpu/gpu usage for functions.
"""
import gc
import time
import psutil
import torch
import subprocess
import os
import threading

def get_gpu_utilization():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return int(output.strip().split('\n')[0])
    except Exception:
        return -1  # Unable to fetch

def profile_function(func, n=1):
    """
    Profiles function `func` on GPU/CPU usage. Runs `n` times.
    Returns last result and mean peak stats over all runs.
    """
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass

    process = psutil.Process(os.getpid())

    cpu_peak_list = []
    mem_peak_list = []
    gpu_mem_list = []
    gpu_reserved_list = []
    gpu_util_list = []
    total_time = 0

    stop_flag = False
    local_cpu_peak = 0
    local_mem_peak = 0

    def track_cpu():
        nonlocal local_cpu_peak, local_mem_peak, stop_flag
        while not stop_flag:
            cpu = process.cpu_percent(interval=0.1)
            mem = process.memory_info().rss / (1024 * 1024)  # MB
            local_cpu_peak = max(local_cpu_peak, cpu)
            local_mem_peak = max(local_mem_peak, mem)
            time.sleep(0.05)

    last_result = None

    for i in range(n):
        local_cpu_peak = 0
        local_mem_peak = 0
        stop_flag = False

        cpu_thread = threading.Thread(target=track_cpu)
        cpu_thread.start()

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        result = func()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        last_result = result

        stop_flag = True
        cpu_thread.join()

        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        gpu_util = get_gpu_utilization()

        cpu_peak_list.append(local_cpu_peak)
        mem_peak_list.append(local_mem_peak)
        gpu_mem_list.append(gpu_mem)
        gpu_reserved_list.append(gpu_reserved)
        gpu_util_list.append(gpu_util)

    def mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0
    stats = {
        'avg_execution_time_sec\t': round(total_time / n,6),
        'cpu_peak_percent\t': mean(cpu_peak_list),
        'cpu_peak_memory_mb\t': mean(mem_peak_list),
        'gpu_util_percent\t': mean(gpu_util_list),
        'gpu_peak_memory_allocated_mb': mean(gpu_mem_list),
        'gpu_peak_memory_reserved_mb': mean(gpu_reserved_list),
        'runs\t\t\t': n
    }

    print("Mean performance stats along runs")
    for p in stats:
        print(p, '\t', stats[p])

    return last_result, stats