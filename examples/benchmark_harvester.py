import warp as wp
import numpy as np
import time
import os
from warp_jsb.experience import ExperienceHarvester
from warp_jsb.eom import AircraftState, ControlState

wp.init()

def benchmark_scenario(num_aircraft, window_size, layout, sync_mode):
    device = "cuda"
    print(f"\nAUDIT: Layout={layout:15} | Sync={str(sync_mode):5} | Agents={num_aircraft}")
    
    # 1. Setup
    harvester = ExperienceHarvester(num_aircraft, window_size, layout=layout, sync_mode=sync_mode, device=device)
    states = wp.zeros(num_aircraft, dtype=AircraftState, device=device)
    controls = wp.zeros(num_aircraft, dtype=ControlState, device=device)
    
    # 2. Performance Test (100 Records)
    start = time.time()
    for _ in range(100):
        harvester.record(states, controls)
    wp.synchronize()
    end = time.time()
    
    duration = end - start
    steps_per_sec = (num_aircraft * 100) / duration
    print(f" -> GPU Throughput: {steps_per_sec:,.0f} samples/sec")
    
    # 3. Integrity Test
    # We'll write unique values per step to verify order
    # (Simplified check: ensure we didn't crash)
    
    # 4. Disk IO Audit
    filename = f"audit_temp_{layout}_{sync_mode}"
    io_start = time.time()
    harvester.save_to_disk(filename)
    io_end = time.time()
    
    # 5. Cleanup temp files
    time.sleep(0.5)
    obs_file = f"{filename}_obs.npy"
    act_file = f"{filename}_acts.npy"
    file_size_gb = (os.path.getsize(obs_file) + os.path.getsize(act_file)) / 1e9
    
    print(f" -> Disk IO: {file_size_gb:.2f} GB written in {io_end-io_start:.2f}s ({file_size_gb/(io_end-io_start):.2f} GB/s)")
    
    try:
        if os.path.exists(obs_file): os.remove(obs_file)
        if os.path.exists(act_file): os.remove(act_file)
    except:
        print(" -> Note: Cleanup delayed by OS lock.")
    
    return steps_per_sec

def run_audit():
    # We'll use 1 million aircraft for the audit to get meaningful IO results
    # 1M Agents * 10 Window * 11 Features = ~440MB.
    num_aircraft = 1000000 
    window_size = 10
    
    results = {}
    
    # Test combinations
    scenarios = [
        ("agent_first", True),
        ("agent_first", False),
        ("feature_first", True),
        ("feature_first", False)
    ]
    
    for layout, sync in scenarios:
        res = benchmark_scenario(num_aircraft, window_size, layout, sync)
        results[f"{layout}_{sync}"] = res
        
    print("\n" + "="*40)
    print(" HARVESTER AUDIT SUMMARY")
    print("="*40)
    best_layout = max(results, key=results.get)
    print(f" WINNER: {best_layout} ({results[best_layout]:,.0f} samples/sec)")
    print("="*40)

if __name__ == "__main__":
    run_audit()
