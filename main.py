import warp as wp
import numpy as np
import os
import time

from warp_jsb.eom import AircraftState, ControlState, integrate_state_rk4_kernel
from warp_jsb.aero_generated import AeroModelHandles

wp.init()

def load_aero_handles(data_dir, device):
    handles = AeroModelHandles()
    # List of all expected tables from manifest
    tables = [
        "aero_coefficient_CDDf", "aero_coefficient_CDwbh", "aero_coefficient_CYb",
        "aero_coefficient_CYp", "aero_coefficient_CYr", "aero_coefficient_CLwbh",
        "aero_coefficient_CLDf", "aero_coefficient_Clb", "aero_coefficient_Clr",
        "aero_coefficient_Cmdf", "aero_coefficient_Cnb"
    ]
    
    for t in tables:
        data = np.load(os.path.join(data_dir, f"{t}.npy"))
        meta = np.load(os.path.join(data_dir, f"{t}_meta.npy"))
        
        # Set attributes on the struct
        setattr(handles, f"{t}_table", wp.from_numpy(data, dtype=wp.float32, device=device))
        setattr(handles, f"{t}_meta", wp.from_numpy(meta, dtype=wp.float32, device=device))
        
    return handles

def run_simulation(num_aircraft=10000, num_steps=100, dt=0.01):
    device = "cuda" if wp.is_cuda_available() else "cpu"
    data_dir = 'd:\\Nvidia-JSB\\data\\c172p'
    
    # 1. Load Aero Handles
    handles = load_aero_handles(data_dir, device)
    
    # 2. Initialize State
    states = wp.zeros(num_aircraft, dtype=AircraftState, device=device)
    controls = wp.zeros(num_aircraft, dtype=ControlState, device=device)
    
    # Metrics
    S, b, c = 16.16, 10.91, 1.49 # SI meters
    rho = 1.225
    
    print(f"Starting RK4 SIMULATION of {num_aircraft} aircraft (35+ Aero Functions) on {device}...")
    
    start = time.time()
    for _ in range(num_steps):
        # Full RK4 Integration (handles Aero internally)
        wp.launch(
            kernel=integrate_state_rk4_kernel,
            dim=num_aircraft,
            inputs=[states, controls, handles, dt, S, b, c, rho],
            device=device
        )
        
    wp.synchronize()
    end = time.time()
    
    print(f"RK4 Model Sim took {end-start:.4f} seconds.")
    print(f"Throughput: {num_aircraft * num_steps / (end-start):.0f} aircraft-steps/sec")

if __name__ == "__main__":
    run_simulation()
