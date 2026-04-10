import warp as wp
import numpy as np
import os
import time

from warp_jsb.eom import AircraftState, ControlState, integrate_full_state_rk4_kernel, ContactPoint
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
    
    prop_tables = ["C_THRUST", "C_POWER"]
    
    for t in tables + prop_tables:
        data = np.load(os.path.join(data_dir, f"{t}.npy"))
        meta = np.load(os.path.join(data_dir, f"{t}_meta.npy"))
        
        # Determine target attribute name
        attr_prefix = "prop_" if t in prop_tables else ""
        
        # Set attributes on the struct
        setattr(handles, f"{attr_prefix}{t}_table", wp.from_numpy(data, dtype=wp.float32, device=device))
        setattr(handles, f"{attr_prefix}{t}_meta", wp.from_numpy(meta, dtype=wp.float32, device=device))
        
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
    
    # 4. Contacts (Simplified initialization for main.py)
    h_contacts = [
        ContactPoint(pos_body=wp.vec3(-0.17, 0.0, -0.49), is_bogey=True, spring_k=1800.0*14.59, damping_c=600.0*14.59, static_friction=0.8, dynamic_friction=0.5, max_steer=0.17),
        ContactPoint(pos_body=wp.vec3(1.48, -1.09, -0.39), is_bogey=True, spring_k=5400.0*14.59, damping_c=1600.0*14.59, static_friction=0.8, dynamic_friction=0.5, max_steer=0.0),
        ContactPoint(pos_body=wp.vec3(1.48, 1.09, -0.39), is_bogey=True, spring_k=5400.0*14.59, damping_c=1600.0*14.59, static_friction=0.8, dynamic_friction=0.5, max_steer=0.0)
    ]
    contacts = wp.array(h_contacts, dtype=ContactPoint, device=device)
    
    print(f"Starting FULL SYSTEM RK4 SIMULATION of {num_aircraft} aircraft on {device}...")
    
    start = time.time()
    for _ in range(num_steps):
        # Full RK4 Integration (handles Aero, Prop, FCS, Gear internally)
        wp.launch(
            kernel=integrate_full_state_rk4_kernel,
            dim=num_aircraft,
            inputs=[states, controls, handles, contacts, dt, S, b, c, rho],
            device=device
        )
        
    wp.synchronize()
    end = time.time()
    
    print(f"RK4 Model Sim took {end-start:.4f} seconds.")
    print(f"Throughput: {num_aircraft * num_steps / (end-start):.0f} aircraft-steps/sec")

if __name__ == "__main__":
    run_simulation()
