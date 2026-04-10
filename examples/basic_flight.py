import warp as wp
import numpy as np
import os
import time

from warp_jsb.eom import AircraftState, ControlState, integrate_full_state_rk4_kernel
from warp_jsb.ground_reactions import ContactPoint
from warp_jsb.aero_generated import AeroModelHandles
from warp_jsb.logger import FleetLogger
from warp_jsb.experience import ExperienceHarvester

wp.init()

@wp.kernel
def apply_actions_kernel(
    actions: wp.array(dtype=wp.float32, ndim=2),
    controls: wp.array(dtype=ControlState)
):
    tid = wp.tid()
    
    # Map normalized actions [-1, 1] to ControlState
    ctrl = ControlState()
    ctrl.aileron  = actions[tid, 0]
    ctrl.elevator = actions[tid, 1]
    ctrl.rudder   = actions[tid, 2]
    ctrl.throttle = (actions[tid, 3] + 1.0) * 0.5
    
    # Default engine configs
    ctrl.mixture = 1.0
    ctrl.flaps   = 0.0
    ctrl.brake   = 0.0
    ctrl.steer   = 0.0
    
    controls[tid] = ctrl

def load_aero_handles(data_dir, device):
    handles = AeroModelHandles()
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
        attr_prefix = "prop_" if t in prop_tables else ""
        setattr(handles, f"{attr_prefix}{t}_table", wp.from_numpy(data, dtype=wp.float32, device=device))
        setattr(handles, f"{attr_prefix}{t}_meta", wp.from_numpy(meta, dtype=wp.float32, device=device))
    return handles

def run_simulation(num_aircraft=100000, num_steps=100, dt=0.01):
    device = "cuda" if wp.is_cuda_available() else "cpu"
    data_dir = 'd:\\Nvidia-JSB\\data\\c172p'
    
    # 1. Load Aero Handles
    handles = load_aero_handles(data_dir, device)
    
    # 2. Initialize State
    states = wp.zeros(num_aircraft, dtype=AircraftState, device=device)
    controls = wp.zeros(num_aircraft, dtype=ControlState, device=device)
    
    @wp.kernel
    def init_states_kernel(states: wp.array(dtype=AircraftState)):
        tid = wp.tid()
        s = states[tid]
        s.quat = wp.quat_identity()
        s.mass = 1880.0 / 2.20462
        s.fuel_mass = 100.0
        s.rpm = 2400.0
        states[tid] = s
    wp.launch(init_states_kernel, dim=num_aircraft, inputs=[states], device=device)
    
    # 3. Contacts
    def create_contact(pos, is_bogey, k, c, sf, df, ms):
        cp = ContactPoint()
        cp.pos_body = pos
        cp.is_bogey = is_bogey
        cp.spring_k = k
        cp.damping_c = c
        cp.static_friction = sf
        cp.dynamic_friction = df
        cp.max_steer = ms
        return cp

    h_contacts = [
        create_contact(wp.vec3(-0.17, 0.0, -0.49), True, 1800.0*14.59, 600.0*14.59, 0.8, 0.5, 0.17),
        create_contact(wp.vec3(1.48, -1.09, -0.39), True, 5400.0*14.59, 1600.0*14.59, 0.8, 0.5, 0.0),
        create_contact(wp.vec3(1.48, 1.09, -0.39), True, 5400.0*14.59, 1600.0*14.59, 0.8, 0.5, 0.0)
    ]
    contacts = wp.array(h_contacts, dtype=ContactPoint, device=device)
    
    # 4. Environment & Parameters
    S, b, c = 16.16, 10.91, 1.49 
    rho = 1.225
    p_amb = 2116.22 
    r_aero = wp.vec3(0.0, 0.0, 0.0)
    r_prop = wp.vec3(-2.0, 0.0, 0.0)
    
    # 5. Experience Harvesting (10-Step Time Series)
    window_size = 10
    print(f"\nInitializing {window_size}-step Time-Series Harvester for {num_aircraft} agents (Agent-First)...")
    harvester = ExperienceHarvester(num_aircraft, window_size=window_size, obs_dim=11, act_dim=4, device=device)
    
    print(f"Starting FULL SYSTEM SIMULATION...")
    
    start = time.time()
    for i in range(num_steps):
        # Actions
        actions_np = np.random.uniform(-1.0, 1.0, size=(num_aircraft, 4)).astype(np.float32)
        actions = wp.from_numpy(actions_np, dtype=wp.float32, device=device)
        wp.launch(apply_actions_kernel, dim=num_aircraft, inputs=[actions, controls], device=device)

        # Physics
        wp.launch(
            kernel=integrate_full_state_rk4_kernel,
            dim=num_aircraft,
            inputs=[states, controls, handles, contacts, dt, S, b, c, rho, p_amb, r_aero, r_prop],
            device=device
        )
        
        # Harvest (Snapshot into circular buffer)
        harvester.record(states, controls)
        
    wp.synchronize()
    end = time.time()
    
    # 6. Verification
    s_final = states.numpy()[0]
    print(f"\nFinal State Telemetry (Aircraft 0):")
    print(f" - Altitude: {s_final['alt_ft']:.2f} ft")
    print(f" - Airspeed: {s_final['v_kts']:.2f} kts")
    print(f" - Attitude Rad: {s_final['euler_rad']}")
    
    logger = FleetLogger(num_aircraft, device)
    stats = logger.compute(states)
    print(f"\n========================================")
    print(f" FLEET MISSION REPORT ({num_aircraft} Aircraft)")
    print(f"========================================")
    print(f" Altitude Mean: {stats['mean_alt_ft']:.2f} ft")
    print(f" Airspeed Max: {stats['max_v_kts']:.2f} kts")
    print(f"========================================\n")
    
    # 7. Dataset Export
    harvester.save_to_disk("pioneer_sequence_data")
    
    print(f"RK4 Model Sim took {end-start:.4f} seconds.")
    print(f"Throughput: {num_aircraft * num_steps / (end-start):.0f} aircraft-steps/sec")

if __name__ == "__main__":
    run_simulation()
