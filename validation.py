import jsbsim
import warp as wp
import numpy as np
import os
import time

from warp_jsb.eom import AircraftState, ControlState, integrate_state_rk4_kernel
from warp_jsb.aero_generated import AeroModelHandles

# JSBSim Configuration
AIRCRAFT = 'c172p'
ROOT_DIR = 'd:\\Nvidia-JSB'

def init_jsbsim():
    fdm = jsbsim.FGFDMExec(ROOT_DIR)
    fdm.load_model(AIRCRAFT)
    fdm.set_dt(1.0/100.0)
    
    # Initialize state
    fdm['ic/h-sl-ft'] = 8202.0 # 2500m
    fdm['ic/ve-fps'] = 164.0  # 50m/s
    fdm.run_ic()
    
    return fdm

def init_warp(device):
    from main import load_aero_handles
    handles = load_aero_handles(os.path.join(ROOT_DIR, 'data', AIRCRAFT), device)
    
    state = wp.zeros(1, dtype=AircraftState, device=device)
    controls = wp.zeros(1, dtype=ControlState, device=device)
    
    h_state = AircraftState()
    h_state.pos = wp.vec3(0.0, 0.0, -2500.0)
    h_state.quat = wp.quat(0.0, 0.0, 0.0, 1.0)
    h_state.vel_body = wp.vec3(50.0, 0.0, 0.0)
    h_state.mass = 1100.0
    h_state.inertia = wp.mat33(1285.0, 0.0, 0.0, 0.0, 1824.0, 0.0, 0.0, 0.0, 2667.0)
    h_state.inertia_inv = wp.mat33(1.0/1285.0, 0.0, 0.0, 0.0, 1.0/1824.0, 0.0, 0.0, 0.0, 1.0/2667.0)
    
    wp.copy(state, wp.array([h_state], dtype=AircraftState, device=device))
    return state, controls, handles

def run_comparison(steps=200):
    device = "cuda" if wp.is_cuda_available() else "cpu"
    jsb = init_jsbsim()
    w_state, w_ctrl, w_handles = init_warp(device)
    
    print(f"{'Step':<5} | {'JSB Alt (m)':<12} | {'Warp Alt (m)':<12} | {'Vel Err %':<8}")
    print("-" * 55)
    
    for i in range(steps):
        # 1. Capture JSB
        jsb_alt = jsb['position/h-sl-ft'] * 0.3048
        jsb_vel = jsb['velocities/ve-fps'] * 0.3048
        
        # 2. Extract Sync Density
        rho_jsb = jsb['atmosphere/rho-slugs_ft3'] * 515.379
        
        # 3. Step Warp RK4
        wp.launch(
            kernel=integrate_state_rk4_kernel,
            dim=1,
            inputs=[w_state, w_ctrl, w_handles, 0.01, 16.16, 10.91, 1.49, float(rho_jsb)],
            device=device
        )
        
        # 4. Capture Warp
        h_s = w_state.numpy()[0]
        warp_alt = -h_s['pos'][2]
        warp_vel = np.linalg.norm(h_s['vel_body'])
        
        vel_err = abs(jsb_vel - warp_vel) / max(jsb_vel, 1.0) * 100.0
        
        if i % 20 == 0:
            print(f"{i:<5} | {jsb_alt:<12.2f} | {warp_alt:<12.2f} | {vel_err:<8.2f}%")
            
        jsb.run()
        
    wp.synchronize()

if __name__ == "__main__":
    run_comparison()
