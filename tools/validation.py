import jsbsim
import warp as wp
import numpy as np
import os
import csv
import time

from warp_jsb.eom import AircraftState, ControlState, integrate_full_state_rk4_kernel
from warp_jsb.ground_reactions import ContactPoint
from warp_jsb.aero_generated import AeroModelHandles

# JSBSim Configuration
AIRCRAFT = 'c172p'
ROOT_DIR = 'd:\\Nvidia-JSB'

def quat_to_euler(q):
    # Standard Tait-Bryan ZYX (psi, theta, phi)
    x, y, z, w = q[0], q[1], q[2], q[3]
    phi = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    theta = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
    psi = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.degrees(phi), np.degrees(theta), np.degrees(psi)

class TelemetryLogger:
    def __init__(self, filename="validation_report.csv"):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        # 13 States: [Pos N, E, D, Euler Phi, Theta, Psi, Vel U, V, W, Rate P, Q, R, RPM]
        header = [
            "step", "time", 
            "jsb_n", "warp_n", "jsb_e", "warp_e", "jsb_d", "warp_d",
            "jsb_phi", "warp_phi", "jsb_theta", "warp_theta", "jsb_psi", "warp_psi",
            "jsb_u", "warp_u", "jsb_v", "warp_v", "jsb_w", "warp_w",
            "jsb_p", "warp_p", "jsb_q", "warp_q", "jsb_r", "warp_r",
            "jsb_rpm", "warp_rpm"
        ]
        self.writer.writerow(header)
        self.final_jsb = None
        self.final_warp = None
        
    def log(self, step, t, jsb, warp_state):
        fps2mps = 0.3048
        rad2deg = 57.2958
        
        # 1. Position
        jsb_n = jsb['position/from-start-neu-n-ft'] * fps2mps
        jsb_e = jsb['position/from-start-neu-e-ft'] * fps2mps
        jsb_d = -jsb['position/h-sl-ft'] * fps2mps
        
        # 2. Attitude (Euler)
        jsb_phi = jsb['attitude/phi-rad'] * rad2deg
        jsb_theta = jsb['attitude/theta-rad'] * rad2deg
        jsb_psi = jsb['attitude/psi-rad'] * rad2deg
        w_phi, w_theta, w_psi = quat_to_euler(warp_state['quat'])
        
        # 3. Velocities (Body)
        jsb_u, jsb_v, jsb_w = jsb['velocities/u-fps']*fps2mps, jsb['velocities/v-fps']*fps2mps, jsb['velocities/w-fps']*fps2mps
        
        # 4. Angular Rates (Body)
        jsb_p, jsb_q, jsb_r = jsb['velocities/p-rad_sec'], jsb['velocities/q-rad_sec'], jsb['velocities/r-rad_sec']
        
        jsb_rpm = jsb['propulsion/engine/engine-rpm']
        
        data = [
            step, f"{t:.2f}",
            f"{jsb_n:.2f}", f"{warp_state['pos'][0]:.2f}",
            f"{jsb_e:.2f}", f"{warp_state['pos'][1]:.2f}",
            f"{jsb_d:.2f}", f"{warp_state['pos'][2]:.2f}",
            f"{jsb_phi:.2f}", f"{w_phi:.2f}",
            f"{jsb_theta:.2f}", f"{w_theta:.2f}",
            f"{jsb_psi:.2f}", f"{w_psi:.2f}",
            f"{jsb_u:.3f}", f"{warp_state['vel_body'][0]:.3f}",
            f"{jsb_v:.3f}", f"{warp_state['vel_body'][1]:.3f}",
            f"{jsb_w:.3f}", f"{warp_state['vel_body'][2]:.3f}",
            f"{jsb_p:.4f}", f"{warp_state['omega_body'][0]:.4f}",
            f"{jsb_q:.4f}", f"{warp_state['omega_body'][1]:.4f}",
            f"{jsb_r:.4f}", f"{warp_state['omega_body'][2]:.4f}",
            f"{jsb_rpm:.1f}", f"{warp_state['rpm']:.1f}"
        ]
        self.writer.writerow(data)
        
        self.final_jsb = [jsb_n, jsb_e, jsb_d, jsb_phi, jsb_theta, jsb_psi, jsb_u, jsb_v, jsb_w, jsb_p, jsb_q, jsb_r, jsb_rpm]
        self.final_warp = [warp_state['pos'][0], warp_state['pos'][1], warp_state['pos'][2], w_phi, w_theta, w_psi,
                           warp_state['vel_body'][0], warp_state['vel_body'][1], warp_state['vel_body'][2],
                           warp_state['omega_body'][0], warp_state['omega_body'][1], warp_state['omega_body'][2], warp_state['rpm']]

        if step % 50 == 0:
            print(f"{step:<4} | Alt Err: {abs(jsb_d - warp_state['pos'][2]):.2f}m | u Err: {abs(jsb_u - warp_state['vel_body'][0]):.2f}m/s | RPM Err: {abs(jsb_rpm - warp_state['rpm']):.1f}")

    def print_summary(self):
        print("\n" + "="*85)
        print(f"{'STATE COMPONENT':<22} | {'JSBSIM':>15} | {'WARP NATIVE':>15} | {'RESIDUAL':>15}")
        print("-" * 85)
        labels = ["Pos North (m)", "Pos East (m)", "Pos Down (m)", "Euler Phi (deg)", "Euler Theta (deg)", "Euler Psi (deg)", 
                  "Vel U (m/s)", "Vel V (m/s)", "Vel W (m/s)", "Rate P (rad/s)", "Rate Q (rad/s)", "Rate R (rad/s)", "Engine RPM"]
        for i in range(13):
            val_jsb = self.final_jsb[i]
            val_warp = self.final_warp[i]
            res = abs(val_jsb - val_warp)
            print(f"{labels[i]:<22} | {val_jsb:>15.5f} | {val_warp:>15.5f} | {res:>15.5f}")
        print("="*85)

    def close(self):
        self.file.close()

def load_aero_handles(device):
    data_dir = os.path.join(ROOT_DIR, "data/c172p")
    handles = AeroModelHandles()
    mappings = {
        "aero_coefficient_CDDf": "aero_coefficient_CDDf",
        "aero_coefficient_CDwbh": "aero_coefficient_CDwbh",
        "aero_coefficient_CYb": "aero_coefficient_CYb",
        "aero_coefficient_CYp": "aero_coefficient_CYp",
        "aero_coefficient_CYr": "aero_coefficient_CYr",
        "aero_coefficient_CLwbh": "aero_coefficient_CLwbh",
        "aero_coefficient_CLDf": "aero_coefficient_CLDf",
        "aero_coefficient_Clb": "aero_coefficient_Clb",
        "aero_coefficient_Clr": "aero_coefficient_Clr",
        "aero_coefficient_Cmdf": "aero_coefficient_Cmdf",
        "aero_coefficient_Cnb": "aero_coefficient_Cnb",
        "prop_C_THRUST": "C_THRUST",
        "prop_C_POWER": "C_POWER"
    }
    for field, filename in mappings.items():
        table = np.load(os.path.join(data_dir, f"{filename}.npy"))
        meta = np.load(os.path.join(data_dir, f"{filename}_meta.npy"))
        setattr(handles, f"{field}_table", wp.array(table, dtype=wp.float32, device=device))
        setattr(handles, f"{field}_meta", wp.array(meta, dtype=wp.float32, device=device))
    return handles

def create_contact(pos, is_bogey, sk, dc, sf, df, ms):
    c = ContactPoint()
    c.pos_body = pos
    c.is_bogey = is_bogey
    c.spring_k = sk
    c.damping_c = dc
    c.static_friction = sf
    c.dynamic_friction = df
    c.max_steer = ms
    return c

def init_jsbsim():
    fdm = jsbsim.FGFDMExec(ROOT_DIR)
    fdm.load_model(AIRCRAFT)
    fdm.set_dt(1.0/100.0)
    fdm['ic/h-sl-ft'] = 5000.0
    fdm['ic/u-fps'] = 150.0
    fdm['ic/rpm'] = 2400.0 # Force hot start
    fdm.run_ic()
    fdm['fcs/throttle-cmd-norm'] = 0.8
    fdm['fcs/mixture-cmd-norm'] = 0.9
    fdm['propulsion/magneto_cmd'] = 3
    fdm['propulsion/starter_cmd'] = 1
    fdm['propulsion/engine/set-running'] = 1
    return fdm

from scipy.spatial.transform import Rotation

def init_warp(device, jsb):
    handles = load_aero_handles(device)
    contacts_list = [
        create_contact(wp.vec3(0.0, 0.0, 0.0), True, 1e6, 1e4, 0.5, 0.5, 0.5),
        create_contact(wp.vec3(-5.0, -4.0, 0.0), True, 1e6, 1e4, 0.5, 0.5, 0.0),
        create_contact(wp.vec3(-5.0, 4.0, 0.0), True, 1e6, 1e4, 0.5, 0.5, 0.0),
    ]
    contacts = wp.array(contacts_list, dtype=ContactPoint, device=device)
    state = wp.zeros(1, dtype=AircraftState, device=device)
    controls = wp.zeros(1, dtype=ControlState, device=device)
    h_state = AircraftState()
    fps2mps = 0.3048
    h_state.pos = wp.vec3(jsb['position/from-start-neu-n-ft']*fps2mps, jsb['position/from-start-neu-e-ft']*fps2mps, -jsb['position/h-sl-ft']*fps2mps)
    
    # Attitude precise sync from Eulers using Scipy
    phi, theta, psi = jsb['attitude/phi-rad'], jsb['attitude/theta-rad'], jsb['attitude/psi-rad']
    rot = Rotation.from_euler('ZYX', [psi, theta, phi])
    q = rot.as_quat() # x, y, z, w
    h_state.quat = wp.quat(q[0], q[1], q[2], q[3])
    
    h_state.vel_body = wp.vec3(jsb['velocities/u-fps']*fps2mps, jsb['velocities/v-fps']*fps2mps, jsb['velocities/w-fps']*fps2mps)
    h_state.omega_body = wp.vec3(jsb['velocities/p-rad_sec'], jsb['velocities/q-rad_sec'], jsb['velocities/r-rad_sec'])
    h_state.mass = jsb['inertia/weight-lbs'] / 2.20462
    
    s2kgm2 = 1.35582
    ixx, iyy, izz = jsb['inertia/ixx-slugs_ft2']*s2kgm2, jsb['inertia/iyy-slugs_ft2']*s2kgm2, jsb['inertia/izz-slugs_ft2']*s2kgm2
    ixy, ixz, iyz = jsb['inertia/ixy-slugs_ft2']*s2kgm2, jsb['inertia/ixz-slugs_ft2']*s2kgm2, jsb['inertia/iyz-slugs_ft2']*s2kgm2
    i_mat = np.array([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])
    h_state.inertia = wp.mat33(i_mat.flatten())
    h_state.inertia_inv = wp.mat33(np.linalg.inv(i_mat).flatten())
    # Force RPM synchronization (JSBSim IC -> Warp h_state)
    target_rpm = 2400.0
    jsb['propulsion/engine/engine-rpm'] = target_rpm
    jsb['propulsion/engine/set-running'] = 1
    h_state.rpm = target_rpm
    h_state.fuel_mass = jsb['propulsion/tank/contents-lbs'] / 2.20462
    
    wp.copy(state, wp.array([h_state], dtype=AircraftState, device=device))
    h_ctrl = ControlState()
    h_ctrl.throttle = 0.8
    h_ctrl.mixture = 0.9
    wp.copy(controls, wp.array([h_ctrl], dtype=ControlState, device=device))
    return state, controls, handles, contacts

def run_comparison(steps=1000):
    device = "cuda" if wp.is_cuda_available() else "cpu"
    jsb = init_jsbsim()
    w_state, w_ctrl, w_handles, w_contacts = init_warp(device, jsb)
    logger = TelemetryLogger("validation_report.csv")
    print(f"Validation Start (Final 13-DOF Audit):")
    dt = 0.01
    for i in range(steps):
        rho_jsb = jsb['atmosphere/rho-slugs_ft3'] * 515.379
        p_amb_inhg = jsb['atmosphere/P-psf'] * 0.014139
        
        # Dynamic Moment Arms (Structural -> Body)
        # Structural: X-aft, Y-right, Z-up
        # Body: X-fwd, Y-right, Z-down
        cg_x = jsb['inertia/cg-x-in']
        cg_y = jsb['inertia/cg-y-in']
        cg_z = jsb['inertia/cg-z-in']
        
        # AERORP: [43.2, 0, 59.4]
        # PROP: [-37.7, 0, 26.6]
        i2m = 0.0254
        r_aero_v = wp.vec3(-(43.2 - cg_x) * i2m, (0.0 - cg_y) * i2m, -(59.4 - cg_z) * i2m)
        r_prop_v = wp.vec3(-(-37.7 - cg_x) * i2m, (0.0 - cg_y) * i2m, -(26.6 - cg_z) * i2m)
        
        wp.launch(
            kernel=integrate_full_state_rk4_kernel, 
            dim=1, 
            inputs=[
                w_state, w_ctrl, w_handles, w_contacts, dt, 
                16.165, 10.91, 1.49, float(rho_jsb), float(p_amb_inhg),
                r_aero_v, r_prop_v
            ], 
            device=device
        )
        logger.log(i, i*dt, jsb, w_state.numpy()[0])
        jsb.run()
    logger.print_summary()
    logger.close()
    wp.synchronize()
    print("\nValidation Complete.")

if __name__ == "__main__":
    run_comparison()
