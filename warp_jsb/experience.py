import warp as wp
import numpy as np
from warp_jsb.eom import AircraftState, ControlState

# --- ASYNC KERNELS (Per-Agent Head Tracking) ---

@wp.kernel
def encode_experience_async_AF_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    obs_buffer: wp.array(dtype=wp.float32, ndim=3), 
    act_buffer: wp.array(dtype=wp.float32, ndim=3), 
    write_heads: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    s = states[tid]
    c = controls[tid]
    time_idx = write_heads[tid]
    
    # 20-D SOTA Mapping: Matches pioneer_survival.py directly
    # [Agent][Time][Feature]
    obs_buffer[tid, time_idx, 0] = s.euler_rad[0] / 1.0          # phi
    obs_buffer[tid, time_idx, 1] = wp.clamp(s.euler_rad[1] / 0.5, -1.0, 1.0) # theta
    obs_buffer[tid, time_idx, 2] = s.euler_rad[2] / 3.14159     # psi (norm)
    obs_buffer[tid, time_idx, 3] = s.omega_body[0] * 2.0        # p
    obs_buffer[tid, time_idx, 4] = s.omega_body[1] * 3.0        # q
    obs_buffer[tid, time_idx, 5] = s.omega_body[2] * 2.0        # r
    obs_buffer[tid, time_idx, 6] = (s.v_kts - 110.0) / 50.0      # v_kts error
    obs_buffer[tid, time_idx, 7] = s.accel_body[0] / 10.0       # u_dot (fps^2)
    obs_buffer[tid, time_idx, 8] = c.throttle * 2.0 - 1.0       # throttle
    obs_buffer[tid, time_idx, 9] = s.alt_ft                     # alt (unscaled for playback)
    
    # Target Errors (Placeholders for now, filled by navigator in loop)
    obs_buffer[tid, time_idx, 10] = 0.0 # alt_err
    obs_buffer[tid, time_idx, 11] = 0.0 # heading_err
    obs_buffer[tid, time_idx, 12] = s.pos[0] * 3.28084          # X/North (Trajectory)
    obs_buffer[tid, time_idx, 13] = s.pos[1] * 3.28084          # Y/East (Trajectory)
    obs_buffer[tid, time_idx, 14] = s.pos[2] * -3.28084         # Z/Alt (Trajectory)
    
    # Native High-Fidelity Features
    obs_buffer[tid, time_idx, 15] = 0.0 # Kappa
    obs_buffer[tid, time_idx, 16] = wp.clamp(s.vel_ned[2] / 10.0, -1.0, 1.0) # v_down (fps)
    obs_buffer[tid, time_idx, 17] = wp.clamp(s.alpha / 0.3, -1.0, 1.0)
    obs_buffer[tid, time_idx, 18] = wp.clamp(s.beta / 0.1, -1.0, 1.0)
    obs_buffer[tid, time_idx, 19] = s.rpm / 2400.0              # rpm
    
    # Acts
    act_buffer[tid, time_idx, 0] = c.aileron
    act_buffer[tid, time_idx, 1] = c.elevator
    act_buffer[tid, time_idx, 2] = c.rudder
    act_buffer[tid, time_idx, 3] = c.throttle

@wp.kernel
def encode_experience_sync_AF_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    obs_buffer: wp.array(dtype=wp.float32, ndim=3),
    act_buffer: wp.array(dtype=wp.float32, ndim=3),
    time_idx: wp.int32
):
    tid = wp.tid()
    s = states[tid]
    c = controls[tid]
    
    obs_buffer[tid, time_idx, 0] = s.euler_rad[0] / 1.0
    obs_buffer[tid, time_idx, 1] = wp.clamp(s.euler_rad[1] / 0.5, -1.0, 1.0)
    obs_buffer[tid, time_idx, 2] = s.euler_rad[2] / 3.14159
    obs_buffer[tid, time_idx, 3] = s.omega_body[0] * 2.0
    obs_buffer[tid, time_idx, 4] = s.omega_body[1] * 3.0
    obs_buffer[tid, time_idx, 5] = s.omega_body[2] * 2.0
    obs_buffer[tid, time_idx, 6] = (s.v_kts - 110.0) / 50.0
    obs_buffer[tid, time_idx, 7] = s.accel_body[0] / 10.0
    obs_buffer[tid, time_idx, 8] = c.throttle * 2.0 - 1.0
    obs_buffer[tid, time_idx, 9] = s.alt_ft
    obs_buffer[tid, time_idx, 10] = 0.0
    obs_buffer[tid, time_idx, 11] = 0.0
    obs_buffer[tid, time_idx, 12] = s.pos[0] * 3.28084
    obs_buffer[tid, time_idx, 13] = s.pos[1] * 3.28084
    obs_buffer[tid, time_idx, 14] = s.pos[2] * -3.28084
    obs_buffer[tid, time_idx, 15] = 0.0
    obs_buffer[tid, time_idx, 16] = wp.clamp(s.vel_ned[2] / 10.0, -1.0, 1.0)
    obs_buffer[tid, time_idx, 17] = wp.clamp(s.alpha / 0.3, -1.0, 1.0)
    obs_buffer[tid, time_idx, 18] = wp.clamp(s.beta / 0.1, -1.0, 1.0)
    obs_buffer[tid, time_idx, 19] = s.rpm / 2400.0

    act_buffer[tid, time_idx, 0] = c.aileron
    act_buffer[tid, time_idx, 1] = c.elevator
    act_buffer[tid, time_idx, 2] = c.rudder
    act_buffer[tid, time_idx, 3] = c.throttle

class ExperienceHarvester:
    """
    Experience Harvester v2 (SOTA Edition)
    Captures 20-D Feature Vectors for bit-perfect DRL Playback.
    """
    def __init__(self, num_aircraft, window_size=10, obs_dim=20, act_dim=4, layout="agent_first", sync_mode=True, device="cuda"):
        self.num_aircraft = num_aircraft
        self.window_size = window_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layout = layout 
        self.sync_mode = sync_mode
        self.device = device
        
        if layout == "agent_first":
            self.obs_buffer = wp.zeros((num_aircraft, window_size, obs_dim), dtype=wp.float32, device=device)
            self.act_buffer = wp.zeros((num_aircraft, window_size, act_dim), dtype=wp.float32, device=device)
        else:
            self.obs_buffer = wp.zeros((obs_dim, num_aircraft, window_size), dtype=wp.float32, device=device)
            self.act_buffer = wp.zeros((act_dim, num_aircraft, window_size), dtype=wp.float32, device=device)
            
        self.global_head = 0
        self.write_heads = wp.zeros(num_aircraft, dtype=wp.int32, device=device)
        
    def record(self, states_array, controls_array):
        if self.sync_mode:
            kernel = encode_experience_sync_AF_kernel if self.layout == "agent_first" else None
            if kernel:
                wp.launch(kernel, dim=self.num_aircraft, inputs=[states_array, controls_array, self.obs_buffer, self.act_buffer, self.global_head], device=self.device)
            self.global_head = (self.global_head + 1) % self.window_size
        else:
            kernel = encode_experience_async_AF_kernel if self.layout == "agent_first" else None
            if kernel:
                wp.launch(kernel, dim=self.num_aircraft, inputs=[states_array, controls_array, self.obs_buffer, self.act_buffer, self.write_heads], device=self.device)
                
                @wp.kernel
                def inc_heads_kernel(heads: wp.array(dtype=wp.int32), window: wp.int32):
                    tid = wp.tid()
                    heads[tid] = (heads[tid] + 1) % window
                wp.launch(inc_heads_kernel, dim=self.num_aircraft, inputs=[self.write_heads, self.window_size], device=self.device)

    def to_numpy(self):
        return self.obs_buffer.numpy(), self.act_buffer.numpy()

    def save_to_disk(self, filename_prefix="pioneer_sota_data"):
        obs, acts = self.to_numpy()
        np.save(f"{filename_prefix}_obs.npy", obs)
        np.save(f"{filename_prefix}_acts.npy", acts)
        print(f"SOTA Export: Saved {self.num_aircraft} agents with full 20-D features.")
