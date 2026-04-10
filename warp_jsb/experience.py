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
    
    # [Agent][Time][Feature]
    obs_buffer[tid, time_idx, 0] = s.euler_rad[0]
    obs_buffer[tid, time_idx, 1] = s.euler_rad[1]
    obs_buffer[tid, time_idx, 2] = s.euler_rad[2]
    obs_buffer[tid, time_idx, 3] = s.omega_body[0]
    obs_buffer[tid, time_idx, 4] = s.omega_body[1]
    obs_buffer[tid, time_idx, 5] = s.omega_body[2]
    obs_buffer[tid, time_idx, 6] = s.v_kts
    obs_buffer[tid, time_idx, 7] = s.alt_ft
    obs_buffer[tid, time_idx, 8] = s.alpha
    obs_buffer[tid, time_idx, 9] = s.beta
    obs_buffer[tid, time_idx, 10] = s.rpm
    
    act_buffer[tid, time_idx, 0] = c.aileron
    act_buffer[tid, time_idx, 1] = c.elevator
    act_buffer[tid, time_idx, 2] = c.rudder
    act_buffer[tid, time_idx, 3] = c.throttle

@wp.kernel
def encode_experience_async_FF_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    obs_buffer: wp.array(dtype=wp.float32, ndim=3), 
    act_buffer: wp.array(dtype=wp.float32, ndim=3), 
    write_heads: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    s = states[tid]
    c = controls[tid]
    t = write_heads[tid]
    
    # [Feature][Agent][Time]
    obs_buffer[0, tid, t] = s.euler_rad[0]
    obs_buffer[1, tid, t] = s.euler_rad[1]
    obs_buffer[2, tid, t] = s.euler_rad[2]
    obs_buffer[3, tid, t] = s.omega_body[0]
    obs_buffer[4, tid, t] = s.omega_body[1]
    obs_buffer[5, tid, t] = s.omega_body[2]
    obs_buffer[6, tid, t] = s.v_kts
    obs_buffer[7, tid, t] = s.alt_ft
    obs_buffer[8, tid, t] = s.alpha
    obs_buffer[9, tid, t] = s.beta
    obs_buffer[10, tid, t] = s.rpm
    
    act_buffer[0, tid, t] = c.aileron
    act_buffer[1, tid, t] = c.elevator
    act_buffer[2, tid, t] = c.rudder
    act_buffer[3, tid, t] = c.throttle

# --- SYNC KERNELS (Global Shared Head Tracking) ---

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
    
    obs_buffer[tid, time_idx, 0] = s.euler_rad[0]
    obs_buffer[tid, time_idx, 1] = s.euler_rad[1]
    obs_buffer[tid, time_idx, 2] = s.euler_rad[2]
    obs_buffer[tid, time_idx, 3] = s.omega_body[0]
    obs_buffer[tid, time_idx, 4] = s.omega_body[1]
    obs_buffer[tid, time_idx, 5] = s.omega_body[2]
    obs_buffer[tid, time_idx, 6] = s.v_kts
    obs_buffer[tid, time_idx, 7] = s.alt_ft
    obs_buffer[tid, time_idx, 8] = s.alpha
    obs_buffer[tid, time_idx, 9] = s.beta
    obs_buffer[tid, time_idx, 10] = s.rpm
    
    act_buffer[tid, time_idx, 0] = c.aileron
    act_buffer[tid, time_idx, 1] = c.elevator
    act_buffer[tid, time_idx, 2] = c.rudder
    act_buffer[tid, time_idx, 3] = c.throttle

@wp.kernel
def encode_experience_sync_FF_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    obs_buffer: wp.array(dtype=wp.float32, ndim=3),
    act_buffer: wp.array(dtype=wp.float32, ndim=3),
    time_idx: wp.int32
):
    tid = wp.tid()
    s = states[tid]
    c = controls[tid]
    
    obs_buffer[0, tid, time_idx] = s.euler_rad[0]
    obs_buffer[1, tid, time_idx] = s.euler_rad[1]
    obs_buffer[2, tid, time_idx] = s.euler_rad[2]
    obs_buffer[3, tid, time_idx] = s.omega_body[0]
    obs_buffer[4, tid, time_idx] = s.omega_body[1]
    obs_buffer[5, tid, time_idx] = s.omega_body[2]
    obs_buffer[6, tid, time_idx] = s.v_kts
    obs_buffer[7, tid, time_idx] = s.alt_ft
    obs_buffer[8, tid, time_idx] = s.alpha
    obs_buffer[9, tid, time_idx] = s.beta
    obs_buffer[10, tid, time_idx] = s.rpm
    
    act_buffer[0, tid, time_idx] = c.aileron
    act_buffer[1, tid, time_idx] = c.elevator
    act_buffer[2, tid, time_idx] = c.rudder
    act_buffer[3, tid, time_idx] = c.throttle

class ExperienceHarvester:
    """
    Experience Harvester v2 (Optimized Audit Edition)
    Optimized for high-speed sequence-aware DRL training on GPU.
    """
    def __init__(self, num_aircraft, window_size=10, obs_dim=11, act_dim=4, layout="agent_first", sync_mode=True, device="cuda"):
        self.num_aircraft = num_aircraft
        self.window_size = window_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layout = layout # "agent_first" or "feature_first"
        self.sync_mode = sync_mode
        self.device = device
        
        # 1. Memory Allocation
        if layout == "agent_first":
            self.obs_buffer = wp.zeros((num_aircraft, window_size, obs_dim), dtype=wp.float32, device=device)
            self.act_buffer = wp.zeros((num_aircraft, window_size, act_dim), dtype=wp.float32, device=device)
        else:
            self.obs_buffer = wp.zeros((obs_dim, num_aircraft, window_size), dtype=wp.float32, device=device)
            self.act_buffer = wp.zeros((act_dim, num_aircraft, window_size), dtype=wp.float32, device=device)
            
        # 2. Sequential State
        self.global_head = 0
        self.write_heads = wp.zeros(num_aircraft, dtype=wp.int32, device=device)
        
    def record(self, states_array, controls_array):
        if self.sync_mode:
            # OPTIMIZED: Synchronous Global write
            kernel = encode_experience_sync_AF_kernel if self.layout == "agent_first" else encode_experience_sync_FF_kernel
            wp.launch(kernel, dim=self.num_aircraft, inputs=[states_array, controls_array, self.obs_buffer, self.act_buffer, self.global_head], device=self.device)
            self.global_head = (self.global_head + 1) % self.window_size
        else:
            # ASYNC: Per-agent write head lookup
            kernel = encode_experience_async_AF_kernel if self.layout == "agent_first" else encode_experience_async_FF_kernel
            wp.launch(kernel, dim=self.num_aircraft, inputs=[states_array, controls_array, self.obs_buffer, self.act_buffer, self.write_heads], device=self.device)
            
            # Increment heads
            @wp.kernel
            def inc_heads_kernel(heads: wp.array(dtype=wp.int32), window: wp.int32):
                tid = wp.tid()
                heads[tid] = (heads[tid] + 1) % window
            wp.launch(inc_heads_kernel, dim=self.num_aircraft, inputs=[self.write_heads, self.window_size], device=self.device)

    def reset_agents(self, reset_mask: wp.array(dtype=wp.bool)):
        if self.sync_mode:
            print("Warning: reset_agents has no effect in sync_mode=True (Global Reset required)")
            return
        @wp.kernel
        def reset_heads_kernel(heads: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool)):
            tid = wp.tid()
            if mask[tid]: heads[tid] = 0
        wp.launch(reset_heads_kernel, dim=self.num_aircraft, inputs=[self.write_heads, reset_mask], device=self.device)

    def to_numpy(self):
        return self.obs_buffer.numpy(), self.act_buffer.numpy()

    def save_to_disk(self, filename_prefix="pioneer_sequence"):
        obs, acts = self.to_numpy()
        np.save(f"{filename_prefix}_obs.npy", obs)
        np.save(f"{filename_prefix}_acts.npy", acts)
        print(f"Audit: Saved {self.num_aircraft} samples to disk.")
