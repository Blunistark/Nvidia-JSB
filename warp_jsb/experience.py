import warp as wp
import numpy as np
from warp_jsb.eom import AircraftState, ControlState

@wp.kernel
def default_experience_encoder_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    obs_buffer: wp.array(dtype=wp.float32, ndim=2), # [num_aircraft, obs_dim]
    act_buffer: wp.array(dtype=wp.float32, ndim=2)  # [num_aircraft, act_dim]
):
    tid = wp.tid()
    s = states[tid]
    c = controls[tid]
    
    # --- Custom Mapping (Pioneer Standard 20-D subset) ---
    # We can pack anything we want here. 
    # For this 'Customizable' version, we record raw physical markers:
    
    # 1. Observations
    obs_buffer[tid, 0] = s.euler_rad[0] # Roll
    obs_buffer[tid, 1] = s.euler_rad[1] # Pitch
    obs_buffer[tid, 2] = s.euler_rad[2] # Yaw
    obs_buffer[tid, 3] = s.omega_body[0] # p
    obs_buffer[tid, 4] = s.omega_body[1] # q
    obs_buffer[tid, 5] = s.omega_body[2] # r
    obs_buffer[tid, 6] = s.v_kts
    obs_buffer[tid, 7] = s.alt_ft
    obs_buffer[tid, 8] = s.alpha
    obs_buffer[tid, 9] = s.beta
    obs_buffer[tid, 10] = s.rpm
    
    # 2. Actions
    act_buffer[tid, 0] = c.aileron
    act_buffer[tid, 1] = c.elevator
    act_buffer[tid, 2] = c.rudder
    act_buffer[tid, 3] = c.throttle

class ExperienceHarvester:
    """
    High-Speed GPU Data Harvester for Pioneer FDM.
    Supports capturing State-Action pairs from millions of agents.
    """
    def __init__(self, num_aircraft, obs_dim=11, act_dim=4, device="cuda"):
        self.num_aircraft = num_aircraft
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        # Pre-allocate GPU buffers
        self.obs_buffer = wp.zeros((num_aircraft, obs_dim), dtype=wp.float32, device=device)
        self.act_buffer = wp.zeros((num_aircraft, act_dim), dtype=wp.float32, device=device)
        
    def record(self, states_array, controls_array, encoder_kernel=default_experience_encoder_kernel):
        """
        Launches the encoder kernel to harvest data from the physics state.
        """
        wp.launch(
            kernel=encoder_kernel,
            dim=self.num_aircraft,
            inputs=[states_array, controls_array, self.obs_buffer, self.act_buffer],
            device=self.device
        )
        
    def to_numpy(self):
        """
        Pulls the state-action pairs to CPU as a tuple of Numpy arrays.
        """
        return self.obs_buffer.numpy(), self.act_buffer.numpy()

    def save_to_disk(self, filename_prefix="pioneer_data"):
        """
        Saves the harvested data to Numpy files.
        """
        obs, acts = self.to_numpy()
        np.save(f"{filename_prefix}_obs.npy", obs)
        np.save(f"{filename_prefix}_acts.npy", acts)
        print(f"Saved {self.num_aircraft} experience pairs to {filename_prefix}")
