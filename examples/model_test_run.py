import warp as wp
import numpy as np
import os
import time
import math
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym

from warp_jsb.eom import AircraftState, ControlState, integrate_full_state_rk4_kernel
from warp_jsb.ground_reactions import ContactPoint
from warp_jsb.aero_generated import AeroModelHandles
from warp_jsb.experience import ExperienceHarvester
from warp_jsb.curriculum import DynamicSkillGenerator, SkillType

wp.init()

# --- INFRASTRUCTURE ---

def load_aero_handles(data_dir, device):
    handles = AeroModelHandles()
    tables = ["aero_coefficient_CDDf", "aero_coefficient_CDwbh", "aero_coefficient_CYb",
              "aero_coefficient_CYp", "aero_coefficient_CYr", "aero_coefficient_CLwbh",
              "aero_coefficient_CLDf", "aero_coefficient_Clb", "aero_coefficient_Clr",
              "aero_coefficient_Cmdf", "aero_coefficient_Cnb"]
    for t in tables + ["C_THRUST", "C_POWER"]:
        data = np.load(os.path.join(data_dir, f"{t}.npy"))
        meta = np.load(os.path.join(data_dir, f"{t}_meta.npy"))
        attr = f"prop_{t}_table" if t in ["C_THRUST", "C_POWER"] else f"{t}_table"
        meta_attr = f"prop_{t}_meta" if t in ["C_THRUST", "C_POWER"] else f"{t}_meta"
        setattr(handles, attr, wp.from_numpy(data, dtype=wp.float32, device=device))
        setattr(handles, meta_attr, wp.from_numpy(meta, dtype=wp.float32, device=device))
    return handles

def get_20d_pioneer_obs(state, generator, cruise_speed=110.0):
    """
    Constructs the 20-D Observation using the dynamic curriculum generator.
    """
    phi, theta, psi = state['euler_rad']
    p, q, r = state['omega_body']
    v_kts = state['v_kts']
    curr_alt = state['alt_ft']
    alpha = state['alpha']
    beta = state['beta']
    u_dot = state['accel_body'][0]
    v_down = state['vel_ned'][2]
    
    # Target and Lookaheads from Curriculum Generator
    target_h, target_a = generator.ph, generator.pa
    l1_h, l1_a = generator.peek_future_steps(50)  # 1.0s lookahead
    l2_h, l2_a = generator.peek_future_steps(100) # 2.0s lookahead
    
    # Curvature (Kappa) approximation
    # Kappa = |v x a| / |v|^3
    v_vec = np.array([generator.vh, generator.va])
    a_vec = np.array([generator.ah, generator.aa])
    kappa = np.abs(np.cross(v_vec, a_vec)) / (np.linalg.norm(v_vec)**3 + 1e-6)

    obs = np.array([
        phi / 1.0,
        np.clip(theta / 0.5, -1, 1),
        ((psi + math.pi) % (2 * math.pi) - math.pi) / math.pi,
        p * 2.0, q * 3.0, r * 2.0,
        (v_kts - cruise_speed) / 50.0,
        u_dot / 10.0,
        0.0, # throttle_prev placeholder
        np.clip((target_a - curr_alt) / 200.0, -1, 1),
        (0.0 - phi) / 1.0, # Corrected: Roll to level
        ((math.radians(target_h) - psi + math.pi) % (2 * math.pi) - math.pi) / math.pi,
        ((math.radians(l1_h) - psi + math.pi) % (2 * math.pi) - math.pi) / math.pi,
        np.clip((l1_a - curr_alt) / 500.0, -1, 1), # Corrected: 500.0 norm
        ((math.radians(l2_h) - psi + math.pi) % (2 * math.pi) - math.pi) / math.pi,
        np.clip((l2_a - curr_alt) / 500.0, -1, 1), # Corrected: 500.0 norm
        0.0, # Kappa as placeholder (matched to training)
        np.clip(v_down / 10.0, -1, 1),
        np.clip(alpha / 0.3, -1, 1),
        np.clip(beta / 0.1, -1, 1)
    ], dtype=np.float32)
    
    return np.clip(np.nan_to_num(obs), -1, 1)

@wp.kernel
def apply_model_actions_kernel(actions: wp.array(dtype=wp.float32, ndim=2), controls: wp.array(dtype=ControlState)):
    tid = wp.tid()
    ctrl = ControlState()
    ctrl.aileron  = actions[tid, 0]
    ctrl.elevator = actions[tid, 1]
    ctrl.rudder   = actions[tid, 2]
    ctrl.throttle = (actions[tid, 3] + 1.0) * 0.5
    ctrl.mixture = 1.0
    controls[tid] = ctrl

# --- MARATHON MODES ---

def run_marathon(mode="sequential", num_aircraft=5, dt=0.02):
    device = "cuda"
    model_path = r"D:\DRL-Project-CE - Copy\output\2026-04-10_075357\models\ppo_staged_best.zip"
    stats_path = r"D:\DRL-Project-CE - Copy\output\2026-04-10_075357\models\ppo_staged_best_vec_normalize.pkl"
    
    model = PPO.load(model_path)
    def make_dummy():
        env = gym.Env()
        env.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        return env
    dummy_env = DummyVecEnv([make_dummy])
    norm_env = VecNormalize.load(stats_path, dummy_env)
    norm_env.training = False
    
    import warp_jsb
    handles = load_aero_handles(warp_jsb.get_c172p_assets_path(), device)
    states = wp.zeros(num_aircraft, dtype=AircraftState, device=device)
    controls = wp.zeros(num_aircraft, dtype=ControlState, device=device)
    contacts = wp.zeros(0, dtype=ContactPoint, device=device)
    
    # Initialize Generators (One per agent)
    generators = [DynamicSkillGenerator(dt=dt) for _ in range(num_aircraft)]
    
    # Initialize Harvester
    num_steps = 7200 if mode == "sequential" else 3000
    harvester = ExperienceHarvester(num_aircraft, window_size=num_steps, obs_dim=20, device=device)
    
    @wp.kernel
    def init_flight_kernel(states: wp.array(dtype=AircraftState)):
        tid = wp.tid()
        s = AircraftState()
        s.quat = wp.quat_identity()
        s.pos = wp.vec3(0.0, float(tid) * 100.0, -1524.0)
        s.vel_body = wp.vec3(56.0, 0.0, 0.0)
        s.mass = 1880.0 / 2.20462
        s.fuel_mass = 100.0
        s.rpm = 2300.0
        states[tid] = s
    wp.launch(init_flight_kernel, dim=num_aircraft, inputs=[states], device=device)
    
    S, b, c = 16.16, 10.91, 1.49
    rho, p_amb = 1.0, 1760.0 # Standard Day
    
    levels = [SkillType.CRUISE, SkillType.PITCH, SkillType.ROLL, SkillType.DYNAMIC, SkillType.ADVANCED, SkillType.TACTICAL]
    current_level_idx = 0
    steps_per_level = 1200
    
    print(f"\nSTARTING MARATHON: Mode={mode.upper()}")
    start_time = time.time()
    
    for step in range(num_steps):
        # 1. Update Generators and Level Transitions
        if mode == "sequential" and step % steps_per_level == 0:
            skill = levels[current_level_idx]
            diff = 0.2 + (0.8 * (current_level_idx / 5.0))
            print(f"Phase 1: Transitioning to {skill} | Difficulty: {diff:.2f}")
            for gen in generators: gen.set_skill(skill, difficulty=diff)
            current_level_idx += 1
        elif mode == "stochastic" and step % 500 == 0:
            skill = random.choice(levels)
            print(f"Phase 2 Gauntlet: Random Redirect -> {skill}")
            for gen in generators: gen.set_skill(skill, difficulty=1.0)

        # 2. Get Observations & Inference
        h_states = states.numpy()
        actions = []
        for i in range(num_aircraft):
            generators[i].get_next_step()
            obs = get_20d_pioneer_obs(h_states[i], generators[i])
            action, _ = model.predict(norm_env.normalize_obs(obs), deterministic=True)
            actions.append(action)
            
        actions_gpu = wp.from_numpy(np.array(actions), dtype=wp.float32, device=device)
        wp.launch(apply_model_actions_kernel, dim=num_aircraft, inputs=[actions_gpu, controls], device=device)
        wp.launch(integrate_full_state_rk4_kernel, dim=num_aircraft, 
                  inputs=[states, controls, handles, contacts, dt, S, b, c, rho, p_amb, wp.vec3(0.0), wp.vec3(-2.0)], device=device)
        harvester.record(states, controls)
        
    wp.synchronize()
    filename = f"marathon_{mode}"
    harvester.save_to_disk(filename)
    print(f"Marathon Success: captured into {filename}.npy in {time.time()-start_time:.2f}s.")

if __name__ == "__main__":
    # Run Both Phases as requested
    run_marathon(mode="sequential")
    run_marathon(mode="stochastic")
