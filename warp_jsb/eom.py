import warp as wp
import numpy as np
from warp_jsb.aero_generated import evaluate_aero_model, AeroModelHandles
from warp_jsb.fcs import compute_fcs_components, FCSState
from warp_jsb.propulsion import compute_prop_forces_and_induced, update_piston_engine
from warp_jsb.ground_reactions import compute_single_contact_force, ContactPoint

@wp.struct
class ControlState:
    aileron: wp.float32
    elevator: wp.float32
    rudder: wp.float32
    flaps: wp.float32
    throttle: wp.float32
    mixture: wp.float32
    brake: wp.float32
    steer: wp.float32

@wp.struct
class AircraftState:
    pos: wp.vec3
    quat: wp.quat
    vel_body: wp.vec3
    omega_body: wp.vec3
    mass: wp.float32
    inertia: wp.mat33
    inertia_inv: wp.mat33
    fuel_mass: wp.float32
    alpha: wp.float32
    beta: wp.float32
    stall_hyst: wp.float32
    rpm: wp.float32
    
    # High-level properties for DRL Observations
    alt_ft: wp.float32
    v_kts: wp.float32
    euler_rad: wp.vec3 # [roll, pitch, yaw]

@wp.struct
class StateDeriv:
    d_pos: wp.vec3
    d_quat: wp.quat
    d_vel: wp.vec3
    d_omega: wp.vec3
    d_fuel: wp.float32
    d_rpm: wp.float32

@wp.func
def compute_aero_angles(vel_body: wp.vec3):
    u = vel_body[0]
    v = vel_body[1]
    w = vel_body[2]
    v_mag = wp.length(vel_body)
    if v_mag < 0.1:
        return 0.0, 0.0, 0.0
    alpha = wp.atan2(w, u)
    beta = wp.asin(wp.clamp(v / v_mag, -1.0, 1.0))
    return alpha, beta, v_mag

@wp.func
def compute_full_dynamics_derivative(
    s: AircraftState,
    ctrl: ControlState,
    handles: AeroModelHandles,
    S: wp.float32, b: wp.float32, c: wp.float32,
    rho: wp.float32,
    contacts: wp.array(dtype=ContactPoint),
    p_amb: wp.float32,
    r_aero: wp.vec3,
    r_prop: wp.vec3
) -> StateDeriv:
    elevator_rad, aileron_rad, rudder_rad, flaps_norm = compute_fcs_components(
        ctrl.elevator, 0.0, ctrl.aileron, 0.0, ctrl.rudder, 0.0, ctrl.flaps, 0.01
    )
    v_ned = wp.quat_rotate(s.quat, s.vel_body)
    dq_dt = 0.5 * wp.mul(s.quat, wp.quat(s.omega_body[0], s.omega_body[1], s.omega_body[2], 0.0))
    v_fps = wp.length(s.vel_body) * 3.28084
    rho_slugs = rho * 0.00194032
    
    thrust_lbs, torque_prop, v_i_fps = compute_prop_forces_and_induced(
        v_fps, rho_slugs, s.rpm, 6.25, handles
    )
    hp_engine, torque_engine = update_piston_engine(
        ctrl.throttle, ctrl.mixture, rho_slugs, s.rpm, p_amb
    )
    
    v_slip_fps = v_fps + v_i_fps * 2.0
    qbar_psf = 0.5 * rho_slugs * (v_fps * v_fps)
    qbar_induced_psf = 0.5 * rho_slugs * (v_slip_fps * v_slip_fps)
    
    alpha, beta, v_mag = compute_aero_angles(s.vel_body)
    inv_v = 1.0 / wp.max(v_mag, 0.1)
    bi2vel = b * 0.5 * inv_v
    ci2vel = c * 0.5 * inv_v
    h_mac = wp.max(0.0, -s.pos[2]) / c
    
    # Pass BOTH qbar and qbar_induced to evaluate_aero_model
    D_lbs, Y_lbs, L_lbs, l_ftlbs, m_ftlbs, n_ftlbs = evaluate_aero_model(
        alpha, beta, qbar_psf, qbar_induced_psf, s.omega_body[0], s.omega_body[1], s.omega_body[2],
        bi2vel, ci2vel, h_mac, s.stall_hyst, elevator_rad, aileron_rad, rudder_rad, flaps_norm, handles
    )
    
    # Propeller Asymmetry Physics
    # 1. P-Factor (Cross-wind on disk due to AoA)
    p_factor_const = 12.5 
    n_pfactor_ftlbs = thrust_lbs * p_factor_const * wp.sin(alpha)
    
    # 2. Spiral Slipstream (Blast hitting vertical tail)
    n_slip_ftlbs = qbar_induced_psf * S * b * (-0.004) # Calibrated bias
    
    f_aero_b = wp.vec3((L_lbs * wp.sin(alpha) - D_lbs * wp.cos(alpha)) * 4.44822, Y_lbs * 4.44822, (-L_lbs * wp.cos(alpha) - D_lbs * wp.sin(alpha)) * 4.44822)
    t_aero_pure_b = wp.vec3(l_ftlbs * 1.35582, m_ftlbs * 1.35582, (n_ftlbs + n_pfactor_ftlbs + n_slip_ftlbs) * 1.35582)
    t_aero_arm_b = wp.cross(r_aero, f_aero_b)
    
    f_prop_b = wp.vec3(thrust_lbs * 4.44822, 0.0, 0.0)
    t_prop_arm_b = wp.cross(r_prop, f_prop_b)
    t_prop_react_b = wp.vec3(-torque_prop * 1.35582, 0.0, 0.0)
    
    f_ground_b = wp.vec3(0.0, 0.0, 0.0)
    t_ground_b = wp.vec3(0.0, 0.0, 0.0)
    for i in range(len(contacts)):
        f_c, t_c = compute_single_contact_force(contacts[i], s.pos, s.quat, s.vel_body, s.omega_body, ctrl.brake, ctrl.steer)
        f_ground_b += f_c
        t_ground_b += t_c
        
    total_f_b = f_aero_b + f_prop_b + f_ground_b
    g_m_s2 = 9.80665
    gravity_ned = wp.vec3(0.0, 0.0, g_m_s2)
    gravity_body = wp.quat_rotate(wp.quat_inverse(s.quat), gravity_ned)
    
    d_vel = (total_f_b / s.mass) + gravity_body - wp.cross(s.omega_body, s.vel_body)
    
    total_t_b = t_aero_pure_b + t_aero_arm_b + t_prop_arm_b + t_prop_react_b + t_ground_b
    i_omega = s.inertia * s.omega_body
    d_omega = s.inertia_inv * (total_t_b - wp.cross(s.omega_body, i_omega))
    deriv = StateDeriv()
    deriv.d_pos = v_ned
    deriv.d_quat = dq_dt
    deriv.d_vel = d_vel
    deriv.d_omega = d_omega
    deriv.d_fuel = 0.0
    
    deriv.d_rpm = (torque_engine - torque_prop) / 1.67 * (60.0 / (2.0 * 3.14159)) 
    
    return deriv

@wp.kernel
def integrate_full_state_rk4_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    handles: AeroModelHandles,
    contacts: wp.array(dtype=ContactPoint),
    dt: wp.float32,
    S: wp.float32, b: wp.float32, c: wp.float32,
    rho: wp.float32,
    p_amb: wp.float32,
    r_aero: wp.vec3,
    r_prop: wp.vec3
):
    tid = wp.tid()
    s = states[tid]
    ctrl = controls[tid]
    
    # 1. k1
    k1 = compute_full_dynamics_derivative(s, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    # 2. k2
    s2 = s
    s2.pos = s.pos + k1.d_pos * (dt * 0.5)
    s2.quat = wp.normalize(s.quat + k1.d_quat * (dt * 0.5))
    s2.vel_body = s.vel_body + k1.d_vel * (dt * 0.5)
    s2.omega_body = s.omega_body + k1.d_omega * (dt * 0.5)
    k2 = compute_full_dynamics_derivative(s2, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    # 3. k3
    s3 = s
    s3.pos = s.pos + k2.d_pos * (dt * 0.5)
    s3.quat = wp.normalize(s.quat + k2.d_quat * (dt * 0.5))
    s3.vel_body = s.vel_body + k2.d_vel * (dt * 0.5)
    s3.omega_body = s.omega_body + k2.d_omega * (dt * 0.5)
    k3 = compute_full_dynamics_derivative(s3, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    # 4. k4
    s4 = s
    s4.pos = s.pos + k3.d_pos * dt
    s4.quat = wp.normalize(s.quat + k3.d_quat * dt)
    s4.vel_body = s.vel_body + k3.d_vel * dt
    s4.omega_body = s.omega_body + k3.d_omega * dt
    k4 = compute_full_dynamics_derivative(s4, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    # Final Integration Update
    s_new = s
    s_new.pos = s.pos + (k1.d_pos + k2.d_pos * 2.0 + k3.d_pos * 2.0 + k4.d_pos) * (dt / 6.0)
    s_new.quat = wp.normalize(s.quat + (k1.d_quat + k2.d_quat * 2.0 + k3.d_quat * 2.0 + k4.d_quat) * (dt / 6.0))
    s_new.vel_body = s.vel_body + (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) * (dt / 6.0)
    s_new.omega_body = s.omega_body + (k1.d_omega + k2.d_omega * 2.0 + k3.d_omega * 2.0 + k4.d_omega) * (dt / 6.0)
    s_new.rpm = wp.max(0.1, s.rpm + (k1.d_rpm + k2.d_rpm * 2.0 + k3.d_rpm * 2.0 + k4.d_rpm) * (dt / 6.0))
    s_new.fuel_mass = s.fuel_mass + (k1.d_fuel + k2.d_fuel * 2.0 + k3.d_fuel * 2.0 + k4.d_fuel) * (dt / 6.0)
    
    # --- Observation Bridge (High-level telemetry) ---
    s_new.alt_ft = -s_new.pos[2] * 3.28084
    s_new.v_kts = wp.length(s_new.vel_body) * 1.94384
    s_new.euler_rad = wp.quat_to_rpy(s_new.quat) # Radians [roll, pitch, yaw]
    
    # Cleanup Aero Angles after step
    alpha, beta, v_mag = compute_aero_angles(s_new.vel_body)
    s_new.alpha = alpha
    s_new.beta = beta
    
    states[tid] = s_new
