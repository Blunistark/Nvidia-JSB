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
    
    # Forensic Telemetry for DRL Parity
    accel_body: wp.vec3 # (u_dot, v_dot, w_dot) in fps^2
    vel_ned: wp.vec3    # (v_north, v_east, v_down) in fps

@wp.struct
class StateDeriv:
    d_pos: wp.vec3
    d_quat: wp.quat
    d_vel: wp.vec3
    d_omega: wp.vec3
    d_fuel: wp.float32
    d_rpm: wp.float32

@wp.func
def quat_rotate_vector(q: wp.quat, v: wp.vec3):
    # Standard quat-vector rotation: v' = q * v * q_inv
    qv = wp.vec3(q[0], q[1], q[2])
    qw = q[3]
    t = wp.cross(qv, v) * 2.0
    return v + t * qw + wp.cross(qv, t)

@wp.func
def compute_aero_angles(vel_body: wp.vec3):
    u = vel_body[0]
    v = vel_body[1]
    w = vel_body[2]
    v_mag = wp.length(vel_body)
    
    alpha = 0.0
    if u > 0.001 or u < -0.001:
        alpha = wp.atan2(w, u)
    
    beta = 0.0
    if v_mag > 0.001:
        beta = wp.asin(v / v_mag)
        
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
    # 1. Kinematics
    v_ned = quat_rotate_vector(s.quat, s.vel_body)
    omega_quad = wp.quat(s.omega_body[0], s.omega_body[1], s.omega_body[2], 0.0)
    dq_dt = s.quat * omega_quad * 0.5

    # 2. Aero Angles & Rates
    alpha, beta, v_mag = compute_aero_angles(s.vel_body)
    q_bar = 0.5 * rho * v_mag * v_mag
    v_mag_fps = v_mag * 3.28084
    
    bi2vel = 0.0
    ci2vel = 0.0
    if v_mag_fps > 0.1:
        bi2vel = b / (2.0 * v_mag_fps)
        ci2vel = c / (2.0 * v_mag_fps)

    # 3. Aero Model (15-parameter synchronization)
    drag, side, lift, roll, pitch, yaw = evaluate_aero_model(
        alpha, beta, q_bar, q_bar, 
        s.omega_body[0], s.omega_body[1], s.omega_body[2],
        bi2vel, ci2vel, c, s.stall_hyst,
        ctrl.elevator, ctrl.aileron, ctrl.rudder, ctrl.flaps,
        handles
    )
    f_aero = wp.vec3(-drag, side, -lift)
    m_aero = wp.vec3(roll, pitch, yaw)

    # 4. Propulsion (Dual Phase Engine -> Prop)
    p_amb_inhg = p_amb * 0.014139 # Psf to inHg
    hp_engine, torque_engine = update_piston_engine(
        ctrl.throttle, ctrl.mixture, rho, s.rpm, p_amb_inhg
    )
    
    diameter_ft = 6.33
    thrust, torque_prop, v_i = compute_prop_forces_and_induced(
        v_mag_fps, rho, s.rpm, diameter_ft, handles
    )
    
    f_prop = wp.vec3(thrust, 0.0, 0.0)
    m_prop = wp.vec3(-torque_prop, 0.0, 0.0)

    # 5. External - Ground Reactions
    f_ext = wp.vec3(0.0, 0.0, 0.0)
    m_ext = wp.vec3(0.0, 0.0, 0.0)
    for i in range(len(contacts)):
        f, m = compute_single_contact_force(
            contacts[i], s.pos, s.quat, s.vel_body, s.omega_body, ctrl.brake, ctrl.steer
        )
        f_ext = f_ext + f
        m_ext = m_ext + m
    
    # 6. Gravity (Body Frame)
    g_ned = wp.vec3(0.0, 0.0, 9.80665)
    g_body = quat_rotate_vector(wp.quat_inverse(s.quat), g_ned)

    # 7. EOM
    d_vel = (f_aero + f_prop + f_ext) / s.mass - wp.cross(s.omega_body, s.vel_body) + g_body
    d_omega = s.inertia_inv * (m_aero + m_prop + m_ext - wp.cross(s.omega_body, s.inertia * s.omega_body))

    deriv = StateDeriv()
    deriv.d_pos = v_ned
    deriv.d_quat = dq_dt
    deriv.d_vel = d_vel
    deriv.d_omega = d_omega
    deriv.d_fuel = 0.0 
    deriv.d_rpm = (torque_engine - torque_prop) / 0.167 * (60.0 / (2.0 * 3.14159)) 
    
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
    
    # RK4 Stages
    k1 = compute_full_dynamics_derivative(s, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    s2 = s
    s2.pos = s.pos + k1.d_pos * (dt * 0.5)
    s2.quat = wp.normalize(s.quat + k1.d_quat * (dt * 0.5))
    s2.vel_body = s.vel_body + k1.d_vel * (dt * 0.5)
    s2.omega_body = s.omega_body + k1.d_omega * (dt * 0.5)
    k2 = compute_full_dynamics_derivative(s2, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    s3 = s
    s3.pos = s.pos + k2.d_pos * (dt * 0.5)
    s3.quat = wp.normalize(s.quat + k2.d_quat * (dt * 0.5))
    s3.vel_body = s.vel_body + k2.d_vel * (dt * 0.5)
    s3.omega_body = s.omega_body + k2.d_omega * (dt * 0.5)
    k3 = compute_full_dynamics_derivative(s3, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    s4 = s
    s4.pos = s.pos + k3.d_pos * dt
    s4.quat = wp.normalize(s.quat + k3.d_quat * dt)
    s4.vel_body = s.vel_body + k3.d_vel * dt
    s4.omega_body = s.omega_body + k3.d_omega * dt
    k4 = compute_full_dynamics_derivative(s4, ctrl, handles, S, b, c, rho, contacts, p_amb, r_aero, r_prop)
    
    # 5. Final State Update
    s.pos = s.pos + (k1.d_pos + k2.d_pos * 2.0 + k3.d_pos * 2.0 + k4.d_pos) * (dt / 6.0)
    s.quat = wp.normalize(s.quat + (k1.d_quat + k2.d_quat * 2.0 + k3.d_quat * 2.0 + k4.d_quat) * (dt / 6.0))
    s.vel_body = s.vel_body + (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) * (dt / 6.0)
    s.omega_body = s.omega_body + (k1.d_omega + k2.d_omega * 2.0 + k3.d_omega * 2.0 + k4.d_omega) * (dt / 6.0)
    s.rpm = wp.max(0.1, s.rpm + (k1.d_rpm + k2.d_rpm * 2.0 + k3.d_rpm * 2.0 + k4.d_rpm) * (dt / 6.0))
    s.fuel_mass = s.fuel_mass + (k1.d_fuel + k2.d_fuel * 2.0 + k3.d_fuel * 2.0 + k4.d_fuel) * (dt / 6.0)
    
    # NEW: Store Forensic Telemetry (Use k4 or smoothed average)
    s.accel_body = (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) / 6.0 * 3.28084 # fps^2
    s.vel_ned = quat_rotate_vector(s.quat, s.vel_body) * 3.28084 # fps
    
    # Properties for DRL
    s.alt_ft = s.pos[2] * -3.28084
    s.v_kts = wp.length(s.vel_body) * 1.94384
    s.euler_rad = wp.quat_to_rpy(s.quat)
    
    # Cleanup Aero Angles
    a, b, v_mag = compute_aero_angles(s.vel_body)
    s.alpha = a
    s.beta = b
    
    states[tid] = s
