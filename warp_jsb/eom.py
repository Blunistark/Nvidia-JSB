import warp as wp
import numpy as np
from warp_jsb.aero_generated import evaluate_aero_model, AeroModelHandles

@wp.struct
class ControlState:
    aileron: wp.float32
    elevator: wp.float32
    rudder: wp.float32
    flaps: wp.float32
    throttle: wp.float32

@wp.struct
class AircraftState:
    # Rigid body state (SI)
    pos: wp.vec3
    quat: wp.quat
    vel_body: wp.vec3
    omega_body: wp.vec3
    
    # Mass properties
    mass: wp.float32
    inertia: wp.mat33
    inertia_inv: wp.mat33
    
    # Aerodynamic state
    alpha: wp.float32
    beta: wp.float32
    qbar: wp.float32
    
    # Non-dimensionalizations
    bi2vel: wp.float32
    ci2vel: wp.float32
    h_mac: wp.float32
    
    stall_hyst: wp.float32

@wp.struct
class StateDeriv:
    d_pos: wp.vec3
    d_quat: wp.quat
    d_vel: wp.vec3
    d_omega: wp.vec3

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
def compute_dynamics_derivative(
    pos: wp.vec3,
    quat: wp.quat,
    vel_body: wp.vec3,
    omega_body: wp.vec3,
    mass: wp.float32,
    inertia: wp.mat33,
    inertia_inv: wp.mat33,
    ctrl: ControlState,
    handles: AeroModelHandles,
    S: wp.float32, b: wp.float32, c: wp.float32,
    stall_hyst: wp.float32,
    rho: wp.float32
) -> StateDeriv:
    
    # 1. Kinematics
    v_ned = wp.quat_rotate(quat, vel_body)
    dq_dt = 0.5 * wp.mul(quat, wp.quat(omega_body[0], omega_body[1], omega_body[2], 0.0))
    
    # 2. Aero Properties (Intermediate)
    alpha, beta, v_mag = compute_aero_angles(vel_body)
    # Correct Dynamic Pressure using passed rho
    qbar_pa = 0.5 * rho * v_mag * v_mag
    qbar_psf = qbar_pa * 0.0208854
    
    inv_v = 1.0 / wp.max(v_mag, 0.1)
    bi2vel = b * 0.5 * inv_v
    ci2vel = c * 0.5 * inv_v
    h_mac = wp.max(0.0, -pos[2]) / c
    
    # 3. Aerodynamics (Returns Imperial units)
    D_lbs, Y_lbs, L_lbs, l_ftlbs, m_ftlbs, n_ftlbs = evaluate_aero_model(
        alpha, beta, qbar_psf,
        omega_body[0], omega_body[1], omega_body[2],
        bi2vel, ci2vel, h_mac, stall_hyst,
        ctrl.elevator, ctrl.aileron, ctrl.rudder, ctrl.flaps,
        handles
    )
    
    # Convert to SI
    force_aero_b = wp.vec3(
        (L_lbs * wp.sin(alpha) - D_lbs * wp.cos(alpha)) * 4.44822,
        Y_lbs * 4.44822,
        (-L_lbs * wp.cos(alpha) - D_lbs * wp.sin(alpha)) * 4.44822
    )
    torque_aero_b = wp.vec3(l_ftlbs * 1.35582, m_ftlbs * 1.35582, n_ftlbs * 1.35582)
    
    # 4. Total Forces (Gravity + Aero)
    gravity_ned = wp.vec3(0.0, 0.0, 9.80665)
    gravity_body = wp.quat_rotate(wp.quat_inverse(quat), gravity_ned)
    
    accel_body = (force_aero_b / mass) + gravity_body
    coriolis = wp.cross(omega_body, vel_body)
    d_vel = accel_body - coriolis
    
    # 5. Total Torques
    i_omega = inertia * omega_body
    d_omega = inertia_inv * (torque_aero_b - wp.cross(omega_body, i_omega))
    
    deriv = StateDeriv()
    deriv.d_pos = v_ned
    deriv.d_quat = dq_dt
    deriv.d_vel = d_vel
    deriv.d_omega = d_omega
    return deriv

@wp.kernel
def integrate_state_rk4_kernel(
    states: wp.array(dtype=AircraftState),
    controls: wp.array(dtype=ControlState),
    handles: AeroModelHandles,
    dt: wp.float32,
    S: wp.float32, b: wp.float32, c: wp.float32,
    rho: wp.float32  # Added rho for accurate integration
):
    tid = wp.tid()
    s = states[tid]
    ctrl = controls[tid]
    
    # RK4 Step
    k1 = compute_dynamics_derivative(s.pos, s.quat, s.vel_body, s.omega_body, s.mass, s.inertia, s.inertia_inv, ctrl, handles, S, b, c, s.stall_hyst, rho)
    
    k2 = compute_dynamics_derivative(
        s.pos + k1.d_pos * (dt * 0.5),
        wp.normalize(s.quat + k1.d_quat * (dt * 0.5)),
        s.vel_body + k1.d_vel * (dt * 0.5),
        s.omega_body + k1.d_omega * (dt * 0.5),
        s.mass, s.inertia, s.inertia_inv, ctrl, handles, S, b, c, s.stall_hyst, rho
    )
    
    k3 = compute_dynamics_derivative(
        s.pos + k2.d_pos * (dt * 0.5),
        wp.normalize(s.quat + k2.d_quat * (dt * 0.5)),
        s.vel_body + k2.d_vel * (dt * 0.5),
        s.omega_body + k2.d_omega * (dt * 0.5),
        s.mass, s.inertia, s.inertia_inv, ctrl, handles, S, b, c, s.stall_hyst, rho
    )
    
    k4 = compute_dynamics_derivative(
        s.pos + k3.d_pos * dt,
        wp.normalize(s.quat + k3.d_quat * dt),
        s.vel_body + k3.d_vel * dt,
        s.omega_body + k3.d_omega * dt,
        s.mass, s.inertia, s.inertia_inv, ctrl, handles, S, b, c, s.stall_hyst, rho
    )
    
    # Final state update
    pos_new = s.pos + (k1.d_pos + k2.d_pos * 2.0 + k3.d_pos * 2.0 + k4.d_pos) * (dt / 6.0)
    quat_new = wp.normalize(s.quat + (k1.d_quat + k2.d_quat * 2.0 + k3.d_quat * 2.0 + k4.d_quat) * (dt / 6.0))
    vel_new = s.vel_body + (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) * (dt / 6.0)
    omega_new = s.omega_body + (k1.d_omega + k2.d_omega * 2.0 + k3.d_omega * 2.0 + k4.d_omega) * (dt / 6.0)
    
    # Update properties
    alpha, beta, v_mag = compute_aero_angles(vel_new)
    inv_v = 1.0 / wp.max(v_mag, 0.1)
    
    states[tid].pos = pos_new
    states[tid].quat = quat_new
    states[tid].vel_body = vel_new
    states[tid].omega_body = omega_new
    states[tid].alpha = alpha
    states[tid].beta = beta
    states[tid].bi2vel = b * 0.5 * inv_v
    states[tid].ci2vel = c * 0.5 * inv_v
    states[tid].h_mac = wp.max(0.0, -pos_new[2]) / c
