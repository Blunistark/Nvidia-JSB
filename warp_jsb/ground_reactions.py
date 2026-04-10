import warp as wp

@wp.struct
class ContactPoint:
    pos_body: wp.vec3
    is_bogey: wp.bool
    spring_k: wp.float32
    damping_c: wp.float32
    static_friction: wp.float32
    dynamic_friction: wp.float32
    max_steer: wp.float32 # for nose wheel

@wp.func
def compute_single_contact_force(
    c: ContactPoint,
    state_pos: wp.vec3,
    state_quat: wp.quat,
    state_vel_body: wp.vec3,
    state_omega_body: wp.vec3,
    brake_cmd: wp.float32,
    steer_cmd: wp.float32
):
    # 1. Transform contact point to NED
    p_ned = state_pos + wp.quat_rotate(state_quat, c.pos_body)
    v_ned_cp = wp.quat_rotate(state_quat, state_vel_body + wp.cross(state_omega_body, c.pos_body))
    
    # 2. Compression (NED Z is Down, so -Z is up)
    # Height above ground (assume flat ground at Z=0)
    z_ground = 0.0
    compression = wp.max(0.0, p_ned[2] - z_ground)
    
    if compression <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
        
    # 3. Vertical Force (Spring/Damping)
    # v_z in NED is positive down. 
    # Positive compression means p_ned[2] > 0.
    # F_z (Up) = k*comp + c*v_z_ned
    v_z = v_ned_cp[2]
    f_z_ned = -(c.spring_k * compression + c.damping_c * v_z)
    
    # Clamp to positive only (ground can only push up)
    if f_z_ned > 0.0: f_z_ned = 0.0
    
    # 4. Friction (Simplified for now)
    # Project NED velocity to ground plane (X, Y)
    v_slip = wp.vec3(v_ned_cp[0], v_ned_cp[1], 0.0)
    v_slip_mag = wp.length(v_slip)
    
    f_fric_ned = wp.vec3(0.0, 0.0, 0.0)
    if v_slip_mag > 0.01:
        # F_friction = mu * F_normal
        mu = c.static_friction if v_slip_mag < 0.1 else c.dynamic_friction
        if c.is_bogey:
            mu += brake_cmd * 0.5 # Braking adds friction
            
        f_norm = -f_z_ned
        f_fric_mag = mu * f_norm
        f_fric_ned = -wp.normalize(v_slip) * f_fric_mag
        
    total_f_ned = wp.vec3(f_fric_ned[0], f_fric_ned[1], f_z_ned)
    
    # Transform back to Body Frame
    f_body = wp.quat_rotate(wp.quat_inverse(state_quat), total_f_ned)
    torque_body = wp.cross(c.pos_body, f_body)
    
    return f_body, torque_body
