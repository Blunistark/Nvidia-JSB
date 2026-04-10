import warp as wp
from warp_jsb.eom import AircraftState, ControlState
from warp_jsb.aero_generated import evaluate_aero_model, AeroModelHandles

@wp.kernel
def compute_full_aero_forces_kernel(
    states: wp.array(dtype=AircraftState),
    forces_body: wp.array(dtype=wp.vec3),
    moments_body: wp.array(dtype=wp.vec3),
    controls: wp.array(dtype=ControlState),
    handles: AeroModelHandles,
    S: wp.float32,
    b: wp.float32,
    c: wp.float32
):
    tid = wp.tid()
    s = states[tid]
    ctrl = controls[tid]
    
    # 1. Calc Dynamic Pressure (SI -> Pa)
    rho = 1.225 # kg/m^3
    v_mag = wp.length(s.vel_body)
    qbar_pa = 0.5 * rho * v_mag * v_mag
    
    # 2. Convert SI units to JSBSim (Imperial) for the generated model
    # The generated model expects qbar in psf and returns Forces in lbs, Moments in ft-lbs
    qbar_psf = qbar_pa * 0.0208854
    
    # 3. Evaluate all 35+ functions (Returns Imperial units)
    D_lbs, Y_lbs, L_lbs, l_ftlbs, m_ftlbs, n_ftlbs = evaluate_aero_model(
        s.alpha, s.beta, qbar_psf,
        s.omega_body[0], s.omega_body[1], s.omega_body[2],
        s.bi2vel, s.ci2vel, s.h_mac, s.stall_hyst,
        ctrl.elevator, ctrl.aileron, ctrl.rudder, ctrl.flaps,
        handles
    )
    
    # 4. Convert Imperial results back to SI for the Integrator
    # 1 lb = 4.44822 N
    # 1 ft-lb = 1.35582 Nm
    D = D_lbs * 4.44822
    Y = Y_lbs * 4.44822
    L = L_lbs * 4.44822
    row_mom = l_ftlbs * 1.35582
    pit_mom = m_ftlbs * 1.35582
    yaw_mom = n_ftlbs * 1.35582
    
    # 5. Transform Stability to Body
    fx = L * wp.sin(s.alpha) - D * wp.cos(s.alpha)
    fz = -L * wp.cos(s.alpha) - D * wp.sin(s.alpha)
    fy = Y
    
    forces_body[tid] = wp.vec3(fx, fy, fz)
    moments_body[tid] = wp.vec3(row_mom, pit_mom, yaw_mom)
