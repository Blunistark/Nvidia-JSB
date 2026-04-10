import warp as wp
import numpy as np

@wp.struct
class EngineState:
    rpm: wp.float32
    throttle: wp.float32
    mixture: wp.float32
    running: wp.bool
    fuel_flow: wp.float32 # kg/s

@wp.func
def calculate_propeller_forces(
    J: wp.float32, 
    rho: wp.float32, 
    rpm: wp.float32, 
    diameter: wp.float32,
    ct_table: wp.array(dtype=wp.float32), # 1D table for Ct vs J
    ct_meta: wp.array(dtype=wp.float32),
    cp_table: wp.array(dtype=wp.float32), # 1D table for Cp vs J
    cp_meta: wp.array(dtype=wp.float32)
):
    from warp_jsb.lut import sample_lut_1d
    
    n = rpm / 60.0 # rev/s
    if n < 1.0:
        return wp.vec3(0.0, 0.0, 0.0), 0.0 # No thrust if not spinning
        
    ct = sample_lut_1d(ct_table, ct_meta, J)
    cp = sample_lut_1d(cp_table, cp_meta, J)
    
    # Thrust = Ct * rho * n^2 * D^4
    thrust_mag = ct * rho * (n * n) * (diameter * diameter * diameter * diameter)
    
    # Power = Cp * rho * n^3 * D^5
    # Torque = Power / (2 * pi * n) = (Cp * rho * n^2 * D^5) / (2 * pi)
    torque_mag = (cp * rho * (n * n) * (diameter * diameter * diameter * diameter * diameter)) / (2.0 * 3.14159)
    
    return wp.vec3(thrust_mag, 0.0, 0.0), torque_mag

@wp.kernel
def update_engine_kernel(
    engines: wp.array(dtype=EngineState),
    dt: wp.float32
):
    tid = wp.tid()
    # Simple RPM logic: map throttle to target RPM
    target_rpm = engines[tid].throttle * 2700.0
    engines[tid].rpm = wp.lerp(engines[tid].rpm, target_rpm, 0.1) # Simple lag
