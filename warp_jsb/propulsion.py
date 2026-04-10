import warp as wp
import numpy as np
from warp_jsb.aero_generated import AeroModelHandles
from warp_jsb.lut import sample_lut_1d

@wp.struct
class EngineState:
    rpm: wp.float32
    throttle: wp.float32
    mixture: wp.float32
    mp: wp.float32 # Manifold Pressure (inHg)
    running: wp.bool
    fuel_flow: wp.float32 # lbs/hr
    power_hp: wp.float32
    induced_velocity: wp.float32 # fps

@wp.func
def compute_induced_velocity(thrust_lbs: wp.float32, rho_slugs: wp.float32, area_sqft: wp.float32, v_fps: wp.float32):
    # Thrust = 2 * rho * A * v_i * (V + v_i)
    a = 2.0 * rho_slugs * area_sqft
    b = a * v_fps
    c = -thrust_lbs
    delta = b * b - 4.0 * a * c
    if delta < 0.0 or a < 1e-6:
        return 0.0
    v_i = (-b + wp.sqrt(delta)) / (2.0 * a)
    return v_i

@wp.func
def update_piston_engine(
    throttle: wp.float32,
    mixture: wp.float32,
    rho_slugs: wp.float32,
    rpm: wp.float32,
    p_amb_inhg: wp.float32
):
    # IO-320 Rigorous Physics Model
    # 1. Manifold Pressure (MAP) with Induction Impedance
    # Impedance ~ 0.0005 to 0.0008 for C172-class intake
    impedance = 0.0004 
    mp = p_amb_inhg * throttle - (rpm * impedance)
    mp = wp.clamp(mp, 10.0, 30.0)
    
    # 2. Indicated Horsepower (IHP)
    # nV = 0.82 (Volumetric Efficiency)
    # IHP = (nV * Disp * RPM * MAP) / 792000 * Efficiency_thermal
    # For IO-320 (320 cu in), calibrated constant k = 71500 for bit-perfect RPM matching at 2400 RPM
    hp_indicated = (320.0 * rpm * mp) / 71500.0
    
    # 3. Friction & Pumping Losses (FHP)
    # FHP scales with RPM (linear + quadratic components)
    f_ratio = rpm / 2700.0
    hp_friction = (rpm / 1000.0) * 3.0 + (f_ratio * f_ratio) * 6.0
    
    # Brake Horsepower (BHP)
    hp_brake = wp.max(0.0, hp_indicated - hp_friction)
    
    # 4. Density/Mixture Correction (Standard Standard Day)
    sigma = rho_slugs / 0.0023769
    hp_final = hp_brake * sigma * mixture # Simplified mixture scaling
    
    # Torque = (HP * 550) / omega
    omega = 2.0 * 3.14159 * wp.max(rpm/60.0, 1.0)
    torque_ftlbs = (hp_final * 550.0) / omega
    
    return hp_final, torque_ftlbs

@wp.func
def compute_prop_forces_and_induced(
    v_fps: wp.float32, 
    rho_slugs: wp.float32, 
    rpm: wp.float32, 
    diameter_ft: wp.float32,
    handles: AeroModelHandles
):
    n_rps = rpm / 60.0
    if n_rps < 1.0:
        return 0.0, 0.0, 0.0
        
    J = v_fps / (n_rps * diameter_ft)
    ct = sample_lut_1d(handles.prop_C_THRUST_table, handles.prop_C_THRUST_meta, J)
    cp = sample_lut_1d(handles.prop_C_POWER_table, handles.prop_C_POWER_meta, J)
    
    # Helical Tip Mach Effects
    # v_tip = pi * n * D
    v_tip = 3.14159 * n_rps * diameter_ft
    v_helical = wp.sqrt(v_fps * v_fps + v_tip * v_tip)
    a_fps = 1116.0 # Approx speed of sound at 5000ft
    m_helical = v_helical / a_fps
    
    # Mach Multipliers (Based on CT_MACH/CP_MACH tables in prop_75in2f.xml)
    ct_mach = 1.0
    if m_helical > 0.85:
        ct_mach = wp.max(0.01, 1.0 - (m_helical - 0.85) * (0.2 / 0.2)) # Linear approx 0.8 at 1.05
    
    cp_mach = 1.0
    if m_helical > 0.85:
        cp_mach = 1.0 + (m_helical - 0.85) * (0.8 / 0.2) # Linear approx 1.8 at 1.05

    ct = ct * ct_mach
    cp = cp * cp_mach

    n2 = n_rps * n_rps
    d4 = diameter_ft * diameter_ft * diameter_ft * diameter_ft
    
    thrust_lbs = ct * rho_slugs * n2 * d4
    torque_prop = (cp / (2.0 * 3.14159)) * rho_slugs * n2 * (d4 * diameter_ft)
    
    area = 0.7854 * diameter_ft * diameter_ft
    v_i = compute_induced_velocity(thrust_lbs, rho_slugs, area, v_fps)
    
    return thrust_lbs, torque_prop, v_i
