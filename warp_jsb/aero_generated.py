import warp as wp
from warp_jsb.lut import sample_lut_1d, sample_lut_2d

# Generated Aerodynamic Model

@wp.struct
class AeroModelHandles:
    aero_coefficient_CDDf_table: wp.array(dtype=wp.float32)
    aero_coefficient_CDDf_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CDwbh_table: wp.array(dtype=wp.float32)
    aero_coefficient_CDwbh_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CYb_table: wp.array(dtype=wp.float32)
    aero_coefficient_CYb_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CYp_table: wp.array(dtype=wp.float32)
    aero_coefficient_CYp_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CYr_table: wp.array(dtype=wp.float32)
    aero_coefficient_CYr_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CLwbh_table: wp.array(dtype=wp.float32)
    aero_coefficient_CLwbh_meta: wp.array(dtype=wp.float32)
    aero_coefficient_CLDf_table: wp.array(dtype=wp.float32)
    aero_coefficient_CLDf_meta: wp.array(dtype=wp.float32)
    aero_coefficient_Clb_table: wp.array(dtype=wp.float32)
    aero_coefficient_Clb_meta: wp.array(dtype=wp.float32)
    aero_coefficient_Clr_table: wp.array(dtype=wp.float32)
    aero_coefficient_Clr_meta: wp.array(dtype=wp.float32)
    aero_coefficient_Cmdf_table: wp.array(dtype=wp.float32)
    aero_coefficient_Cmdf_meta: wp.array(dtype=wp.float32)
    aero_coefficient_Cnb_table: wp.array(dtype=wp.float32)
    aero_coefficient_Cnb_meta: wp.array(dtype=wp.float32)

@wp.func
def evaluate_aero_model(
    alpha: wp.float32,
    beta: wp.float32,
    qbar: wp.float32,
    p: wp.float32,
    q: wp.float32,
    r: wp.float32,
    bi2vel: wp.float32,
    ci2vel: wp.float32,
    h_mac: wp.float32,
    stall_hyst: wp.float32,
    elevator: wp.float32,
    aileron: wp.float32,
    rudder: wp.float32,
    flaps: wp.float32,
    handles: AeroModelHandles
):
    aero_coefficient_CDo = (qbar * 174.0 * 0.027)
    aero_coefficient_CDDf = (qbar * 174.0 * 1.0 * sample_lut_1d(handles.aero_coefficient_CDDf_table, handles.aero_coefficient_CDDf_meta, (flaps * 30.0)))
    aero_coefficient_CDwbh = (qbar * 174.0 * 1.0 * sample_lut_2d(handles.aero_coefficient_CDwbh_table, handles.aero_coefficient_CDwbh_meta, alpha, (flaps * 30.0)))
    aero_coefficient_CDDe = (qbar * 174.0 * wp.abs(elevator * 0.4) * 0.06)
    aero_coefficient_CDbeta = (qbar * 174.0 * wp.abs(beta) * 0.17)
    aero_coefficient_CYb = (qbar * 174.0 * sample_lut_2d(handles.aero_coefficient_CYb_table, handles.aero_coefficient_CYb_meta, beta, (flaps * 30.0)))
    aero_coefficient_CYda = (qbar * 174.0 * (aileron * 0.3) * 0.0)
    aero_coefficient_CYdr = (qbar * 174.0 * (rudder * 0.3) * 0.187)
    aero_coefficient_CYp = (qbar * 174.0 * bi2vel * p * sample_lut_2d(handles.aero_coefficient_CYp_table, handles.aero_coefficient_CYp_meta, alpha, (flaps * 30.0)))
    aero_coefficient_CYr = (qbar * 174.0 * bi2vel * r * sample_lut_2d(handles.aero_coefficient_CYr_table, handles.aero_coefficient_CYr_meta, alpha, (flaps * 30.0)))
    aero_coefficient_CLwbh = (qbar * 174.0 * 1.0 * sample_lut_2d(handles.aero_coefficient_CLwbh_table, handles.aero_coefficient_CLwbh_meta, alpha, stall_hyst))
    aero_coefficient_CLDf = (qbar * 174.0 * 1.0 * sample_lut_1d(handles.aero_coefficient_CLDf_table, handles.aero_coefficient_CLDf_meta, (flaps * 30.0)))
    aero_coefficient_CLDe = (qbar * 174.0 * (elevator * 0.4) * 0.43)
    aero_coefficient_CLadot = (qbar * 174.0 * 0.0 * ci2vel * 1.7)
    aero_coefficient_CLq = (qbar * 174.0 * q * ci2vel * 3.9)
    aero_coefficient_Clb = (qbar * 174.0 * 35.8 * sample_lut_1d(handles.aero_coefficient_Clb_table, handles.aero_coefficient_Clb_meta, beta))
    aero_coefficient_Clp = (qbar * 174.0 * 35.8 * bi2vel * p * -0.484)
    aero_coefficient_Clr = (qbar * 174.0 * 35.8 * bi2vel * r * sample_lut_2d(handles.aero_coefficient_Clr_table, handles.aero_coefficient_Clr_meta, alpha, (flaps * 30.0)))
    aero_coefficient_ClDa = (qbar * 174.0 * 35.8 * (aileron * 0.3) * 0.229)
    aero_coefficient_Cldr = (qbar * 174.0 * 35.8 * (rudder * 0.3) * 0.0147)
    aero_coefficient_Cmo = (qbar * 174.0 * 4.9 * 0.1)
    aero_coefficient_Cmalpha = (qbar * 174.0 * 4.9 * alpha * -1.8)
    aero_coefficient_Cmq = (qbar * 174.0 * 4.9 * ci2vel * q * -12.4)
    aero_coefficient_Cmadot = (qbar * 174.0 * 4.9 * ci2vel * 0.0 * -7.27)
    aero_coefficient_Cmde = (qbar * 174.0 * 4.9 * (elevator * 0.4) * -1.122)
    aero_coefficient_Cmdf = (qbar * 174.0 * 4.9 * sample_lut_1d(handles.aero_coefficient_Cmdf_table, handles.aero_coefficient_Cmdf_meta, (flaps * 30.0)))
    aero_coefficient_Cnb = (qbar * 174.0 * 35.8 * sample_lut_1d(handles.aero_coefficient_Cnb_table, handles.aero_coefficient_Cnb_meta, beta))
    aero_coefficient_Cnp = (qbar * 174.0 * 35.8 * bi2vel * p * -0.0278)
    aero_coefficient_Cnr = (qbar * 174.0 * 35.8 * bi2vel * r * -0.0937)
    aero_coefficient_Cnda = (qbar * 174.0 * 35.8 * (aileron * 0.3) * -0.0053)
    aero_coefficient_Cndr = (qbar * 174.0 * 35.8 * (rudder * 0.3) * -0.043)
    # Total Sums
    total_DRAG = aero_coefficient_CDo + aero_coefficient_CDDf + aero_coefficient_CDwbh + aero_coefficient_CDDe + aero_coefficient_CDbeta
    total_SIDE = aero_coefficient_CYb + aero_coefficient_CYda + aero_coefficient_CYdr + aero_coefficient_CYp + aero_coefficient_CYr
    total_LIFT = aero_coefficient_CLwbh + aero_coefficient_CLDf + aero_coefficient_CLDe + aero_coefficient_CLadot + aero_coefficient_CLq
    total_ROLL = aero_coefficient_Clb + aero_coefficient_Clp + aero_coefficient_Clr + aero_coefficient_ClDa + aero_coefficient_Cldr
    total_PITCH = aero_coefficient_Cmo + aero_coefficient_Cmalpha + aero_coefficient_Cmq + aero_coefficient_Cmadot + aero_coefficient_Cmde + aero_coefficient_Cmdf
    total_YAW = aero_coefficient_Cnb + aero_coefficient_Cnp + aero_coefficient_Cnr + aero_coefficient_Cnda + aero_coefficient_Cndr
    return total_DRAG, total_SIDE, total_LIFT, total_ROLL, total_PITCH, total_YAW