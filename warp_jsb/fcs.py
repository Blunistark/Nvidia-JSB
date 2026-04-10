import warp as wp

@wp.func
def compute_fcs_components(
    pitch_cmd: wp.float32, pitch_trim: wp.float32,
    roll_cmd: wp.float32, roll_trim: wp.float32,
    yaw_cmd: wp.float32, yaw_trim: wp.float32,
    flap_cmd: wp.float32,
    dt: wp.float32
):
    # 1. Pitch Channel
    pitch_sum = wp.clamp(pitch_cmd + pitch_trim, -1.0, 1.0)
    # Range: -28 to 23 degrees. Gain: 0.01745 (deg to rad approx)
    elevator_pos = pitch_sum * 0.01745 * (28.0 if pitch_sum < 0.0 else 23.0)
    # Wait, the XML says <gain>0.01745</gain> <range><min>-28</min><max>23</max></range>
    # In JSBSim, the aerosurface_scale works by mapping [-1, 1] to the range.
    
    # Correct scale logic:
    # If sum > 0, map [0, 1] to [0, max]
    # If sum < 0, map [-1, 0] to [min, 0]
    elev_deg = 0.0
    if pitch_sum > 0.0:
        elev_deg = pitch_sum * 23.0
    else:
        elev_deg = pitch_sum * 28.0
    elevator_rad = elev_deg * 0.0174533
    
    # 2. Roll Channel
    roll_sum = wp.clamp(roll_cmd + roll_trim, -1.0, 1.0)
    # Aileron Left: -20 to 15
    # Aileron Right: -15 to 20 (inverted)
    aileron_deg = 0.0
    if roll_sum > 0.0:
        aileron_deg = roll_sum * 15.0
    else:
        aileron_deg = roll_sum * 20.0
    aileron_rad = aileron_deg * 0.0174533
    
    # 3. Yaw Channel
    yaw_sum = wp.clamp(yaw_cmd + yaw_trim, -1.0, 1.0)
    rudder_deg = yaw_sum * 16.0
    rudder_rad = rudder_deg * 0.0174533
    
    # 4. Flaps (Kinematic Delay missing here, using direct for now)
    flaps_deg = flap_cmd * 30.0
    
    return elevator_rad, aileron_rad, rudder_rad, flaps_deg / 30.0

@wp.struct
class FCSState:
    elevator_pos_rad: wp.float32
    aileron_pos_rad: wp.float32
    rudder_pos_rad: wp.float32
    flaps_pos_norm: wp.float32
