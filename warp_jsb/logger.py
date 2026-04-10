import warp as wp
import numpy as np
from warp_jsb.eom import AircraftState

@wp.struct
class FleetMetrics:
    avg_alt_ft: wp.float32
    avg_v_kts: wp.float32
    max_alt_ft: wp.float32
    min_alt_ft: wp.float32
    max_v_kts: wp.float32
    std_alt_ft: wp.float32

@wp.kernel
def aggregate_fleet_metrics_kernel(
    states: wp.array(dtype=AircraftState), 
    summary: wp.array(dtype=wp.float32) # [sum_alt, sum_v, sum_alt_sq, max_alt, min_alt, max_v]
):
    tid = wp.tid()
    s = states[tid]
    
    alt = s.alt_ft
    v = s.v_kts
    
    # 1. Global Reductions (Sums for Mean)
    wp.atomic_add(summary, 0, alt)
    wp.atomic_add(summary, 1, v)
    wp.atomic_add(summary, 2, alt * alt) # For Variance/STD
    
    # 2. Min/Max Tracking
    wp.atomic_max(summary, 3, alt)
    wp.atomic_min(summary, 4, alt)
    wp.atomic_max(summary, 5, v)

class FleetLogger:
    def __init__(self, num_aircraft, device="cuda"):
        self.num_aircraft = num_aircraft
        self.device = device
        # [sum_alt, sum_v, sum_alt_sq, max_alt, min_alt, max_v]
        self.summary_data = wp.zeros(6, dtype=wp.float32, device=device)
        
    def reset(self):
        self.summary_data.zero_()
        
    def compute(self, states_array):
        self.reset()
        wp.launch(
            kernel=aggregate_fleet_metrics_kernel,
            dim=self.num_aircraft,
            inputs=[states_array, self.summary_data],
            device=self.device
        )
        
        # Pull 6 floats to CPU
        stats = self.summary_data.numpy()
        
        n = float(self.num_aircraft)
        mean_alt = stats[0] / n
        mean_v = stats[1] / n
        
        # Variance = Mean(X^2) - Mean(X)^2
        var_alt = (stats[2] / n) - (mean_alt * mean_alt)
        std_alt = np.sqrt(max(0.0, var_alt))
        
        return {
            "mean_alt_ft": mean_alt,
            "mean_v_kts": mean_v,
            "max_alt_ft": stats[3],
            "min_alt_ft": stats[4],
            "max_v_kts": stats[5],
            "std_alt_ft": std_alt
        }
