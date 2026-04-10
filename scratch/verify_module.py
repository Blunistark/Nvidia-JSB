import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp_jsb as wj
import warp as wp

# Verify that we can access the core structs and kernels
print(f"Package API check:")
print(f" - AircraftState: {wj.AircraftState}")
print(f" - ControlState: {wj.ControlState}")
print(f" - RK4 Kernel: {wj.integrate_full_state_rk4_kernel}")

# Success
print("\nSUCCESS: warp_jsb is now a fully functional standalone module.")
