# Pioneer FDM Technical Usage Guide

Welcome to the **Pioneer FDM**—the high-performance, differentiable flight dynamics backend for NVIDIA Warp. This guide explains how to integrate the Pioneer physics engine into your Reinforcement Learning or simulation projects.

## 1. Installation

The easiest way to use Pioneer FDM as a module is to install it in "editable" mode:

```bash
cd Nvidia-JSB
pip install -e .
```

This allows you to `import warp_jsb` from any script on your system.

## 2. Basic Architecture

Pioneer FDM is designed to run **thousands or millions** of aircraft in parallel on a single GPU.

### Key Data Structures
*   `AircraftState`: 13-DOF state (pos, quat, vel, omega, engine_state).
*   `ControlState`: Input vector (aileron, elevator, rudder, flaps, throttle, mixture, brake, steer).
*   `AeroModelHandles`: Cached GPU buffers for aerodynamic look-up tables.

## 3. The "Manual Struct" Pattern (CRITICAL)

In NVIDIA Warp 1.12.1, you cannot use keyword arguments in struct constructors. You must use the **Instantiate-then-Assign** pattern:

```python
import warp_jsb as wj

# INCORRECT: Will throw TypeError
# cp = wj.ContactPoint(pos_body=...) 

# CORRECT: Manual Assignment
cp = wj.ContactPoint()
cp.pos_body = wp.vec3(0.0, 0.0, 0.0)
cp.spring_k = 1000.0
```

## 4. Initializing the GPU Simulation

### Step 1: Load Aero Data
You must load the aerodynamic tables once and pass the "Handles" to the kernel.

```python
from warp_jsb import AeroModelHandles
# ... populate handles with wp.from_numpy() from the /data directory
```

### Step 2: Allocate State Buffers
Initialize arrays for all your parallel agents.

```python
num_aircraft = 1000000 # Scaling to 1M agents
states = wp.zeros(num_aircraft, dtype=wj.AircraftState, device="cuda")
controls = wp.zeros(num_aircraft, dtype=wj.ControlState, device="cuda")
```

## 5. The Physics Loop

At each step, launch the **`integrate_full_state_rk4_kernel`**. This kernel handles aerodynamics, propulsion, control surfaces, and landing gear in a single GPU call.

```python
import warp as wp
from warp_jsb import integrate_full_state_rk4_kernel

# The integration step
wp.launch(
    kernel=integrate_full_state_rk4_kernel,
    dim=num_aircraft,
    inputs=[
        states,    # AircraftState array
        controls,  # ControlState array
        handles,   # AeroModelHandles
        contacts,  # ContactPoint array (landing gear)
        dt,        # Time step (e.g., 0.01)
        S, b, c,   # Ref area, span, chord
        rho,       # Air density
        p_amb,     # Ambient pressure (psf)
        r_aero,    # Aero moment arm (vec3)
        r_prop     # Prop moment arm (vec3)
    ],
    device="cuda"
)
```

## 6. Scaling for Performance

To achieve the benchmarked **200M+ steps/sec** throughput:
1.  **Avoid CPU-GPU transfers**: Keep your Reward and Observation logic in Warp kernels or use PyTorch GPU tensors.
2.  **Batch Actions**: Apply all agent actions to the `controls` array in one launch before the physics step.
3.  **Kernel Caching**: The first run will compile (taking ~20s); subsequent runs will be near-instant.

---
For a complete working example, see [examples/basic_flight.py](examples/basic_flight.py).
