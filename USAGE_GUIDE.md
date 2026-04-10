# Pioneer FDM Technical Usage Guide

Welcome to the **Pioneer FDM**—the high-performance, differentiable flight dynamics backend for NVIDIA Warp. This guide explains how to integrate the Pioneer physics engine into your Reinforcement Learning or simulation projects.

## 1. Installation

The easiest way to use Pioneer FDM as a module is to install it in "editable" mode:

```bash
cd Nvidia-JSB
pip install -e .
```

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

# CORRECT: Manual Assignment
cp = wj.ContactPoint()
cp.pos_body = wp.vec3(0.0, 0.0, 0.0)
cp.spring_k = 1000.0
```

## 4. Initializing the GPU Simulation

### Step 1: Load Aero Data
```python
from warp_jsb import AeroModelHandles
# ... populate handles with wp.from_numpy() from the /data directory
```

### Step 2: Allocate State Buffers
```python
num_aircraft = 1000000 
states = wp.zeros(num_aircraft, dtype=wj.AircraftState, device="cuda")
controls = wp.zeros(num_aircraft, dtype=wj.ControlState, device="cuda")
```

## 5. Parallel Telemetry & Logging

### Fleet Dashboard
To compute fleet-wide statistics (mean altitude, max airspeed) without moving data to the CPU:

```python
from warp_jsb.logger import FleetLogger
logger = FleetLogger(num_aircraft)
stats = logger.compute(states) # Returns a dict of mean/std/min/max
```

## 6. Experience Harvesting (DRL Datasets)

The **`ExperienceHarvester`** is a high-speed sequence recorder designed for Offline RL and Behavior Cloning.

```python
from warp_jsb.experience import ExperienceHarvester

# Setup a 10-step Time-Series buffer
harvester = ExperienceHarvester(
    num_aircraft, 
    window_size=10, 
    layout="agent_first", # (A, T, F) or "feature_first" (F, A, T)
    sync_mode=True        # Global sequence tracking
)

# In your simulation loop:
# physics.step()...
harvester.record(states, controls)
```

### Layout Optimization
- **Agent-First (`A, T, F`)**: Best for **Real-time Recording** (Speed: ~545M samples/sec).
- **Feature-First (`F, A, T`)**: Best for **DRL Training** (Optimized for RNN/Transformer inputs).

### Asynchronous Resets
If your aircraft crash and reset independently, use `sync_mode=False`. This allows each agent to track its own circular history:

```python
# Reset the history for agent 42
mask = wp.zeros(num_aircraft, dtype=wp.bool, device="cuda")
mask[42] = True
harvester.reset_agents(mask)
```

## 7. Performance Benchmarks (RTX 4060)

| Mode | Throughput (Samples/Sec) |
| :--- | :--- |
| **Physics (RK4)** | ~146,000,000 |
| **Experience (Sync)** | ~545,000,000 |
| **Experience (Async)**| ~256,000,000 |

---
For a complete working example, see [examples/basic_flight.py](examples/basic_flight.py) and [examples/benchmark_harvester.py](examples/benchmark_harvester.py).
