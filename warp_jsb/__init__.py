import os
from .eom import AircraftState, ControlState, integrate_full_state_rk4_kernel
from .propulsion import update_piston_engine, compute_prop_forces_and_induced
from .aero_generated import evaluate_aero_model, AeroModelHandles
from .experience import ExperienceHarvester
from .curriculum import DynamicSkillGenerator

__version__ = "0.1.0"

def get_c172p_assets_path():
    """Returns the absolute path to the bundled C172P assets."""
    return os.path.join(os.path.dirname(__file__), "assets", "c172p")
