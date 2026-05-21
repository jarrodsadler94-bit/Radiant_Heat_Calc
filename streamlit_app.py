import streamlit as st
import math
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ============================================================
# CONSTANTS
# ============================================================

SIGMA_KW = 5.67e-11  # Stefan-Boltzmann constant, kW/m².K⁴
SIGMA_W = 5.67e-8    # Stefan-Boltzmann constant, W/m².K⁴

MAX_REASONABLE_TEMP_K = 2500.0
APP_VERSION = "1.1.0"

# ============================================================
# MATERIAL PRESETS
# cp is in kJ/kg.K for user input consistency.
# k is in W/m.K.
# ============================================================

MATERIALS = {
    "Custom": {"rho": 7850.0, "cp": 0.49, "k": 45.0},
    "Carbon Steel": {"rho": 7850.0, "cp": 0.49, "k": 45.0},
    "Aluminium (Alloy)": {"rho": 2770.0, "cp": 0.88, "k": 175.0},
    "Copper": {"rho": 8960.0, "cp": 0.39, "k": 400.0},
    "Window Glass": {"rho": 2500.0, "cp": 0.84, "k": 1.0},
    "Concrete (Normal)": {"rho": 2300.0, "cp": 0.88, "k": 1.4},
    "Gypsum Board": {"rho": 800.0, "cp": 1.09, "k": 0.25},
    "Timber (Softwood)": {"rho": 500.0, "cp": 1.63, "k": 0.12},
}

THRESHOLDS = {
    4.7: "Low-level exposure reference",
    12.6: "Common injury / ignition screening reference",
    23.0: "High radiant heat exposure reference",
    35.0: "Severe exposure / equipment damage screening reference",
}

# ============================================================
# SAVED CALCULATION STATE
# ============================================================

DEFAULT_CONFIG = {
    # Source inputs
    "Ts_C": 1000.0,
    "W": 6.1,
    "H": 2.8,
    "eps_s": 1.0,
    "Zs_base": 0.0,

    # Receiver location inputs
    "L": 3.0,
    "Zr": 1.5,
    "Xr": 0.0,
    "theta_deg": 0.0,
    "Tt_initial_C": 20.0,
    "eps_t": 0.90,

    # Radiation options
    "Tamb_radiation_C": 20.0,
    "subtract_ambient": False,

    # Thermally thin inputs
    "Tamb_C_thin": 20.0,
    "hc_thin": 0.015,
    "duration_thin": 600.0,
    "dt_thin": 1.0,
    "mat_choice_thin": "Custom",
    "rho_thin": 7850.0,
    "cp_thin": 0.49,
    "thickness_thin": 0.005,

    # Thermally thick inputs
    "Tamb_C_thick": 20.0,
    "hc_thick": 15.0,
    "duration_thick": 3600.0,
    "nodes_thick": 10,
    "mat_choice_thick": "Concrete (Normal)",
    "rho_thick": 2300.0,
    "cp_thick": 0.88,
    "k_thick": 1.4,
    "thickness_thick": 0.150,

    # Map inputs
    "map_width": 20.0,
    "max_distance": 20.0,
    "map_resolution": 50,
}

CONFIG_KEYS = list(DEFAULT_CONFIG.keys())


def initialise_session_state() -> None:
    """Initialise all input widget state from DEFAULT_CONFIG."""
    for key, value in DEFAULT_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalise_loaded_config(raw_config: dict) -> dict:
    """
    Converts an uploaded JSON object into a clean app input config.

    Expected JSON structure is either:
    - {"app": ..., "inputs": {...}}
    - {...direct input keys...}
    """
    if not isinstance(raw_config, dict):
        raise ValueError("The uploaded JSON file must contain a JSON object.")

    input_config = raw_config.get("inputs", raw_config)
    if not isinstance(input_config, dict):
        raise ValueError("The uploaded JSON file does not contain a valid 'inputs' object.")

    cleaned_config = DEFAULT_CONFIG.copy()

    for key in CONFIG_KEYS:
        if key in input_config:
            cleaned_config[key] = input_config[key]

    # Validate material choices against current presets.
    if cleaned_config["mat_choice_thin"] not in MATERIALS:
        cleaned_config["mat_choice_thin"] = "Custom"

    if cleaned_config["mat_choice_thick"] not in MATERIALS:
        cleaned_config["mat_choice_thick"] = "Concrete (Normal)"

    # Validate slider/list constrained fields.
    cleaned_config["map_resolution"] = int(cleaned_config.get("map_resolution", 50))
    cleaned_config["map_resolution"] = min(150, max(25, cleaned_config["map_resolution"]))

    cleaned_config["nodes_thick"] = int(cleaned_config.get("nodes_thick", 10))
    cleaned_config["nodes_thick"] = min(100, max(3, cleaned_config["nodes_thick"]))

    cleaned_config["subtract_ambient"] = bool(cleaned_config.get("subtract_ambient", False))

    return cleaned_config


def apply_config_to_session_state(config: dict) -> None:
    """Applies a validated config to Streamlit session state."""
    for key in CONFIG_KEYS:
        st.session_state[key] = config[key]


def build_export_config() -> dict:
    """Builds a JSON-serialisable object containing the current calculator input state."""
    inputs = {key: st.session_state.get(key, DEFAULT_CONFIG[key]) for key in CONFIG_KEYS}

    return {
        "app": "Radiant Heat Flux & Thermal Response Calculator",
        "app_version": APP_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,
    }


def config_to_json_bytes(config: dict) -> bytes:
    return json.dumps(config, indent=2).encode("utf-8")

# ============================================================
# GENERAL HELPERS
# ============================================================

def c_to_k(temp_c: float) -> float:
    return temp_c + 273.15


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ============================================================
# RADIATION / VIEW FACTOR CALCULATIONS
# ============================================================

def phi_para(h: float, v: float, L: float) -> float:
    """Auxiliary term for view factor to a vertical rectangular source."""
    if L <= 0:
        return 0.0

    denom_h = math.sqrt(h**2 + L**2)
    denom_v = math.sqrt(v**2 + L**2)

    return (h / denom_h) * math.atan(v / denom_h) + (v / denom_v) * math.atan(h / denom_v)


def phi_horiz(h: float, v: float, L: float) -> float:
    """Auxiliary term for angle/orientation correction."""
    if L <= 0:
        return 0.0

    denom_h = math.sqrt(h**2 + L**2)
    return -0.5 * (L / denom_h) * math.atan(v / denom_h)


def calculate_view_factor_general(
    W: float,
    H: float,
    L: float,
    Xr: float,
    Zr: float,
    Zs_base: float,
    theta_deg: float,
) -> float:
    """
    Calculates an orientation-adjusted view factor from a flat vertical rectangular source
    to a differential receiver.

    Geometry convention:
    - W: source width, m
    - H: source height, m
    - L: perpendicular distance from source plane, m
    - Xr: horizontal offset from source centreline, m
    - Zr: receiver elevation, m
    - Zs_base: base elevation of source, m
    - theta_deg: receiver angle, degrees. 0 degrees is the base receiver orientation.
    """
    if W <= 0 or H <= 0 or L <= 0:
        return 0.0

    h1 = -W / 2.0 - Xr
    h2 = W / 2.0 - Xr
    v1 = Zs_base - Zr
    v2 = (Zs_base + H) - Zr

    F_para = (1.0 / (2.0 * math.pi)) * (
        phi_para(h2, v2, L)
        - phi_para(h1, v2, L)
        - phi_para(h2, v1, L)
        + phi_para(h1, v1, L)
    )

    F_horiz = (1.0 / (2.0 * math.pi)) * (
        phi_horiz(h2, v2, L)
        - phi_horiz(h1, v2, L)
        - phi_horiz(h2, v1, L)
        + phi_horiz(h1, v1, L)
    )

    theta_rad = math.radians(theta_deg)
    F_total = (F_para * math.cos(theta_rad)) + (F_horiz * math.sin(theta_rad))

    return max(0.0, F_total)


def calculate_incident_flux_kw(
    W: float,
    H: float,
    L: float,
    Xr: float,
    Zr: float,
    Zs_base: float,
    theta_deg: float,
    Ts_K: float,
    eps_s: float,
    Tamb_K: float | None = None,
    subtract_ambient: bool = False,
) -> tuple[float, float]:
    """Returns view factor and incident radiant heat flux in kW/m²."""
    F = calculate_view_factor_general(W, H, L, Xr, Zr, Zs_base, theta_deg)

    if subtract_ambient and Tamb_K is not None:
        source_emission_kw = eps_s * SIGMA_KW * max(0.0, Ts_K**4 - Tamb_K**4)
    else:
        source_emission_kw = eps_s * SIGMA_KW * Ts_K**4

    q_incident_kw = F * source_emission_kw
    return F, q_incident_kw


def find_distance_for_flux(
    target_flux_kw: float,
    W: float,
    H: float,
    Ts_K: float,
    eps_s: float,
    Xr: float,
    Zr: float,
    Zs_base: float,
    theta_deg: float,
    Tamb_K: float | None = None,
    subtract_ambient: bool = False,
    L_min: float = 0.001,
    L_max: float = 500.0,
) -> float | None:
    """Finds the perpendicular distance required to reduce incident flux to the target value."""
    if target_flux_kw <= 0:
        return None

    _, q_at_min = calculate_incident_flux_kw(
        W, H, L_min, Xr, Zr, Zs_base, theta_deg, Ts_K, eps_s, Tamb_K, subtract_ambient
    )

    _, q_at_max = calculate_incident_flux_kw(
        W, H, L_max, Xr, Zr, Zs_base, theta_deg, Ts_K, eps_s, Tamb_K, subtract_ambient
    )

    if target_flux_kw > q_at_min:
        return None

    if target_flux_kw < q_at_max:
        return None

    L_low = L_min
    L_high = L_max
    L_mid = (L_low + L_high) / 2.0

    for _ in range(100):
        L_mid = (L_low + L_high) / 2.0
        _, q_mid = calculate_incident_flux_kw(
            W, H, L_mid, Xr, Zr, Zs_base, theta_deg, Ts_K, eps_s, Tamb_K, subtract_ambient
        )

        if q_mid > target_flux_kw:
            L_low = L_mid
        else:
            L_high = L_mid

    return L_mid

# ============================================================
# THERMAL RESPONSE CALCULATIONS
# ============================================================

def run_thermally_thin_simulation(
    q_incident_kw: float,
    eps_t: float,
    T_initial_C: float,
    Tamb_C: float,
    hc_kw: float,
    duration_s: float,
    dt_s: float,
    rho: float,
    cp_kj: float,
    thickness_m: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Lumped thermal response model for thermally thin targets.

    Unit basis:
    - q is kW/m², equivalent to kJ/s.m².
    - rho * cp * thickness is kJ/m².K.
    - dT is therefore in K.
    """
    warnings = []

    if dt_s <= 0:
        raise ValueError("Time step must be greater than zero.")
    if duration_s <= 0:
        raise ValueError("Exposure duration must be greater than zero.")
    if rho <= 0 or cp_kj <= 0 or thickness_m <= 0:
        raise
