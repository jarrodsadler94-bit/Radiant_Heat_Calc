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
        raise ValueError("Density, specific heat and thickness must be greater than zero.")

    Tt_K = c_to_k(T_initial_C)
    Tamb_K = c_to_k(Tamb_C)

    time_data = []
    temp_data = []
    absorbed_flux_data = []
    reradiation_data = []
    convection_data = []
    net_flux_data = []

    n_steps = int(duration_s / dt_s)

    for step in range(n_steps + 1):
        t = step * dt_s

        q_absorbed_kw = eps_t * q_incident_kw
        q_rerad_kw = eps_t * SIGMA_KW * (Tt_K**4 - Tamb_K**4)
        q_conv_kw = hc_kw * (Tt_K - Tamb_K)
        q_net_kw = q_absorbed_kw - q_rerad_kw - q_conv_kw

        time_data.append(t)
        temp_data.append(Tt_K - 273.15)
        absorbed_flux_data.append(q_absorbed_kw)
        reradiation_data.append(q_rerad_kw)
        convection_data.append(q_conv_kw)
        net_flux_data.append(q_net_kw)

        dT = (q_net_kw * dt_s) / (rho * cp_kj * thickness_m)
        Tt_K += dT

        if Tt_K > MAX_REASONABLE_TEMP_K:
            warnings.append("Simulation exceeded 2500 K. Results after this point were not calculated.")
            break

        if math.isnan(Tt_K) or math.isinf(Tt_K):
            warnings.append("Simulation became numerically unstable. Results after this point were not calculated.")
            break

    df = pd.DataFrame(
        {
            "Time (s)": time_data,
            "Target Temperature (°C)": temp_data,
            "Absorbed Flux (kW/m²)": absorbed_flux_data,
            "Re-radiation Loss (kW/m²)": reradiation_data,
            "Convective Loss (kW/m²)": convection_data,
            "Net Flux (kW/m²)": net_flux_data,
        }
    )

    return df, warnings


def update_exposed_surface(
    T0: float,
    T1: float,
    q_net_front_W: float,
    k: float,
    rho: float,
    cp_J: float,
    dx: float,
    dt: float,
) -> float:
    """
    Explicit half-control-volume update for the exposed surface node.

    q_net_front_W is positive into the exposed surface.
    conduction_in is positive when the adjacent internal node is hotter than the surface node.
    """
    conduction_in_W = k * (T1 - T0) / dx
    return T0 + (2.0 * dt / (rho * cp_J * dx)) * (q_net_front_W + conduction_in_W)


def update_unexposed_surface(
    T_last: float,
    T_prev: float,
    q_loss_back_W: float,
    k: float,
    rho: float,
    cp_J: float,
    dx: float,
    dt: float,
) -> float:
    """
    Explicit half-control-volume update for the unexposed surface node.

    q_loss_back_W is positive heat loss from the unexposed surface to the environment.
    conduction_in is positive when the internal adjacent node is hotter than the unexposed surface.
    """
    conduction_in_W = k * (T_prev - T_last) / dx
    return T_last + (2.0 * dt / (rho * cp_J * dx)) * (conduction_in_W - q_loss_back_W)


def run_thermally_thick_fdm_simulation(
    q_incident_kw: float,
    eps_t: float,
    T_initial_C: float,
    Tamb_C: float,
    hc_W: float,
    duration_s: float,
    nodes: int,
    rho: float,
    cp_kj: float,
    k_W: float,
    thickness_m: float,
) -> tuple[pd.DataFrame, float, float, list[str]]:
    """Explicit 1D finite difference model for thermally thick targets."""
    warnings = []

    if duration_s <= 0:
        raise ValueError("Exposure duration must be greater than zero.")
    if nodes < 3:
        raise ValueError("At least 3 nodes are required.")
    if rho <= 0 or cp_kj <= 0 or k_W <= 0 or thickness_m <= 0:
        raise ValueError("Density, specific heat, conductivity and thickness must be greater than zero.")

    cp_J = cp_kj * 1000.0
    q_incident_W = q_incident_kw * 1000.0
    Tamb_K = c_to_k(Tamb_C)
    dx = thickness_m / (nodes - 1)
    alpha = k_W / (rho * cp_J)

    dt_stable = 0.5 * (dx**2) / alpha
    dt_actual = min(0.5, dt_stable * 0.9)
    Fo = alpha * dt_actual / (dx**2)

    T = np.full(nodes, c_to_k(T_initial_C), dtype=float)
    T_new = np.zeros(nodes, dtype=float)

    time_data = []
    temp_exposed = []
    temp_mid = []
    temp_unexposed = []

    absorbed_flux_data = []
    reradiation_front_data = []
    convection_front_data = []
    net_front_flux_data = []

    t = 0.0
    mid_node = nodes // 2
    record_interval = max(1, int(round(10.0 / dt_actual)))
    step = 0

    while t <= duration_s:
        if step % record_interval == 0 or step == 0:
            time_data.append(t)
            temp_exposed.append(T[0] - 273.15)
            temp_mid.append(T[mid_node] - 273.15)
            temp_unexposed.append(T[-1] - 273.15)

        q_absorbed_W = eps_t * q_incident_W
        q_rerad_front_W = eps_t * SIGMA_W * (T[0]**4 - Tamb_K**4)
        q_conv_front_W = hc_W * (T[0] - Tamb_K)
        q_net_front_W = q_absorbed_W - q_rerad_front_W - q_conv_front_W

        if step % record_interval == 0 or step == 0:
            absorbed_flux_data.append(q_absorbed_W / 1000.0)
            reradiation_front_data.append(q_rerad_front_W / 1000.0)
            convection_front_data.append(q_conv_front_W / 1000.0)
            net_front_flux_data.append(q_net_front_W / 1000.0)

        T_new[0] = update_exposed_surface(
            T0=T[0],
            T1=T[1],
            q_net_front_W=q_net_front_W,
            k=k_W,
            rho=rho,
            cp_J=cp_J,
            dx=dx,
            dt=dt_actual,
        )

        for i in range(1, nodes - 1):
            T_new[i] = T[i] + Fo * (T[i - 1] - 2.0 * T[i] + T[i + 1])

        q_rerad_back_W = eps_t * SIGMA_W * (T[-1]**4 - Tamb_K**4)
        q_conv_back_W = hc_W * (T[-1] - Tamb_K)
        q_loss_back_W = q_rerad_back_W + q_conv_back_W

        T_new[-1] = update_unexposed_surface(
            T_last=T[-1],
            T_prev=T[-2],
            q_loss_back_W=q_loss_back_W,
            k=k_W,
            rho=rho,
            cp_J=cp_J,
            dx=dx,
            dt=dt_actual,
        )

        T = T_new.copy()
        t += dt_actual
        step += 1

        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            warnings.append("FDM simulation became numerically unstable. Results after this point were not calculated.")
            break

        if np.any(T > MAX_REASONABLE_TEMP_K):
            warnings.append("FDM simulation exceeded 2500 K. Results after this point were not calculated.")
            break

    df = pd.DataFrame(
        {
            "Time (s)": time_data,
            "Exposed Face (°C)": temp_exposed,
            f"Mid-depth ({thickness_m / 2.0:.3f} m) (°C)": temp_mid,
            "Unexposed Face (°C)": temp_unexposed,
            "Absorbed Flux (kW/m²)": absorbed_flux_data,
            "Front Re-radiation Loss (kW/m²)": reradiation_front_data,
            "Front Convective Loss (kW/m²)": convection_front_data,
            "Net Front Flux (kW/m²)": net_front_flux_data,
        }
    )

    return df, dt_actual, Fo, warnings

# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Radiant Heat Flux Calculator", layout="wide")
initialise_session_state()

st.title("Radiant Heat Flux & Thermal Response Calculator")

with st.sidebar:
    st.header("Calculation File")

    project_name = st.text_input("Calculation name", value="radiant_heat_flux_calculation", key="calculation_name")

    uploaded_config_file = st.file_uploader(
        "Upload saved JSON calculation",
        type=["json"],
        help="Upload a previously downloaded calculator JSON file to reload the input state.",
    )

    if uploaded_config_file is not None:
        try:
            raw_config = json.load(uploaded_config_file)
            cleaned_config = normalise_loaded_config(raw_config)
            apply_config_to_session_state(cleaned_config)
            st.success("Saved calculation loaded. The inputs have been restored.")
        except Exception as exc:
            st.error(f"Could not load JSON file: {exc}")

    current_config = build_export_config()
    safe_project_name = project_name.strip().replace(" ", "_") or "radiant_heat_flux_calculation"

    st.download_button(
        "Download current calculation as JSON",
        data=config_to_json_bytes(current_config),
        file_name=f"{safe_project_name}.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button("Reset inputs to defaults", use_container_width=True):
        apply_config_to_session_state(DEFAULT_CONFIG.copy())
        st.success("Inputs reset to defaults.")

with st.expander("Model assumptions and limitations", expanded=False):
    st.markdown(
        """
        This calculator represents the radiant source as a flat, vertical, rectangular, isothermal emitter.

        The model assumes:
        - a uniform source temperature;
        - a uniform source emissivity;
        - a differential receiver surface;
        - no atmospheric attenuation;
        - no shielding or obstruction;
        - no flame tilt, wind effect, smoke obscuration, intermittent flame behaviour or source pulsation;
        - grey-body target behaviour, where target absorptivity is assumed equal to target emissivity;
        - one-dimensional heat transfer through the target for the thermally thick finite difference model.

        Results should be treated as screening-level engineering calculations unless independently verified against the relevant project basis, standard, or fire engineering method.
        """
    )

col_left, col_right = st.columns([1, 1])

# ============================================================
# INPUTS
# ============================================================

with col_left:
    st.header("Source (Emitter)")

    c1, c2 = st.columns(2)
    with c1:
        Ts_C = st.number_input("Temperature (°C)", step=10.0, key="Ts_C")
        W = st.number_input("Width (m)", min_value=0.01, step=0.1, key="W")
        H = st.number_input("Height (m)", min_value=0.01, step=0.1, key="H")
    with c2:
        eps_s = st.number_input("Source Emissivity (ε_s)", min_value=0.01, max_value=1.00, step=0.01, key="eps_s")
        Zs_base = st.number_input("Source Base Elevation (m)", step=0.1, key="Zs_base")

    st.header("Target (Receiver Location)")

    c4, c5 = st.columns(2)
    with c4:
        L = st.number_input("Perpendicular Distance (m)", min_value=0.001, step=0.1, key="L")
        Zr = st.number_input("Receiver Height (m)", step=0.1, key="Zr")
        Xr = st.number_input("Horizontal Offset (m)", step=0.1, key="Xr")
    with c5:
        theta_deg = st.number_input(
            "Receiver Angle (deg)",
            min_value=-90.0,
            max_value=90.0,
            step=5.0,
            key="theta_deg",
            help="App convention: 0° is the base receiver orientation used by the view factor calculation. Positive and negative rotations modify the orientation correction term.",
        )
        Tt_initial_C = st.number_input("Initial Target Temperature (°C)", step=1.0, key="Tt_initial_C")
        eps_t = st.number_input(
            "Target Emissivity / Absorptivity (ε_t)",
            min_value=0.01,
            max_value=1.00,
            step=0.01,
            key="eps_t",
            help="The thermal response model assumes grey-body behaviour, where absorptivity is approximately equal to emissivity.",
        )

    st.header("Radiation Options")
    Tamb_radiation_C = st.number_input("Background / Ambient Temperature (°C)", step=1.0, key="Tamb_radiation_C")
    subtract_ambient = st.checkbox(
        "Subtract ambient background radiation from source emission",
        key="subtract_ambient",
        help="If selected, source emission is calculated using εσ(Ts⁴ - Tamb⁴). If not selected, source emission is calculated using εσTs⁴.",
    )

# ============================================================
# CORE SPATIAL CALCULATION
# ============================================================

Ts_K = c_to_k(Ts_C)
Tamb_radiation_K = c_to_k(Tamb_radiation_C)

F_current, q_incident_current_kw = calculate_incident_flux_kw(
    W=W,
    H=H,
    L=L,
    Xr=Xr,
    Zr=Zr,
    Zs_base=Zs_base,
    theta_deg=theta_deg,
    Ts_K=Ts_K,
    eps_s=eps_s,
    Tamb_K=Tamb_radiation_K,
    subtract_ambient=subtract_ambient,
)

q_absorbed_current_kw = eps_t * q_incident_current_kw

# ============================================================
# OUTPUTS
# ============================================================

with col_right:
    st.header("Current Geometry Result")

    r1, r2, r3 = st.columns(3)
    r1.metric("View Factor", f"{F_current:.5f}")
    r2.metric("Incident Flux", f"{q_incident_current_kw:.2f} kW/m²")
    r3.metric("Absorbed Flux", f"{q_absorbed_current_kw:.2f} kW/m²")

    st.caption("Incident flux is the radiant heat flux arriving at the receiver location. Absorbed flux applies the target emissivity / absorptivity factor used by the thermal response calculations.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Thermally Thin", "Thermally Thick", "Separation Distances", "2D Contour Map (Plan)"]
    )

    # ========================================================
    # TAB 1: THERMALLY THIN
    # ========================================================

    with tab1:
        st.markdown("**Suitable for highly conductive, thin materials where a lumped thermal mass assumption is acceptable.**")

        c6, c7 = st.columns(2)
        with c6:
            Tamb_C = st.number_input("Ambient Temperature (°C)", step=1.0, key="Tamb_C_thin")
            hc = st.number_input(
                "Convective Coefficient (kW/m².K)",
                min_value=0.0,
                step=0.005,
                format="%.3f",
                key="hc_thin",
            )
        with c7:
            duration = st.number_input("Exposure Duration (s)", min_value=0.1, step=60.0, key="duration_thin")
            dt = st.number_input("Time Step (s)", min_value=0.001, step=0.5, key="dt_thin")

        mat_choice_thin = st.selectbox("Material Preset", list(MATERIALS.keys()), key="mat_choice_thin")

        # Update material fields when the selected preset changes, except Custom.
        if mat_choice_thin != "Custom":
            st.session_state["rho_thin"] = MATERIALS[mat_choice_thin]["rho"]
            st.session_state["cp_thin"] = MATERIALS[mat_choice_thin]["cp"]

        c8, c9, c10 = st.columns(3)
        with c8:
            rho = st.number_input("Density (kg/m³)", min_value=1.0, key="rho_thin")
        with c9:
            cp = st.number_input("Specific Heat, Cp (kJ/kg.K)", min_value=0.001, key="cp_thin")
        with c10:
            thickness = st.number_input("Thickness (m)", min_value=0.0001, step=0.001, format="%.4f", key="thickness_thin")

        if st.button("Run Lumped Mass Simulation", use_container_width=True):
            try:
                sim_df, sim_warnings = run_thermally_thin_simulation(
                    q_incident_kw=q_incident_current_kw,
                    eps_t=eps_t,
                    T_initial_C=Tt_initial_C,
                    Tamb_C=Tamb_C,
                    hc_kw=hc,
                    duration_s=duration,
                    dt_s=dt,
                    rho=rho,
                    cp_kj=cp,
                    thickness_m=thickness,
                )

                st.metric("Peak Target Temperature", f"{sim_df['Target Temperature (°C)'].max():.1f} °C")

                chart_df = sim_df[["Time (s)", "Target Temperature (°C)"]].set_index("Time (s)")
                st.line_chart(chart_df)

                for warning in sim_warnings:
                    st.warning(warning)

                st.download_button(
                    "Download Thermally Thin Results as CSV",
                    data=convert_df_to_csv(sim_df),
                    file_name="thermally_thin_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except ValueError as exc:
                st.error(str(exc))

    # ========================================================
    # TAB 2: THERMALLY THICK
    # ========================================================

    with tab2:
        st.markdown("**Suitable for insulating or thick materials where through-depth temperature gradients are important.**")

        c11, c12 = st.columns(2)
        with c11:
            Tamb_C_fdm = st.number_input("Ambient Temperature (°C)", step=1.0, key="Tamb_C_thick")
            hc_W = st.number_input(
                "Convective Coefficient (W/m².K)",
                min_value=0.0,
                step=1.0,
                help="The finite difference model uses SI units for the convective coefficient.",
                key="hc_thick",
            )
        with c12:
            duration_fdm = st.number_input("Exposure Duration (s)", min_value=0.1, step=60.0, key="duration_thick")
            nodes = st.number_input("Number of Nodes", min_value=3, max_value=100, step=1, key="nodes_thick")

        mat_choice_thick = st.selectbox("Material Preset", list(MATERIALS.keys()), key="mat_choice_thick")

        # Update material fields when the selected preset changes, except Custom.
        if mat_choice_thick != "Custom":
            st.session_state["rho_thick"] = MATERIALS[mat_choice_thick]["rho"]
            st.session_state["cp_thick"] = MATERIALS[mat_choice_thick]["cp"]
            st.session_state["k_thick"] = MATERIALS[mat_choice_thick]["k"]

        c13, c14, c15 = st.columns(3)
        with c13:
            rho_fdm = st.number_input("Density (kg/m³)", min_value=1.0, key="rho_thick")
        with c14:
            cp_fdm = st.number_input("Specific Heat, Cp (kJ/kg.K)", min_value=0.001, key="cp_thick")
        with c15:
            k_fdm = st.number_input("Conductivity (W/m.K)", min_value=0.001, step=0.1, key="k_thick")

        thickness_fdm = st.number_input("Thickness (m)", min_value=0.001, step=0.010, format="%.3f", key="thickness_thick")

        if st.button("Run 1D FDM Simulation", use_container_width=True):
            try:
                fdm_df, dt_actual, Fo, fdm_warnings = run_thermally_thick_fdm_simulation(
                    q_incident_kw=q_incident_current_kw,
                    eps_t=eps_t,
                    T_initial_C=Tt_initial_C,
                    Tamb_C=Tamb_C_fdm,
                    hc_W=hc_W,
                    duration_s=duration_fdm,
                    nodes=int(nodes),
                    rho=rho_fdm,
                    cp_kj=cp_fdm,
                    k_W=k_fdm,
                    thickness_m=thickness_fdm,
                )

                st.caption(f"Auto-calculated stable time step: {dt_actual:.3f} s. Fourier number: {Fo:.3f}.")
                st.metric("Peak Exposed Surface Temperature", f"{fdm_df['Exposed Face (°C)'].max():.1f} °C")

                temp_cols = [col for col in fdm_df.columns if "°C" in col]
                chart_df = fdm_df[["Time (s)"] + temp_cols].set_index("Time (s)")
                st.line_chart(chart_df)

                for warning in fdm_warnings:
                    st.warning(warning)

                st.download_button(
                    "Download Thermally Thick Results as CSV",
                    data=convert_df_to_csv(fdm_df),
                    file_name="thermally_thick_fdm_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except ValueError as exc:
                st.error(str(exc))

    # ========================================================
    # TAB 3: SEPARATION DISTANCES
    # ========================================================

    with tab3:
        st.subheader("Critical Separation Distances")
        st.metric("Incident Flux at Current Distance", f"{q_incident_current_kw:.2f} kW/m²")

        results = []
        for threshold_kw, description in THRESHOLDS.items():
            dist = find_distance_for_flux(
                target_flux_kw=threshold_kw,
                W=W,
                H=H,
                Ts_K=Ts_K,
                eps_s=eps_s,
                Xr=Xr,
                Zr=Zr,
                Zs_base=Zs_base,
                theta_deg=theta_deg,
                Tamb_K=Tamb_radiation_K,
                subtract_ambient=subtract_ambient,
            )

            results.append(
                {
                    "Threshold (kW/m²)": threshold_kw,
                    "Description": description,
                    "Minimum Distance (m)": f"{dist:.2f}" if dist is not None else "N/A",
                }
            )

        separation_df = pd.DataFrame(results)
        st.table(separation_df)

        st.download_button(
            "Download Separation Distance Results as CSV",
            data=convert_df_to_csv(separation_df),
            file_name="separation_distance_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption("Threshold descriptions are generic screening labels only. Adopt project-specific criteria from the applicable standard, guideline or fire engineering brief.")

    # ========================================================
    # TAB 4: PLAN VIEW CONTOUR MAP
    # ========================================================

    with tab4:
        st.subheader("2D Incident Flux Contour Map (Plan View)")
        st.markdown(f"**Mapping incident flux footprint at elevation Z = {Zr:.2f} m.**")

        map_width = st.number_input("Map Width, X-axis (m)", min_value=0.1, step=1.0, key="map_width")
        max_distance = st.number_input("Maximum Distance, Y-axis (m)", min_value=0.1, step=1.0, key="max_distance")
        resolution = st.slider("Grid Resolution", min_value=25, max_value=150, step=25, key="map_resolution")

        if st.button("Generate Plan View Map", use_container_width=True):
            x_vals = np.linspace(-map_width / 2.0, map_width / 2.0, resolution)
            l_vals = np.linspace(0.1, max_distance, resolution)
            X_grid, L_grid = np.meshgrid(x_vals, l_vals)
            Flux_grid = np.zeros_like(X_grid)

            progress_bar = st.progress(0)

            for i in range(resolution):
                for j in range(resolution):
                    _, q_pt_kw = calculate_incident_flux_kw(
                        W=W,
                        H=H,
                        L=float(L_grid[i, j]),
                        Xr=float(X_grid[i, j]),
                        Zr=Zr,
                        Zs_base=Zs_base,
                        theta_deg=theta_deg,
                        Ts_K=Ts_K,
                        eps_s=eps_s,
                        Tamb_K=Tamb_radiation_K,
                        subtract_ambient=subtract_ambient,
                    )
                    Flux_grid[i, j] = q_pt_kw

                progress_bar.progress((i + 1) / resolution)

            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            cp = ax.contourf(X_grid, L_grid, Flux_grid, levels=30, cmap="inferno")
            cbar = fig.colorbar(cp)
            cbar.set_label("Incident Flux (kW/m²)", rotation=270, labelpad=15, color="black")
            cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")

            thresholds_to_plot = [
                t for t in THRESHOLDS.keys()
                if np.nanmin(Flux_grid) <= t <= np.nanmax(Flux_grid)
            ]

            if thresholds_to_plot:
                CS = ax.contour(
                    X_grid,
                    L_grid,
                    Flux_grid,
                    levels=thresholds_to_plot,
                    colors="white",
                    linestyles="dashed",
                    linewidths=1.5,
                )
                ax.clabel(
                    CS,
                    inline=True,
                    inline_spacing=5,
                    fontsize=10,
                    fmt="%1.1f kW/m²",
                    colors="white",
                )

            ax.set_xlabel("Horizontal Offset from Source Centreline, X (m)", color="black")
            ax.set_ylabel("Perpendicular Distance from Source, L (m)", color="black")
            ax.set_title(f"Radiant Footprint at Elevation Z = {Zr:.2f} m, Angle = {theta_deg:.1f}°", color="black")

            ax.tick_params(axis="x", colors="black")
            ax.tick_params(axis="y", colors="black")

            ax.plot([-W / 2.0, W / 2.0], [0, 0], color="cyan", linewidth=4, label="Emitter Source Base")

            legend = ax.legend(loc="upper right", facecolor="white", edgecolor="black")
            for text in legend.get_texts():
                text.set_color("black")

            st.pyplot(fig, clear_figure=True, transparent=False)

            map_df = pd.DataFrame(
                {
                    "X Offset (m)": X_grid.flatten(),
                    "Distance L (m)": L_grid.flatten(),
                    "Incident Flux (kW/m²)": Flux_grid.flatten(),
                }
            )

            st.download_button(
                "Download Map Data as CSV",
                data=convert_df_to_csv(map_df),
                file_name="radiant_footprint_map_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
            st.download_button(
                label="Download Map as PNG",
                data=buf.getvalue(),
                file_name="radiant_footprint_map.png",
                mime="image/png",
                use_container_width=True,
            )

            progress_bar.empty()
