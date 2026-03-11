import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Stefan-Boltzmann Constant
SIGMA_KW = 5.67e-11 # kW/m²·K⁴
SIGMA_W = 5.67e-8   # W/m²·K⁴

# --- MATERIAL PRESETS ---
MATERIALS = {
    "Custom": {"rho": 7850.0, "cp": 0.49, "k": 45.0},
    "Carbon Steel": {"rho": 7850.0, "cp": 0.49, "k": 45.0},
    "Aluminum (Alloy)": {"rho": 2770.0, "cp": 0.88, "k": 175.0},
    "Copper": {"rho": 8960.0, "cp": 0.39, "k": 400.0},
    "Window Glass": {"rho": 2500.0, "cp": 0.84, "k": 1.0},
    "Concrete (Normal)": {"rho": 2300.0, "cp": 0.88, "k": 1.4},
    "Gypsum Board": {"rho": 800.0, "cp": 1.09, "k": 0.25},
    "Timber (Softwood)": {"rho": 500.0, "cp": 1.63, "k": 0.12}
}

# --- HELPER FUNCTIONS ---
def phi_para(h, v, L):
    if L <= 0: return 0.0
    denom_h = math.sqrt(h**2 + L**2)
    denom_v = math.sqrt(v**2 + L**2)
    return (h / denom_h) * math.atan(v / denom_h) + (v / denom_v) * math.atan(h / denom_v)

def phi_horiz(h, v, L):
    if L <= 0: return 0.0
    denom_h = math.sqrt(h**2 + L**2)
    return -0.5 * (L / denom_h) * math.atan(v / denom_h)

def calculate_view_factor_general(W, H, L, Xr, Zr, Zs_base, theta_deg):
    h1, h2 = -W/2 - Xr, W/2 - Xr
    v1, v2 = Zs_base - Zr, (Zs_base + H) - Zr
    
    F_para = (1 / (2 * math.pi)) * (phi_para(h2, v2, L) - phi_para(h1, v2, L) - phi_para(h2, v1, L) + phi_para(h1, v1, L))
    F_horiz = (1 / (2 * math.pi)) * (phi_horiz(h2, v2, L) - phi_horiz(h1, v2, L) - phi_horiz(h2, v1, L) + phi_horiz(h1, v1, L))
    
    theta_rad = math.radians(theta_deg)
    F_total = (F_para * math.cos(theta_rad)) + (F_horiz * math.sin(theta_rad))
    return max(0.0, F_total)

def find_distance_for_flux(target_flux_kw, W, H, Ts_K, eps_s, Xr, Zr, Zs_base, theta_deg):
    E_max = eps_s * SIGMA_KW * (Ts_K**4)
    if target_flux_kw >= E_max: return None
    target_F = target_flux_kw / E_max
    L_low, L_high = 0.001, 500.0  
    for _ in range(100): 
        L_mid = (L_low + L_high) / 2
        F_mid = calculate_view_factor_general(W, H, L_mid, Xr, Zr, Zs_base, theta_deg)
        if F_mid > target_F: L_low = L_mid  
        else: L_high = L_mid 
    return L_mid

def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# --- APP LAYOUT ---
st.set_page_config(page_title="Radiant Heat Flux Calculator", layout="wide")
st.title("Radiant Heat Flux & Thermal Response Calculator")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("Source (Emitter)")
    c1, c2 = st.columns(2)
    with c1:
        Ts_C = st.number_input("Temperature (°C)", value=1000.0, step=10.0, key="Ts")
        W = st.number_input("Width (m)", value=6.1, step=0.1)
        H = st.number_input("Height (m)", value=2.8, step=0.1)
    with c2:
        eps_s = st.number_input("Source Emissivity (ε_s)", value=1.00, min_value=0.01, max_value=1.00, step=0.01)
        Zs_base = st.number_input("Source Base Elev. (m)", value=0.0, step=0.1)

    st.header("Target (Receiver Location)")
    c4, c5 = st.columns(2)
    with c4:
        L = st.number_input("Perpendicular Distance (m)", value=3.0, step=0.1, min_value=0.1)
        Zr = st.number_input("Receiver Height (m)", value=1.5, step=0.1)
        Xr = st.number_input("Horizontal Offset (m)", value=0.0, step=0.1)
    with c5:
        theta_deg = st.number_input("Receiver Angle (deg)", value=0.0, step=5.0)
        Tt_initial_C = st.number_input("Initial Target Temp (°C)", value=20.0, step=1.0)
        eps_t = st.number_input("Target Emissivity (ε_t)", value=0.90, min_value=0.01, max_value=1.00, step=0.01)

# --- SPATIAL CALCULATIONS ---
Ts_K = Ts_C + 273.15
F_current = calculate_view_factor_general(W, H, L, Xr, Zr, Zs_base, theta_deg)
q_inc_current = F_current * eps_s * SIGMA_KW * (Ts_K**4)

with col_right:
    st.header("Receiver Analysis")
    
    # Set up Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Thermally Thin", "Thermally Thick", "Separation Distances", "2D Contour Map (Plan)"])
    
    with tab1:
        st.markdown("**Suitable for highly conductive, thin materials (e.g., sheet metal, glass).**")
        c6, c7 = st.columns(2)
        with c6:
            Tamb_C = st.number_input("Ambient Temp (°C)", value=20.0, step=1.0, key="tamb_thin")
            hc = st.number_input("Conv. Coeff (kW/m²·K)", value=0.015, step=0.005, format="%.3f")
        with c7:
            duration = st.number_input("Exposure (s)", value=600, step=60, key="dur_thin")
            dt = st.number_input("Time Step (s)", value=1.0, step=0.5, key="dt_thin")

        mat_choice_thin = st.selectbox("Material Preset", list(MATERIALS.keys()), key="mat_thin")
        c8, c9, c10 = st.columns(3)
        with c8: rho = st.number_input("Density (kg/m³)", value=MATERIALS[mat_choice_thin]["rho"], key="rho_thin")
        with c9: cp = st.number_input("Cp (kJ/kg·K)", value=MATERIALS[mat_choice_thin]["cp"], key="cp_thin")
        with c10: thickness = st.number_input("Thickness (m)", value=0.005, step=0.001, format="%.3f", key="th_thin")

        if st.button("Run Lumped Mass Simulation", use_container_width=True):
            time_data, temp_data = [], []
            Tt_K = Tt_initial_C + 273.15
            Tamb_K = Tamb_C + 273.15
            t = 0.0
            while t <= duration:
                q_rerad = eps_t * SIGMA_KW * ((Tt_K**4) - (Tamb_K**4))
                q_conv = hc * (Tt_K - Tamb_K)
                q_net_total = (eps_t * q_inc_current) - q_rerad - q_conv
                
                time_data.append(t)
                temp_data.append(Tt_K - 273.15) 
                dT = (q_net_total * dt) / (rho * cp * thickness)
                Tt_K += dT
                t += dt

            st.metric(label="Peak Target Temp", value=f"{max(temp_data):.1f} °C")
            sim_df = pd.DataFrame({"Time (s)": time_data, "Temp (°C)": temp_data}).set_index("Time (s)")
            st.line_chart(sim_df)

    with tab2:
        st.markdown("**Suitable for insulating or thick materials (e.g., concrete, timber, gypsum).**")
        c11, c12 = st.columns(2)
        with c11:
            Tamb_C_fdm = st.number_input("Ambient Temp (°C)", value=20.0, step=1.0, key="tamb_thick")
            hc_W = st.number_input("Conv. Coeff (W/m²·K)", value=15.0, step=1.0, help="Note: SI units (W) used for FDM")
        with c12:
            duration_fdm = st.number_input("Exposure (s)", value=3600, step=60, key="dur_thick")
            nodes = st.number_input("Number of Nodes", value=10, min_value=3, max_value=100, step=1)

        mat_choice_thick = st.selectbox("Material Preset", list(MATERIALS.keys()), index=5, key="mat_thick")
        c13, c14, c15 = st.columns(3)
        with c13: rho_fdm = st.number_input("Density (kg/m³)", value=MATERIALS[mat_choice_thick]["rho"], key="rho_thick")
        with c14: cp_fdm = st.number_input("Cp (kJ/kg·K)", value=MATERIALS[mat_choice_thick]["cp"], key="cp_thick")
        with c15: k_fdm = st.number_input("Conductivity (W/m·K)", value=MATERIALS[mat_choice_thick]["k"], step=0.1)
        
        thickness_fdm = st.number_input("Thickness (m)", value=0.150, step=0.010, format="%.3f", key="th_thick")

        if st.button("Run 1D FDM Simulation", use_container_width=True):
            cp_J = cp_fdm * 1000
            q_inc_W = q_inc_current * 1000
            Tamb_K_fdm = Tamb_C_fdm + 273.15
            dx = thickness_fdm / (nodes - 1)
            alpha = k_fdm / (rho_fdm * cp_J)
            
            dt_stable = 0.5 * (dx**2) / alpha
            dt_actual = min(0.5, dt_stable * 0.9) 
            Fo = alpha * dt_actual / (dx**2)
            
            st.caption(f"Auto-calculated stable time step: {dt_actual:.3f} s (Fourier No: {Fo:.3f})")

            T = np.full(nodes, Tt_initial_C + 273.15)
            T_new = np.zeros(nodes)
            time_data_fdm, temp_exposed, temp_mid, temp_unexposed = [], [], [], []
            t, mid_node = 0.0, nodes // 2

            while t <= duration_fdm:
                if int(t) % 10 == 0 or t == 0: 
                    time_data_fdm.append(t)
                    temp_exposed.append(T[0] - 273.15)
                    temp_mid.append(T[mid_node] - 273.15)
                    temp_unexposed.append(T[-1] - 273.15)

                q_rerad = eps_t * SIGMA_W * ((T[0]**4) - (Tamb_K_fdm**4))
                q_conv_front = hc_W * (T[0] - Tamb_K_fdm)
                q_net_front = (eps_t * q_inc_W) - q_rerad - q_conv_front
                
                T_new[0] = T[0] + (2 * dt_actual / (rho_fdm * cp_J * dx)) * (q_net_front - k_fdm * (T[0] - T[1]) / dx)

                for i in range(1, nodes - 1):
                    T_new[i] = T[i] + Fo * (T[i-1] - 2*T[i] + T[i+1])

                q_rerad_back = eps_t * SIGMA_W * ((T[-1]**4) - (Tamb_K_fdm**4))
                q_conv_back = hc_W * (T[-1] - Tamb_K_fdm)
                q_loss_back = q_rerad_back + q_conv_back
                
                T_new[-1] = T[-1] + (2 * dt_actual / (rho_fdm * cp_J * dx)) * (k_fdm * (T[-2] - T[-1]) / dx - q_loss_back)

                T = np.copy(T_new)
                t += dt_actual

            st.metric(label="Peak Exposed Surface Temp", value=f"{max(temp_exposed):.1f} °C")
            fdm_df = pd.DataFrame({
                "Time (s)": time_data_fdm, 
                "Exposed Face (°C)": temp_exposed,
                f"Mid-Depth ({thickness_fdm/2:.3f}m) (°C)": temp_mid,
                "Unexposed Face (°C)": temp_unexposed
            }).set_index("Time (s)")
            
            st.line_chart(fdm_df)

    with tab3:
        st.subheader("Critical Separation Distances")
        st.metric(label="Incident Flux at Current Distance", value=f"{q_inc_current:.2f} kW/m²")
        thresholds = [4.7, 12.6, 23.0, 35.0]
        results_dict = {"Threshold (kW/m²)": [], "Min. Distance (m)": []}
        
        for target in thresholds:
            dist = find_distance_for_flux(target, W, H, Ts_K, eps_s, Xr, Zr, Zs_base, theta_deg)
            results_dict["Threshold (kW/m²)"].append(target)
            results_dict["Min. Distance (m)"].append(f"{dist:.2f}" if dist else "N/A")
            
        st.table(pd.DataFrame(results_dict))

    with tab4:
        st.subheader("2D Incident Flux Contour Map (Plan View)")
        st.markdown(f"**Mapping incident flux footprint at Elevation Z = {Zr} m**")
        
        map_width = st.number_input("Map Width (X-axis) (m)", value=20.0, step=1.0)
        max_distance = st.number_input("Max Distance (Y-axis) (m)", value=20.0, step=1.0)
        resolution = 50  # Fixed grid resolution
            
        if st.button("Generate Plan View Map", use_container_width=True):
            x_vals = np.linspace(-map_width/2, map_width/2, resolution)
            l_vals = np.linspace(0.1, max_distance, resolution) 
            X_grid, L_grid = np.meshgrid(x_vals, l_vals)
            Flux_grid = np.zeros_like(X_grid)
            
            progress_bar = st.progress(0)
            
            for i in range(resolution):
                for j in range(resolution):
                    F_pt = calculate_view_factor_general(W, H, L_grid[i, j], X_grid[i, j], Zr, Zs_base, theta_deg)
                    Flux_grid[i, j] = F_pt * eps_s * SIGMA_KW * (Ts_K**4)
                progress_bar.progress((i + 1) / resolution)
                
            # Force figure and axes to have a white background
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            cp = ax.contourf(X_grid, L_grid, Flux_grid, levels=30, cmap='inferno')
            
            # Formatting the colorbar text to remain legible on white background
            cbar = fig.colorbar(cp)
            cbar.set_label('Incident Flux (kW/m²)', rotation=270, labelpad=15, color='black')
            cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
            
            # Overlay thresholds
            thresholds_to_plot = [t for t in [4.7, 12.6, 23.0, 35.0] if t <= np.max(Flux_grid) and t >= np.min(Flux_grid)]
            if thresholds_to_plot:
                CS = ax.contour(X_grid, L_grid, Flux_grid, levels=thresholds_to_plot, colors='white', linestyles='dashed', linewidths=1.5)
                # inline_spacing adds padding so the contour line doesn't strike through the text
                ax.clabel(CS, inline=True, inline_spacing=5, fontsize=10, fmt='%1.1f kW/m²', colors='white')
                
            ax.set_xlabel("Horizontal Offset from Source Centerline (X) (m)", color='black')
            ax.set_ylabel("Perpendicular Distance from Source (L) (m)", color='black')
            ax.set_title(f"Radiant Footprint at Elevation Z={Zr}m (Angle={theta_deg}°)", color='black')
            
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            ax.plot([-W/2, W/2], [0, 0], color='cyan', linewidth=4, label='Emitter Source Base')
            
            legend = ax.legend(loc='upper right', facecolor='white', edgecolor='black')
            for text in legend.get_texts():
                text.set_color("black")
            
            st.pyplot(fig, clear_figure=True, transparent=False)
            
            # Save to buffer for download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
            st.download_button(
                label="Download Map as PNG",
                data=buf.getvalue(),
                file_name="radiant_footprint_map.png",
                mime="image/png",
                use_container_width=True
            )
            
            progress_bar.empty()
