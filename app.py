
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

st.set_page_config(page_title="Unified Mobility Model Fitting", layout="wide")

# Constants
K_B_EV = 8.617333262e-5  # eV/K

# Unified mobility model
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l_T = l300 * (T / 300.0) ** (-p)
    geom_factor = l_T / (l_T + w_GB)
    thermionic_factor = np.exp(-phi_GB / (K_B_EV * T))
    return mu_w * thermionic_factor * geom_factor

# Fit function
def fit_mobility_model(T_data, mu_data, init_params, vary_p):
    model = Model(mobility_model)
    params = Parameters()
    params.add('mu_w', value=init_params['mu_w'], min=10, max=1000)
    params.add('phi_GB', value=init_params['phi_GB'], min=0, max=0.5)
    params.add('l300', value=init_params['l300'], min=1, max=100)
    params.add('w_GB', value=init_params['w_GB'], min=1, max=50)
    params.add('p', value=init_params['p'], min=1.0, max=3.0, vary=not vary_p)

    result = model.fit(mu_data, params, T=T_data)
    return result

# Material-based guidance
material_defaults = {
    "Oxide (ZnO, SrTiO₃)": {'mu_w': 100, 'phi_GB': 0.15, 'l300': 15, 'w_GB': 5, 'p': 2.0},
    "Chalcogenide (Bi₂Te₃, SnSe)": {'mu_w': 400, 'phi_GB': 0.05, 'l300': 30, 'w_GB': 10, 'p': 1.6},
    "Intermetallic (Mg₂Si, PbTe)": {'mu_w': 300, 'phi_GB': 0.05, 'l300': 60, 'w_GB': 5, 'p': 1.5},
    "Amorphous/Nanocomposite": {'mu_w': 50, 'phi_GB': 0.1, 'l300': 10, 'w_GB': 5, 'p': 2.0},
    "Other / Custom": {'mu_w': 300, 'phi_GB': 0.1, 'l300': 20, 'w_GB': 5, 'p': 1.5},
}

st.title("📈 Unified Mobility Model Fitting Tool")

st.markdown("Upload experimental temperature vs. mobility data to fit the unified grain-boundary-limited transport model.")

material_type = st.selectbox("Material Type", list(material_defaults.keys()))
init_params = material_defaults[material_type]

st.sidebar.header("Model Parameters")
mu_w = st.sidebar.number_input("μw (cm²/V·s)", value=init_params['mu_w'], help="Weighted mobility. Try 50–400.")
phi_GB = st.sidebar.number_input("ΦGB (eV)", value=init_params['phi_GB'], help="Grain-boundary barrier height. 0 for single crystals.")
l300 = st.sidebar.number_input("ℓ₃₀₀ (nm)", value=init_params['l300'], help="Mean free path at 300K.")
w_GB = st.sidebar.number_input("wGB (nm)", value=init_params['w_GB'], help="Grain boundary width.")
fix_p = st.sidebar.checkbox("Fix p (phonon scattering exponent)?", value=True)
p = st.sidebar.number_input("p (if fixed)", value=init_params['p'], help="1.5: acoustic; 2.0+: polar phonons")

uploaded_file = st.file_uploader("Upload CSV file with Temperature and Mobility columns", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if data.shape[1] != 2:
        st.error("CSV must have exactly 2 columns: Temperature, Mobility.")
    else:
        T_data = data.iloc[:,0].values
        mu_data = data.iloc[:,1].values

        st.write("### Experimental Data Preview")
        st.dataframe(data)

        init_dict = {'mu_w': mu_w, 'phi_GB': phi_GB, 'l300': l300, 'w_GB': w_GB, 'p': p}
        result = fit_mobility_model(T_data, mu_data, init_dict, vary_p=fix_p)

        T_fit = np.linspace(min(T_data), max(T_data), 300)
        mu_fit = result.eval(T=T_fit)
        mu_std = result.eval_uncertainty(sigma=1, T=T_fit)

        # Plot fit
        st.subheader("Model Fit with Confidence Band")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(T_data, mu_data, 'bo', label="Experimental Data")
        ax.plot(T_fit, mu_fit, 'r-', label="Model Fit")
        ax.fill_between(T_fit, mu_fit - mu_std, mu_fit + mu_std, color='red', alpha=0.3, label="±1σ Confidence")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mobility (cm²/V·s)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Plot residuals
        residuals = mu_data - result.eval(T=T_data)
        st.subheader("Residuals")
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.plot(T_data, residuals, 'ko-')
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Residual (cm²/V·s)")
        ax2.grid(True)
        st.pyplot(fig2)

        # Fit report
        st.subheader("Fit Report")
        st.code(result.fit_report())
