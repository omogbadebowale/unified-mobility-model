
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import itertools

# Constants
K_B_EV = 8.617333262e-5

# Mobility model function
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l300 *= 1e-9  # Convert from nm to meters
    w_GB *= 1e-9
    l_T = l300 * (T / 300.0) ** (-p)
    geom = l_T / (l_T + w_GB)
    thermo = np.exp(-phi_GB / (K_B_EV * T))
    return mu_w * thermo * geom

# Run fit with selected parameters
def run_fit(T, mu, params_dict, float_keys):
    model = Model(mobility_model)
    params = Parameters()
    for key in ["mu_w", "phi_GB", "l300", "w_GB", "p"]:
        vary = key in float_keys
        val = params_dict[key]
        minval, maxval = {
            "mu_w": (10, 1000),
            "phi_GB": (0.0005, 0.3),
            "l300": (1, 100),
            "w_GB": (1, 100),
            "p": (1.0, 3.0)
        }[key]
        params.add(key, value=val, min=minval, max=maxval, vary=vary)
    return model.fit(mu, params, T=T)

# Streamlit UI
st.set_page_config("Unified Mobility Model", layout="wide", page_icon="📊")
st.title("📊 Unified Polycrystalline Mobility Model Fitting Tool")

materials = {
    "ZnO / SrTiO₃ (Oxide)": {"mu_w": 100, "phi_GB": 0.15, "l300": 15, "w_GB": 5, "p": 2.0},
    "Bi₂Te₃ / SnSe (Chalcogenide)": {"mu_w": 400, "phi_GB": 0.05, "l300": 30, "w_GB": 10, "p": 1.6},
    "Mg₂Si / PbTe (Intermetallic)": {"mu_w": 300, "phi_GB": 0.05, "l300": 60, "w_GB": 5, "p": 1.5},
    "Custom": {"mu_w": 300, "phi_GB": 0.1, "l300": 20, "w_GB": 5, "p": 1.5}
}

material = st.selectbox("📁 Select Material Class", list(materials))
defaults = materials[material]

guided = st.checkbox("🧪 Enable Guided Fit Mode", value=True)

guided_presets = {
    "ZnO / SrTiO₃ (Oxide)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
    "Bi₂Te₃ / SnSe (Chalcogenide)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
    "Mg₂Si / PbTe (Intermetallic)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
    "Custom": {"float": ["mu_w", "phi_GB", "l300"], "fix": ["w_GB", "p"]}
}

if guided:
    float_keys = guided_presets[material]["float"]
    st.info(f"Guided fit mode enabled. Floating: {', '.join(float_keys)}")
else:
    float_keys = st.multiselect("🔧 Select Parameters to Fit", ["mu_w", "phi_GB", "l300", "w_GB", "p"], default=["mu_w", "phi_GB"])

# Sidebar Inputs
mu_w = st.sidebar.number_input("μw (cm²/V·s)", value=defaults["mu_w"])
phi_GB = st.sidebar.number_input("ΦGB (eV)", value=defaults["phi_GB"])
l300 = st.sidebar.number_input("ℓ300 (nm)", value=defaults["l300"])
w_GB = st.sidebar.number_input("wGB (nm)", value=defaults["w_GB"])
p = st.sidebar.slider("p (phonon scattering exponent)", 1.0, 3.0, value=defaults["p"], step=0.05)

# File upload
uploaded = st.file_uploader("📂 Upload CSV (Temperature, Mobility)", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    if data.shape[1] != 2:
        st.error("CSV must have exactly 2 columns.")
    else:
        T = data.iloc[:, 0].values
        mu = data.iloc[:, 1].values
        st.write("### 📋 Experimental Data", data)

        if len(T) < 10:
            st.warning("⚠️ Less than 10 data points may lead to unstable fits.")

        result = run_fit(T, mu, {
            "mu_w": mu_w, "phi_GB": phi_GB, "l300": l300, "w_GB": w_GB, "p": p
        }, float_keys)

        T_fit = np.linspace(min(T), max(T), 500)
        mu_fit = result.eval(T=T_fit)
        mu_unc = result.eval_uncertainty(sigma=1, T=T_fit)

        st.subheader("📈 Fit Visualization")
        fig, ax = plt.subplots()
        ax.plot(T, mu, 'bo', label="Data")
        ax.plot(T_fit, mu_fit, 'r-', label="Model Fit")
        ax.fill_between(T_fit, mu_fit - mu_unc, mu_fit + mu_unc, alpha=0.3, color="red", label="±1σ")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mobility (cm²/V·s)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Residuals")
        residuals = mu - result.eval(T=T)
        fig2, ax2 = plt.subplots()
        ax2.plot(T, residuals, 'ko-')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Residual (cm²/V·s)")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("🧾 Fit Report")
        st.text(result.fit_report())
