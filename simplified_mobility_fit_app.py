
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Constants
K_B_EV = 8.617333262e-5

# Model function
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l300 *= 1e-9  # nm to m
    w_GB *= 1e-9  # nm to m
    l_T = l300 * (T / 300.0) ** (-p)
    geom = l_T / (l_T + w_GB)
    thermo = np.exp(-phi_GB / (K_B_EV * T))
    return mu_w * thermo * geom

# Fit only Φ_GB and w_GB
def run_fit(T, mu, mu_w, l300, p, phi_GB_guess, w_GB_guess):
    model = Model(mobility_model)
    params = Parameters()
    params.add("mu_w", value=mu_w, vary=False)
    params.add("l300", value=l300, vary=False)
    params.add("p", value=p, vary=False)
    params.add("phi_GB", value=phi_GB_guess, min=0.005, max=0.3)
    params.add("w_GB", value=w_GB_guess, min=1, max=20)
    result = model.fit(mu, params, T=T)
    return result

# Streamlit UI
st.set_page_config("Simplified Mobility Fit", layout="wide")
st.title("🔬 Simplified Mobility Fitting Tool (Φ_GB, w_GB Only)")

st.sidebar.markdown("### Fixed Parameters")
mu_w = st.sidebar.number_input("μw (cm²/V·s)", value=100)
l300 = st.sidebar.number_input("ℓ300 (nm)", value=20)
p = st.sidebar.slider("p (phonon exponent)", 1.0, 3.0, value=2.0)

st.sidebar.markdown("### Parameters to Fit")
phi_GB_guess = st.sidebar.number_input("Initial Φ_GB (eV)", value=0.1)
w_GB_guess = st.sidebar.number_input("Initial w_GB (nm)", value=5.0)

uploaded = st.file_uploader("📂 Upload CSV with Temperature,Mobility", type="csv")
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

        result = run_fit(T, mu, mu_w, l300, p, phi_GB_guess, w_GB_guess)
        T_fit = np.linspace(min(T), max(T), 300)
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
