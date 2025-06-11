# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.title("Unified Mobility Model for Polycrystalline Materials")

# 1. Upload your CSV
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns 'T' (K) and 'mu' (cm²/Vs)",
    type="csv"
)
if not uploaded_file:
    st.warning("Please upload a mobility CSV file (T, mu).")
    st.stop()

data = pd.read_csv(uploaded_file)
T = data['T'].values
mu = data['mu'].values

# 2. Auto-estimate μ_w from the high-T plateau (last 5 pts)
order     = np.argsort(T)
mu_sorted = mu[order]
mu_w_est  = float(np.mean(mu_sorted[-5:]))

# 3. Sidebar: define bounds for all five parameters
st.sidebar.header("Fit Bounds (all parameters free)")

def get_bounds(label, lo_default, hi_default, factor=1.0):
    lo = st.sidebar.number_input(f"{label} min", value=lo_default)
    hi = st.sidebar.number_input(f"{label} max", value=hi_default)
    return lo * factor, hi * factor

bounds = {
    # ±20% around the plateau estimate
    'mu_w': get_bounds("μ_w (cm²/Vs)", mu_w_est*0.8, mu_w_est*1.2),
    # Grain-boundary barrier in eV
    'phi':  get_bounds("Φ_GB (eV)", 0.0, 0.5),
    # Mean free path at 300K in nm → meters
    'l300': get_bounds("ℓ₃₀₀ (nm)", 1.0, 100.0, factor=1e-9),
    # Scattering exponent
    'p':    get_bounds("p", 1.0, 3.0),
    # Boundary width in nm → meters
    'w':    get_bounds("w_GB (nm)", 0.1, 10.0, factor=1e-9),
}

# 4. Define the unified mobility model
k_B = 8.617333262e-5  # eV/K
def unified_model(T, mu_w, phi, l300, p, w):
    l_T = l300 * (T/300.0)**(-p)
    P_GB = np.exp(-phi / (k_B * T))
    G = l_T / (l_T + w)
    return mu_w * P_GB * G

# 5. Build initial guess & bounds arrays
p0 = [(lo+hi)/2 for lo,hi in bounds.values()]
lower, upper = zip(*bounds.values())

# 6. Perform the fit
try:
    popt, pcov = curve_fit(
        unified_model, T, mu,
        p0=p0, bounds=(lower, upper)
    )
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    st.error(f"Fit failed: {e}")
    st.stop()

# 7. Show fitted parameters & uncertainties
param_names = ['μ_w (cm²/Vs)', 'Φ_GB (eV)', 'ℓ₃₀₀ (m)', 'p', 'w_GB (m)']
results_df = pd.DataFrame({
    'Estimate':    popt,
    'Uncertainty': perr
}, index=param_names)

st.subheader("Fitted Parameters")
st.table(results_df)

# 8. Show correlation matrix
corr = pcov / np.outer(perr, perr)
corr_df = pd.DataFrame(corr, index=param_names, columns=param_names)
st.subheader("Parameter Correlation Matrix")
st.table(corr_df)

# 9. Plot data vs. fit
fig, ax = plt.subplots()
ax.scatter(T, mu, color='black', label='Data')
T_fit = np.linspace(T.min(), T.max(), 400)
ax.plot(T_fit, unified_model(T_fit, *popt), color='blue', label='Model Fit')
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mobility (cm²/Vs)")
ax.set_title("Unified Mobility Model Fit")
ax.legend()
ax.grid(True)
st.pyplot(fig)
