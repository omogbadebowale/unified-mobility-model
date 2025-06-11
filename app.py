# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.title("Unified Mobility Model for Polycrystalline Materials")

# 1. Upload data
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns 'T' (K) and 'mu' (cm²/Vs)", type="csv"
)
if uploaded_file is None:
    st.warning("Please upload a mobility CSV file to proceed.")
    st.stop()

data = pd.read_csv(uploaded_file)
T = data['T'].values
mu = data['mu'].values

# 2. Sidebar: parameter fixing
st.sidebar.header("Parameter Anchors")
fix_mu_w = st.sidebar.checkbox("Fix μ_w", value=False)
if fix_mu_w:
    mu_w_val = st.sidebar.number_input("Fixed μ_w (cm²/Vs)", value=200.0, min_value=0.0)

fix_l300 = st.sidebar.checkbox("Fix ℓ₃₀₀", value=False)
if fix_l300:
    l300_val_nm = st.sidebar.number_input("Fixed ℓ₃₀₀ (nm)", value=30.0, min_value=0.1)
    l300_val = l300_val_nm * 1e-9

fix_phi = st.sidebar.checkbox("Fix Φ_GB", value=False)
if fix_phi:
    phi_val = st.sidebar.number_input("Fixed Φ_GB (eV)", value=0.12, min_value=0.0)

fix_p = st.sidebar.checkbox("Fix p", value=False)
if fix_p:
    p_val = st.sidebar.number_input("Fixed p", value=2.0, min_value=0.0)

fix_wgb = st.sidebar.checkbox("Fix w_GB", value=False)
if fix_wgb:
    w_val_nm = st.sidebar.number_input("Fixed w_GB (nm)", value=2.0, min_value=0.1)
    w_val = w_val_nm * 1e-9

# 3. Sidebar: bounds for free parameters
st.sidebar.header("Bounds for Free Parameters")
def get_bounds(label, default_min, default_max, factor=1.0):
    lo = st.sidebar.number_input(f"{label} min", value=default_min * factor)
    hi = st.sidebar.number_input(f"{label} max", value=default_max * factor)
    return lo / factor, hi / factor

bounds = {}
if not fix_mu_w:
    bounds['mu_w'] = get_bounds("μ_w (cm²/Vs)", 50, 2000)
if not fix_phi:
    bounds['phi'] = get_bounds("Φ_GB (eV)", 0.0, 0.5)
if not fix_l300:
    bounds['l300'] = get_bounds("ℓ₃₀₀ (nm)", 5, 100, factor=1e-9)
if not fix_p:
    bounds['p'] = get_bounds("p", 1.0, 3.0)
if not fix_wgb:
    bounds['w'] = get_bounds("w_GB (nm)", 0.5, 10.0, factor=1e-9)

# 4. Build unified mobility model dynamically
k_B = 8.617333262e-5  # eV/K
def unified_model(T, *params):
    idx = 0
    mu_w = mu_w_val if fix_mu_w else params[idx]; idx += not fix_mu_w
    phi  = phi_val  if fix_phi  else params[idx]; idx += not fix_phi
    l300 = l300_val if fix_l300 else params[idx]; idx += not fix_l300
    p    = p_val    if fix_p    else params[idx]; idx += not fix_p
    wgb  = w_val    if fix_wgb  else params[idx]
    l_T = l300 * (T/300)**(-p)
    P_GB = np.exp(-phi / (k_B * T))
    G = l_T / (l_T + wgb)
    return mu_w * P_GB * G

# 5. Prepare initial guess and bounds arrays
p0, lower, upper = [], [], []
for key, (lo, hi) in bounds.items():
    p0.append((lo + hi) / 2)
    lower.append(lo)
    upper.append(hi)

# 6. Fit model
try:
    popt, pcov = curve_fit(
        unified_model, T, mu, p0=p0, bounds=(lower, upper)
    )
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    st.error(f"Fit failed: {e}")
    st.stop()

# 7. Display results
param_names = [
    ("μ_w (cm²/Vs)", "mu_w"),
    ("Φ_GB (eV)",      "phi"),
    ("ℓ₃₀₀ (m)",       "l300"),
    ("p",              "p"),
    ("w_GB (m)",       "w")
]

rows = []
for name, key in param_names:
    if key in bounds:
        idx = list(bounds.keys()).index(key)
        est, err = popt[idx], perr[idx]
    else:
        est = {"mu_w":mu_w_val,"phi":phi_val,
               "l300":l300_val,"p":p_val,"w":w_val}[key]
        err = 0.0
    rows.append((name, est, err))

results_df = pd.DataFrame(rows, columns=["Parameter","Estimate","Uncertainty"]).set_index("Parameter")
st.subheader("Fitted Parameters")
st.table(results_df)

st.subheader("Parameter Correlation Matrix")
corr_mat = pcov / np.outer(perr, perr)
corr_df = pd.DataFrame(corr_mat, index=bounds.keys(), columns=bounds.keys())
st.table(corr_df)

# 8. Plot fit
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
