# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.title("Anchored Unified Mobility Model")

# Sidebar inputs
st.sidebar.header("Data Inputs & Priors")

# 1) Polycrystal mobility + carrier concentration
poly_file = st.sidebar.file_uploader(
    "Upload polycrystal μ(T) CSV (columns: T [K], mu [cm²/Vs])",
    type="csv"
)
n = st.sidebar.number_input(
    "Carrier concentration n (cm⁻³)", value=1e18, format="%.3e"
)

# 2) Barrier anchor: either σ(T) or manual Φ_GB
st.sidebar.subheader("Barrier Anchor")
use_sigma = st.sidebar.checkbox("Provide σ(T)?", value=False)
if use_sigma:
    sigma_file = st.sidebar.file_uploader(
        "Upload low-T σ(T) CSV (columns: T [K], sigma [S/m])",
        type="csv"
    )
else:
    Phi_prior = st.sidebar.number_input("Manual Φ_GB (eV)", value=0.12, min_value=0.0)
    Phi_delta = st.sidebar.number_input("±Δ Φ_GB",        value=0.02, min_value=0.0)

# 3) Phonon anchor: either single-crystal μ(T) or manual μ_w/p
st.sidebar.subheader("Phonon Anchor")
use_single = st.sidebar.checkbox("Provide single-crystal μ(T)?", value=False)
if use_single:
    single_file = st.sidebar.file_uploader(
        "Upload single-crystal μ(T) CSV (T, mu)",
        type="csv"
    )
else:
    mu_w_prior = st.sidebar.number_input("Manual μ_w (cm²/Vs)", value=200.0)
    mu_w_delta = st.sidebar.number_input("±Δ μ_w",           value=20.0)
    p_prior    = st.sidebar.number_input("Manual p",         value=2.0)
    p_delta    = st.sidebar.number_input("±Δ p",             value=0.2)

# Validate inputs
if not poly_file:
    st.warning("Please upload the polycrystal μ(T) CSV.")
    st.stop()
if use_sigma and not sigma_file:
    st.warning("Please upload σ(T) CSV for barrier extraction.")
    st.stop()
if use_single and not single_file:
    st.warning("Please upload single-crystal μ(T) CSV for phonon anchor.")
    st.stop()

# Load data
df_poly = pd.read_csv(poly_file)
T_poly  = df_poly["T"].values
mu_poly = df_poly["mu"].values

kB = 8.617333262e-5  # eV/K
e  = 1.602e-19       # C

# — Stage A: Barrier extraction —————————————
if use_sigma:
    df_sig = pd.read_csv(sigma_file)
    T_sig, sigma = df_sig["T"].values, df_sig["sigma"].values
    invT = 1.0 / T_sig
    lnσ  = np.log(sigma)
    def arrh(x, Phi, C): return -Phi/(kB * x) + C
    poptA, covA = curve_fit(arrh, invT, lnσ, p0=[0.1, 0.0])
    Phi_GB = poptA[0]
    Phi_err = np.sqrt(covA[0,0])
else:
    Phi_GB  = Phi_prior
    Phi_err = Phi_delta

st.subheader("Stage A: Barrier Height")
st.write(f"Φ_GB = **{Phi_GB:.3f}** ± **{Phi_err:.3f}** eV")

# — Stage B: Phonon anchor —————————————
if use_single:
    df_single = pd.read_csv(single_file)
    T_s, mu_s = df_single["T"].values, df_single["mu"].values
    def phonon_model(T, mu_w, p): return mu_w * (T/300.0)**(-p)
    poptB, covB = curve_fit(phonon_model, T_s, mu_s, p0=[200,2])
    mu_w_fit, p_fit = poptB
    mu_w_err, p_err = np.sqrt(np.diag(covB))
    mu_w_bounds = (mu_w_fit * 0.9, mu_w_fit * 1.1)
    p_bounds    = (p_fit    * 0.9, p_fit    * 1.1)
    st.subheader("Stage B: Phonon Anchor (Fitted)")
    st.write(f"μ_w = **{mu_w_fit:.1f}** ± **{mu_w_err:.1f}** cm²/Vs")
    st.write(f"p   = **{p_fit:.2f}** ± **{p_err:.2f}**")
else:
    mu_w_fit = mu_w_prior
    p_fit    = p_prior
    mu_w_bounds = (mu_w_prior - mu_w_delta, mu_w_prior + mu_w_delta)
    p_bounds    = (p_prior    - p_delta,    p_prior    + p_delta)
    st.subheader("Stage B: Phonon Anchor (Manual)")
    st.write(f"μ_w = **{mu_w_prior:.1f}** ± **{mu_w_delta:.1f}** cm²/Vs")
    st.write(f"p   = **{p_prior:.2f}** ± **{p_delta:.2f}**")

# — Stage C: Polycrystal fit —————————————
def unified_model(T, l300, w):
    lT = l300 * (T/300.0)**(-p_fit)
    P  = np.exp(-Phi_GB / (kB * T))
    G  = lT / (lT + w)
    return mu_w_fit * P * G

# bounds: l300 ∈ [1,100] nm, w ∈ [0.1,10] nm
l300_bounds = (1e-9,   100e-9)
w_bounds    = (0.1e-9, 10e-9)
p0C = [30e-9, 2e-9]

poptC, covC = curve_fit(
    unified_model, T_poly, mu_poly,
    p0=p0C,
    bounds=([l300_bounds[0], w_bounds[0]],
            [l300_bounds[1], w_bounds[1]])
)
l300_fit, w_fit = poptC
l300_err, w_err = np.sqrt(np.diag(covC))

st.subheader("Stage C: Polycrystal Fit")
st.write(f"ℓ₃₀₀ = **{l300_fit*1e9:.1f}** ± **{l300_err*1e9:.1f}** nm")
st.write(f"w_GB  = **{w_fit*1e9:.2f}** ± **{w_err*1e9:.2f}** nm")

# Plot final fit
fig, ax = plt.subplots()
ax.scatter(T_poly, mu_poly, c="k", label="Poly μ(T)")
T_fit = np.linspace(T_poly.min(), T_poly.max(), 300)
ax.plot(T_fit, unified_model(T_fit, *poptC),
         color="red", label="Unified Fit")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mobility (cm²/Vs)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
