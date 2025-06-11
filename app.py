# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.title("Anchored Unified Mobility Model")

# Sidebar inputs
st.sidebar.header("Data & Priors")

# Polycrystal mobility + carrier n
poly_file = st.sidebar.file_uploader(
    "1) Polycrystal μ(T) CSV (T [K], mu [cm²/Vs])", type="csv"
)
n = st.sidebar.number_input(
    "Carrier concentration n (cm⁻³)", value=1e18, format="%.3e"
)

# Optional single-crystal mobility
use_single = st.sidebar.checkbox("Provide single-crystal μ(T)?", value=False)
if use_single:
    single_file = st.sidebar.file_uploader(
        "2) Single-crystal μ(T) CSV (T [K], mu [cm²/Vs])", type="csv"
    )
else:
    mu_w_prior = st.sidebar.number_input("μ_w prior (cm²/Vs)", value=200.0)
    mu_w_delta = st.sidebar.number_input("±Δ μ_w", value=20.0)
    p_prior    = st.sidebar.number_input("p prior", value=2.0)
    p_delta    = st.sidebar.number_input("±Δ p", value=0.2)

if not poly_file:
    st.warning("Please upload your polycrystal μ(T) CSV and enter n.")
    st.stop()

# Load poly data
df_poly = pd.read_csv(poly_file)
T_poly  = df_poly["T"].values
mu_poly = df_poly["mu"].values

kB = 8.617333262e-5  # eV/K
e  = 1.602e-19       # C

# — Stage A: Barrier from low-T μ(T) -------------------
# Convert μ to σ = n·e·μ
sigma = n * 1e6 * e * (mu_poly * 1e-4)  # μ cm²→m², n cm⁻³→m⁻³

# Use low-T cutoff slider
Tcut = st.sidebar.slider(
    "Barrier-fit T cutoff (K)", 
    int(T_poly.min()), int(T_poly.max()), int(0.5 * T_poly.max())
)
mask = T_poly <= Tcut
invT = 1.0 / T_poly[mask]
lnσ  = np.log(sigma[mask])

# Fit lnσ = –Φ/(kB T) + C
def arrh(x, Phi, C): return -Phi/(kB * x) + C
poptA, covA = curve_fit(arrh, invT, lnσ, p0=[0.1, 0.0])
Phi_GB, C0 = poptA
Phi_err = np.sqrt(covA[0,0])

st.subheader("Stage A: Grain-Boundary Barrier")
st.write(f"Φ_GB = **{Phi_GB:.3f} ± {Phi_err:.3f}** eV")

# — Stage B: Phonon anchor -----------------------------
st.subheader("Stage B: Phonon Anchor")

if use_single:
    if not single_file:
        st.warning("Please upload single-crystal μ(T) CSV.")
        st.stop()
    df_single = pd.read_csv(single_file)
    T_s, mu_s = df_single["T"].values, df_single["mu"].values
    def phonon_model(T, mu_w, p): return mu_w * (T/300)**(-p)
    poptB, covB = curve_fit(phonon_model, T_s, mu_s, p0=[200,2])
    mu_w_fit, p_fit = poptB
    mu_w_err, p_err = np.sqrt(np.diag(covB))
    st.write(f"μ_w = **{mu_w_fit:.1f} ± {mu_w_err:.1f}** cm²/Vs")
    st.write(f"p   = **{p_fit:.2f} ± {p_err:.2f}**")
    mu_w_bounds = (mu_w_fit * 0.9, mu_w_fit * 1.1)
    p_bounds    = (p_fit * 0.9,    p_fit * 1.1)
else:
    mu_w_fit = mu_w_prior
    p_fit    = p_prior
    st.write(f"Using μ_w = **{mu_w_prior:.1f} ± {mu_w_delta:.1f}** cm²/Vs")
    st.write(f"Using p   = **{p_prior:.2f} ± {p_delta:.2f}**")
    mu_w_bounds = (mu_w_prior - mu_w_delta, mu_w_prior + mu_w_delta)
    p_bounds    = (p_prior - p_delta,       p_prior + p_delta)

# — Stage C: Fit ℓ300 & w_GB -----------------------------
st.subheader("Stage C: Polycrystal Fit")

def unified_model(T, l300, w):
    lT = l300 * (T/300)**(-p_fit)
    P  = np.exp(-Phi_GB / (kB * T))
    G  = lT / (lT + w)
    return mu_w_fit * P * G

# bounds for l300 (1–100 nm) and w_GB (0.1–10 nm)
l300_bounds = (1e-9,   100e-9)
w_bounds    = (0.1e-9, 10e-9)

p0C = [30e-9, 2e-9]
lowerC, upperC = [l300_bounds[0], w_bounds[0]], [l300_bounds[1], w_bounds[1]]
poptC, covC = curve_fit(
    unified_model, T_poly, mu_poly,
    p0=p0C, bounds=(lowerC, upperC)
)
l300_fit, w_fit = poptC
l300_err, w_err = np.sqrt(np.diag(covC))

st.write(f"ℓ₃₀₀ = **{l300_fit*1e9:.1f} ± {l300_err*1e9:.1f}** nm")
st.write(f"w_GB  = **{w_fit*1e9:.2f} ± {w_err*1e9:.2f}** nm")

# — Plot all stages ------------------------------------
fig, ax = plt.subplots()
ax.scatter(T_poly, mu_poly, c="k", label="Poly μ(T)")
Tfit = np.linspace(T_poly.min(), T_poly.max(), 400)
ax.plot(Tfit, unified_model(Tfit, *poptC), "-r", label="Poly Fit")
ax.set_xlabel("T (K)")
ax.set_ylabel("μ (cm²/Vs)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
