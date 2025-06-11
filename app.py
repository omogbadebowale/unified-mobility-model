# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.title("Two-Stage Unified Mobility Model Fit")

# 1) Upload data
uploaded = st.sidebar.file_uploader("Upload CSV with columns 'T' (K) and 'mu' (cm²/Vs)", type="csv")
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)
T = df["T"].values
mu = df["mu"].values

# 2) Literals / estimates
# Estimate μ_w from the highest‐T plateau
idx = np.argsort(T)
mu_plateau = np.mean(mu[idx][-5:])
# Single‐crystal phonon parameters from literature
l300_est = 30e-9     # 30 nm
p_est    = 2.0       # acoustic‐phonon exponent

# 3) Stage 1: Fit Φ_GB and w_GB
def model_stage1(T, phi, w):
    kB = 8.617333262e-5
    lT = l300_est * (T/300)**(-p_est)
    P  = np.exp(-phi/(kB*T))
    G  = lT/(lT + w)
    return mu_plateau * P * G

# bounds: phi in [0,0.5] eV, w in [0.1,10] nm
b1 = ([0.0, 0.1e-9], [0.5, 10e-9])
p1, cov1 = curve_fit(model_stage1, T, mu, p0=[0.1, 2e-9], bounds=b1)
phi_fit, w_fit = p1
sigma1 = np.sqrt(np.diag(cov1))

# 4) Stage 2: Fit μ_w and ℓ₃₀₀
def model_stage2(T, mu_w, l300):
    kB = 8.617333262e-5
    lT = l300 * (T/300)**(-p_est)
    P  = np.exp(-phi_fit/(kB*T))
    G  = lT/(lT + w_fit)
    return mu_w * P * G

# bounds: μ_w ±20% around plateau, ℓ₃₀₀ in [1,100] nm
b2 = ([0.8*mu_plateau, 1e-9], [1.2*mu_plateau, 100e-9])
p2, cov2 = curve_fit(model_stage2, T, mu, p0=[mu_plateau, l300_est], bounds=b2)
mu_w_fit, l300_fit = p2
sigma2 = np.sqrt(np.diag(cov2))

# 5) Display results
st.subheader("Stage 1: Barrier Fit")
st.table(pd.DataFrame({
    "Estimate": [phi_fit, w_fit],
    "Uncertainty": sigma1
}, index=["Φ_GB (eV)", "w_GB (m)"]))

st.subheader("Stage 2: Phonon Fit")
st.table(pd.DataFrame({
    "Estimate": [mu_w_fit, l300_fit],
    "Uncertainty": sigma2
}, index=["μ_w (cm²/Vs)", "ℓ₃₀₀ (m)"]))

# 6) Correlations
corr1 = cov1/np.outer(sigma1, sigma1)
corr2 = cov2/np.outer(sigma2, sigma2)
st.subheader("Stage 1 Correlation")
st.write(corr1)
st.subheader("Stage 2 Correlation")
st.write(corr2)

# 7) Plot
fig, ax = plt.subplots()
ax.scatter(T, mu, color="k", label="Data")
T_fine = np.linspace(T.min(), T.max(), 400)
ax.plot(T_fine, model_stage1(T_fine, *p1), "--r", label="Stage1 Fit")
ax.plot(T_fine, model_stage2(T_fine, *p2), "-b", label="Stage2 Fit")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mobility (cm²/Vs)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
