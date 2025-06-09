
import streamlit as st
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
import matplotlib.pyplot as plt

st.title("Universal Mobility Model Fitting Tool")

st.markdown("""
Upload your experimental mobility data (CSV file with two columns: `T` for temperature and `mu` for mobility).
This universal tool fits a grain-boundary-limited mobility model with smart defaults and optional overrides.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    T_data = data["T"].values
    mu_exp = data["mu"].values

    # Constants
    kB = 8.617333262145e-5  # eV/K

    # Mobility model definition
    def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
        l_T = l300 * (T / 300) ** (-p)
        geom_factor = l_T / (l_T + w_GB)
        thermionic_factor = np.exp(-phi_GB / (kB * T))
        return mu_w * thermionic_factor * geom_factor

    model = Model(mobility_model)
    st.sidebar.header("Fit Settings")

    fix_p = st.sidebar.checkbox("Fix exponent p", value=True)
    p_val = st.sidebar.slider("p (phonon exponent)", 1.0, 3.0, 1.5, 0.1)
    fix_phi = st.sidebar.checkbox("Fix phi_GB", value=False)
    fix_l300 = st.sidebar.checkbox("Fix l300", value=False)
    fix_wGB = st.sidebar.checkbox("Fix w_GB", value=False)

    params = Parameters()
    params.add("mu_w", value=300, min=0, max=3000)
    params.add("phi_GB", value=0.1, min=0, max=0.5, vary=not fix_phi)
    params.add("l300", value=20, min=1, max=100, vary=not fix_l300)
    params.add("w_GB", value=5, min=1, max=50, vary=not fix_wGB)
    params.add("p", value=p_val, min=1.0, max=3.0, vary=not fix_p)

    result = model.fit(mu_exp, params, T=T_data)

    st.subheader("Fit Report")
    st.text(result.fit_report())

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_data, mu_exp, "bo", label="Experimental")
    ax.plot(T_data, result.best_fit, "r-", label="Model fit")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mobility (cm²/V·s)")
    ax.set_title("Mobility Fit")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.success("Fitting complete. Check parameter uncertainties and correlations for reliability.")

st.markdown("---")
st.markdown("Based on: Yusuf et al., *Unified Mobility Model for Grain-Boundary-Limited Transport in Polycrystalline Materials*. [DOI: 10.5281/zenodo.15617024](https://doi.org/10.5281/zenodo.15617024)")
