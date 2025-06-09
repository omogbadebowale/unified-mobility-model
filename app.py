import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Unified Mobility Model
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p=1.5):
    kB = 8.617333262145e-5
    l_T = l300 * (T / 300)**(-p)
    geometric = l_T / (l_T + w_GB)
    thermionic = np.exp(-phi_GB / (kB * T))
    return mu_w * thermionic * geometric

# Fitting Function
def fit_with_constraints(T_data, mu_data):
    model = Model(mobility_model)
    params = Parameters()

    # Setup physical constraints
    params.add('mu_w', value=300, min=100, max=600)
    params.add('l300', value=20, min=10, max=50)
    params.add('w_GB', value=5, vary=False)
    params.add('p', value=1.5, vary=False)

    if np.min(T_data) > 400:
        params.add('phi_GB', value=0.05, vary=False)
    else:
        params.add('phi_GB', value=0.1, min=0.05, max=0.15)

    result = model.fit(mu_data, params, T=T_data)
    return result

# Streamlit UI
st.title("🧪 Unified Mobility Model Fitting Tool")

uploaded = st.file_uploader("Upload CSV with columns 'T' and 'mu'", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    T_data = data['T'].values
    mu_data = data['mu'].values

    st.write("### Experimental Data")
    st.dataframe(data)

    result = fit_with_constraints(T_data, mu_data)
    st.write("### Fit Report")
    st.text(result.fit_report())

    # Diagnostic Warnings
    if result.redchi > 10 or result.rsquared < 0.9:
        st.warning("⚠ Fit quality is low. Consider simplifying your model or reviewing your data.")
    for name, p in result.params.items():
        if p.stderr and abs(p.value) > 0 and p.stderr / abs(p.value) > 0.5:
            st.warning(f"⚠ High uncertainty for {name}: {p.stderr / abs(p.value):.1%}")

    # Plot Fit
    T_fit = np.linspace(min(T_data)-20, max(T_data)+20, 300)
    mu_fit = result.eval(T=T_fit)

    fig, ax = plt.subplots()
    ax.scatter(T_data, mu_data, label="Data", color="blue")
    ax.plot(T_fit, mu_fit, label="Model Fit", color="red")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mobility (cm²/V·s)")
    ax.legend()
    st.pyplot(fig)

    # Residuals
    residuals = mu_data - result.eval(T=T_data)
    fig2, ax2 = plt.subplots()
    ax2.scatter(T_data, residuals)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title("Fit Residuals")
    ax2.set_xlabel("T")
    ax2.set_ylabel("Residuals")
    st.pyplot(fig2)
