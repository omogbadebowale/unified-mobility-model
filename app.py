import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Unified Mobility Model Function
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p=1.5):
    kB = 8.617333262145e-5  # eV/K
    l_T = l300 * (T / 300)**(-p)
    geometric = l_T / (l_T + w_GB)
    thermionic = np.exp(-phi_GB / (kB * T))
    return mu_w * thermionic * geometric

# Fitting function with physical constraints
def fit_with_constraints(T_data, mu_data):
    enable_barrier = np.min(T_data) < 400
    model = Model(mobility_model)
    params = Parameters()
    params.add('mu_w', value=400, min=50, max=1000)
    params.add('l300', value=20, min=5, max=100)
    params.add('w_GB', value=5, min=3, max=10)
    params.add('p', value=1.5, vary=False)

    if enable_barrier:
        params.add('phi_GB', value=0.1, min=0.05, max=0.15)
    else:
        params.add('phi_GB', value=0, vary=False)

    result = model.fit(mu_data, params, T=T_data)
    return result

# Streamlit App
st.title("Unified Mobility Model Fitting Tool")
uploaded_file = st.file_uploader("Upload your data (CSV with 'T' and 'mu')", type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    T_data = data['T'].values
    mu_data = data['mu'].values

    st.write("### Uploaded Data")
    st.dataframe(data)

    result = fit_with_constraints(T_data, mu_data)

    st.write("### Fit Report")
    st.text(result.fit_report())

    # Plotting fit
    T_fit = np.linspace(min(T_data) - 20, max(T_data) + 20, 300)
    mu_fit = result.eval(T=T_fit)

    fig, ax = plt.subplots()
    ax.scatter(T_data, mu_data, label='Data', color='blue')
    ax.plot(T_fit, mu_fit, label='Fit', color='red')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Mobility (cm²/V·s)')
    ax.set_title('Mobility Fit')
    ax.legend()
    st.pyplot(fig)

    # Plotting residuals
    residuals = mu_data - result.eval(T=T_data)
    fig2, ax2 = plt.subplots()
    ax2.scatter(T_data, residuals, color='green')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('Residuals of the Fit')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Residuals')
    st.pyplot(fig2)

    # Warnings on parameter uncertainty
    st.write("### Parameter Stability Check")
    for name, p in result.params.items():
        if p.stderr and abs(p.value) > 0:
            rel_err = p.stderr / abs(p.value)
            if rel_err > 0.5:
                st.warning(f"⚠ {name}: High uncertainty ({rel_err:.1%})")
