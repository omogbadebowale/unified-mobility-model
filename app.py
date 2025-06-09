import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mobility_model import run_fit
from utils import plot_fit, evaluate_fit_quality

st.set_page_config(layout="wide", page_title="Grain-Boundary Mobility Model")

st.title("Unified Mobility Model Fitting for Polycrystalline Materials")

# Upload data
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel with Temperature and Mobility columns)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if 'Temperature' not in df.columns or 'Mobility' not in df.columns:
        st.error("File must contain 'Temperature' and 'Mobility' columns.")
        st.stop()

    T_data = df['Temperature'].values
    mu_data = df['Mobility'].values

    st.sidebar.header("Model Parameters")
    mu_w = st.sidebar.slider("Initial μw", 50, 1000, 300)
    phi_GB = st.sidebar.slider("Initial ΦGB (eV)", 0.0, 0.5, 0.05)
    l300 = st.sidebar.slider("Initial ℓ₃₀₀ (nm)", 1, 100, 20)
    w_GB = st.sidebar.slider("Initial w_GB (nm)", 1, 50, 5)
    p = st.sidebar.slider("Initial phonon exponent p", 1.0, 3.0, 1.5, 0.1)

    fix_p = st.sidebar.checkbox("Fix p")
    fix_params = []
    if fix_p:
        fix_params.append('p')

    fallback = st.sidebar.checkbox("Use simplified thermionic model only")

    if st.button("Fit Model"):
        result = run_fit(
            T_data, mu_data,
            initial_params={'mu_w': mu_w, 'phi_GB': phi_GB, 'l300': l300, 'w_GB': w_GB, 'p': p},
            fix_params=fix_params,
            fallback=fallback
        )
        st.subheader("Fitting Results")
        st.text(result.fit_report())

        fig = plot_fit(T_data, mu_data, result)
        st.pyplot(fig)

        R2, RMSE = evaluate_fit_quality(mu_data, result.best_fit)
        st.markdown(f"**R²** = {R2:.4f}, **RMSE** = {RMSE:.2f} cm²/V·s")
