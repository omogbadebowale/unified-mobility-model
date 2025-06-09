import streamlit as st
import numpy as np
import pandas as pd
from lmfit import Model
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Unified Mobility Model Fitting Tool")

st.markdown("""
Upload your experimental mobility data (CSV file with two columns: `T` for temperature and `mu` for mobility).
The app will fit the unified grain-boundary-limited mobility model and extract key parameters.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'T' not in data.columns or 'mu' not in data.columns:
            st.error("CSV must contain 'T' and 'mu' columns.")
        else:
            T_data = data['T'].values
            mu_exp = data['mu'].values

            # Constants
            kB = 8.617333262145e-5  # eV/K

            # Unified Mobility Model
            def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
                l_T = l300 * (T / 300)**(-p)
                geom_factor = l_T / (l_T + w_GB)
                thermionic_factor = np.exp(-phi_GB / (kB * T))
                mu_eff = mu_w * thermionic_factor * geom_factor
                return mu_eff

            model = Model(mobility_model)
            params = model.make_params(mu_w=300, phi_GB=0.1, l300=20, w_GB=5, p=1.5)
            params['mu_w'].set(min=0, max=2000)
            params['phi_GB'].set(min=0, max=0.5)
            params['l300'].set(min=1, max=100)
            params['w_GB'].set(min=1, max=50)
            params['p'].set(min=1.0, max=3.0)

            result = model.fit(mu_exp, params, T=T_data)

            st.subheader("Fit Results")
            st.text(result.fit_report())

            # Plot
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(T_data, mu_exp, 'bo', label='Experimental Data')
            ax.plot(T_data, result.best_fit, 'r-', label='Model Fit')
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Mobility (cm²/V·s)')
            ax.set_title('Unified Mobility Model Fit')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.success("Fitting completed successfully.")

            st.markdown("""
            **Model Interpretation Caution:** 
            This model may produce excellent fit quality but extracted parameters can be highly uncertain due to parameter correlations, limited data, or narrow temperature range. 
            Interpret fitted values carefully and consider fixing known parameters if available.
            """)

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("---")
st.markdown("Developed based on: Gbadebo Taofeek Yusuf et al., Unified Mobility Model for Grain-Boundary-Limited Transport in Polycrystalline Materials. DOI: [10.5281/zenodo.15617024](https://doi.org/10.5281/zenodo.15617024)")
