
import streamlit as st
import numpy as np
import pandas as pd
from lmfit import Model
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Grain-Boundary Mobility Model Fitting",
    page_icon="📈",
    layout="wide",
)

# Title and Description
st.title("📊 Unified Mobility Model Fitting Tool")
st.markdown("""
This interactive tool fits the **Grain-Boundary-Limited Mobility Model** to your experimental data. 

Upload your **CSV file** with two columns:
- `T` → Temperature (in Kelvin)
- `mu` → Carrier mobility (in cm²/V·s)

🔬 This model incorporates thermionic barrier, grain-boundary scattering, and phonon-limited transport.
""")

# Sidebar Information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Crystal_structure_ZnO.png/220px-Crystal_structure_ZnO.png", width=220)
    st.header("Instructions")
    st.markdown("""
    - Upload your data as a CSV file.
    - The fitting model will extract key physical parameters.
    - Scroll down to view model fit, equations, and fit quality.
    """)

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'T' not in data.columns or 'mu' not in data.columns:
            st.error("CSV must contain 'T' and 'mu' columns.")
        else:
            T_data = data['T'].values
            mu_exp = data['mu'].values

            # Sort data by T for smooth plotting
            sort_idx = np.argsort(T_data)
            T_data = T_data[sort_idx]
            mu_exp = mu_exp[sort_idx]

            # Constants
            kB = 8.617333262145e-5  # eV/K

            # Unified Mobility Model with fixed p = 1.5
            def mobility_model(T, mu_w, phi_GB, l300, w_GB):
                p = 1.5
                l_T = l300 * (T / 300)**(-p)
                geom_factor = l_T / (l_T + w_GB)
                thermionic_factor = np.exp(-phi_GB / (kB * T))
                mu_eff = mu_w * thermionic_factor * geom_factor
                return mu_eff

            model = Model(mobility_model)
            params = model.make_params(mu_w=300, phi_GB=0.1, l300=20, w_GB=5)
            params['mu_w'].set(min=0, max=2000)
            params['phi_GB'].set(min=0, max=0.5)
            params['l300'].set(min=1, max=100)
            params['w_GB'].set(min=1, max=50)

            # Fitting
            result = model.fit(mu_exp, params, T=T_data)

            # Results
            st.subheader("📄 Fit Summary Report")
            st.text(result.fit_report())

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(T_data, mu_exp, 'bo', label='Experimental Data')
            ax.plot(T_data, result.best_fit, 'r-', label='Fitted Model')
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Mobility (cm²/V·s)')
            ax.set_title('Grain-Boundary-Limited Mobility Fit')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # Parameter Table
            st.subheader("📌 Fitted Parameters")
            fit_table = pd.DataFrame({
                'Parameter': list(result.params.keys()),
                'Value': [f"{v.value:.4g}" for v in result.params.values()],
                'Std. Error': [f"± {v.stderr:.2g}" if v.stderr else "-" for v in result.params.values()]
            })
            st.dataframe(fit_table)

            st.success("✅ Model fitting completed successfully.")

            st.markdown("""
            ### ⚠️ Interpretation Note:
            - The scattering exponent `p` has been fixed to 1.5 to reflect acoustic phonon dominance.
            - Strong correlations may occur between parameters; use known values (e.g., SEM grain size) when possible.
            - For more robust fitting, increase data points or constrain parameters.
            """)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("""
👨‍🔬 Developed by **Gbadebo Taofeek Yusuf et al.**  
🔗 [Zenodo Dataset & Code](https://doi.org/10.5281/zenodo.15617024)  
📚 Referenced in: *Unified Mobility Model for Grain-Boundary-Limited Transport in Polycrystalline Materials*
""")
