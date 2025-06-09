
import streamlit as st
import numpy as np
import pandas as pd
from lmfit import Model
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Grain-Boundary Mobility Model Fitting",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Unified Mobility Model Fitting Tool")
st.markdown("""
Upload a CSV file with columns:
- `T` (Temperature in Kelvin)
- `mu` (Mobility in cm²/V·s)

This tool fits a grain-boundary-limited mobility model to your data.
""")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Crystal_structure_ZnO.png/220px-Crystal_structure_ZnO.png", width=220)
    st.header("Settings")
    constrained = st.checkbox("Use Constrained Fitting (fix ℓ₃₀₀ & w_GB)", value=True)
    l300_fixed = st.number_input("Fixed ℓ₃₀₀ (nm)", value=20.0, min_value=1.0) if constrained else None
    wGB_fixed = st.number_input("Fixed w_GB (nm)", value=10.0, min_value=1.0) if constrained else None

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'T' not in data.columns or 'mu' not in data.columns:
            st.error("CSV must contain 'T' and 'mu' columns.")
        else:
            T_data = data['T'].values
            mu_exp = data['mu'].values
            sort_idx = np.argsort(T_data)
            T_data, mu_exp = T_data[sort_idx], mu_exp[sort_idx]

            kB = 8.617333262145e-5

            def mobility_model(T, mu_w, phi_GB, l300, w_GB):
                p = 1.5
                l_T = l300 * (T / 300)**(-p)
                G = l_T / (l_T + w_GB)
                P = np.exp(-phi_GB / (kB * T))
                return mu_w * P * G

            model = Model(mobility_model)

            if constrained:
                params = model.make_params(mu_w=300, phi_GB=0.1)
                params.add("l300", value=l300_fixed, vary=False)
                params.add("w_GB", value=wGB_fixed, vary=False)
            else:
                params = model.make_params(mu_w=300, phi_GB=0.1, l300=20, w_GB=5)
                params['l300'].set(min=1, max=100)
                params['w_GB'].set(min=1, max=50)

            params['mu_w'].set(min=0, max=2000)
            params['phi_GB'].set(min=0, max=0.5)

            result = model.fit(mu_exp, params, T=T_data)

            st.subheader("📄 Fit Summary Report")
            st.text(result.fit_report())

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(T_data, mu_exp, 'bo', label='Experimental Data')
            ax.plot(T_data, result.best_fit, 'r-', label='Model Fit')
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Mobility (cm²/V·s)')
            ax.set_title('Unified Mobility Model Fit')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.subheader("📌 Fitted Parameters")
            fit_table = pd.DataFrame({
                'Parameter': list(result.params.keys()),
                'Value': [f"{v.value:.4g}" for v in result.params.values()],
                'Std. Error': [f"± {v.stderr:.2g}" if v.stderr else "-" for v in result.params.values()]
            })
            st.dataframe(fit_table)

            st.success("✅ Fit completed with " + ("constrained" if constrained else "free") + " parameters.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("""
Developed by **Gbadebo Taofeek Yusuf et al.**  
🔗 [Zenodo Dataset](https://doi.org/10.5281/zenodo.15617024)  
📚 *Unified Mobility Model for Grain-Boundary-Limited Transport in Polycrystalline Materials*
""")
