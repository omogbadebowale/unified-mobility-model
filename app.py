import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mobility_model import fit_mobility, fit_power_law

st.set_page_config(page_title="Mobility Model Fitting", layout="wide")
st.title("📊 Grain Boundary & Power Law Mobility Fitting")
st.markdown("Upload your experimental data and select a model to fit.")

# Upload data
uploaded_file = st.file_uploader("Upload CSV with 'T' and 'mu' columns", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'T' not in data.columns or 'mu' not in data.columns:
        st.error("CSV must contain 'T' and 'mu' columns.")
    else:
        T = data['T'].values
        mu = data['mu'].values

        st.subheader("📈 Raw Data Preview")
        st.dataframe(data)

        # Model selection
        model_type = st.selectbox("Choose Model", ["Grain Boundary Model", "Power Law Model"])

        # Grain Boundary options
        fix_wGB = None
        if model_type == "Grain Boundary Model":
            if st.checkbox("Fix w_GB (nm)?"):
                fix_wGB = st.number_input("Enter fixed w_GB value (nm)", min_value=0.01, max_value=100.0, value=10.0)

        # Fit model
        if st.button("🔍 Fit Model"):
            if model_type == "Grain Boundary Model":
                with st.spinner("Fitting GB model..."):
                    results = fit_mobility(T, mu, fix_w_GB=fix_wGB)
            else:
                with st.spinner("Fitting Power Law model..."):
                    results = fit_power_law(T, mu)

            st.success("Model fitting complete!")

            # Display results
            st.subheader("📌 Fitted Parameters")
            for param, val in results.items():
                if isinstance(val, tuple):
                    st.write(f"**{param}**: {val[0]:.4g} ± {val[1]:.2g}")
                else:
                    st.write(f"**{param}**: {val:.4g}")

            # Plot results
            st.subheader("📉 Fit Visualization")
            T_fit = pd.Series(range(int(min(T)), int(max(T)) + 1))
            if model_type == "Grain Boundary Model":
                from mobility_model import kB
                if isinstance(results["w_GB (nm)"], tuple):
                    mu_w, phi, w_GB, l_300, p = [results[k][0] for k in results]
                else:
                    mu_w, phi, w_GB, l_300, p = [results[k] for k in results]
                l_T = l_300 * (T_fit / 300) ** (-p)
                G_T = l_T / (l_T + w_GB)
                P_GB = np.exp(-phi / (kB * T_fit))
                mu_fit = mu_w * P_GB * G_T
            else:
                A, n = [results[k][0] for k in results]
                mu_fit = A * T_fit ** (-n)

            fig, ax = plt.subplots()
            ax.plot(T, mu, 'o', label='Experimental')
            ax.plot(T_fit, mu_fit, '-', label='Model Fit')
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Mobility (cm²/Vs)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Download results
            if st.button("⬇ Download Parameters"):
                df_out = pd.DataFrame([
                    {"Parameter": k, "Value": v[0] if isinstance(v, tuple) else v, "Uncertainty": v[1] if isinstance(v, tuple) else 0}
                    for k, v in results.items()
                ])
                csv = df_out.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, file_name="fitted_parameters.csv", mime="text/csv")
