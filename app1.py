import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

kB = 8.617333262145e-5  # Boltzmann constant in eV/K

def mu_eff_model(T, mu_w, phi_GB, w_GB, l_300, p):
    l_T = l_300 * (T / 300) ** (-p)
    G_T = l_T / (l_T + w_GB)
    P_GB = np.exp(-phi_GB / (kB * T))
    return mu_w * P_GB * G_T

def fit_mobility(T, mu, fix_w_GB=None):
    T = np.array(T)
    mu = np.array(mu)
    init_guess = [250, 0.08, 10, 40, 1.8]
    bounds = ([0, 0, 0.1, 1, 0.1], [1e4, 1, 100, 1e3, 5])

    if fix_w_GB is not None:
        def model_fixed(T, mu_w, phi_GB, l_300, p):
            return mu_eff_model(T, mu_w, phi_GB, fix_w_GB, l_300, p)
        popt, pcov = curve_fit(model_fixed, T, mu, p0=init_guess[:1]+init_guess[1:2]+init_guess[3:], bounds=(bounds[0][:1]+bounds[0][1:2]+bounds[0][3:], bounds[1][:1]+bounds[1][1:2]+bounds[1][3:]), maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "mu_w": (popt[0], perr[0]),
            "phi_GB": (popt[1], perr[1]),
            "w_GB (nm)": fix_w_GB,
            "l_300": (popt[2], perr[2]),
            "p": (popt[3], perr[3])
        }
    else:
        popt, pcov = curve_fit(mu_eff_model, T, mu, p0=init_guess, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "mu_w": (popt[0], perr[0]),
            "phi_GB": (popt[1], perr[1]),
            "w_GB (nm)": (popt[2], perr[2]),
            "l_300": (popt[3], perr[3]),
            "p": (popt[4], perr[4])
        }

def fit_power_law(T, mu):
    def model(T, A, n):
        return A * T ** (-n)
    popt, pcov = curve_fit(model, T, mu, p0=[1e5, 1.5], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    return {
        "A": (popt[0], perr[0]),
        "n": (popt[1], perr[1])
    }

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
            T_fit = pd.Series(np.linspace(min(T), max(T), 300))
            if model_type == "Grain Boundary Model":
                if isinstance(results["w_GB (nm)"], tuple):
                    mu_w, phi, w_GB, l_300, p = [results[k][0] for k in results]
                elif isinstance(results["w_GB (nm)"], (float, int)):
                    mu_w, phi, l_300, p = [results[k][0] for k in results if isinstance(results[k], tuple)]
                    w_GB = results["w_GB (nm)"]
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
