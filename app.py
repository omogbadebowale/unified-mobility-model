
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

K_B_EV = 8.617333262e-5

def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l_T = l300 * (T / 300.0) ** (-p)
    geom = l_T / (l_T + w_GB)
    thermo = np.exp(-phi_GB / (K_B_EV * T))
    return mu_w * thermo * geom

def run_fit(T, mu, params_dict, fix_p):
    model = Model(mobility_model)
    params = Parameters()
    params.add("mu_w", value=params_dict["mu_w"], min=10, max=1000)
    params.add("phi_GB", value=params_dict["phi_GB"], min=0, max=0.3)
    params.add("l300", value=params_dict["l300"], min=5, max=100)
    params.add("w_GB", value=params_dict["w_GB"], min=1, max=30)
    params.add("p", value=params_dict["p"], min=1.0, max=3.0, vary=not fix_p)
    return model.fit(mu, params, T=T)

st.set_page_config("Unified Mobility Fitting", layout="wide")
st.title("📈 Unified Mobility Model Fitting Tool")

materials = {
    "Oxide (ZnO, SrTiO₃)": {"mu_w": 100, "phi_GB": 0.15, "l300": 15, "w_GB": 5, "p": 2.0},
    "Chalcogenide (Bi₂Te₃, SnSe)": {"mu_w": 400, "phi_GB": 0.05, "l300": 30, "w_GB": 10, "p": 1.6},
    "Intermetallic (Mg₂Si, PbTe)": {"mu_w": 300, "phi_GB": 0.05, "l300": 60, "w_GB": 5, "p": 1.5},
    "Custom": {"mu_w": 300, "phi_GB": 0.1, "l300": 20, "w_GB": 5, "p": 1.5}
}
material = st.selectbox("Material Type", list(materials))
defaults = materials[material]

# Guided Fit Mode
guided = st.checkbox("🧪 Enable Guided Fit Mode (Recommended)", value=True)
if guided:
    st.markdown("✔️ Guided mode is using a preset configuration based on material type.")
    guided_presets = {
        "Oxide (ZnO, SrTiO₃)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
        "Chalcogenide (Bi₂Te₃, SnSe)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
        "Intermetallic (Mg₂Si, PbTe)": {"float": ["mu_w", "phi_GB"], "fix": ["l300", "w_GB", "p"]},
        "Custom": {"float": ["mu_w", "phi_GB", "l300"], "fix": ["w_GB", "p"]}
    }
    current = guided_presets[material]
    float_set = set(current["float"])
else:
    st.markdown("⚠️ You are in advanced mode. Adjust parameters manually.")
    float_set = set()

mu_w = st.sidebar.number_input("μw (cm²/V·s)", value=defaults["mu_w"])
phi_GB = st.sidebar.number_input("ΦGB (eV)", value=defaults["phi_GB"])
l300 = st.sidebar.number_input("ℓ300 (nm)", value=defaults["l300"])
w_GB = st.sidebar.number_input("wGB (nm)", value=defaults["w_GB"])
fix_p = st.sidebar.checkbox("Fix phonon exponent p?", value=True)
p = st.sidebar.number_input("p (if fixed)", value=defaults["p"])

uploaded = st.file_uploader("Upload CSV file with Temperature,Mobility", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    if data.shape[1] != 2:
        st.error("CSV must have exactly 2 columns.")
    else:
        T = data.iloc[:, 0].values
        mu = data.iloc[:, 1].values
        st.write("### Experimental Data", data)

        result = run_fit(T, mu, {
            "mu_w": mu_w, "phi_GB": phi_GB, "l300": l300, "w_GB": w_GB, "p": p
        }, fix_p)

        T_fit = np.linspace(min(T), max(T), 500)
        mu_fit = result.eval(T=T_fit)
        mu_unc = result.eval_uncertainty(sigma=1, T=T_fit)

        st.subheader("Model Fit with Confidence Interval")
        fig, ax = plt.subplots()
        ax.plot(T, mu, 'bo', label="Data")
        ax.plot(T_fit, mu_fit, 'r-', label="Model Fit")
        ax.fill_between(T_fit, mu_fit - mu_unc, mu_fit + mu_unc, alpha=0.3, color="red", label="±1σ")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mobility (cm²/V·s)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Residuals")
        residuals = mu - result.eval(T=T)
        fig2, ax2 = plt.subplots()
        ax2.plot(T, residuals, 'ko-')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Residual (cm²/V·s)")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Fit Report")
        st.text(result.fit_report())

        # Fit Quality Summary
        def display_fit_quality(result):
            st.subheader("🧾 Fit Quality Summary")
            st.write(f"**R²:** {result.rsquared:.3f}")
            st.write(f"**Reduced χ²:** {result.redchi:.2f}")
            warning_lines = []
            for name, param in result.params.items():
                if param.stderr is not None and param.value != 0:
                    pct_uncertainty = abs(param.stderr / param.value) * 100
                    if pct_uncertainty > 100:
                        warning_lines.append(f"⚠️ `{name}` uncertainty > 100% ({pct_uncertainty:.1f}%)")
            if warning_lines:
                st.error("Unstable Parameters Detected:\n" + "\n".join(warning_lines))
            else:
                st.success("All parameter uncertainties < 100% ✅")
        display_fit_quality(result)
