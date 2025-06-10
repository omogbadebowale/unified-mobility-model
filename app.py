############################################
# Unified Mobility Model – self‑contained Streamlit app
# ----------------------------------------------------
# * Upload a CSV with at least two columns: Temperature (K) and Mobility (cm^2/V·s).
# * Optional columns: grain_size_nm, carrier_density_cm3.  Presence/absence of these
#   dictates which model parameters are allowed to vary so the fit remains identifiable.
# * Priors for common TE materials are read from an inline YAML block.  Users can
#   override or supply a custom formula in the sidebar.
# * Uses lmfit for Levenberg‑Marquardt optimisation; an MCMC option is provided but
#   disabled by default to keep the interface snappy.
############################################

import io
from pathlib import Path
import textwrap

import pandas as pd
import numpy as np
import yaml
import streamlit as st
from lmfit import Model, Parameters, Minimizer, minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Unified Mobility Model", layout="centered")
st.title("📈 Unified Mobility Model – Grain‑Boundary‑Limited Transport")

# -----------------------------------------------------------------------------
# 0.  Priors database (inline YAML for demo; can be external file in production)
# -----------------------------------------------------------------------------
PRIORS_YAML = textwrap.dedent(
    """
    ZnO:
      mu_w: 300          # cm^2/V·s
      phi_GB: 0.15       # eV
      l300: 20           # nm
      w_GB: 5            # nm
      p: 1.5             # acoustic‑phonon limit
    Bi2Te3:
      mu_w: 1000
      phi_GB: 0.01
      l300: 50
      w_GB: 100
      p: 1.5
    Mg2Si:
      mu_w: 500
      phi_GB: 0.05
      l300: 60
      w_GB: 5
      p: 2.3
    SnSe:
      mu_w: 100
      phi_GB: 0.08
      l300: 20
      w_GB: 10
      p: 1.6
    """
)
MATERIAL_PRIORS = yaml.safe_load(PRIORS_YAML)

# -----------------------------------------------------------------------------
# 1.  Physics model
# -----------------------------------------------------------------------------
K_B = 8.617_333_262e-5  # eV/K

def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    """Unified mobility model (Eq. 6 in manuscript).  Units:
    T [K], l300 & w_GB [nm]."""
    l_T = l300 * (T / 300) ** (-p)
    thermionic = np.exp(-phi_GB / (K_B * T))
    geometric = l_T / (l_T + w_GB)
    return mu_w * thermionic * geometric

MODEL = Model(mobility_model)

# -----------------------------------------------------------------------------
# 2.  Helper – choose which parameters to vary based on available data
# -----------------------------------------------------------------------------

def choose_vary_flags(df):
    # Only temperature & mobility provided – cannot decouple l300/w_GB/p without priors.
    vary = {
        "mu_w": True,
        "phi_GB": True,
        "l300": False,
        "w_GB": False,
        "p": False,
    }
    if "grain_size_nm" in df.columns:
        vary["w_GB"] = True  # grain size acts as additional lever ⇒ can free w_GB
    if "carrier_density_cm3" in df.columns:
        vary["p"] = st.sidebar.checkbox("Allow p to vary", value=False)
    return vary

# -----------------------------------------------------------------------------
# 3.  Streamlit sidebar – material & fitting options
# -----------------------------------------------------------------------------

st.sidebar.header("Material & Fit Options")
material_options = list(MATERIAL_PRIORS.keys()) + ["Custom"]
material = st.sidebar.selectbox("Material formula", material_options)

priors = MATERIAL_PRIORS.get(material, {
    "mu_w": 300,
    "phi_GB": 0.1,
    "l300": 20,
    "w_GB": 5,
    "p": 1.5,
})

run_mcmc = st.sidebar.checkbox("Run Bayesian MCMC (slower)", value=False)

# -----------------------------------------------------------------------------
# 4.  File upload & preview
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload CSV with Temperature [K] & Mobility [cm^2/V·s]", type="csv")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip() for c in df.columns]
        # Standardise expected header names
        df = df.rename(columns={
            df.columns[0]: "T",
            df.columns[1]: "mu",
        })
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df.head())

    # -------------------------------------------------------------------------
    # 5.  Build lmfit Parameters, setting priors & vary flags
    # -------------------------------------------------------------------------
    vary_flags = choose_vary_flags(df)
    params = Parameters()
    bounds = {
        "mu_w": (priors["mu_w"] * 0.1, priors["mu_w"] * 10),
        "phi_GB": (0, 0.5),
        "l300": (1, 200),
        "w_GB": (1, 200),
        "p": (1.0, 3.0),
    }
    for key in ["mu_w", "phi_GB", "l300", "w_GB", "p"]:
        params.add(key, value=priors[key], min=bounds[key][0], max=bounds[key][1], vary=vary_flags[key])

    # -------------------------------------------------------------------------
    # 6.  Perform the fit
    # -------------------------------------------------------------------------
    T_data = df["T"].values
    mu_data = df["mu"].values

    if st.button("🔬 Fit model"):
        if run_mcmc:
            # Simple wrapper around lmfit.envelope_mcmc for demonstration
            # (Production: use emcee API and corner plot.)
            from lmfit import Minimizer

            def residual(pars):
                return mobility_model(T_data, **pars) - mu_data

            minimizer = Minimizer(residual, params)
            fit_res = minimizer.minimize(method="leastsq")
            mc_res = minimizer.emcee(params=fit_res.params, steps=500, burn=100)
            result = mc_res  # For API compatibility below
        else:
            result = MODEL.fit(mu_data, params, T=T_data, method="leastsq")

        # ---------------------------------------------------------------------
        # 7.  Display results
        # ---------------------------------------------------------------------
        st.subheader("Fit report")
        st.text(result.fit_report())

        # Plot
        T_fit = np.linspace(T_data.min(), T_data.max(), 300)
        mu_fit = mobility_model(T_fit, **result.best_values)

        fig, ax = plt.subplots()
        ax.plot(T_data, mu_data, "o", label="Data")
        ax.plot(T_fit, mu_fit, "-", label="Fit")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mobility (cm²/V·s)")
        ax.legend()
        st.pyplot(fig)

        # Metrics
        R2 = 1 - np.sum((mu_data - result.best_fit) ** 2) / np.sum((mu_data - mu_data.mean()) ** 2)
        RMSE = np.sqrt(np.mean((mu_data - result.best_fit) ** 2))
        st.write(f"**R²:** {R2:.4f}  |  **RMSE:** {RMSE:.2f} cm²/V·s")

        # Warn on large reduced chi‑sq or missing stderr
        if hasattr(result, "redchi") and result.redchi > 10:
            st.warning("High reduced χ² – fit may be unstable.")
        if any(p.stderr is None for p in result.params.values() if p.vary):
            st.warning("One or more parameters are poorly constrained (σ unknown). Consider fixing them or adding more data columns.")

        # Download button for parameters
        param_df = pd.DataFrame({
            "param": [p.name for p in result.params.values()],
            "value": [p.value for p in result.params.values()],
            "stderr": [p.stderr for p in result.params.values()],
        })
        st.download_button(
            "📥 Download fitted parameters", data=param_df.to_csv(index=False), file_name="mobility_fit_params.csv", mime="text/csv"
        )

        # Permalink (naïve: write to .txt in .streamlit‑storage for now)
        if st.button("🔗 Generate permalink"):
            storage_dir = Path(".permastore")
            storage_dir.mkdir(exist_ok=True)
            file_id = f"fit_{material}_{pd.Timestamp.utcnow().isoformat()}.txt".replace(":", "-")
            fpath = storage_dir / file_id
            with open(fpath, "w", encoding="utf-8") as fp:
                fp.write(result.fit_report())
            st.success(f"Link saved at {fpath.resolve()}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 Unified Mobility – built with Streamlit, lmfit, and ❤️")
