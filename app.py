import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mobility_model import run_fit
from material_presets import material_classes

st.set_page_config(page_title="Unified Mobility Model", layout="centered")
st.title("📈 Unified Mobility Model Fitting for Polycrystalline Materials")

uploaded_file = st.file_uploader("Upload your CSV file (Temperature,Mobility)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=",")
        df.columns = df.columns.str.strip()

        if df.shape[1] < 2:
            st.error("❌ File must have two columns: Temperature and Mobility.")
            st.stop()

        T_data = df.iloc[:, 0].values
        mu_data = df.iloc[:, 1].values

        st.subheader("Data Preview")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    material = st.sidebar.selectbox("Select Material", list(material_classes.keys()))
    preset = material_classes[material]

    if material != "Custom":
        st.sidebar.markdown(f"**Class:** {preset['class']}")
        pset = preset["params"]
        fix_params = preset["fixed"]

        mu_w = st.sidebar.slider("μw (cm²/V·s)", *pset["mu_w_bounds"], value=pset["mu_w_bounds"][0])
        initial_params = {
            "mu_w": mu_w,
            "phi_GB": pset["phi_GB"],
            "l300": pset["l300"],
            "w_GB": pset["w_GB"],
            "p": pset["p"]
        }
    else:
        mu_w = st.sidebar.slider("μw", 100, 1000, 300)
        phi_GB = st.sidebar.slider("ΦGB (eV)", 0.05, 0.3, 0.1)
        l300 = st.sidebar.slider("ℓ₃₀₀ (nm)", 5, 100, 20)
        w_GB = st.sidebar.slider("w_GB (nm)", 2, 30, 5)
        p = st.sidebar.slider("Phonon exponent p", 1.0, 3.0, 1.5)
        fix_params = []
        if st.sidebar.checkbox("Fix p"):
            fix_params.append("p")
        initial_params = {
            "mu_w": mu_w,
            "phi_GB": phi_GB,
            "l300": l300,
            "w_GB": w_GB,
            "p": p
        }

    if st.button("Fit Model"):
        result = run_fit(T_data, mu_data, initial_params, fix_params)

        st.subheader("Fitting Results")
        st.text(result.fit_report())

        fig, ax = plt.subplots()
        ax.plot(T_data, mu_data, 'o', label='Data')
        T_fit = sorted(T_data)
        mu_fit = result.model.eval(result.params, T=T_fit)
        ax.plot(T_fit, mu_fit, '-', label='Fit')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Mobility (cm²/V·s)')
        ax.legend()
        st.pyplot(fig)

        R2 = 1 - sum((mu_data - result.best_fit)**2) / sum((mu_data - mu_data.mean())**2)
        RMSE = ((mu_data - result.best_fit)**2).mean()**0.5
        st.markdown(f"**R²** = {R2:.4f}, **RMSE** = {RMSE:.2f} cm²/V·s")

        if result.redchi > 10 or any(p.stderr is None for p in result.params.values()):
            st.warning("⚠️ Overfitting possible. Try fixing or constraining more parameters.")
