"""
Surrogate‑DFT Molecular Designer
================================

A Streamlit web app that lets you sketch / upload a molecular or crystal
structure and obtain *instant* surrogate‑DFT predictions (formation energy,
band gap, ΔE<sub>hull</sub>) from a pre‑trained **M3GNet** graph‑neural
network.

--------------------------------------------------------------------------
Deployment quick‑start (Streamlit Community Cloud)
--------------------------------------------------------------------------
1. **Repository layout**
   ```text
   your‑repo/
   ├── app.py               # this file
   ├── requirements.txt     # Python deps (see below)
   ├── packages.txt         # OPTIONAL: system libs (e.g. libopenblas‑dev)
   └── README.md            # project overview
   ```

2. **requirements.txt** (tested May 2025)
   ```text
   streamlit>=1.35
   stmol
   pymatgen
   matgl
   torch
   rdkit-pypi
   ```
   All packages have manylinux wheels, so no compiler is required on the
   free tier. If `matgl` can’t find BLAS/LAPACK, add `libopenblas-dev` to
   `packages.txt`.

3. **Launch** → https://streamlit.io/cloud → *New app* → pick repo/branch.
   The first run downloads ~60 MB of model weights into the container cache;
   subsequent sessions load instantly.

Optional extras
---------------
* **streamlit‑ketcher** for a 2‑D drag‑and‑drop editor.
* **DFT queue**: put SSH or API creds in `~/.streamlit/secrets.toml` and
  implement `submit_dft_job()`.

This file doubles as both runnable code and deployment guide—edit freely and
commit; Streamlit Cloud’s hot‑reloader will rebuild automatically.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Union

import streamlit as st

# ── Chemistry / ML imports ────────────────────────────────────────────────
try:
    from stmol import mol_view
except ImportError:  # graceful fallback for environments where stmol fails
    st.warning("📦 Installing stmol failed; 3‑D viewer disabled.")
    mol_view = None  # type: ignore[assignment]

from pymatgen.core import Molecule, Structure
from matgl.ext.matbench import load_model

# Optional: HPC submission helper (user‑supplied)
# from utils import submit_dft_job

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Surrogate‑DFT Molecular Designer",
                   layout="wide")

st.title("⚛️ Surrogate‑DFT Molecular Designer")

st.markdown(
    """
    **1. Sketch or upload** a structure  →  **2. Inspect 3‑D**  →  **3. Get
    near‑DFT predictions in milliseconds**.
    """,
    unsafe_allow_html=True,
)

# ── Load surrogate model (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading surrogate model …")
def get_model():
    """Download & cache the GNN weights on first use (≈ 60 MB)."""
    return load_model("M3GNet_universal")

model = get_model()

# ── Input panel ───────────────────────────────────────────────────────────
col_in, col_view = st.columns([1, 2])

with col_in:
    st.header("1  Provide a structure")

    smiles: str = st.text_input(
        "SMILES string (organic molecules)",
        placeholder="e.g. C1=CC=CC=C1 (benzene)",
    )

    uploaded_cif = st.file_uploader(
        "or upload CIF / POSCAR", type=["cif", "poscar", "vasp", "txt"]
    )

    # TODO: integrate streamlit‑ketcher for true drag‑and‑drop editing

with col_view:
    st.header("2  3‑D visualisation")

    structure: Union[Molecule, Structure, None] = None

    if smiles:
        try:
            structure = Molecule.from_smiles(smiles)
            if mol_view:
                mol_view(structure, height=450, width=450)
        except Exception as exc:
            st.error(f"Could not parse SMILES: {exc}")

    elif uploaded_cif is not None:
        try:
            cif_text = uploaded_cif.read().decode()
            structure = Structure.from_str(cif_text, "cif")
            if mol_view:
                mol_view(structure, height=450, width=450)
        except Exception as exc:
            st.error(f"Could not read CIF/POSCAR: {exc}")

    else:
        st.info("Enter a SMILES string *or* upload a CIF/POSCAR to see the 3‑D view.")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────
st.header("3  Predicted properties")


def predict(props_input: Union[Molecule, Structure]) -> Dict[str, Any]:
    """Run a forward pass through the GNN surrogate and return a dict."""
    graph = model.graph_converter.convert(props_input)
    return model.predict_graph(graph)

if structure is not None:
    try:
        preds = predict(structure)

        col_E, col_gap, col_hull = st.columns(3)
        col_E.metric("Formation E (eV/atom)", f"{preds.E_form:.3f}")
        col_gap.metric("Band gap (eV)", f"{preds.bandgap:.2f}")
        col_hull.metric("ΔE above hull (eV/atom)", f"{preds.deeh:.3f}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

else:
    st.caption("⬆️ Awaiting input …")

# ── Optional: full DFT submission ─────────────────────────────────────────
with st.expander("Run full DFT (optional)"):
    st.caption("Sends the current structure to an external DFT queue (e.g. VASP). Plug in your own backend in `submit_dft_job()`.")
    if st.button("Submit job"):
        if structure is None:
            st.error("No structure to submit!")
        else:
            st.warning("Job‑submission backend not implemented in demo mode.")
            # job_id = submit_dft_job(structure)
            # st.success(f"Submitted job {job_id}")

# ── Footer ────────────────────────────────────────────────────────────────
st.sidebar.success("👈 Modify *app.py* or *requirements.txt* and push to GitHub – Streamlit Cloud will redeploy automatically.")
