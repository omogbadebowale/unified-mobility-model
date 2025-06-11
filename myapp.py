"""
Surrogate‑DFT Molecular Designer
================================

A Streamlit web app that lets you sketch or upload a molecular/crystal
structure and get *instant* surrogate‑DFT predictions (formation energy,
band gap, ΔE<sub>hull</sub>) from a pre‑trained **M3GNet** graph‑neural
network.

--------------------------------------------------------------------------
Deployment quick‑start (Streamlit Community Cloud)
--------------------------------------------------------------------------
1. **Repo layout**
   ```text
   your‑repo/
   ├── app.py               # this file
   ├── requirements.txt     # Python deps (see below)
   ├── packages.txt         # OPTIONAL system libs (e.g. libopenblas‑dev)
   └── README.md
   ```

2. **requirements.txt** – pin proven versions to avoid wheel clashes
   ```text
   streamlit==1.35.0
   # core libs
   numpy==1.26.4
   scipy==1.11.4
   # domain + ML
   pymatgen==2023.5.10
   matgl==0.8.3
   torch==2.2.2
   # viewers / helpers (optional but nice)
   stmol==0.1.2
   rdkit-pypi==2024.3.1
   ```
   Wheels exist for all of the above, so no compiler toolchain is required.
   If `matgl` complains about missing BLAS/LAPACK, add this to **packages.txt**:
   ```text
   libopenblas-dev
   ```

3. **Deploy** → <https://streamlit.io/cloud> → *New app* → pick repo/branch →
   main file `app.py` → **Deploy**. First build grabs the 60 MB M3GNet model.

----------------------------------------------------------------------
❗ Troubleshooting
----------------------------------------------------------------------
* **stmol fails** → viewer disabled but app still works.
* **pymatgen fails** → check logs, ensure pinned version above. Import errors
  are now caught and surfaced in‑app for easier debugging.
* **Memory limit** → Settings → Hardware → “Medium” (free 3 GB).

--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Any, Dict, Union

import streamlit as st

# ── Optional 3‑D viewer (stmol) ───────────────────────────────────────────
try:
    from stmol import mol_view
except Exception:
    st.warning("ℹ️ `stmol` unavailable – 3‑D viewer disabled.")
    mol_view = None  # type: ignore

# ── Core chemistry libs (pymatgen) ────────────────────────────────────────
try:
    from pymatgen.core import Molecule, Structure
except Exception as pmg_err:  # noqa: BLE001  (broad import OK here)
    Molecule = None  # type: ignore
    Structure = None  # type: ignore
    st.error(
        "❌ `pymatgen` failed to import. Check that it’s listed in `requirements.txt` "
        "and that the wheel installed correctly.\n\nLog excerpt: "
        f"{pmg_err}"
    )

# ── ML surrogate (matgl / M3GNet) ─────────────────────────────────────────
try:
    from matgl.ext.matbench import load_model

    @st.cache_resource(show_spinner="Loading surrogate model …")
    def get_model():
        return load_model("M3GNet_universal")

    model = get_model()
except Exception as model_err:  # noqa: BLE001
    model = None  # type: ignore
    st.error("❌ Surrogate model failed to load – predictions disabled.\n\n" f"{model_err}")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Surrogate‑DFT Molecular Designer", layout="wide")

st.title("⚛️ Surrogate‑DFT Molecular Designer")

st.markdown(
    """
    1. **Provide** a structure → 2. **Inspect** 3‑D → 3. **Get** near‑DFT
    property predictions.
    """,
    unsafe_allow_html=True,
)

# ── Input & viewer columns ────────────────────────────────────────────────
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

    # TODO: integrate streamlit‑ketcher for 2‑D drag‑and‑drop editing

with col_view:
    st.header("2  3‑D visualisation")

    structure: Union["Molecule", "Structure", None] = None

    if Molecule is None or Structure is None:
        st.info("`pymatgen` unavailable – viewer disabled.")
    else:
        if smiles:
            try:
                structure = Molecule.from_smiles(smiles)
                if mol_view is not None:
                    mol_view(structure, height=450, width=450)
            except Exception as exc:
                st.error(f"Could not parse SMILES: {exc}")

        elif uploaded_cif is not None:
            try:
                cif_text = uploaded_cif.read().decode()
                structure = Structure.from_str(cif_text, "cif")
                if mol_view is not None:
                    mol_view(structure, height=450, width=450)
            except Exception as exc:
                st.error(f"Could not read CIF/POSCAR: {exc}")

        else:
            st.info("Enter a SMILES string *or* upload a CIF/POSCAR to see the 3‑D view.")

st.divider()

# ── Prediction panel ──────────────────────────────────────────────────────
st.header("3  Predicted properties")

if structure is not None and model is not None:

    def predict(props_input: Union["Molecule", "Structure"]) -> Dict[str, Any]:
        graph = model.graph_converter.convert(props_input)  # type: ignore[attr-defined]
        return model.predict_graph(graph)  # type: ignore[attr-defined]

    try:
        preds = predict(structure)

        col_E, col_gap, col_hull = st.columns(3)
        col_E.metric("Formation E (eV/atom)", f"{preds.E_form:.3f}")
        col_gap.metric("Band gap (eV)", f"{preds.bandgap:.2f}")
        col_hull.metric("ΔE above hull (eV/atom)", f"{preds.deeh:.3f}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
elif model is None:
    st.info("Predictions unavailable – model did not load.")
else:
    st.caption("⬆️ Awaiting input …")

# ── Optional DFT submission stub ──────────────────────────────────────────
with st.expander("Run full DFT (optional)"):
    st.caption(
        "Dispatch the current structure to an external DFT queue (e.g. VASP). "
        "Plug in your own backend in `submit_dft_job()`."
    )
    if st.button("Submit job"):
        if structure is None:
            st.error("No structure to submit!")
        else:
            st.warning("Job‑submission backend not implemented in demo mode.")
            # job_id = submit_dft_job(structure)
            # st.success(f"Submitted job {job_id}")

# ── Sidebar footer ────────────────────────────────────────────────────────
st.sidebar.success(
    "👈 Modify *app.py* or *requirements.txt* then **push to GitHub** – "
    "Streamlit Cloud will redeploy automatically."
)
