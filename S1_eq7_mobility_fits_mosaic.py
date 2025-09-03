# =============================================================================
# Supplementary Code — Unified Mobility Model (Eq. 7)
# =============================================================================
# Title:    Unified Mobility Model for Grain-Boundary-Limited Transport
#           in Polycrystalline Thermoelectric Materials
#
# Authors:  Gbadebo Taofeek Yusuf1,4*, Sukhwinder Singh2, Alexandros Askounis1, Zlatka Stoeva3,
#           Fideline Tchuenbou-Magaia1
# Affiliations:
#   1) Energy & Green Technology Research Group, Centre for Engineering
#      Innovation & Research, University of Wolverhampton, UK
#   2) Magnetics & Materials Research Group, School of Engineering,
#      Cardiff University, UK
#   3) DZP Technologies Limited, Cambridge, UK
#   4) Dept. of Science Laboratory Technology (Physics), Osun State Polytechnic,
#      Iree, Nigeria
#
# Purpose:   Mobility fits and the 2×3 publication-grade figure.
# Outputs:
#   - outputs/physical_MAP_parameters.csv
#   - outputs/Fig_2x3_physical_MAP.png
#   - outputs/Fig_2x3_physical_MAP.pdf
#
# CLI usage:
#   python Supplementary_UnifiedMobility_EQ7.py
#   # or with custom CSV (columns: Sample,T_K,mu_cm2V_s):
#   python Supplementary_UnifiedMobility_EQ7.py --csv path/to/data.csv --outdir outputs
#
# Jupyter usage: run this cell; unknown kernel args are ignored safely.
#
# Dependencies (tested):
#   numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scipy>=1.10
# =============================================================================

# ==== 0) Imports & CLI =======================================================
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ==== 1) Embedded default datasets ===========================================
# Panel order: (a) Bi2Te3  (b) ZnO  (c) Mg2Si  (d) SnSe  (e) Fe2VAl0.9Si  (f) NbCoSn
EMBEDDED_DATA = pd.DataFrame(
    {
        "Sample": (
            ["Bi2Te3"] * 9
            + ["ZnO"] * 15
            + ["Mg2Si"] * 11
            + ["SnSe"] * 12
            + ["Fe2VAl0.9Si"] * 26
            + ["NbCoSn"] * 9
        ),
        "T_K": (
            # Bi2Te3
            [100,125,150,175,200,225,250,275,300] +
            # ZnO
            [400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100] +
            # Mg2Si
            [314.2318619,364.0219156,411.1505682,457.4600168,504.8694179,
             552.760506,600.743472,648.088843,695.9385522,743.5500842,791.5886809] +
            # SnSe
            [300,350,400,450,500,550,600,650,700,750,800,850] +
            # Fe2VAl0.9Si
            [58.4,73.9,91.3,108,124.5,141.8,157.7,175.1,191.6,208.2,225.6,241.2,
             258.8,275.2,291.8,309.3,325.1,342.4,359.8,375.7,393,408.7,426.1,443.5,459.3,476.5] +
            # NbCoSn
            [300,350,400,450,500,550,600,650,700]
        ),
        "mu_cm2V_s": (
            # Bi2Te3
            [1000,950,890,830,770,700,630,550,470] +
            # ZnO
            [12,13,14,14,14,14,16,18,20,21,22,23,23,24,25] +
            # Mg2Si
            [136.3864764,125.2822581,113.7903226,107.1520311,95.5319751,
             83.8831406,72.35243651,66.26099707,60.43757467,54.65800478,48.79190386] +
            # SnSe
            [600,400,300,250,200,150,130,126,124,121,115,111] +
            # Fe2VAl0.9Si
            [53.7,52.3,49.7,43.7,38.6,34.1,30.8,26.5,24.1,22,21.1,18,17.5,15.6,14,
             11.9,10.8,9.5,8.9,7.8,7.2,6.6,5.8,4.9,4.2,4.1] +
            # NbCoSn
            [438.18,383.41,346.89,319.5,292.12,273.86,237.35,219.09,200.83]
        ),
    }
)


# ==== 2) Model (Eq. 7) and helpers ===========================================
kB = 8.617333262e-5  # eV/K

def mu_eq7(T, PhiGB, ell300, wGB, p, muw0):
    """Unified mobility model (Eq. 7)."""
    T = np.asarray(T, float)
    ellT = ell300 * (T/300.0)**(-p)
    return muw0 * np.exp(-PhiGB/(kB*T)) * (ellT/(ellT + wGB))

def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def mad_sigma(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad

def s2b(x, lo, hi):  # sigmoid -> bounded
    return lo + (hi - lo) / (1 + np.exp(-x))

def b2s(y, lo, hi):  # inverse of s2b
    y = np.clip((y - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return np.log(y/(1 - y))


# ==== 3) Fit configuration (physically constrained, robust) ==================
@dataclass
class FitConfig:
    PHI_LO: float = 0.0
    PHI_HI: float = 0.20
    P_LO: float   = 1.4
    P_HI: float   = 2.6
    PRIOR_WEIGHT: float = 0.5  # gentle priors to avoid degeneracy
    # Priors in native/log spaces
    phi_prior: Tuple[float,float]     = (0.03, 0.05)       # eV
    log_ell_prior: Tuple[float,float] = (np.log(40.), 0.8) # ln(nm)
    log_w_prior:   Tuple[float,float] = (np.log(15.), 0.8) # ln(nm)
    p_prior: Tuple[float,float]       = (2.0, 0.4)


# ==== 4) Single-sample fit ====================================================
def fit_sample(T, mu, cfg: FitConfig = FitConfig()) -> Dict:
    """
    Robust, physics-constrained fit for one dataset.
    Returns dict with parameters, R2, and a dense curve for plotting.
    """
    T = np.asarray(T, float); mu = np.asarray(mu, float)
    pri_log_muw = np.log(max(1.0, float(np.median(mu))))
    theta = np.array([
        b2s(0.05, cfg.PHI_LO, cfg.PHI_HI),  # x_phi
        np.log(40.0),                       # log_ell
        np.log(15.0),                       # log_w
        b2s(2.0, cfg.P_LO, cfg.P_HI),       # x_p
        pri_log_muw                         # log_muw0
    ], float)

    def unpack(th):
        x_phi, log_ell, log_w, x_p, log_muw = th
        Phi = s2b(x_phi, cfg.PHI_LO, cfg.PHI_HI)
        ell = float(np.exp(log_ell))
        w   = float(np.exp(log_w))
        p   = s2b(x_p, cfg.P_LO, cfg.P_HI)
        muw = float(np.exp(log_muw))
        return Phi, ell, w, p, muw

    def residuals(th):
        Phi, ell, w, p, muw = unpack(th)
        mu_hat = mu_eq7(T, Phi, ell, w, p, muw)
        sig_y = mad_sigma(mu)
        if not np.isfinite(sig_y) or sig_y < 1e-9:
            sig_y = max(1.0, 0.05*(mu.max() - mu.min()))
        res_data = (mu_hat - mu) / sig_y
        # Weak priors for identifiability:
        res_prior = cfg.PRIOR_WEIGHT * np.array([
            (Phi - cfg.phi_prior[0])       / cfg.phi_prior[1],
            (th[1] - cfg.log_ell_prior[0]) / cfg.log_ell_prior[1],
            (th[2] - cfg.log_w_prior[0])   / cfg.log_w_prior[1],
            (p     - cfg.p_prior[0])       / cfg.p_prior[1],
            (th[4] - pri_log_muw) / 1.0
        ])
        return np.concatenate([res_data, res_prior])

    sol = least_squares(residuals, theta, loss="soft_l1", f_scale=1.0, max_nfev=400000)
    Phi, ell, w, p, muw = unpack(sol.x)
    mu_fit = mu_eq7(T, Phi, ell, w, p, muw)
    T_dense = np.linspace(T.min(), T.max(), 400)
    mu_dense = mu_eq7(T_dense, Phi, ell, w, p, muw)
    return dict(
        PhiGB_eV=float(Phi), ell300_nm=float(ell), wGB_nm=float(w),
        p=float(p), muw0_cm2Vs=float(muw), R2=float(r2(mu, mu_fit)),
        T_dense=T_dense, mu_dense=mu_dense
    )


# ==== 5) Plotting (2×3 square panels) ========================================
def plot_2x3(panels, fits, out_png, out_pdf, panel_size=3.15, dpi=600):
    """Square, publication-grade 2×3 composite with centered panel letters."""
    labels = ["(a)","(b)","(c)","(d)","(e)","(f)"]
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2.0
    })
    fig, axes = plt.subplots(2, 3, figsize=(3*panel_size, 2*panel_size),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    axes = axes.ravel()
    for i, (name, T, mu) in enumerate(panels):
        ax = axes[i]; ax.set_box_aspect(1)
        f = fits[name]
        ax.plot(T, mu, 'o', mfc='none', mec='blue', mew=1.2, ms=4.6, linestyle='None', label="Data")
        ax.plot(f["T_dense"], f["mu_dense"], color='red', label="Fit")
        ax.text(0.50, 0.98, labels[i], transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.15", alpha=0.9))
        ax.set_title(f"{name}  (R$^2$={f['R2']:.3f})")
        xpad = 0.03*(T.max()-T.min()); ypad = 0.12*(mu.max()-mu.min())
        ax.set_xlim(T.min()-xpad, T.max()+xpad); ax.set_ylim(mu.min()-ypad, mu.max()+ypad)
        ax.minorticks_on(); ax.tick_params(direction="out", length=4, width=1)
        ax.legend(frameon=False, loc="best")
    for j in [3,4,5]: axes[j].set_xlabel("T (K)")
    for j in [0,3]:   axes[j].set_ylabel(r"$\mu_{\mathrm{eff}}$ (cm$^2$ V$^{-1}$ s$^{-1}$)")
    fig.savefig(out_png, dpi=dpi, facecolor="white"); fig.savefig(out_pdf, dpi=dpi)
    return fig


# ==== 6) Utilities ============================================================
def fmt_phi(x: float) -> str:
    return "≈0" if x < 1e-3 else f"{x:.3f}"

def fmt3(x: float) -> str:
    return f"{x:.3f}"

def load_data_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"Sample","T_K","mu_cm2V_s"}
    if not need.issubset(df.columns):
        missing = ", ".join(sorted(need - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")
    return df[["Sample","T_K","mu_cm2V_s"]].copy()


# ==== 7) Main: fit, save table, save figure ==================================
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data
    if args.csv is None:
        data = EMBEDDED_DATA.copy()
    else:
        data = load_data_from_csv(Path(args.csv))

    # enforce manuscript panel order:
    order = ["Bi2Te3","ZnO","Mg2Si","SnSe","Fe2VAl0.9Si","NbCoSn"]

    fits: Dict[str, Dict] = {}
    panels = []
    for name in order:
        df = data[data["Sample"] == name]
        if df.empty:
            raise ValueError(f"No rows found for sample '{name}' in the input data.")
        T = df["T_K"].to_numpy(float)
        mu = df["mu_cm2V_s"].to_numpy(float)
        res = fit_sample(T, mu); fits[name] = res
        panels.append((name, T, mu))

    # tidy parameter table
    table = pd.DataFrame(
        [{"Sample": n, **{k:v for k,v in d.items() if k not in ("T_dense","mu_dense")}}
         for n,d in fits.items()]
    )

    # pretty print to console
    show = table.copy()
    print("\nFitted parameters (robust, constrained):\n")
    print(show.assign(
        PhiGB_eV   = show["PhiGB_eV"].apply(fmt_phi),
        ell300_nm  = show["ell300_nm"].apply(fmt3),
        wGB_nm     = show["wGB_nm"].apply(fmt3),
        p          = show["p"].apply(fmt3),
        muw0_cm2Vs = show["muw0_cm2Vs"].apply(fmt3),
        R2         = show["R2"].apply(fmt3)
    )[["Sample","PhiGB_eV","ell300_nm","wGB_nm","p","muw0_cm2Vs","R2"]].to_string(index=False))

    # save raw numeric table
    table.to_csv(outdir/"physical_MAP_parameters.csv", index=False)

    # figure
    plot_2x3(panels, fits,
             out_png = outdir/"Fig_2x3_physical_MAP.png",
             out_pdf = outdir/"Fig_2x3_physical_MAP.pdf",
             panel_size=args.panel, dpi=args.dpi)

    print("\nWrote:", outdir/"physical_MAP_parameters.csv")
    print("Wrote:", outdir/"Fig_2x3_physical_MAP.png")
    print("Wrote:", outdir/"Fig_2x3_physical_MAP.pdf")


# ==== 8) Entry point (Jupyter/CLI-safe) ======================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce unified mobility fits (Eq. 7) and the 2×3 figure."
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV with columns: Sample,T_K,mu_cm2V_s")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory for outputs (default: outputs)")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI")
    parser.add_argument("--panel", type=float, default=3.15,
                        help="Panel size in inches (square)")

    # parse_known_args lets this script run inside Jupyter (ignores '-f ...json')
    args, unknown = parser.parse_known_args()

    # Only print a notice if unknown args are not the standard Jupyter ones
    def _is_jupyter_noise(seq):
        # Jupyter typically passes: ['-f', '...kernel-<id>.json']
        return any(x == "-f" for x in seq) and any(str(x).endswith(".json") for x in seq)

    if unknown and not _is_jupyter_noise(unknown):
        print(f"Ignoring unknown arguments: {unknown}")

    main(args)
