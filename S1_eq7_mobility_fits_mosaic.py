# ===============================================================
# Mobility fits — Eq. (7) with APS-like styling
# Panels:
#   (a) ZnO: Ta-doped (~3%)
#   (b) Bi2Te3 (n)
#   (c) Mg2Si (baseline) + Mg2Si (1% Sb) in one axes
#   (d) SnSe (p)
#   (e) Fe2V (n) + Fe2V (p) in one axes
#   (f) ZrNiSn  ← NEW
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import math, json

# ---------------- Utilities ----------------
def safe_pow(base, exp, min_base=1e-12):
    b = np.asarray(base, float)
    return np.power(np.clip(b, min_base, np.inf), exp)

def sanitize_xy(T, y):
    T = np.asarray(T, float); y = np.asarray(y, float)
    m = np.isfinite(T) & np.isfinite(y) & (T > 0)
    T, y = T[m], y[m]
    o = np.argsort(T)
    return T[o], y[o]

def fit_metrics(y, yhat, k_params):
    resid = y - yhat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r2  = 1.0 - rss/tss if tss > 0 else np.nan
    n   = len(y)
    aic = 2*k_params + n*math.log(max(rss/n, 1e-300))
    aicc = aic + (2*k_params*(k_params+1))/max(n - k_params - 1, 1e-9)
    bic = k_params*math.log(max(n,1)) + n*math.log(max(rss/n, 1e-300))
    return r2, aicc, bic

# ---------------- Unified model ----------------
kB = 8.617333262e-5  # eV/K

def l_of_T(T, l300, p):
    return l300 * safe_pow(T/300.0, -p)

def mu_w_of_T(T, mu0, q):
    return mu0 * safe_pow(T/300.0, -q)

def mu_eff_model(T, mu0, q, Phi, l300, p, wGB):
    lT = l_of_T(T, l300, p)
    G  = lT / (lT + wGB)
    return mu_w_of_T(T, mu0, q) * np.exp(-Phi/(kB*T)) * G

# ---------------- Fit driver ----------------
def fit_best_spec(T, mu, wGB_grid, p_grid):
    T, mu = sanitize_xy(T, mu)
    best = None
    for wGB in wGB_grid:
        for p_fixed in p_grid:
            def f(T, mu0, q, Phi, l300):
                return mu_eff_model(T, mu0, q, Phi, l300, p_fixed, wGB)
            p0     = [max(mu[0], 1.0), 0.6, 0.02, 30.0]
            bounds = ([0.0, -2.0, 0.0,  5.0],
                      [1e6,  2.0, 0.25, 500.0])
            try:
                popt, _ = curve_fit(f, T, mu, p0=p0, bounds=bounds, maxfev=800000)
                yhat = f(T, *popt)
                r2, aicc, bic = fit_metrics(mu, yhat, k_params=4)
                cand = dict(
                    r2=r2, aicc=aicc, bic=bic, T=T, mu=mu, yhat=yhat,
                    params=dict(mu0=float(popt[0]), q=float(popt[1]), Phi=float(popt[2]),
                                l300=float(popt[3]), p=float(p_fixed), wGB=float(wGB)),
                    spec=f"wGB={wGB:g} nm, p={p_fixed:g}"
                )
                if best is None or cand["aicc"] < best["aicc"]:
                    best = cand
            except Exception:
                pass
    if best is None:
        raise RuntimeError("No successful fit — widen grids.")
    return best

# ---------------- Datasets ----------------
datasets = [
    # (a) ZnO — Ta-doped (~3%)
    ("ZnO: Ta-doped (~3%)",
     np.array([305.1020408, 371.9954649, 422.4489796, 471.7687075, 522.2222222,
               570.9750567, 621.4285714, 671.8820862, 721.7687075, 770.521542], float),
     np.array([23.63489499, 20.63004847, 19.75767367, 15.97738288, 14.62035541,
               12.48788368, 10.93699515,  9.095315024, 7.641357027, 5.799676898], float)),

    # (b) Bi2Te3 (n-type)
    ("Bi2Te3 (n-type)",
     np.array([175,200,225,250,275,300], float),
     np.array([116.2,105.0,93.34,83.25,78.4,64.37], float)),

    # (c1) Mg2Si — baseline (NEW)
    ("Mg2Si (baseline, n-type)",
     np.array([316.8350168,362.8507295,407.7441077,452.0763187,494.1638608,
               532.3232323,573.28844,607.5196409,643.4343434,673.1762065,
               705.1627385,732.6599327,754.5454545], float),
     np.array([121.6438356,115.6164384,112.3287671,103.2876712,93.69863014,
               84.65753425,78.96678967,74.24354244,68.78228782,62.43542435,
               57.71217712,53.72693727,49.15129151], float)),

    # (c2) Mg2Si — 1% Sb (existing)
    ("Mg2Si (1% Sb, n-type)",
     np.array([348.2603816,431.3131313,495.8473625,554.2087542,603.030303,
               650.7295174,691.1335578,716.9472503], float),
     np.array([84.31818182,82.95454545,82.04545455,74.26470588,67.05882353,
               62.35294118,56.32352941,50.58823529], float)),

    # (d) SnSe (p-type)
    ("SnSe (p-type)",
     np.array([299.3702771,322.0403023,372.418136,424.0554156,471.2846348,
               521.0327456,570.7808564,623.0478589,674.6851385,721.9143577], float),
     np.array([159.4594595,142.3423423,109.9099099,83.78378378,63.96396396,
               49.54954955,40.54054054,32.43243243,28.82882883,27.02702703], float)),

    # (e1) Fe2V… — n-type
    ("Fe2V0.95Ta0.05Al0.95Si0.05 (n-type)",
     np.array([2.159827214,19.43844492,37.79697624,58.31533477,77.7537797,
               100.4319654,119.8704104,140.3887689,158.7473002,200.8639309,
               222.462203,238.6609071,255.9395248,275.3779698,300.2159827,
               316.4146868,340.1727862,354.2116631,375.8099352,399.5680346,
               420.0863931,436.2850972,457.8833693], float),
     np.array([33.45821326,33.19884726,31.90201729,30.43227666,28.27089337,
               25.41786744,23.17002882,20.66282421,18.84726225,15.38904899,
               14.17867435,12.53602305,11.06628242,9.423631124,7.867435159,
               5.706051873,4.495677233,2.853025937,2.42074928,1.902017291,
               1.383285303,0.518731988,0.432276657], float)),

    # (e2) Fe2V… — p-type
    ("Fe2VAl0.95Si0.05 (p-type)",
     np.array([4.319654428,21.59827214,37.79697624,58.31533477,80.99352052,
               97.19222462,122.0302376,141.4686825,157.6673866,198.7041037,
               178.1857451,215.9827214,238.6609071,259.1792657,278.6177106,
               301.2958963,316.4146868,339.0928726,358.5313175,381.2095032,
               399.5680346,423.3261339,441.6846652,456.8034557,479.4816415,
               492.4406048,512.9589633], float),
     np.array([55.5907781,53.25648415,51.78674352,49.10662824,42.79538905,
               37.26224784,32.50720461,28.6167147,24.46685879,19.88472622,
               22.04610951,18.41498559,15.64841499,14.870317,12.53602305,
               10.7204611,9.250720461,7.694524496,6.484149856,5.53314121,
               4.755043228,3.976945245,3.112391931,2.593659942,1.469740634,
               0.951008646,0.518731988], float)),

    # (f) ZrNiSn — NEW
    ("ZrNiSn (dense polycrystal)",
     np.array([284.1845303,336.5733539,405.7660999,454.2010372,510.5436716,
               540.1977056,626.1943741,655.8484081,691.4333092,720.0988151,
               744.8105101,770.5107333,799.1762391,823.8879341,870.345966], float),
     np.array([33.91691245,33.91691245,31.91394751,31.69139334,30.94955363,
               27.90801198,26.64688392,26.86943526,24.12463062,23.53115659,
               23.53115659,22.9376854,21.6023745,20.56379778,19.6735924], float))
]

# dataset-specific grids (physics-guided)
def grids_for(name):
    if "ZnO" in name:
        return (6,8,10,12,15), (1.8,2.0,2.2)          # Ta–ZnO
    if "Bi2Te3" in name:
        return (50,100,200), (1.2,1.5,1.8)
    if "ZrNiSn" in name:
        return (15,20,30,40), (0.5,0.6,0.8,1.0)       # alloy-like slope ~T^-0.5
    return (10,15,20,30,40), (1.5,1.8,2.0,2.2)

# ---------------- Fit all ----------------
fits = []
for name, T, mu in datasets:
    wgrid, pgrid = grids_for(name)
    best = fit_best_spec(T, mu, wGB_grid=wgrid, p_grid=pgrid)
    fits.append((name, best))
    print(f"{name:33s} | R^2={best['r2']:.3f} | {best['spec']} | params={best['params']}")

fitmap = {name: bf for name, bf in fits}

# ---------------- Styling ----------------
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
})

BLUE  = "#1f77b4"  # series A (data)
BLACK = "#222222"  # series B (data)
RED   = "#d62728"  # fit A
GRAY  = "#555555"  # fit B

def panel_box(ax):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(which="both", direction="in", length=3.0, width=0.9, top=True, right=True)

def mu_eff_from_params(Tx, P):
    lT = l_of_T(Tx, P["l300"], P["p"])
    G  = lT/(lT + P["wGB"])
    return mu_w_of_T(Tx, P["mu0"], P["q"]) * np.exp(-P["Phi"]/(kB*Tx)) * G

# ---------------- Plot mosaic ----------------
labels = ["a","b","c","d","e","f"]
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(9.6, 6.4), constrained_layout=True)

# (a) ZnO
ax = axs[0,0]; name = "ZnO: Ta-doped (~3%)"; bf = fitmap[name]
T, mu, P = bf["T"], bf["mu"], bf["params"]; Tfine = np.linspace(max(1.0, T.min()), T.max(), 800)
ax.plot(T, mu, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE, mew=1.0, color=BLUE)
ax.plot(Tfine, mu_eff_from_params(Tfine, P), lw=2.0, color=RED)
panel_box(ax); ax.text(0.5,1.02,"(a)", transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
ax.text(0.97,0.04,f"R$^2$={bf['r2']:.3f}", transform=ax.transAxes, ha="right", va="bottom")
ax.set_ylabel(r"$\mu_{\mathrm{eff}}$ (cm$^2$ V$^{-1}$ s$^{-1}$)")

# (b) Bi2Te3
ax = axs[0,1]; name = "Bi2Te3 (n-type)"; bf = fitmap[name]
T, mu, P = bf["T"], bf["mu"], bf["params"]; Tfine = np.linspace(max(1.0, T.min()), T.max(), 800)
ax.plot(T, mu, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE, mew=1.0, color=BLUE)
ax.plot(Tfine, mu_eff_from_params(Tfine, P), lw=2.0, color=RED)
panel_box(ax); ax.text(0.5,1.02,"(b)", transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
ax.text(0.97,0.04,f"R$^2$={bf['r2']:.3f}", transform=ax.transAxes, ha="right", va="bottom")

# (c) Mg2Si combined
ax_c = axs[0,2]
nameA = "Mg2Si (baseline, n-type)"
nameB = "Mg2Si (1% Sb, n-type)"
bfA, bfB = fitmap[nameA], fitmap[nameB]
TA, muA, PA = bfA["T"], bfA["mu"], bfA["params"]; TAf = np.linspace(max(1.0, TA.min()), TA.max(), 800)
TB, muB, PB = bfB["T"], bfB["mu"], bfB["params"]; TBf = np.linspace(max(1.0, TB.min()), TB.max(), 800)
ax_c.plot(TA, muA, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE,  mew=1.0, color=BLUE,  label="Mg$_2$Si (baseline)")
ax_c.plot(TAf, mu_eff_from_params(TAf, PA), lw=2.0, color=RED,  label="Model (baseline)")
ax_c.plot(TB, muB, linestyle="none", marker="s", ms=3.2, mfc="white", mec=BLACK, mew=1.0, color=BLACK, label="Mg$_2$Si (1% Sb)")
ax_c.plot(TBf, mu_eff_from_params(TBf, PB), lw=2.0, color=GRAY, label="Model (1% Sb)")
panel_box(ax_c); ax_c.text(0.5,1.02,"(c)", transform=ax_c.transAxes, ha="center", va="bottom", fontweight="bold")
ax_c.text(0.97,0.04,f"R$^2$ base={bfA['r2']:.3f}\nR$^2$ Sb={bfB['r2']:.3f}", transform=ax_c.transAxes, ha="right", va="bottom")
ax_c.legend(frameon=False, fontsize=8, loc="upper right")

# (d) SnSe
ax = axs[1,0]; name = "SnSe (p-type)"; bf = fitmap[name]
T, mu, P = bf["T"], bf["mu"], bf["params"]; Tfine = np.linspace(max(1.0, T.min()), T.max(), 800)
ax.plot(T, mu, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE, mew=1.0, color=BLUE)
ax.plot(Tfine, mu_eff_from_params(Tfine, P), lw=2.0, color=RED)
panel_box(ax); ax.text(0.5,1.02,"(d)", transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
ax.text(0.97,0.04,f"R$^2$={bf['r2']:.3f}", transform=ax.transAxes, ha="right", va="bottom")
ax.set_xlabel("T (K)"); ax.set_ylabel(r"$\mu_{\mathrm{eff}}$ (cm$^2$ V$^{-1}$ s$^{-1}$)")

# (e) Fe2V combined
ax_e = axs[1,1]
nm_n = "Fe2V0.95Ta0.05Al0.95Si0.05 (n-type)"
nm_p = "Fe2VAl0.95Si0.05 (p-type)"
bf_n, bf_p = fitmap[nm_n], fitmap[nm_p]
Tn, mun, Pn = bf_n["T"], bf_n["mu"], bf_n["params"]; Tnf = np.linspace(max(1.0, Tn.min()), Tn.max(), 800)
Tp, mup, Pp = bf_p["T"], bf_p["mu"], bf_p["params"]; Tpf = np.linspace(max(1.0, Tp.min()), Tp.max(), 800)
ax_e.plot(Tn, mun, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE,  mew=1.0, color=BLUE,  label="Fe$_2$V... (n)")
ax_e.plot(Tnf, mu_eff_from_params(Tnf, Pn), lw=2.0, color=RED,  label="Model (n)")
ax_e.plot(Tp, mup, linestyle="none", marker="s", ms=3.2, mfc="white", mec=BLACK, mew=1.0, color=BLACK, label="Fe$_2$VAl (p)")
ax_e.plot(Tpf, mu_eff_from_params(Tpf, Pp), lw=2.0, color=GRAY, label="Model (p)")
panel_box(ax_e); ax_e.text(0.5,1.02,"(e)", transform=ax_e.transAxes, ha="center", va="bottom", fontweight="bold")
ax_e.set_xlabel("T (K)"); ax_e.set_ylabel(r"$\mu_{\mathrm{eff}}$ (cm$^2$ V$^{-1}$ s$^{-1}$)")
ax_e.legend(frameon=False, fontsize=8, loc="upper right")

# (f) ZrNiSn — NEW
ax_f = axs[1,2]; name = "ZrNiSn (dense polycrystal)"; bf = fitmap[name]
T, mu, P = bf["T"], bf["mu"], bf["params"]; Tfine = np.linspace(max(1.0, T.min()), T.max(), 800)
ax_f.plot(T, mu, linestyle="none", marker="o", ms=3.5, mfc="white", mec=BLUE, mew=1.0, color=BLUE)
ax_f.plot(Tfine, mu_eff_from_params(Tfine, P), lw=2.0, color=RED)
panel_box(ax_f); ax_f.text(0.5,1.02,"(f)", transform=ax_f.transAxes, ha="center", va="bottom", fontweight="bold")
ax_f.text(0.97,0.04,f"R$^2$={bf['r2']:.3f}", transform=ax_f.transAxes, ha="right", va="bottom")
ax_f.set_xlabel("T (K)")

# tidy x-labels on top row
for ax in axs[0,:]: ax.set_xlabel("")

# ---------------- Save ----------------
out_dir = Path("./mosaic_with_zrnisn"); out_dir.mkdir(parents=True, exist_ok=True)
fig = plt.gcf()
fig.savefig(out_dir/"mobility_fits_mosaic.png", dpi=600, bbox_inches="tight")
fig.savefig(out_dir/"mobility_fits_mosaic.pdf", bbox_inches="tight")
plt.show()
print("Saved to:", out_dir.resolve())

# ---------------- Optional: export SI fit curves for Fe-series and ZrNiSn ----------------
for nm in (nm_n, nm_p, "ZrNiSn (dense polycrystal)"):
    bf = fitmap[nm]
    T = bf["T"]; P = bf["params"]
    Tf = np.linspace(max(1.0, T.min()), T.max(), 600)
    lT = l_of_T(Tf, P["l300"], P["p"])
    G  = lT/(lT + P["wGB"])
    muf = mu_w_of_T(Tf, P["mu0"], P["q"]) * np.exp(-P["Phi"]/(kB*Tf)) * G
    stem = nm.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    np.savetxt(out_dir/f"{stem}__fit_series.csv",
               np.column_stack([Tf, muf]),
               header="T_K,mu_fit_cm2_per_Vs", delimiter=",", comments="")
    with open(out_dir/f"{stem}__fit_params.json","w") as f:
        json.dump(dict(name=nm, **P, R2=bf["r2"], spec=bf["spec"]), f, indent=2)

