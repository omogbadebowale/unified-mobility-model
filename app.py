# mobility_model.py
import numpy as np
from scipy.optimize import differential_evolution, curve_fit

kB = 8.617333262145e-5  # eV/K

# Core GB-limited model (log-form for stability)
def _log_mu_eff_model(params, T, log_mu_obs):
    log_mu_w, phi_GB, log_w_GB, log_l_300, p = params
    mu_w = np.exp(log_mu_w)
    w_GB = np.exp(log_w_GB)
    l_300 = np.exp(log_l_300)
    l_T = l_300 * (T / 300) ** (-p)
    G_T = l_T / (l_T + w_GB)
    P_GB = np.exp(-phi_GB / (kB * T))
    mu_pred = mu_w * P_GB * G_T
    log_mu_pred = np.log(mu_pred)
    return np.sum((log_mu_pred - log_mu_obs) ** 2)

# Public interface for model fitting
def fit_mobility(T, mu, fix_w_GB=None):
    log_mu = np.log(mu)
    
    bounds = [
        (np.log(10), np.log(1000)),   # log(mu_w)
        (0.001, 1.0),                 # phi_GB
        (np.log(0.1), np.log(100)),  # log(w_GB)
        (np.log(1), np.log(500)),    # log(l_300)
        (0.1, 5.0)                   # p
    ]

    # Differential evolution for stability
    result = differential_evolution(
        _log_mu_eff_model,
        bounds,
        args=(T, log_mu),
        strategy='best1bin',
        maxiter=1000,
        polish=True,
        seed=42
    )

    # Extract DE result
    log_mu_w, phi_GB, log_w_GB, log_l_300, p = result.x
    mu_w = np.exp(log_mu_w)
    w_GB = np.exp(log_w_GB)
    l_300 = np.exp(log_l_300)

    # Optionally fix w_GB for hybrid fitting
    if fix_w_GB is not None:
        def model_fixed(T, mu_w, phi_GB, l_300, p):
            l_T = l_300 * (T / 300) ** (-p)
            G_T = l_T / (l_T + fix_w_GB)
            P_GB = np.exp(-phi_GB / (kB * T))
            return mu_w * P_GB * G_T

        popt, pcov = curve_fit(
            model_fixed, T, mu,
            p0=[mu_w, phi_GB, l_300, p],
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "mu_w (cm^2/Vs)": (popt[0], perr[0]),
            "phi_GB (eV)": (popt[1], perr[1]),
            "w_GB (nm)": (fix_w_GB, 0.0),
            "l_300 (nm)": (popt[2], perr[2]),
            "p": (popt[3], perr[3])
        }

    # Return DE-only fit without uncertainty
    return {
        "mu_w (cm^2/Vs)": mu_w,
        "phi_GB (eV)": phi_GB,
        "w_GB (nm)": w_GB,
        "l_300 (nm)": l_300,
        "p": p
    }
