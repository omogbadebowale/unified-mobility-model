# ======================================================
# Unified Mobility Model Fitting for ZnO
# Authors: Gbadebo Taofeek Yusuf, Sukhwinder Singh, Alexandros Askounis, Zlatka Stoeva, Fideline Tchuenbou-Magaia
# Description: Model fitting code for ZnO using semi-empirical mobility model.
# This code is part of the repository: Unified Mobility Model for Grain-Boundary-Limited Transport
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the unified mobility model
def unified_mobility_model(T, mu_w, Phi_GB, w_GB, p):
    k_B = 8.6173e-5  # Boltzmann constant in eV/K
    # Thermionic emission and geometric transmission
    P_GB = np.exp(-Phi_GB / (k_B * T))  # thermionic emission term
    G_T = (15 / (w_GB + 15))  # Assuming a fixed bulk mean free path for simplicity
    return mu_w * P_GB * G_T

# Data for ZnO (temperature and corresponding mobility values)
T_ZnO = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100])
mobility_ZnO = np.array([12, 13, 14, 14, 14, 14, 16, 18, 20, 21, 22, 23, 23, 24, 25])

# Initial guesses for parameters (mu_w, Phi_GB, w_GB, p)
params_init_ZnO = [0.30, 0.15, 11.79, 1.5]  # Starting guesses for fitting

# Fit the model to the data
params_ZnO, _ = curve_fit(unified_mobility_model, T_ZnO, mobility_ZnO, p0=params_init_ZnO)

# Extract the fitted parameters
mu_w_fit_ZnO, Phi_GB_fit_ZnO, w_GB_fit_ZnO, p_fit_ZnO = params_ZnO

# Plot the data and the fitted model
T_fit_ZnO = np.linspace(min(T_ZnO), max(T_ZnO), 1000)  # Generate finer temperature points for smooth fit
mobility_fit_ZnO = unified_mobility_model(T_fit_ZnO, *params_ZnO)

plt.figure(figsize=(8,6))

# Plot the experimental data as blue dots
plt.plot(T_ZnO, mobility_ZnO, 'bo', label='ZnO')

# Plot the fitted model as a red curve
plt.plot(T_fit_ZnO, mobility_fit_ZnO, 'r-', label='Model')

# Customize the plot
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Mobility (cm²/V·s)', fontsize=12)
plt.title('ZnO Mobility vs. Temperature with Model Fit', fontsize=14)
plt.grid(True)
plt.legend(loc='best')

# Show the plot
plt.show()
