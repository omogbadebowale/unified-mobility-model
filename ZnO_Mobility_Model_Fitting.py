# ======================================================
# Unified Mobility Model Fitting for ZnO
# Authors: Gbadebo Taofeek Yusuf, Sukhwinder Singh, Alexandros Askounis, Zlatka Stoeva, Fideline Tchuenbou-Magaia
# Description: Model fitting code for ZnO using semi-empirical mobility model.
# This code is part of the repository: Unified Mobility Model for Grain-Boundary-Limited Transport
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

# Experimental data for ZnO (Temperature in K and mobility in cm²/V·s)
T_data = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700])
mu_exp = np.array([20, 45, 70, 100, 130, 150, 175, 185, 190])

# Constants
kB = 8.617333262145e-5  # eV/K

# Define the full mobility model
def mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l_T = l300 * (T / 300)**(-p)
    geom_factor = l_T / (l_T + w_GB)
    thermionic_factor = np.exp(-phi_GB / (kB * T))
    mu_eff = mu_w * thermionic_factor * geom_factor
    return mu_eff

# Wrap the model for lmfit
model = Model(mobility_model)

# Initial parameter guesses
params = model.make_params(
    mu_w=300,       # weighted mobility guess
    phi_GB=0.1,     # barrier height guess (eV)
    l300=20,        # mean free path at 300K (nm)
    w_GB=5,         # grain boundary width (nm)
    p=1.5           # phonon scattering exponent
)

# Set reasonable bounds for parameters
params['mu_w'].set(min=0, max=1000)
params['phi_GB'].set(min=0, max=0.5)
params['l300'].set(min=1, max=100)
params['w_GB'].set(min=1, max=20)
params['p'].set(min=1.0, max=3.0)

# Perform the fit
result = model.fit(mu_exp, params, T=T_data)

# Print the fitting report
print(result.fit_report())

# Plotting the data and the fit
plt.figure(figsize=(8,5))
plt.plot(T_data, mu_exp, 'bo', label='Experimental data')
plt.plot(T_data, result.best_fit, 'r-', label='Model fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Mobility (cm²/V·s)')
plt.title('ZnO Mobility Model Fit')
plt.legend()
plt.grid(True)
plt.show()
