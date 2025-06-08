# ======================================================
# Unified Mobility Model Fitting for SnSe
# Author: (Your Name)
# Description: Model fitting code for SnSe using semi-empirical mobility model.
# This code is part of the repository: Unified Mobility Model for Grain-Boundary-Limited Transport
# ======================================================


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

# Experimental SnSe data you provided
T_data = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
mu_exp = np.array([30, 45, 55, 65, 75, 80, 85, 88, 90, 93, 95])

# Constants
e = 1.602e-19  # elementary charge in Coulombs
kB = 1.381e-23  # Boltzmann constant in J/K
m_e = 9.109e-31  # free electron mass in kg

# Fixed parameters
m_star = 0.6 * m_e
p_fixed = 1.6

# The mobility model function
def mobility_model(T, phi_GB, l300, w_GB, mu_w):
    l_T = l300 * (T / 300) ** (-p_fixed)
    thermionic = np.exp(-phi_GB / (kB * T / e))  # eV to Joules inside exponential
    geometric = l_T / (l_T + w_GB)
    return mu_w * thermionic * geometric

# Set up the lmfit model
model = Model(mobility_model)

# Initial parameter guesses
params = model.make_params(
    phi_GB=0.08,  # eV
    l300=20,      # nm
    w_GB=10,      # nm
    mu_w=100      # cm^2/V/s (weighted mobility)
)

# Perform the fit
result = model.fit(mu_exp, params, T=T_data)

# Print the fitting report
fit_report = result.fit_report()
print(fit_report)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(T_data, mu_exp, 'bo', label='Experimental Data (SnSe)')
plt.plot(T_data, result.best_fit, 'r-', label='Model Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Mobility (cm²/V·s)')
plt.title('SnSe Mobility Fitting')
plt.legend()
plt.grid(True)
plt.show()
