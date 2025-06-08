# ======================================================
# Unified Mobility Model Fitting for Mg2Si
# Author: (Your Name)
# Description: Model fitting code for Mg2Si using semi-empirical mobility model.
# This code is part of the repository: Unified Mobility Model for Grain-Boundary-Limited Transport
# ======================================================


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

# Experimental data for Mg2Si
T_data = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700])
mu_data = np.array([400, 470, 580, 650, 700, 730, 770, 780, 760])

# Physical constants
kB = 8.617333262e-5  # Boltzmann constant in eV/K

# Weighted mobility (fixed for Mg2Si)
m_star = 0.4  # effective mass (m*/me)
mu_w = 400 * (m_star)**1.5

# Mean free path model
def leff(T, l300, p):
    return l300 * (T/300)**(-p)

# Full mobility model
def mobility(T, Phi_GB, l300, w_GB, p):
    return mu_w * np.exp(-Phi_GB / (kB * T)) * (leff(T, l300, p) / (leff(T, l300, p) + w_GB))

# Fitting using lmfit
model = Model(mobility)
params = model.make_params(Phi_GB=0.05, l300=60, w_GB=5, p=2.0)
params['Phi_GB'].min = 0
params['l300'].min = 1
params['w_GB'].min = 1
params['p'].min = 1.0
params['p'].max = 3.0

result = model.fit(mu_data, params, T=T_data)
print(result.fit_report())

# Generate fit for plotting
T_fit = np.linspace(300, 700, 300)
mu_fit = model.eval(result.params, T=T_fit)

# Plotting the fit result
plt.figure(figsize=(8,5))
plt.scatter(T_data, mu_data, color='blue', label='Experimental Data')
plt.plot(T_fit, mu_fit, color='red', label='Model Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Mobility (cm²/V·s)')
plt.title('Mg₂Si Mobility Fit')
plt.legend()
plt.grid()
plt.show()
