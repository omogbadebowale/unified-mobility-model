# ======================================================
# Unified Mobility Model Fitting for Bi2Te3
# Author: (Your Name)
# Description: Model fitting code for Bi2Te3 using semi-empirical mobility model.
# This code is part of the repository: Unified Mobility Model for Grain-Boundary-Limited Transport
# ======================================================


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Experimental data for Bi2Te3
T_data = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300])
mu_data = np.array([1000, 950, 890, 830, 770, 700, 630, 550, 470])

# Constants
k_B = 8.617333262e-5  # eV/K

# Model function
def mobility_model(T, mu_w, Phi_GB, l_300, w_GB, p):
    l_T = l_300 * (T / 300) ** (-p)
    thermionic = np.exp(-Phi_GB / (k_B * T))
    geometric = l_T / (l_T + w_GB)
    return mu_w * thermionic * geometric

model = Model(mobility_model)

# Set initial parameter guesses and bounds
params = Parameters()
params.add('mu_w', value=1000, min=100, max=2000)
params.add('Phi_GB', value=0.01, min=0, max=0.05)
params.add('l_300', value=50, min=10, max=100)
params.add('w_GB', value=100, min=10, max=200)
params.add('p', value=1.5, vary=False)

# Fit the model
result = model.fit(mu_data, params, T=T_data)

# Generate fit curve
T_fit = np.linspace(100, 300, 500)
mu_fit = result.eval(T=T_fit)

# Plot results
plt.figure(figsize=(8,5))
plt.scatter(T_data, mu_data, label='Experimental Data', color='blue')
plt.plot(T_fit, mu_fit, label='Model Fit', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Mobility (cm²/V·s)')
plt.title('Bi₂Te₃ Mobility Fit')
plt.legend()
plt.grid(True)
plt.show()

# Display fit report
print(result.fit_report())
