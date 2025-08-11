import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the unified mobility model (you can refine this as needed)
def unified_mobility_model(T, mu_w, Phi_GB, w_GB, p):
    k_B = 8.6173e-5  # Boltzmann constant in eV/K
    # Thermionic emission and geometric transmission
    P_GB = np.exp(-Phi_GB / (k_B * T))  # thermionic emission term
    G_T = (15 / (w_GB + 15))  # Assuming a fixed bulk mean free path for simplicity
    return mu_w * P_GB * G_T

# Data for SnSe (temperature and corresponding mobility values)
T = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850])
mobility = np.array([600, 400, 300, 250, 200, 150, 130, 126, 124, 121, 115, 111])

# Fit the model to the data (with initial guesses for parameters)
params_init = [124.41, 0.1, 35.5, 1.5]  # Initial guesses for mu_w, Phi_GB, w_GB, p
params, covariance = curve_fit(unified_mobility_model, T, mobility, p0=params_init)

# Extract fitted parameters
mu_w_fit, Phi_GB_fit, w_GB_fit, p_fit = params

# Plot the data and the fitted model
T_fit = np.linspace(min(T), max(T), 1000)  # Generate finer temperature points for smooth fit
mobility_fit = unified_mobility_model(T_fit, *params)

plt.figure(figsize=(8,6))

# Plot the experimental data as blue dots
plt.plot(T, mobility, 'bo', label='SnSe')

# Plot the fitted model as a red curve
plt.plot(T_fit, mobility_fit, 'r-', label='Model')

# Customize the plot
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Mobility (cm²/V·s)', fontsize=12)
plt.title('Unified Mobility Model Fit for SnSe', fontsize=14)
plt.grid(True)
plt.legend(loc='best')

# Show the plot
plt.show()
