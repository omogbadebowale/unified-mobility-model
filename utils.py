import matplotlib.pyplot as plt
import numpy as np

def plot_fit(T_data, mu_data, fit_result):
    T_fit = np.linspace(min(T_data), max(T_data), 500)
    mu_fit = fit_result.model.eval(fit_result.params, T=T_fit)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_data, mu_data, 'bo', label='Experimental Data')
    ax.plot(T_fit, mu_fit, 'r-', label='Model Fit')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Mobility (cm²/V·s)')
    ax.set_title('Mobility Fit')
    ax.grid(True)
    ax.legend()
    return fig

def evaluate_fit_quality(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return r_squared, rmse
