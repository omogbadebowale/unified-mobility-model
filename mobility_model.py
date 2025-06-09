import numpy as np
from lmfit import Model, Parameters

kB = 8.617333262e-5  # eV/K

def unified_mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l_T = l300 * (T / 300)**(-p)
    geom_factor = l_T / (l_T + w_GB)
    thermionic_factor = np.exp(-phi_GB / (kB * T))
    return mu_w * thermionic_factor * geom_factor

def run_fit(T_data, mu_data, initial_params, fix_params=None):
    model = Model(unified_mobility_model)
    params = Parameters()

    params.add('mu_w', value=initial_params['mu_w'], min=50, max=6000)
    params.add('phi_GB', value=initial_params['phi_GB'], min=0.01, max=0.3)
    params.add('l300', value=initial_params['l300'], min=5, max=100)
    params.add('w_GB', value=initial_params['w_GB'], min=2, max=30)
    params.add('p', value=initial_params['p'], min=1.0, max=3.0)

    if fix_params:
        for key in fix_params:
            if key in params:
                params[key].set(vary=False)

    result = model.fit(mu_data, params, T=T_data)
    return result
