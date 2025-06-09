import numpy as np
from lmfit import Model, Parameters

kB = 8.617333262e-5  # eV/K

def unified_mobility_model(T, mu_w, phi_GB, l300, w_GB, p):
    l_T = l300 * (T / 300)**(-p)
    geom_factor = l_T / (l_T + w_GB)
    thermionic_factor = np.exp(-phi_GB / (kB * T))
    return mu_w * thermionic_factor * geom_factor

def simplified_thermionic_model(T, mu_w, phi_GB):
    return mu_w * np.exp(-phi_GB / (kB * T))

def run_fit(T_data, mu_data, initial_params, fix_params=None, fallback=False):
    if fallback:
        model = Model(simplified_thermionic_model)
        params = Parameters()
        params.add('mu_w', value=initial_params.get('mu_w', 100), min=1)
        params.add('phi_GB', value=initial_params.get('phi_GB', 0.05), min=0)
    else:
        model = Model(unified_mobility_model)
        params = Parameters()
        params.add('mu_w', value=initial_params.get('mu_w', 300), min=0)
        params.add('phi_GB', value=initial_params.get('phi_GB', 0.05), min=0)
        params.add('l300', value=initial_params.get('l300', 20), min=1)
        params.add('w_GB', value=initial_params.get('w_GB', 5), min=1)
        params.add('p', value=initial_params.get('p', 1.5), min=1, max=3)

        if fix_params:
            for p_name in fix_params:
                params[p_name].set(vary=False)

    result = model.fit(mu_data, params, T=T_data)
    return result
