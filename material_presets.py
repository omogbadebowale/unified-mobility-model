# material_presets.py

material_classes = {
    "ZnO": {
        "class": "B: Moderate GB-Limited Semiconductor",
        "params": {
            "phi_GB": 0.15,
            "l300": 15,
            "w_GB": 5,
            "p": 1.5,
            "mu_w_bounds": (100, 600)
        },
        "fixed": ["phi_GB", "l300", "w_GB", "p"]
    },
    "Bi2Te3": {
        "class": "C: High Barrier / Disordered Semiconductor",
        "params": {
            "phi_GB": 0.20,
            "l300": 20,
            "w_GB": 10,
            "p": 2.0,
            "mu_w_bounds": (50, 300)
        },
        "fixed": ["phi_GB", "l300", "w_GB", "p"]
    },
    "SnSe": {
        "class": "C: High Barrier / Disordered Semiconductor",
        "params": {
            "phi_GB": 0.25,
            "l300": 10,
            "w_GB": 10,
            "p": 2.5,
            "mu_w_bounds": (30, 200)
        },
        "fixed": ["phi_GB", "l300", "w_GB", "p"]
    },
    "PbTe": {
        "class": "B: Moderate GB-Limited Semiconductor",
        "params": {
            "phi_GB": 0.12,
            "l300": 25,
            "w_GB": 8,
            "p": 1.8,
            "mu_w_bounds": (100, 500)
        },
        "fixed": ["phi_GB", "l300", "w_GB", "p"]
    },
    "Custom": {
        "class": "Custom Material – User Defines Parameters",
        "params": {},
        "fixed": []
    }
}
