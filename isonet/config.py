from pathlib import Path

ROOT = Path(__file__).resolve().parent.as_posix() + "/"

ENDPOINTS = {
    "Lipophilicity": {
        "type": "regression",
        "target_column": "exp",
    },
}
MODEL_A_CONFIG = {
    "atom_dim": 72,        # atom feature 차원
    "bond_dim": 14,        # bond feature 차원
    "dmpnn_hidden_dim": 512,
    "admet_hidden_dim": 256,
    "num_endpoint": len(ENDPOINTS.keys()), 
    "depth": 5,
    "drop_rate":0.3,
    "endpoints":ENDPOINTS
}