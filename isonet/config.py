from pathlib import Path

ROOT = Path(__file__).resolve().parent.as_posix() + "/"

#예시 양식 겸 기본값 
ENDPOINTS = {
    "Lipophilicity": {
        "type": "regression",
        "target_column": "exp",
        "selection_metric": "rmse",
    },
}
MODEL_A_CONFIG = { # 지금은 안씀
    "atom_dim": 72,        # atom feature 차원
    "bond_dim": 14,        # bond feature 차원
    "hidden_dim": 512,
    "admet_hidden_dim": 256,
    "num_endpoint": len(ENDPOINTS.keys()), 
    "depth": 5,
    "drop_rate":0.3,
    "endpoints":ENDPOINTS
}

ALL_ENDPOINTS = {
    "Lipophilicity": {
        "type": "regression",
        "target_column": "exp",
        "selection_metric": "rmse",
        "smiles_col": "smiles",
    },
    "PPBR_AZ": {
        "type": "regression",
        "target_column": "Y",
        "selection_metric": "mae",
        "smiles_col": "SMILES",
    },
    "ExpansionRx_HLM": {
        "type": "regression",
        "target_column": "HLM CLint",
        "selection_metric": "rmse",
        "smiles_col": "SMILES",
        "target_transform": "log10p1",
    },
    "ExpansionRx_MLM": {
        "type": "regression",
        "target_column": "MLM CLint",
        "selection_metric": "rmse",
        "smiles_col": "SMILES",
        "target_transform": "log10p1",
    },
    "CMRT_order": { # 사전학습용임 endpoint이 아님 
        "type": "binary",
        "target_column": "elution_order",
        "smiles_col": "SMILES",
        "target_transform": "map_0/1"
    },
}

DATAPATHS = {
    "PPBR_AZ": {
        "train" : ROOT + "dataset/raw/admet/PPBR_AZ.csv",
    },
    "ExpansionRx_HLM": {
        "train": ROOT + "dataset/raw/admet/ExpansionRx_train.csv",
        "test":  ROOT + "dataset/raw/admet/ExpansionRx_test.csv",
    }
}