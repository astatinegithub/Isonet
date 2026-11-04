from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS, DATAPATHS
from isonet.finalv1.run_experiment import run_experiment


def main(train_seed, experiment_name, endpoint_names, pretrained=False):
    origin_raw_data = DATAPATHS[endpoint_names[0]]["train"]
    origin_test_path = DATAPATHS[endpoint_names[0]].get("test", None)

    endpoints = {
        name: deepcopy(ALL_ENDPOINTS[name])
        for name in endpoint_names
    }

    BASE_CONFIG = {
        "atom_dim": 72,
        "bond_dim": 14,
        "hidden_dim": 128,
        "admet_hidden_dim": 64,
        "pooling": global_mean_pool,
        "num_endpoint": len(endpoints),
        "depth": 5,
        "drop_rate": 0.3,
        "endpoints": endpoints,

        "encode_grad_off": False,
        "attn_dim": 32,
        "attn_droprate": 0.2,
        "pretrain_component": ["encoder"],
    }

    # A. Encoder 입체정보 OFF, attention OFF
    cfg_A = BASE_CONFIG.copy()
    cfg_A.update({
        "condition_name": "A_no_stereo",
        "encoder_use_cip": False,
        "encoder_use_ez": False,
        "use_stereo_attention": False,
    })

    # D. Encoder CIP + E/Z ON, attention OFF
    cfg_D = BASE_CONFIG.copy()
    cfg_D.update({
        "condition_name": "D_stereo_features",
        "encoder_use_cip": True,
        "encoder_use_ez": True,
        "use_stereo_attention": False,
    })

    # E. Encoder 입체정보 OFF, attention 경로에서만 사용
    cfg_E = BASE_CONFIG.copy()
    cfg_E.update({
        "condition_name": "E_attention_only",
        "encoder_use_cip": False,
        "encoder_use_ez": False,
        "use_stereo_attention": True,
        "include_ez_attention": True,
        "stereo_attention_mode": "learned",
    })

    # G. Encoder CIP + E/Z ON, R/S + E/Z 주변 attention
    cfg_G = BASE_CONFIG.copy()
    cfg_G.update({
        "condition_name": "G_rs_ez_attention",
        "encoder_use_cip": True,
        "encoder_use_ez": True,
        "use_stereo_attention": True,
        "include_ez_attention": True,
        "stereo_attention_mode": "learned",
    })

    # metrics_0: A / metrics_1: D / metrics_2: E / metrics_3: G
    cfgs = [cfg_A, cfg_D, cfg_E, cfg_G]

    pretrain_path = None
    if pretrained:
        pretrain_path = (
            ROOT
            + "experiment/ex03/output/"
            + "pretrain_for_share_small_128_model_with_25seed/"
            + "checkpoint1_1/ssl_checkpoint_val_best.pt"
        )

    run_experiment(
        cfgs=cfgs,
        origin_raw_data=origin_raw_data,
        experiment_name=experiment_name + f"_{train_seed}seed",
        dir_name="routes",
        endpoints=endpoints,
        smiles_col=endpoints[endpoint_names[0]]["smiles_col"],
        epochs=100,
        save_freq=20,
        pretrain_path=pretrain_path,
        origin_test_path=origin_test_path,
        train_seed=train_seed,
        split_seed=25,
    )


if __name__ == "__main__":
    """Encoder 입체정보와 확장 attention의 전달 경로 비교: A·D·E·G"""

    for endpoint in ("ExpansionRx_HLM", "PPBR_AZ"):
        for pretrained in (False, True):
            initialization = "pretrained" if pretrained else "scratch"
            experiment_name = (
                f"{endpoint}_final_v1_routes_{initialization}"
            )

            for seed in (20, 25, 42):
                main(
                    seed,
                    experiment_name,
                    [endpoint],
                    pretrained=pretrained,
                )