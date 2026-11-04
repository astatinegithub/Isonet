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

    # F. Encoder CIP + E/Z ON, 기존 R/S 중심 attention
    cfg_F = BASE_CONFIG.copy()
    cfg_F.update({
        "condition_name": "F_rs_attention",
        "encoder_use_cip": True,
        "encoder_use_ez": True,
        "use_stereo_attention": True,
        "include_ez_attention": False,
        "stereo_attention_mode": "learned",
    })

    # metrics_0: F
    # 비교 대상 G는 routes.py 결과 사용
    cfgs = [cfg_F]

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
    """E/Z 주변으로 집계 범위를 확장한 효과: F와 G 비교"""

    for endpoint in ("ExpansionRx_HLM", "PPBR_AZ"):
        for pretrained in (False, True):
            initialization = "pretrained" if pretrained else "scratch"
            experiment_name = (
                f"{endpoint}_final_v1_ez_scope_{initialization}"
            )

            for seed in (20, 25, 42):
                main(
                    seed,
                    experiment_name,
                    [endpoint],
                    pretrained=pretrained,
                )