import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_pretrain_data
from isonet.model.train.pretrain_models import model_pretraining
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer
from isonet.data.dataload.raw_data_download import load_sdf_data



def run_experiment(cfg, pretrain_data, experiment_name, train_seed,
                   split_seed, epochs, checkpoint_path=None):
    
    
    output_path1_1 = ROOT + f"dataset/processed_data/ex03_{experiment_name}/pretrain_data1.pt"
    Path(ROOT + f"dataset/processed_data/ex03_{experiment_name}").mkdir(parents=True, exist_ok=True)

    if not(os.path.exists(output_path1_1)):
        make_pretrain_data(pretrain_data, output_path1_1, use_stereo=True, max_len=200000)
    train_graphs1_1, val_graphs1_1, _ = split_dataset(
        path=output_path1_1,
        split_ratio=[0.8, 0.2, 0],
        seed=split_seed
    )
    # 모델 생성
    save_dir1_1 = ROOT + f"experiment/ex03/output/{experiment_name}/checkpoint1_1"
    Path(save_dir1_1).mkdir(parents=True, exist_ok=True)

    loss_image_path1_1 = ROOT + f"experiment/ex03/output/{experiment_name}/loss_curve1_1.png"

    model_pretraining(
        cfg=cfg,
        train_graphs=train_graphs1_1,
        val_graphs=val_graphs1_1,
        loss_image_path=loss_image_path1_1,
        save_dir=save_dir1_1,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        plt_close=True,
        seed=train_seed
    )


if __name__ == "__main__":
    """사전학습 실험(?) -> 공용 사전학습 checkpoint 제작이 목표"""
    # pretrain_data = ROOT + "dataset/raw/smiles/Compound_000000001_000500000.sdf"
    # experiment_name = 'pretrain_share_retrain_mean_pool'
    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 128,
            "depth": 5,
            "drop_rate":0.3,
            "pooling":global_mean_pool,
            "atom_types": AtomFeaturizer.default().atom_type_nums,
            "bond_types": BondFeaturizer.default().bond_type_nums,
            "encoder_use_cip": False,      # DMPNN에 원자 CIP를 전달할지
            # "use_stereo_attention": True,  # Attention을 사용할지
        }
    experiment_name = 'pretrain_for_share_small_128_model_with_25seed'

    pretrain_data = ROOT + "dataset/raw/smiles/Compound_000000001_000500000_parts/part_001.sdf"
    run_experiment(
        BASE_CONFIG,
        pretrain_data,
        experiment_name,
        train_seed=25,
        split_seed=25, 
        epochs=100,
        # checkpoint_path=ROOT+'/experiment/ex03/output/pretrain_for_share_no_cip2/checkpoint1_1/ssl_checkpoint_200epoch.pt'
    )
