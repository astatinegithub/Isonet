import os
import torch
from pathlib import Path
from copy import deepcopy

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_pretrain_data
from isonet.model.train.pretrain_models import model_pretraining
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer
from isonet.data.dataload.raw_data_download import load_sdf_data



def run_experiment(cfg, pretrain_data, experiment_name, train_seed, split_seed, checkpoint_path=None):
    
    
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
        epochs=30,
        plt_close=True,
        seed=train_seed
    )


if __name__ == "__main__":
    pretrain_data = ROOT + "dataset/raw/smiles/Compound_000500001_001000000.sdf"
    experiment_name = 'pretrain_share2'
    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 512,
            "depth": 5,
            "drop_rate":0.3,
            "atom_types": AtomFeaturizer.default().atom_type_nums,
            "bond_types": BondFeaturizer.default().bond_type_nums,
        }
    run_experiment(
        BASE_CONFIG,
        pretrain_data,
        experiment_name,
        train_seed=42,
        split_seed=25, 
        checkpoint_path=ROOT+'/experiment/ex03/output/pretrain_share/checkpoint1_1/ssl_checkpoint_49epoch_finish.pt'
    )