import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model



def run_experiment(cfgs:list, origin_raw_data, experiment_name,
                   endpoints, smiles_col, epochs, save_freq=1,
                   pretrain_path=None, train_seed=42, split_seed=42):
    """
    1. base 
    2. base + 인코더 CIP
    3. Stereo Attention
    4. 인코더 CIP+ Stereo Attention
    """
    Path(ROOT + f"dataset/processed_data/ex06_{experiment_name}").mkdir(parents=True, exist_ok=True)
    output_dir = ROOT + f"experiment/ex06/output/{experiment_name}/" 
    for i, cfg in enumerate(cfgs):
        output_path_ = ROOT + f"dataset/processed_data/ex06_{experiment_name}/finetuning_data{i}.pt"
        if not(os.path.exists(output_path_)):
            make_fine_tuning_data(origin_raw_data, output_path_, endpoints, smiles_col=smiles_col, use_stereo=True)
        data_path_ = output_path_
        train_graphs_, val_graphs_, test_graphs_ = split_dataset(
            path=data_path_,
            split_ratio=[0.7, 0.2, 0.1],
            seed=split_seed
        )
        save_dir_ = ROOT + f"experiment/ex06/output/{experiment_name}/checkpoint{i}"
        Path(save_dir_).mkdir(parents=True, exist_ok=True)

        loss_image_path_ = ROOT + f"experiment/ex06/output/{experiment_name}/loss_curve{i}.png"

        model_, last_model_path_ = model_finetuning(
                cfg=cfg,
                train_graphs=train_graphs_,
                val_graphs=val_graphs_,
                loss_image_path=loss_image_path_,
                save_dir=save_dir_,
                pretrain_path=pretrain_path,
                epochs=epochs,
                endpoints=endpoints,
                plt_close=True,
                seed=train_seed,
                save_freq=save_freq,
            )
        checkpoint_ = select_best_model(last_model_path_, save_dir_)
        evaluate_model(
            eval_model=checkpoint_,
            test_graphs=test_graphs_,
            output_dir=output_dir,
            output_file_number=i
        )

def main(train_seed, experiment_name):
    origin_raw_data = ROOT + "dataset/raw/admet/PPBR_AZ.csv"
    endpoint_names = ["PPBR_AZ"]
    endpoints = endpoints = {
        name: deepcopy(ALL_ENDPOINTS[name])
        for name in endpoint_names
    }

    # 저장 dir 이름 변경
    # train_seed = 40
    # experiment_name = f'PPBR_AZ_{train_seed}seed_attn_test_lr1e4_mean'
    

    # 공통설정
    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 512,
            "admet_hidden_dim": 256,
            "pooling": global_mean_pool,
            "num_endpoint": len(endpoints.keys()), 
            "depth": 5,
            "drop_rate":0.3,
            "endpoints": endpoints,
        }
    
    # 1. base 
    cfg1 = BASE_CONFIG.copy()
    cfg1['encoder_use_cip'] = False

    cfg1['use_stereo_attention'] = False


    # 2. base + 인코더 CIP
    cfg2 = BASE_CONFIG.copy()
    cfg2['encoder_use_cip'] = True

    cfg2['use_stereo_attention'] = False


    # 3. Stereo Attention
    cfg3 = BASE_CONFIG.copy()
    cfg3['encoder_use_cip'] = False

    cfg3['use_stereo_attention'] = True
    cfg3['attn_dim'] = 64
    cfg3['attn_droprate'] = 0.2


    # 4. 인코더 CIP+ Stereo Attention
    cfg4 = BASE_CONFIG.copy()
    cfg4['encoder_use_cip'] = True

    cfg4['use_stereo_attention'] = True
    cfg4['attn_dim'] = 64
    cfg4['attn_droprate'] = 0.2

    cfgs = [cfg1, cfg2, cfg3, cfg4]


    run_experiment(
        cfgs=cfgs,
        origin_raw_data=origin_raw_data,
        experiment_name=experiment_name,
        endpoints=endpoints,
        epochs=100,
        save_freq=20,
        # pretrain_path=ROOT + '/experiment/ex03/output/pretrain_share_retrain_mean_pool2/checkpoint1_1/ssl_checkpoint_119epoch.pt',
        # pretrain_path=ROOT + '/experiment/ex03/output/pretrain_share2/checkpoint1_1/ssl_checkpoint_80epoch.pt',
        smiles_col=endpoints[endpoint_names[0]]['smiles_col'],
        train_seed=train_seed,
        split_seed=25,
    )


if __name__ == "__main__":
    """cip언제 어디에 넣는지 비교 실험"""

    train_seed = 20
    main(train_seed, f'PPBR_AZ_{train_seed}seed_cip_where') # 사전학습 경로 재설정 필요
    train_seed = 25
    main(train_seed, f'PPBR_AZ_{train_seed}seed_cip_where')
    train_seed = 42
    main(train_seed, f'PPBR_AZ_{train_seed}seed_cip_where')