import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool, global_add_pool

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model



def run_experiment(cfg1, cfg2, origin_raw_data, experiment_name,
                   endpoints, smiles_col, epochs, save_freq=1,
                   pretrain_path=None, train_seed=42, split_seed=42):
    """
    cfg1 -> stereo attention\\
    cfg2 -> base (there is no stereo attention)
    """
    output_path1 = ROOT + f"dataset/processed_data/ex05_{experiment_name}/finetuning_data1.pt"
    output_path2 = ROOT + f"dataset/processed_data/ex05_{experiment_name}/finetuning_data2.pt"
    Path(ROOT + f"dataset/processed_data/ex05_{experiment_name}").mkdir(parents=True, exist_ok=True)

    if not(os.path.exists(output_path1) & os.path.exists(output_path2)):
        make_fine_tuning_data(origin_raw_data, output_path1, endpoints, smiles_col=smiles_col, use_stereo=True)
        make_fine_tuning_data(origin_raw_data, output_path2, endpoints, smiles_col=smiles_col, use_stereo=True)

    data_path1 = output_path1
    data_path2 = output_path2

    train_graphs1, val_graphs1, test_graphs1 = split_dataset(
        path=data_path1,
        split_ratio=[0.7, 0.2, 0.1],
        seed=split_seed
    )
    train_graphs2, val_graphs2, test_graphs2 = split_dataset(
        path=data_path2,
        split_ratio=[0.7, 0.2, 0.1],
        seed=split_seed
    )

    # 검사
    data1 = train_graphs1.dataset
    data2 = train_graphs2.dataset
    if len(data1) != len(data2):
        raise ValueError("두 조건의 데이터 개수가 다릅니다.")
    for a, b in zip(data1, data2):
        if a.smiles != b.smiles:
            raise ValueError("두 조건의 분자 순서가 다릅니다.")
        torch.testing.assert_close(
            a.y, b.y, rtol=0, atol=0, equal_nan=True
        )
    for part1, part2 in zip(
        (train_graphs1, val_graphs1, test_graphs1),
        (train_graphs2, val_graphs2, test_graphs2),
    ):
        if part1.indices != part2.indices:
            raise ValueError("두 조건의 분할 인덱스가 다릅니다.")


    # 모델 2가지 생성
    save_dir1 = ROOT + f"experiment/ex05/output/{experiment_name}/checkpoint1"
    save_dir2 = ROOT + f"experiment/ex05/output/{experiment_name}/checkpoint2"
    Path(save_dir1).mkdir(parents=True, exist_ok=True)
    Path(save_dir2).mkdir(parents=True, exist_ok=True)

    loss_image_path1 = ROOT + f"experiment/ex05/output/{experiment_name}/loss_curve1.png"
    loss_image_path2 = ROOT + f"experiment/ex05/output/{experiment_name}/loss_curve2.png"

    model1, last_model_path1 = model_finetuning(
        cfg=cfg1,
        train_graphs=train_graphs1,
        val_graphs=val_graphs1,
        loss_image_path=loss_image_path1,
        save_dir=save_dir1,
        pretrain_path=pretrain_path,
        epochs=epochs,
        endpoints=endpoints,
        plt_close=True,
        seed=train_seed,
        save_freq=save_freq,
    )
    model2, last_model_path2 = model_finetuning(
        cfg=cfg2,
        train_graphs=train_graphs2,
        val_graphs=val_graphs2,
        loss_image_path=loss_image_path2,
        save_dir=save_dir2,
        pretrain_path=pretrain_path,
        epochs=epochs,
        endpoints=endpoints,
        plt_close=True,
        seed=train_seed,
        save_freq=save_freq
    )

    # best_model선정
    checkpoint1 = select_best_model(last_model_path1, save_dir1)
    checkpoint2 = select_best_model(last_model_path2, save_dir2)


    # 모델 2가지 평가
    output_dir = ROOT + f"experiment/ex05/output/{experiment_name}/" 

    evaluate_model(
        eval_model=checkpoint1,
        test_graphs=test_graphs1,
        output_dir=output_dir,
        output_file_number=1
    )
    evaluate_model(
        eval_model=checkpoint2,
        test_graphs=test_graphs2,
        output_dir=output_dir,
        output_file_number=2
    )


if __name__ == "__main__":
    origin_raw_data = ROOT + "dataset/raw/admet/PPBR_AZ.csv"
    endpoint_names = ["PPBR_AZ"]
    endpoints = endpoints = {
        name: deepcopy(ALL_ENDPOINTS[name])
        for name in endpoint_names
    }

    # 저장 dir 이름 변경
    train_seed = 40
    experiment_name = f'PPBR_AZ_{train_seed}seed_attn_test_lr1e4_mean'
    

    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 512,
            "admet_hidden_dim": 256,
            "pooling": global_mean_pool,
            "num_endpoint": len(endpoints.keys()), 
            "depth": 5,
            "drop_rate":0.3,
            "endpoints": endpoints
        }
    
    cfg1 = BASE_CONFIG.copy()
    cfg1['use_stereo_attention'] = True
    cfg1['attn_dim'] = 64
    cfg1['attn_droprate'] = 0.2

    cfg2 = BASE_CONFIG.copy()
    cfg2['use_stereo_attention'] = False



    run_experiment(
        cfg1=cfg1,
        cfg2=cfg2,
        origin_raw_data=origin_raw_data,
        experiment_name=experiment_name,
        endpoints=endpoints,
        epochs=100,
        save_freq=10,
        pretrain_path=ROOT + '/experiment/ex03/output/pretrain_share_retrain_mean_pool2/checkpoint1_1/ssl_checkpoint_119epoch.pt',
        # pretrain_path=ROOT + '/experiment/ex03/output/pretrain_share2/checkpoint1_1/ssl_checkpoint_80epoch.pt',
        smiles_col=endpoints[endpoint_names[0]]['smiles_col'],
        train_seed=train_seed,
        split_seed=25,
    )