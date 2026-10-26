import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data, make_pretrain_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.train.pretrain_models import model_pretraining
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer



def run_experiment(cfg, processed_data, finetune_data, experiment_name,
                   endpoints, train_seed, split_seed, pre_epochs=30,
                   fine_epochs=50, smiles_col='smiles', pretrained_checkpoint=None):

    # output_path1_1 = ROOT + f"dataset/processed_data/ex02_{experiment_name}/pretrain_data1.pt"
    output_path1_2 = ROOT + f"dataset/processed_data/ex02_{experiment_name}/finetuning_data1.pt"
    output_path2 = ROOT + f"dataset/processed_data/ex02_{experiment_name}/finetuning_data2.pt"
    Path(ROOT + f"dataset/processed_data/ex02_{experiment_name}").mkdir(parents=True, exist_ok=True)

    if not(os.path.exists(output_path1_2) & os.path.exists(output_path2)):
        make_fine_tuning_data(finetune_data, output_path1_2, endpoints, smiles_col=smiles_col, use_stereo=True)
        make_fine_tuning_data(finetune_data, output_path2, endpoints, smiles_col=smiles_col, use_stereo=True)
        

    data_path1_1 = processed_data
    data_path1_2 = output_path1_2
    data_path2 = output_path2

    train_graphs1_1, val_graphs1_1, test_graphs1_1 = split_dataset(
        path=data_path1_1,
        split_ratio=[0.7, 0.2, 0.1],
        seed=split_seed
    )
    train_graphs1_2, val_graphs1_2, test_graphs1_2 = split_dataset(
            path=data_path1_2,
            split_ratio=[0.7, 0.2, 0.1],
            seed=split_seed
        )
    train_graphs2, val_graphs2, test_graphs2 = split_dataset(
            path=data_path2,
            split_ratio=[0.7, 0.2, 0.1],
            seed=split_seed
        )
    


    # 검사
    data1 = train_graphs1_2.dataset
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
        (train_graphs1_2, val_graphs1_2, test_graphs1_2),
        (train_graphs2, val_graphs2, test_graphs2),
    ):
        if part1.indices != part2.indices:
            raise ValueError("두 조건의 분할 인덱스가 다릅니다.")


    # 모델 2가지 생성
    save_dir1_1 = ROOT + f"experiment/ex02/output/{experiment_name}/checkpoint1_1"
    save_dir1_2 = ROOT + f"experiment/ex02/output/{experiment_name}/checkpoint1_2"
    save_dir2 = ROOT + f"experiment/ex02/output/{experiment_name}/checkpoint2"
    Path(save_dir1_1).mkdir(parents=True, exist_ok=True)
    Path(save_dir1_2).mkdir(parents=True, exist_ok=True)
    Path(save_dir2).mkdir(parents=True, exist_ok=True)

    loss_image_path1_1 = ROOT + f"experiment/ex02/output/{experiment_name}/loss_curve1_1.png"
    loss_image_path1_2 = ROOT + f"experiment/ex02/output/{experiment_name}/loss_curve1_2.png"
    loss_image_path2 = ROOT + f"experiment/ex02/output/{experiment_name}/loss_curve2.png"

    if pretrained_checkpoint is None:
        pretrain_last_model_path = model_pretraining(
            cfg=cfg,
            train_graphs=train_graphs1_1,
            val_graphs=val_graphs1_1,
            loss_image_path=loss_image_path1_1,
            save_dir=save_dir1_1,
            epochs=pre_epochs,
            plt_close=True,
            seed=train_seed
        )
    else: 
        pretrain_last_model_path = pretrained_checkpoint
    model1, last_model_path1 = model_finetuning(
        cfg=cfg,
        train_graphs=train_graphs1_2,
        val_graphs=val_graphs1_2,
        loss_image_path=loss_image_path1_2,
        save_dir=save_dir1_2,
        pretrain_path=pretrain_last_model_path,
        epochs=fine_epochs,
        endpoints=endpoints,
        plt_close=True,
        seed=train_seed
    )
    model2, last_model_path2 = model_finetuning(
        cfg=cfg,
        train_graphs=train_graphs2,
        val_graphs=val_graphs2,
        loss_image_path=loss_image_path2,
        save_dir=save_dir2,
        pretrain_path=None,
        epochs=fine_epochs,
        endpoints=endpoints,
        plt_close=True,
        seed=train_seed
    )

    # best_model선정
    checkpoint1 = select_best_model(last_model_path1, save_dir1_2)
    checkpoint2 = select_best_model(last_model_path2, save_dir2)


    # 모델 2가지 평가
    output_dir = ROOT + f"experiment/ex02/output/{experiment_name}/" 

    evaluate_model(
        eval_model=checkpoint1,
        test_graphs=test_graphs1_2,
        output_dir=output_dir,
        output_file_number=1
    )
    evaluate_model(
        eval_model=checkpoint2,
        test_graphs=test_graphs2,
        output_dir=output_dir,
        output_file_number=2
    )


def make_pretrain_dataset(in_path, out_path):
    if not os.path.exists(out_path):
        make_pretrain_data(in_path, out_path, use_stereo=True, max_len=200000)
    else:
        print('이미 데이터셋이 있음')

if __name__ == "__main__":
    """사전학습 유무 차이보는 실험"""
    pretrain_data = ROOT + "dataset/raw/smiles/Compound_000000001_000500000.sdf"
    processed_data = ROOT + "/dataset/processed_data/processed_smiles/" + pretrain_data.split('/')[-1][:-4]
    finetune_path = ROOT + "dataset/raw/admet/PPBR_AZ.csv"
    endpoint_names = ["PPBR_AZ"]
    endpoints = endpoints = {
        name: deepcopy(ALL_ENDPOINTS[name])
        for name in endpoint_names
    }
    train_seed = 42
    experiment_name = f'PPBR_AZ_{train_seed}seed_re'

    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 512,
            "admet_hidden_dim": 256,
            "num_endpoint": len(endpoints.keys()), 
            "depth": 5,
            "drop_rate":0.3,
            "endpoints": endpoints,
            "atom_types": AtomFeaturizer.default().atom_type_nums,
            "bond_types": BondFeaturizer.default().bond_type_nums,
        }

    BASE_CONFIG["pooling"] = global_mean_pool
    make_pretrain_dataset(pretrain_data, processed_data)

    run_experiment(
        BASE_CONFIG,
        processed_data,
        finetune_path, 
        experiment_name, 
        endpoints, 
        fine_epochs=100,
        smiles_col=endpoints[endpoint_names[0]]['smiles_col'],
        train_seed=train_seed,
        split_seed=25,
    )