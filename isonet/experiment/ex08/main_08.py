import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS
from isonet.data.dataset import split_dataset
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer 
from isonet.data.preprocess import make_fine_tuning_data, make_pretrain_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.train.pretrain_models import model_pretraining
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model



def run_experiment(pre_cfg, cfgs:list, origin_raw_data, experiment_name,
                   endpoints, smiles_col, epochs, pre_epochs,
                   origin_pretrain_data, save_freq=1, pretrain_path=None,
                   train_seed=42, split_seed=42):
    """
    1. 2차 stereo 학습 + stereo attention
    2. 추가 일반학습 + stereo attention
    """
    # 2차 stereo 학습
    output_path1 = ROOT + f"dataset/processed_data/ex08_{experiment_name}/pre_isomer_train_data.pt"
    output_path2 = ROOT + f"dataset/processed_data/ex08_{experiment_name}/pretrain_data.pt"
    Path(ROOT + f"dataset/processed_data/ex08_{experiment_name}").mkdir(parents=True, exist_ok=True)

    origin_raw_data_ = ROOT + "dataset/raw/admet/CMRT_ADH.csv"
    endpoints_ = {"CMRT_order": deepcopy(ALL_ENDPOINTS["CMRT_order"])}



    if not(os.path.exists(output_path1)):
        make_fine_tuning_data(origin_raw_data_, output_path1, endpoints_, smiles_col=smiles_col, use_stereo=True)

    data_path1 = output_path1
    data_path2 = output_path2

    train_graphs1, val_graphs1, test_graphs1 = split_dataset(
        path=data_path1,
        split_ratio=[0.7, 0.2, 0.1],
        seed=split_seed
    )

    train_len = len(train_graphs1)+len(val_graphs1)+len(test_graphs1)

    if not(os.path.exists(output_path2)):
        make_pretrain_data(origin_pretrain_data, output_path2, use_stereo=True, max_len=train_len)

    train_graphs2, val_graphs2, test_graphs2 = split_dataset(
        path=data_path2,
        split_ratio=[0.7, 0.2, 0.1],
        seed=split_seed
    )


    # 모델 생성
    save_dir1 = ROOT + f"experiment/ex08/output/{experiment_name}/checkpoint0"
    save_dir2 = ROOT + f"experiment/ex08/output/{experiment_name}/checkpoint1"
    Path(save_dir1).mkdir(parents=True, exist_ok=True)
    Path(save_dir2).mkdir(parents=True, exist_ok=True)

    loss_image_path1 = ROOT + f"experiment/ex08/output/{experiment_name}/loss_curve0.png"
    loss_image_path2 = ROOT + f"experiment/ex08/output/{experiment_name}/loss_curve1.png"

    model1, last_model_path1 = model_finetuning(
        cfg=pre_cfg,
        train_graphs=train_graphs1,
        val_graphs=val_graphs1,
        loss_image_path=loss_image_path1,
        save_dir=save_dir1,
        pretrain_path=pretrain_path,
        epochs=epochs,
        endpoints=endpoints,
        plt_close=True,
        seed=train_seed,
        save_freq=save_freq
    )
    pretrain_last_model_path = model_pretraining(
        cfg=pre_cfg,
        train_graphs=train_graphs2,
        val_graphs=val_graphs2,
        loss_image_path=loss_image_path2,
        save_dir=save_dir2,
        checkpoint_path=pretrain_path,
        epochs=pre_epochs,
        plt_close=True,
        seed=train_seed
    )
    # best_model선정
    checkpoint1 = select_best_model(last_model_path1, save_dir1)

    # 모델 평가
    output_dir = ROOT + f"experiment/ex08/output/{experiment_name}/" 

    evaluate_model(
        eval_model=checkpoint1,
        test_graphs=test_graphs1,
        output_dir=output_dir,
        output_file_number=1
    )

    pretrain_paths = {
        "nomal_pretrain": pretrain_last_model_path,
        "isomer_pretrain": last_model_path1,
        "base_pretrain": pretrain_path
    }
    
    for i, cfg in enumerate(cfgs):
        output_path_ = ROOT + f"dataset/processed_data/ex08_{experiment_name}/finetuning_data{i+2}.pt"
        if not(os.path.exists(output_path_)):
            make_fine_tuning_data(origin_raw_data, output_path_, endpoints, smiles_col=smiles_col, use_stereo=True)

        data_path_ = output_path_
        train_graphs_, val_graphs_, test_graphs_ = split_dataset(
            path=data_path_,
            split_ratio=[0.7, 0.2, 0.1],
            seed=split_seed
        )
        save_dir_ = ROOT + f"experiment/ex08/output/{experiment_name}/checkpoint{i+2}"
        Path(save_dir_).mkdir(parents=True, exist_ok=True)

        loss_image_path_ = ROOT + f"experiment/ex08/output/{experiment_name}/loss_curve{i+2}.png"

        model_, last_model_path_ = model_finetuning(
                cfg=cfg,
                train_graphs=train_graphs_,
                val_graphs=val_graphs_,
                loss_image_path=loss_image_path_,
                save_dir=save_dir_,
                pretrain_path=pretrain_paths[cfg['pretrain_key']],
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
            output_file_number=i+2
        )


def main(train_seed, experiment_name):
    origin_raw_data = ROOT + "dataset/raw/admet/PPBR_AZ.csv"
    origin_pretrain_data = ROOT + "dataset/raw/smiles/Compound_000000001_000500000_parts/part_003.sdf"
    endpoint_names = ["PPBR_AZ"]
    endpoints = endpoints = {
        name: deepcopy(ALL_ENDPOINTS[name])
        for name in endpoint_names
    }

    # 공통설정
    BASE_CONFIG = {
            "atom_dim": 72,        # atom feature 차원
            "bond_dim": 14,        # bond feature 차원
            "hidden_dim": 512,
            "admet_hidden_dim": 256,
            "pooling": global_mean_pool,
            "num_endpoint": len(endpoints.keys()), 
            "depth": 5,
            "drop_rate": 0.3,
            "endpoints": endpoints,
            "encode_grad_off": False,
            "encoder_use_cip": True,
            "use_stereo_attention": True,
            "atom_types": AtomFeaturizer.default().atom_type_nums,
            "bond_types": BondFeaturizer.default().bond_type_nums,
            "attn_dim": 64,
            "attn_droprate": 0.2
        }
    cfg0 = BASE_CONFIG.copy()

    # 추가학습 (isomer)
    cfg1 = BASE_CONFIG.copy()
    cfg1['pretrain_key'] = 'isomer_pretrain'
    cfg1['pretrain_component'] = ['encoder', 'stereo_attention']

    # 추가학습 (일반)
    cfg2= BASE_CONFIG.copy()
    cfg2['pretrain_key'] = 'nomal_pretrain'
    cfg2["pretrain_component"] = ["encoder"]
    
    # 기본 (추가학습 x)
    cfg3= BASE_CONFIG.copy()
    cfg3['pretrain_key'] = 'base_pretrain'
    


    cfgs = [cfg1, cfg2, cfg3] # 아직 적용안됨


    run_experiment(
        pre_cfg=cfg0,
        cfgs=cfgs,
        origin_raw_data=origin_raw_data,
        experiment_name=experiment_name+f'_{train_seed}seed',
        endpoints=endpoints,
        epochs=100,
        per_epoch=20,
        save_freq=20,
        pretrain_path=ROOT + '/experiment/ex03/output/pretrain_for_share_no_cip3/checkpoint1_1/ssl_checkpoint_250epoch.pt',
        # pretrain_path=ROOT + '/experiment/ex03/output/pretrain_share2/checkpoint1_1/ssl_checkpoint_80epoch.pt',
        smiles_col=endpoints[endpoint_names[0]]['smiles_col'],
        train_seed=train_seed,
        split_seed=25,
    )


if __name__ == "__main__":
    """2차 사전학습 유용성 확인 실험"""

    main(40, 'PPBR_AZ_2nd_pretrain_test')