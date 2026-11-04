import os
import torch
from pathlib import Path
from copy import deepcopy
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT, ALL_ENDPOINTS, DATAPATHS
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model



def run_experiment(cfgs:list, origin_raw_data, experiment_name, dir_name,
                   endpoints, smiles_col, epochs, save_freq=1,
                   pretrain_path=None, origin_test_path=None,
                   train_seed=42, split_seed=42):
    Path(ROOT + f"dataset/processed_data/{experiment_name}").mkdir(parents=True, exist_ok=True)
    output_dir = ROOT + f"finalv1/{dir_name}/output/{experiment_name}/"
    test_path = ROOT + f"dataset/processed_data/{experiment_name}/finetuning_test_data.pt"

    if not (origin_test_path is None):
        test_path = ROOT + f"dataset/processed_data/{experiment_name}/finetuning_test_data.pt"
        if (not os.path.exists(test_path)):
            make_fine_tuning_data(origin_test_path, test_path, endpoints, smiles_col=smiles_col, use_stereo=True) 
    else:
        test_path = None

    for i, cfg in enumerate(cfgs):
        output_path_ = ROOT + f"dataset/processed_data/{experiment_name}/finetuning_data{i}.pt"
        if not(os.path.exists(output_path_)):
            make_fine_tuning_data(origin_raw_data, output_path_, endpoints, smiles_col=smiles_col, use_stereo=True)
        data_path_ = output_path_
        train_graphs_, val_graphs_, test_graphs_ = split_dataset(
            path=data_path_,
            split_ratio=(
                [0.7, 0.2, 0.1]
                if test_path is None
                else [0.8, 0.2]
            ),
            seed=split_seed,
            test_path=test_path
        )
        save_dir_ = ROOT + f"finalv1/{dir_name}/output/{experiment_name}/checkpoint{i}"
        Path(save_dir_).mkdir(parents=True, exist_ok=True)

        loss_image_path_ = ROOT + f"finalv1/{dir_name}/output/{experiment_name}/loss_curve{i}.png"

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


