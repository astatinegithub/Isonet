import torch
from pathlib import Path


from isonet.config import ROOT
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.model.utils.select_best_model import select_best_model
from isonet.eval.downstream_evaluation import evaluate_model


# 원하는 전처리 데이터 있는지 확인후 없으면 생성
origin_raw_data = ROOT + "dataset/raw/admet/Lipophilicity.csv"
output_path1 = ROOT + "dataset/processed_data/ex01_finetuning_data1.pt"
output_path2 = ROOT + "dataset/processed_data/ex01_finetuning_data2.pt"

make_fine_tuning_data(origin_raw_data, output_path1, use_stereo=True)
make_fine_tuning_data(origin_raw_data, output_path2, use_stereo=False)


data_path1 = output_path1
data_path2 = output_path2

train_graphs1, val_graphs1, test_graphs1 = split_dataset(
    path=data_path1,
    split_ratio=[0.7, 0.2, 0.1]
)
train_graphs2, val_graphs2, test_graphs2 = split_dataset(
    path=data_path2,
    split_ratio=[0.7, 0.2, 0.1]
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


# endpoint 설정
endpoints = {
    "Lipophilicity": {
        "type": "regression",
        "target_column": "exp",
        "selection_metric": "rmse",
    },
}


# config 설정
BASE_CONFIG = {
    "atom_dim": 72,        # atom feature 차원
    "bond_dim": 14,        # bond feature 차원
    "dmpnn_hidden_dim": 512,
    "admet_hidden_dim": 256,
    "num_endpoint": len(endpoints.keys()), 
    "depth": 5,
    "drop_rate":0.3,
    "endpoints": endpoints
}


# 모델 2가지 생성
save_dir1 = ROOT + "experiment/ex01/output/checkpoint1"
save_dir2 = ROOT + "experiment/ex01/output/checkpoint2"
Path(save_dir1).mkdir(parents=True, exist_ok=True)
Path(save_dir2).mkdir(parents=True, exist_ok=True)
loss_image_path1 = ROOT + "experiment/ex01/output/loss_curve1.png"
loss_image_path2 = ROOT + "experiment/ex01/output/loss_curve2.png"

model1, last_model_path1 = model_finetuning(
    cfg=BASE_CONFIG,
    train_graphs=train_graphs1,
    val_graphs=val_graphs1,
    loss_image_path=loss_image_path1,
    save_dir=save_dir1,
    pretrain_path=None,
    epochs=20,
    endpoints=endpoints
)
model2, last_model_path2 = model_finetuning(
    cfg=BASE_CONFIG,
    train_graphs=train_graphs2,
    val_graphs=val_graphs2,
    loss_image_path=loss_image_path2,
    save_dir=save_dir2,
    pretrain_path=None,
    epochs=20,
    endpoints=endpoints
)

# best_model선정
checkpoint1 = select_best_model(last_model_path1)
checkpoint2 = select_best_model(last_model_path2)


# 모델 2가지 평가
output_dir = ROOT + "experiment/ex01/output" 

evaluate_model(
    eval_model=checkpoint1['model'],
    test_graphs=test_graphs1,
    output_dir=output_dir,
    output_file_number=1
)
evaluate_model(
    eval_model=checkpoint2['model'],
    test_graphs=test_graphs2,
    output_dir=output_dir,
    output_file_number=2
)