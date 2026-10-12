from isonet.config import ROOT
from isonet.data.dataset import split_dataset 
from isonet.data.preprocess import make_fine_tuning_data
from isonet.model.train.finetuning_learn import model_finetuning
from isonet.eval.downstream_evaluation import evaluate_model


# 원하는 전처리 데이터 있는지 확인후 없으면 생성
origin_raw_data = ROOT + ""

make_fine_tuning_data()
make_fine_tuning_data()


data_path1 = ROOT + ""
data_path2 = ROOT + ""

train_graphs1, val_graphs1, test_graphs1 = split_dataset(
    path=data_path1,
    split_ratio=[0.7, 0.2, 0.1]
)
train_graphs2, val_graphs2, test_graphs2 = split_dataset(
    path=data_path2,
    split_ratio=[0.7, 0.2, 0.1]
)


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
model1 = model_finetuning(
    cfg=BASE_CONFIG,
    train_graphs=train_graphs1,
    val_graphs=val_graphs1,
    loss_image_path=ROOT,
    save_dir=ROOT,
    epochs=20,
    save_freq=4,
    endpoints=endpoints
)
model2 = model_finetuning(
    cfg=BASE_CONFIG,
    train_graphs=train_graphs2,
    val_graphs=val_graphs2,
    loss_image_path=ROOT,
    save_dir=ROOT,
    epochs=20,
    save_freq=4,
    endpoints=endpoints
)



# 모델 2가지 평가
evaluate_model()
evaluate_model()