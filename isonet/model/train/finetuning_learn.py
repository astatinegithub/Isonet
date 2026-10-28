from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from isonet.config import ROOT, ENDPOINTS, MODEL_A_CONFIG
from isonet.model.fine_tuning_model import IsonetModel
from isonet.data.dataset import split_dataset
from isonet.model.utils.utils import save, train_val_loss_graph


def make_dataloader(train_graphs, val_graphs, batch_size=64):
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


def compute_loss(pred, batch, criterion_r, criterion_b, endpoints) -> None:
    losses = []
    cnt_loss = 0
    for i, endpoint in enumerate(endpoints):
        target = batch.y[:, i]              # batch의 i번째 타겟들 가져오기
        pred_i = pred[endpoint].squeeze(-1) # 예측된 값 shape(batch, )로 만들어서 가져오기
        mask = ~torch.isnan(target)         # 마스크 제작 -> nan은 없던거

        if not mask.any():
            continue
        if endpoints[endpoint]['type'] == "regression":
            losses.append(criterion_r(pred_i[mask], target[mask]))
        else:
            losses.append(criterion_b(pred_i[mask], target[mask]))
        cnt_loss += 1

    if not losses:
        raise ValueError("이 배치에는 유효한 ADMET 정답이 없습니다.")
    loss = sum(losses) / cnt_loss
    return loss


def train_one_epoch(model, train_loader, optimizer, device, criterion_r, criterion_b, endpoints) -> float:
    train_total_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        pred = model(batch)
        loss = compute_loss(pred, batch, criterion_r, criterion_b, endpoints)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()

    return train_total_loss


@torch.no_grad()
def evaluate(model, val_loader, device, criterion_r, criterion_b, endpoints, target_idx=0) -> dict[int, dict]:
    model.eval()
    val_total_loss = 0.0
    val_total_count = 0

    for batch in tqdm(val_loader):
        batch = batch.to(device)
        pred = model(batch)
        loss = compute_loss(
            pred, batch, criterion_r, criterion_b, endpoints
        )

        # PPBR_AZ 정답이 있는 분자 수
        n = (~torch.isnan(batch.y[:, 0])).sum().item()
        val_total_loss += loss.item() * n
        val_total_count += n

    return val_total_loss / val_total_count


def model_finetuning(cfg, train_graphs, val_graphs, loss_image_path, save_dir, pretrain_path=None,
                     epochs=10, save_freq=1, lr=1e-3, seed=25, endpoints=ENDPOINTS, plt_close=False) -> tuple[IsonetModel, str]:
    torch.manual_seed(seed)
    if epochs < 1: raise ValueError("epochs는 1 이상이어야 합니다.")
    if save_freq < 1: raise ValueError("save_freq는 1 이상이어야 합니다.")


    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    train_loader, val_loader = make_dataloader(train_graphs, val_graphs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsonetModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_r = nn.MSELoss()
    criterion_b = F.binary_cross_entropy_with_logits

    if cfg.get('encode_grad_off', False): # 파라미터 정리필요
        for param in model.encoder.parameters():
            param.requires_grad = False

    assert list(model.heads.keys()) == list(endpoints), ("모델 head와 endpoint 설정이 일치하지 않습니다.")
    loss_history = {"train_loss":[], "val_loss":[]} # loss 추적용
    if pretrain_path is not None:                       # 사전학습 있을 때 
        checkpoint = torch.load(
            pretrain_path,
            map_location=device,
            weights_only=False
        )
        components = cfg.get('pretrain_component', "encoder")
        for component in components: # 사전학습 불러올때 원하는거 불러오기
            try:
                model.encoder.load_state_dict(checkpoint[component])
            except KeyError as e:
                raise KeyError(f"pretrain_component missmatch! checkpoint has {checkpoint.keys()}\nerror message:{e}")
            
    best_loss = float('inf')

    # 학습
    for epoch in range(epochs):
        train_total_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_r, criterion_b, endpoints)
        val_total_loss = evaluate(model, val_loader, device, criterion_r, criterion_b, endpoints)

        train_loss = train_total_loss / len(train_loader)
        val_loss = val_total_loss # 미리 나누는 걸로 바꿈 / len(val_loader)

        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1} | train loss {train_loss} | valid_loss {val_loss}")

        if (epoch+1)%save_freq==0:
            save(save_dir / f"finetune_train_epoch_{epoch + 1}.pt", model=model, optimizer=optimizer, epoch=epoch, loss_history=loss_history, cfg=cfg)
        if val_loss < best_loss:
            best_loss = float(val_loss)
            save(save_dir / f"finetune_val_best.pt", model=model, optimizer=optimizer, epoch=epoch, loss_history=loss_history, cfg=cfg)
        
    save(save_dir / f"finish_train_epoch_{epoch+1}.pt", model=model, optimizer=optimizer, epoch=epoch, loss_history=loss_history, cfg=cfg)
    train_val_loss_graph(loss_history, loss_image_path, plt_close=plt_close)
    return model, save_dir / f"finish_train_epoch_{epoch+1}.pt"


if __name__ == "__main__":
    ...
    # trian_graphs, val_graphs, _ = split_dataset(ROOT + "dataset/processed_data/Lipophilicity_processed.pt", split_ratio=[0.8, 0.2, 0])
    # model_finetuning(
    #     cfg=MODEL_A_CONFIG,
    #     trian_graphs=trian_graphs,
    #     val_graphs=val_graphs,
    #     loss_image_path=ROOT + "model/expir0/loss_curve.png",
    #     save_dir=ROOT+'model/expir0',
    #     pretrain_path=ROOT + "model/checkpoint/model_A/ssl_checkpoint_29epoch.pt",
    #     epochs=20,
    #     save_freq=1
    # )
