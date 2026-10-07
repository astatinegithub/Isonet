from rdkit import Chem
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from isonet.utils.path import str2path
from isonet.config import ROOT, ENDPOINTS, MODEL_A_CONFIG
from isonet.model.dmpnn import IsonetModel
from isonet.model.fine_tuning_models import ModelA


def make_dataloader(path, split_ratio=0.8, batch_size=64, seed=25):
    graphs = torch.load(path, weights_only=False)
    train_size = int(len(graphs) * split_ratio)
    val_size = len(graphs) - train_size

    # 이거 맞는지 확인 필요
    train_graphs, val_graphs = random_split(
        graphs,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
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


def train_val_loss_graph(loss_history: dict[list], save_path=None, dpi=300):
    epochs = range(1, len(loss_history["train_loss"]) + 1)
    plt.plot(epochs, loss_history['train_loss'], label="Train")
    plt.plot(epochs, loss_history['val_loss'], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


@torch.no_grad()
def evaluate(model, val_loader, device, criterion_r, criterion_b, endpoints):
    val_total_loss = 0
    model.eval()
    for batch in tqdm(val_loader):
        batch = batch.to(device)
        pred = model(batch)
        loss = compute_loss(pred, batch, criterion_r, criterion_b, endpoints)
        val_total_loss += loss.item()
    return val_total_loss


def model_finetuning(cfg, graph_path, checkpoint_path, loss_image_path, save_dir,
                     epochs=10, save_freq=1, lr=1e-3, seed=25, endpoints=ENDPOINTS):
    torch.manual_seed(seed)
    if epochs < 1: raise ValueError("epochs는 1 이상이어야 합니다.")
    if save_freq < 1: raise ValueError("save_freq는 1 이상이어야 합니다.")


    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    train_loader, val_loader = make_dataloader(graph_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_r = nn.MSELoss()
    criterion_b = F.binary_cross_entropy_with_logits

    assert list(model.heads.keys()) == list(endpoints),("모델 head와 endpoint 설정이 일치하지 않습니다.")
    
    loss_history = {"train_loss":[], "val_loss":[]} # loss 추적용
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.encoder.load_state_dict(checkpoint["encoder"])

    
    # 불러오는거 아직 안만듦
    for epoch in range(epochs):
        train_total_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_r, criterion_b, endpoints)
        val_total_loss = evaluate(model, val_loader, device, criterion_r, criterion_b, endpoints)

        train_loss = train_total_loss / len(train_loader)
        val_loss = val_total_loss / len(val_loader)

        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1} | train loss {train_loss} | valid_loss {val_loss}")

        if (epoch+1)%save_freq==0:
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "heads": model.heads.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": loss_history
                },
                save_dir / f"finetune_epoch_{epoch + 1}.pt"
            )

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "heads": model.heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_history
        },
        save_dir / f"finish_train_epoch_{epoch+1}.pt"
    )
    train_val_loss_graph(loss_history, loss_image_path)


if __name__ == "__main__":
    model_finetuning(
        cfg=MODEL_A_CONFIG,
        graph_path=ROOT + "dataset/processed_data/Lipophilicity_processed.pt",
        checkpoint_path=ROOT + "model/checkpoint/model_A/ssl_checkpoint_29epoch.pt",
        loss_image_path=ROOT + "model/expir0/loss_curve.png",
        save_dir=ROOT+'model/expir0',
        epochs=20,
        save_freq=1
    )
