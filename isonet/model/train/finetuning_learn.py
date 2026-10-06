from rdkit import Chem
import time
from tqdm import tqdm
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

torch.manual_seed(25)



graph_path = ROOT + "data"
checkpoint_path = ROOT + "model/checkpoint/ssl_checkpoint.pt"

batch_size = 64
epochs = 10
lr = 1e-3


graphs = torch.load(graph_path, weights_only=False)
train_size = int(len(graphs) * 0.8)
val_size = len(graphs) - train_size


# 이거 맞는지 확인 필요
train_graphs, val_graphs = random_split(
    graphs,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(25)
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelA(MODEL_A_CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction="none")

loss_history = {"train_loss":[], "val_loss":[]} # loss 추적용
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)
model.encoder.load_state_dict(checkpoint["encoder"])

# 불러오는거 아직 안만듦

for epoch in range(epochs):
    train_total_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        pred = model(batch)

        loss = []
        cnt_loss = 0
        for i, endpoint in enumerate(ENDPOINTS):
            target = batch.y[:, i]              # batch의 i번째 타겟들 가져오기
            pred_i = pred[endpoint].squeeze(-1) # 예측된 값 shape(batch, )로 만들어서 가져오기
            mask = ~torch.isnan(target)         # 마스크 제작 -> nan은 없던거

            if mask.any():
                continue

            if pred[endpoint]['type'] == "regression":
                loss.append(criterion(pred_i[mask], target[mask]))
            else:
                loss.append(F.binary_cross_entropy_with_logits(pred_i[mask], target[mask]))
            cnt_loss += 1

        loss = loss.sum() / cnt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        loss_history["train_loss"].append(train_total_loss)

    # valid    
    val_total_loss = 0
    model.eval()
    for batch in tqdm(val_loader):
        batch = batch.to(device)
        pred = model(batch)

        loss = []
        cnt_loss = 0
        for i, endpoint in enumerate(ENDPOINTS):
            target = batch.y[:, i]              # batch의 i번째 타겟들 가져오기
            pred_i = pred[endpoint].squeeze(-1) # 예측된 값 shape(batch, )로 만들어서 가져오기
            mask = ~torch.isnan(target)         # 마스크 제작 -> nan은 없던거

            if mask.any():
                continue

            if pred[endpoint]['type'] == "regression":
                loss.append(criterion(pred_i[mask], target[mask]))
            else:
                loss.append(F.binary_cross_entropy_with_logits(pred_i[mask], target[mask]))
            cnt_loss += 1

        loss = loss.sum() / cnt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_total_loss += loss.item()
        loss_history["val_loss"].append(val_total_loss)
    print(f"Epoch {epoch+1} | train loss {train_total_loss/ len(train_loader)} | valid_loss {val_total_loss/ len(val_loader)}")

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "heads": model.heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_history
        },
        ROOT + f"model/checkpoint/ssl_checkpoint_{epoch}epoch.pt"
    )

plt.plot(loss_history['train_loss'], label="Train")
plt.plot(loss_history['val_loss'], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig(ROOT + "model/checkpoint/loss_curve.png",dpi=300)

plt.show()