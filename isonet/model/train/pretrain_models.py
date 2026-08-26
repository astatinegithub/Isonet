from rdkit import Chem
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


from isonet.utils.path import str2path
from isonet.config import ROOT
from isonet.model.dmpnn import *
from isonet.data.dataset import SSLDataset 
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer 

torch.manual_seed(25)

graph_path = ROOT + "dataset/processed_data/for_test.pt"


batch_size = 64
epochs = 2
lr = 1e-3


# dataset
graphs = torch.load(graph_path, weights_only=False)

train_size = int(len(graphs) * 0.8)
val_size = len(graphs) - train_size

train_graphs, val_graphs = random_split(
    graphs,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(25)
)


train_dataset = SSLDataset(
    train_graphs,
    atom_mask_ratio=0.15,
    bond_mask_ratio=0.15
)

val_dataset = SSLDataset(
    val_graphs,
    atom_mask_ratio=0.15,
    bond_mask_ratio=0.15
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


print('maked a dataloader')


# 모델 설정필요
atom_featurizer = AtomFeaturizer.model_A()
bond_featurizer = BondFeaturizer.model_A()

num_atom_types = len(atom_featurizer.atomic_nums) + 1
num_bond_types = len(bond_featurizer.bond_types) + 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSLModel(
    atom_dim=72,      # atom feature 차원
    bond_dim=14,        # bond feature 차원
    hidden_dim=512,
    atom_types=num_atom_types,
    bond_types=num_bond_types,
    depth=5
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
criterion = nn.CrossEntropyLoss()


train_loss_history = []
val_loss_history = []



batch = next(iter(train_loader))
batch = batch.to(device)

print("model:", next(model.parameters()).device)
print("x:", batch.x.device)
print("edge_attr:", batch.edge_attr.device)



for epoch in range(epochs):
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        atom_pred, bond_pred = model(batch)

        # 실제 SSL target이 있는 위치
        atom_valid = batch.atom_target != -100
        bond_valid = batch.bond_target != -100

        loss = torch.tensor(0.0, device=device)

        if atom_valid.any():
            atom_loss = criterion(atom_pred[atom_valid], batch.atom_target[atom_valid])
            loss += atom_loss

        if bond_valid.any():
            bond_loss = criterion(bond_pred[bond_valid], batch.bond_target[bond_valid])
            loss += bond_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            atom_pred, bond_pred = model(batch)

            atom_valid = batch.atom_target != -100
            bond_valid = batch.bond_target != -100

            loss = 0

            if atom_valid.any():
                loss += criterion(atom_pred[atom_valid], batch.atom_target[atom_valid])

            if bond_valid.any():
                loss += criterion(bond_pred[bond_valid], batch.bond_target[bond_valid])

            val_loss += loss.item()

    print("val loss:", val_loss / len(val_loader))

    train_loss_history.append(train_loss / len(train_loader))
    val_loss_history.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1} | train loss {train_loss/ len(train_loader)} | valid loss {val_loss/ len(val_loader)}", )

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "atom_head": model.atom_head.state_dict(),
            "bond_head": model.bond_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        ROOT + f"model/checkpoint/ssl_checkpoint_{epoch}epoch.pt"
    )

plt.plot(train_loss_history, label="Train")
plt.plot(val_loss_history, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig(ROOT + "model/checkpoint/loss_curve.png",dpi=300)

plt.show()