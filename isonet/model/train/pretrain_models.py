from rdkit import Chem
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
epochs = 30
lr = 1e-3


# dataset
graphs = torch.load(graph_path, weights_only=False)

dataset = SSLDataset(
    graphs,
    atom_mask_ratio=0.15,
    bond_mask_ratio=0.15
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
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


model.train()

for epoch in range(epochs):
    total_loss = 0
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

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | loss {total_loss/ len(train_loader)}", )
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