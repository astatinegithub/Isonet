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


def make_dataloader(path, split_ratio=0.8, batch_size=64, seed=25):
    graphs = torch.load(path, weights_only=False)
    train_size = int(len(graphs) * split_ratio)
    val_size = len(graphs) - train_size

    train_graphs, val_graphs = random_split(
        graphs,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
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

    return train_loader, val_loader


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


def train_one_epoch(model, train_loader, optimizer, device, criterion):
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        batch: MolGraph = batch.to(device)
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
    return train_loss


@torch.no_grad()
def evaluate(model, val_loader, device, criterion):
    val_loss = 0
    model.eval()
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
    return val_loss



def model_pretraining(cfg, graph_path, loss_image_path, 
                      epochs=20, lr=1e-3):
    # dataset
    train_loader, val_loader = make_dataloader(graph_path)
    print('maked a dataloader')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = {'train':[], 'val':[]}

    # train and valid
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion) / len(train_loader) # batch수로 나누기
        val_loss = evaluate(model, val_loader, device, criterion) / len(val_loader) # batch수로 나누기

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        print(f"Epoch {epoch+1} | train loss {train_loss} | valid loss {val_loss}", )

        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "atom_head": model.atom_head.state_dict(),
                "bond_head": model.bond_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss_history": loss_history,
                'config':cfg 
            },
            ROOT + f"model/checkpoint/ssl_checkpoint_{epoch}epoch.pt"
        )

    train_val_loss_graph(loss_history, loss_image_path)


if __name__ == "__main__":
    torch.manual_seed(25)
    ISONET_CONFIG = {
        "atom_dim": 72,        # atom feature 차원
        "bond_dim": 14,        # bond feature 차원
        "hidden_dim": 512,
        "atom_types": AtomFeaturizer.model_A().atom_type_nums,
        "bond_types:": BondFeaturizer.model_A().bond_type_nums,
        "depth": 5
    }
    model_pretraining(
        cfg=ISONET_CONFIG,
        graph_path=ROOT + "dataset/processed_data/for_test.pt",
        loss_image_path=ROOT + "model/checkpoint/loss_curve.png",
        epochs=20
    )