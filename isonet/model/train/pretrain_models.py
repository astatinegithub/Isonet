from rdkit import Chem
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path 

import torch
import torch.nn as nn
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from isonet.config import ROOT
from isonet.model.dmpnn import *
from isonet.data.dataset import SSLDataset, split_dataset
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer 
from isonet.model.utils.utils import save, train_val_loss_graph


def make_dataloader(train_graphs, val_graphs, batch_size=64):
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
        batch: MolGraph = batch.to(device)
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


def model_pretraining(cfg, train_graphs, val_graphs, loss_image_path,
                      save_dir, checkpoint_path=None, epochs=20, lr=1e-3,
                      plt_close=False, seed=25):
    # dataset
    torch.manual_seed(seed)
    train_loader, val_loader = make_dataloader(train_graphs, val_graphs)
    print('maked a dataloader')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = {'train_loss':[], 'val_loss':[]}
    previous_epoch = 0

    if isinstance(checkpoint_path, str|Path): # 이어서 학습
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        previous_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_history = checkpoint['loss_history']

    # train and valid
    best_loss = torch.inf
    for epoch in range(previous_epoch, previous_epoch+epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion) / len(train_loader) # batch수로 나누기
        val_loss = evaluate(model, val_loader, device, criterion) / len(val_loader) # batch수로 나누기

        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1} | train loss {train_loss} | valid loss {val_loss}", )
        save(save_dir + f"/ssl_checkpoint_{epoch+1}epoch.pt", False, model=model, optimizer=optimizer,epoch=epoch, loss_history=loss_history, cfg=cfg)
        
        if val_loss < best_loss:
            best_loss = float(val_loss)
            save(save_dir + f"/ssl_checkpoint_val_best.pt", False, model=model, optimizer=optimizer,epoch=epoch, loss_history=loss_history, cfg=cfg)
    train_val_loss_graph(loss_history, loss_image_path, plt_close=plt_close)
    return save_dir + f"/ssl_checkpoint_val_best.pt"
