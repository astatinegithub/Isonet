import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool
from torch_geometric.data import Data

from isonet.data.dataset import MolGraph
from isonet.model.dmpnn import DMPNN



class ModelA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = DMPNN(
            cfg["atom_dim"],
            cfg["bond_dim"],
            cfg["dmpnn_hidden_dim"],
            cfg["depth"]
        )

        self.pool = global_add_pool # 임시pooling 방식
        self.heads = nn.ModuleDict({
            admet : nn.Sequential(
                            nn.Linear(cfg['dmpnn_hidden_dim'], cfg['admet_hidden_dim']),
                            nn.ReLU(),
                            nn.Dropout(cfg['drop_rate']),
                            nn.Linear(cfg['admet_hidden_dim'], 1)
                        ) for admet in cfg['endpoints']
        })


    def forward(self, data: MolGraph):
        h, _ = self.encoder(data)
        h = self.pool(h, data.batch)
        return {name: head(h) for name, head in self.heads.items()}



class ModelB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...
        
    def forward(self, x):
        ...



class ModelC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...
        
    def forward(self, x):
        ...



def model_select_func(model_type):
    ...