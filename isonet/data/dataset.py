import numpy as np
from tqdm import tqdm
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem.rdchem import Bond

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset

from isonet.config import ROOT
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer


ATOM_FEATURIZER = AtomFeaturizer.model_A()
BOND_FEATURIZER = BondFeaturizer.model_A()


def build_reverse_edge_index(edge_index: list) -> list:
    edge_dict = {}
    for i, (s, d) in enumerate(edge_index):
        edge_dict[(s, d)] = i

    rev_edge = []
    for s, d in edge_index:
        rev_edge.append(edge_dict[(d, s)])

    return rev_edge


def mol2feature(mol: Chem.Mol,
                atomfeaturizer: AtomFeaturizer =ATOM_FEATURIZER,
                bondfeaturizer: BondFeaturizer =BOND_FEATURIZER) -> Data:
    edge_attr   = []
    edge_index  = []

    node_feature = [atomfeaturizer(atom) for atom in mol.GetAtoms()]   

    for bond in mol.GetBonds():
        bond: Bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = bondfeaturizer(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)

    rev_edge = build_reverse_edge_index(edge_index)

    atom_type = torch.tensor([atomfeaturizer.atom_type(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    bond_type = torch.tensor([bondfeaturizer.bond_type(bond) for bond in mol.GetBonds()], dtype=torch.long)


    x = torch.stack(node_feature).float()
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr).float()
    rev_edge = torch.tensor(rev_edge, dtype=torch.long)
    

    return x, edge_index, edge_attr, rev_edge, atom_type, bond_type



class MolGraph(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                rev_edge=None, atom_type=None, bond_type=None,
                 y=None, y_mask=None):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            rev_edge=rev_edge,
            atom_type=atom_type,
            bond_tpye=bond_type,
            y=y,
            y_mask=y_mask
        )

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.rev_edge = rev_edge
        self.atom_type = atom_type
        self.bond_type = bond_type

        # for pretrain variables (task아니어도 일단 가지고 있는게 좋음)
        # self.atom_mask_idx = torch.empty(0, dtype=torch.long)
        # self.bond_mask_idx = torch.empty(0, dtype=torch.long)

        self.atom_target = torch.empty(0, dtype=torch.long)
        self.bond_target = torch.empty(0, dtype=torch.long)


    def __inc__(self, key, value, *args, **kwargs): # rev_edge는 커스텀이라 batch계산을 위해 필요함
        if key == "rev_edge":
            return self.edge_attr.size(0) # bond 수
        if key == "mask_idx":
            return self.x.size(0) # atom 수
        if key == "atom_target":  # 변하면 안되기에 
            return 0
        return super().__inc__(key, value, *args, **kwargs)




class MoleculeDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



class SSLDataset(Dataset): # working on here!                   objective: bond에 대한 ssl추가하기
    def __init__(self, data_list, atom_mask_ratio=0.15,
                 bond_mask_ratio=0.15, atom_select_ratio=0.5):
        super().__init__()

        self.data_list = data_list
        self.atom_mask_ratio = atom_mask_ratio
        self.bond_mask_ratio = bond_mask_ratio

        self.atom_select_ratio = atom_select_ratio


    def len(self):
        return len(self.data_list)


    def get(self, idx):
        data: MolGraph = deepcopy(self.data_list[idx])
        num_atoms = data.x.size(0)
        num_bonds = data.edge_attr.size(0) // 2

        data.atom_target = torch.full((num_atoms,), -100, dtype=torch.long)
        data.bond_target = torch.full((num_bonds,), -100, dtype=torch.long)


        if self.atom_select_ratio > torch.rand(1).item():
            num_mask = max(1,int(num_atoms * self.atom_mask_ratio))
            atom_mask_idx = torch.randperm(num_atoms)[:num_mask]

            data.atom_target[atom_mask_idx] = data.atom_type[atom_mask_idx]
            # data.atom_mask_idx = atom_mask_idx

            # masking
            data.x[atom_mask_idx] = 0
        else:
            if num_bonds == 0:
                return data
            num_mask = max(1,int(num_bonds * self.bond_mask_ratio)) # 마스킹할 갯수
            bond_mask_idx = torch.randperm(num_bonds)[:num_mask]

            forward_idx = bond_mask_idx * 2
            reverse_idx = bond_mask_idx * 2 + 1

            data.bond_target[bond_mask_idx] = data.bond_type[bond_mask_idx]
            # data.bond_mask_idx = bond_mask_idx

            data.edge_attr[forward_idx] = 0
            data.edge_attr[reverse_idx] = 0

            
        return data



# def create_ssl_dataloader(path, batch_size=64,
#                           mask_ratio=0.15, shuffle=True):

#     graphs = torch.load(path)
#     dataset = SSLDataset(graphs, mask_ratio)

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle
#     )
#     return loader



if __name__ == "__main__": # test zone
    import time
    smiles = Chem.SDMolSupplier(ROOT+"dataset/raw/Compound_000000001_000500000.sdf")

    # for i, mol in enumerate(smiles):
    #     if mol is None:
    #         continue

    #     if mol.GetNumAtoms() == 1:
    #         print(i, Chem.MolToSmiles(mol))

    # print(len(smiles))
    # T = time.time()
    # train_dataloader = create_dataloader(
    #     dataset=smiles,
    #     batch_size=16
    # )

    mol = next(smiles)
    data = MolGraph(*mol2feature(mol))
    print(data.x.shape)
    print(data.edge_index.shape)
    print(data.edge_attr.shape)
    print(data.rev_edge)
    # print(time.time()-T)