from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
from tqdm import tqdm
import torch

from concurrent.futures import ProcessPoolExecutor

from isonet.config import ROOT
from isonet.data.dataset import MolGraph, mol2feature
from isonet.data.dataset_validation import *


allowed_atoms = set(list(range(1, 37)) + [53])


def process_item(item):
    mol_binary, target = item
    try:
        mol = Chem.Mol(mol_binary)
        if target is None:
            return mol2graph(mol)
        y, y_mask = target
        return mol2graph(mol, y, y_mask)

    except Exception:
        return None


def mol2graph(mol, y=None, y_mask=None): # 수정중
    # RDKit 변환 실패
    if mol is None:
        return None

    
    graph = MolGraph(*mol2feature(mol))
    graph.smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    if y is not None:
        graph.y = y
    if y_mask is not None:
        graph.y_mask = y_mask

    return graph



class SDFReader:
    def __init__(self, input_path):
        self.mols = Chem.SDMolSupplier(input_path, removeHs=False)


    def __len__(self):
        return len(self.mols)


    def __iter__(self):
        for mol in self.mols:
            yield mol, None
    


def makeMolGraph(reader, output_path, validator: MolValidator,
                 max_len=None, num_workers=8, chunksize=100) -> list:
    items = []
    graphs = []
    removed = 0

    with RDKitLogCapture() as log:
        for mol, target in tqdm(reader, desc="validation", mininterval=0.3):
            if max_len is not None and len(items) >= max_len: break
            logs = log.get()

            if not validator.validate(mol, logs):
                continue

            if mol is None:
                continue

            items.append((mol.ToBinary(), target))

    # 2. parallel mol -> graph
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_item, items, chunksize=chunksize)

        for graph in tqdm(results, total=len(items), desc="processing", mininterval=0.3):
            if graph is None:
                removed += 1
            else:
                graphs.append(graph)

    # with RDKitLogCapture() as log:
    #     for mol, target in tqdm(reader, desc="processing", mininterval=0.3):
    #         if (max_len is not None) and (len(graphs) > max_len):
    #             break 
    #         logs = log.get()

    #         if not validator.validate(mol, logs):
    #             continue


    #         if target is None:
    #             graph = mol2graph(mol)
    #         else: 
    #             y, y_mask = target
    #             graph = mol2graph(mol, y, y_mask)


    #         if graph == None:
    #             removed += 1
    #         else:
    #             graphs.append(graph)

    torch.save(graphs, output_path)

    print("====================")
    print(f"saved : {len(graphs)}")
    print(f"removed : {removed}")
    validator.report()
    print(f"path : {output_path}")



if __name__ == "__main__":
    from time import time
    input_path = ROOT + "dataset/raw/" + "Compound_000000001_000500000.sdf"
    output_path = ROOT + "dataset/processed_data/" + f"for_test_{time()}.pt"

    reader = SDFReader(input_path)
    validator = MolValidator(allowed_atoms)
    
    makeMolGraph(reader, output_path, validator)