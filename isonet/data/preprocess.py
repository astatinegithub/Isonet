from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

from isonet.config import ROOT, ENDPOINTS
from isonet.data.dataset import MolGraph, mol2feature
from isonet.data.dataset_validation import *


allowed_atoms = set(list(range(1, 37)) + [53])


class SDFReader:
    def __init__(self, input_path):
        self.mols = Chem.SDMolSupplier(input_path, removeHs=False)


    def __len__(self):
        return len(self.mols)


    def __iter__(self):
        for mol in self.mols:
            yield mol, None



class ADMETReader:
    def __init__(self, df: pd.DataFrame, target_cols, smiles_col="Drug"):
        self.df = df
        self.target_cols = target_cols
        self.smiles_col = smiles_col

    def __len__(self):
        return len(self.df)


    def __iter__(self):
        cols = [self.smiles_col, *self.target_cols] # 모든 열의 이름을 담아놓은 리스트
        for row in self.df[cols].itertuples(index=False, name=None): # 일반튜플로 df의 한 행씩 가져옴
            smile = row[0]
            values = row[1:]
            mol = Chem.MolFromSmiles(smile) if isinstance(smile, str) else None # smile이 문자가 아니면 None 반환

            y = torch.tensor([[float(v) if pd.notna(v) else torch.nan for v in values]],
                             dtype=torch.float32)
            yield mol, y


def mol2graph(mol, y=None): # 수정중
    # RDKit 변환 실패
    if mol is None:
        return None

    graph = MolGraph(*mol2feature(mol))
    graph.smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    if y is not None:
        graph.y = y

    return graph


def makeMolGraph(reader, output_path, validator: MolValidator, max_len=None) -> list:
    graphs = []
    removed = 0

    with RDKitLogCapture() as log:
        for mol, target in tqdm(reader, desc="processing", mininterval=0.3):
            if (max_len is not None) and (len(graphs) >= max_len):
                break 
            logs = log.get()

            if not validator.validate(mol, logs):
                removed += 1
                continue


            if target is None:
                graph = mol2graph(mol)
            else: 
                y = target
                graph = mol2graph(mol, y)


            if graph == None:
                removed += 1
            else:
                graphs.append(graph)

    torch.save(graphs, output_path)

    print("====================")
    print(f"saved : {len(graphs)}")
    print(f"removed : {removed}")
    validator.report()
    print(f"path : {output_path}")
    return graphs



# 여기에 admet읽는 거 추가할떄 y -> [admet1, admet2, admet3, ...]식으로 저장하되 없으면 nan체워서 넣어야함


if __name__ == "__main__":
    # from time import time
    # input_path = ROOT + "dataset/raw/" + "Compound_000000001_000500000.sdf"
    # output_path = ROOT + "dataset/processed_data/" + f"for_test_{int(time())}.pt"

    # reader = SDFReader(input_path)
    # validator = MolValidator(allowed_atoms)
    
    # makeMolGraph(reader, output_path, validator, max_len=30000)


    input_path = Path(ROOT) / "dataset" / "raw" / "admet" / "Lipophilicity.csv"
    output_path = Path(ROOT) / "dataset" / "processed_data" / "Lipophilicity_processed.pt"


    df = pd.read_csv(input_path)

    target_columns = [
        info["target_column"]
        for info in ENDPOINTS.values()
    ]

    reader = ADMETReader(
        df=df,
        smiles_col="smiles",
        target_cols=target_columns,
    )
    validator = MolValidator(allowed_atoms)
    makeMolGraph(reader, output_path, validator, max_len=30000)
