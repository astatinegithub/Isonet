import pandas as pd
from rdkit import Chem
from collections import defaultdict

from isonet.config import ROOT


def isomer_test(input_path, data_type="pre", smiles_col=None):
    if data_type == "fine":
        df = pd.read_csv(input_path)
        molecules = (
            Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            for smiles in df[smiles_col]
        )
    else:
        molecules = Chem.SDMolSupplier(input_path, removeHs=False)

    # 같은 연결 구조에 속한 입체이성질체들을 모음
    groups = defaultdict(set)
    stereo_count = 0

    for mol in molecules:
        if mol is None:
            continue

        # SDF의 명시적 수소와 SMILES의 수소 표현을 맞춤
        mol = Chem.RemoveHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        stereo_info = list(Chem.FindPotentialStereo(mol))

        # 입체배치가 하나라도 지정되어 있는 분자
        has_specified = any(
            info.specified == Chem.StereoSpecified.Specified
            for info in stereo_info
        )

        if has_specified:
            stereo_count += 1

        # 입체배치가 일부라도 미지정이면 쌍 집계에서 제외
        has_unspecified = any(
            info.specified != Chem.StereoSpecified.Specified
            for info in stereo_info
        )

        if not has_specified or has_unspecified:
            continue

        # 입체정보를 포함한 SMILES
        stereo_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )

        # 입체정보만 제거한 연결 구조
        base_mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(base_mol)
        base_smiles = Chem.MolToSmiles(
            base_mol, canonical=True, isomericSmiles=True
        )

        groups[base_smiles].add(stereo_smiles)

    pair_groups = [variants for variants in groups.values() if len(variants) >= 2]

    # 한 그룹에 이성질체가 n개면 가능한 쌍은 n*(n-1)/2개
    pair_count = sum(
        len(variants) * (len(variants) - 1) // 2
        for variants in pair_groups
    )

    print(f"입체배치가 지정된 분자: {stereo_count}개")
    print(f"이성질체가 2개 이상 있는 그룹: {len(pair_groups)}개")
    print(f"이성질체 쌍: {pair_count}개")


if __name__ == "__main__":
    # isomer_test(
    #     input_path=ROOT + "dataset/raw/smiles/Compound_001000001_001500000.sdf",
    #     data_type="pre",
    # )
    isomer_test(
        input_path=ROOT + "dataset/raw/admet/ExpansionRx_train.csv",
        data_type="fine",
        smiles_col="SMILES",
    )