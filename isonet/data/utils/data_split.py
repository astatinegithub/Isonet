from pathlib import Path
from rdkit import Chem

from isonet.config import ROOT


def sdf_split(sdf_path, chunk_size=200_000):
    sdf_path = Path(sdf_path)
    output_dir = sdf_path.parent / f"{sdf_path.stem}_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)

    count = 0
    skipped = 0
    writer = None

    try:
        with sdf_path.open("rb") as file:
            molecules = Chem.ForwardSDMolSupplier(file, removeHs=False)
            for mol in molecules:
                if mol is None:
                    skipped += 1
                    continue
                # 20만 개마다 새 파일 생성
                if count % chunk_size == 0:
                    if writer is not None:
                        writer.close()

                    part = count // chunk_size + 1
                    output_path = output_dir / f"part_{part:03d}.sdf"
                    writer = Chem.SDWriter(str(output_path))
                    print(f"저장 중: {output_path}")

                writer.write(mol)
                count += 1

    finally:
        if writer is not None:
            writer.close()

    print(f"저장한 분자: {count:,}개")
    print(f"읽기 실패로 제외한 분자: {skipped:,}개")



if __name__ == "__main__":
    sdf_split(ROOT + "dataset/raw/smiles/Compound_000000001_000500000.sdf")