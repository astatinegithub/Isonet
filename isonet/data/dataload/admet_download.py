from pathlib import Path
from urllib.request import urlretrieve

from isonet.config import ROOT



# Biogen은 특정 커밋으로 고정
BIOGEN_COMMIT = "b00df003de117ce9e5b381afd886095c5f2af2d5"

DATASETS = {
    "biogen": {
        "filename": "ADME_public_set_3521.csv",
        "url": (
            "https://raw.githubusercontent.com/"
            "molecularinformatics/Computational-ADME/"
            f"{BIOGEN_COMMIT}/ADME_public_set_3521.csv"
        ),
    },
    "bbbp": {
        "filename": "BBBP.csv",
        "url": (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/"
            "datasets/BBBP.csv"
        ),
    },
    "tox21": {
        "filename": "tox21.csv.gz",
        "url": (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/"
            "datasets/tox21.csv.gz"
        ),
    },
    "clintox": {
        "filename": "clintox.csv.gz",
        "url": (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/"
            "datasets/clintox.csv.gz"
        ),
    },
    "lipophilicity": {
    "filename": "Lipophilicity.csv",
    "url": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/"
        "datasets/Lipophilicity.csv"
        ),
    },
}
# 중복되는건 이렇게 추가
for key, filename in {
    "ppbr_az": "PPBR_AZ.csv",
    "clearance_microsome_az": "Clearance_Microsome_AZ.csv",
    "cyp3a4_veith": "CYP3A4_Veith.csv",
}.items():
    DATASETS[key] = {
        "filename": filename,
        "url": (
            f"https://huggingface.co/datasets/"
            f"scikit-fingerprints/TDC_{key}/resolve/main/"
            f"tdc_{key}.csv"
        ),
    }
    
DATASETS.update({
    "expansionrx_train": {
        "filename": "ExpansionRx_train.csv",
        "url": (
            "https://huggingface.co/datasets/"
            "openadmet/openadmet-expansionrx-challenge-data/"
            "resolve/main/expansion_data_train.csv"
        ),
    },
    "expansionrx_test": {
        "filename": "ExpansionRx_test.csv",
        "url": (
            "https://huggingface.co/datasets/"
            "openadmet/openadmet-expansionrx-challenge-data/"
            "resolve/main/expansion_data_test.csv"
        ),
    },
})

def download_dataset(name: str) -> Path:
    info = DATASETS[name]

    save_dir = Path(ROOT) / "dataset/raw/admet"
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / info["filename"]

    if not path.is_file():
        urlretrieve(info["url"], str(path))

    return path


if __name__ == "__main__":
    for name in DATASETS:
        print(name, download_dataset(name))
        origin_raw_data = download_dataset("ppbr_az")