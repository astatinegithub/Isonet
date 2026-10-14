import urllib.request
import gzip
import shutil 
from pathlib import Path

from isonet.config import ROOT


def load_sdf_data(filename):
    folder_zip_dir = Path(ROOT+"dataset/raw/smils_gz/")
    folder_dir = Path(ROOT+"dataset/raw/smiles/")
    filepath_zip = folder_zip_dir / filename
    filepath = folder_dir / filename[:-3]  # .gz제거한 경로
    folder_zip_dir.mkdir(parents=True, exist_ok=True)
    folder_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/" + filename

    print('save to', filepath)

    urllib.request.urlretrieve(url, filepath_zip)


    with gzip.open(filepath_zip, 'rb') as f_in:
        with open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    load_sdf_data('Compound_000500001_001000000.sdf.gz')
