import urllib.request
import gzip
import shutil 
from pathlib import Path

from isonet.config import ROOT


def load_data(filename):
    folder_zip = "dataset/test_dataset_zip/"
    folder = "dataset/test_dataset/"
    filepath_zip = Path(ROOT) + folder_zip + filename 
    filepath = Path(ROOT) + folder + filename[:-3]  # .gz제거한 경로
    filepath.mkdir(parents=True, exist_ok=True)
    filepath_zip.mkdir(parents=True, exist_ok=True)

    url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/" + filename


    urllib.request.urlretrieve(url, filepath_zip)


    with gzip.open(filepath_zip, 'rb') as f_in:
        with open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)