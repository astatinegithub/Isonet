import os

import torch
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.decomposition import PCA

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT
from isonet.model.dmpnn import DMPNN


# setting
graph_path = ROOT + "dataset/processed_data/for_test2.pt"

checkpoint_path = (
    ROOT
    + "model/checkpoint/ssl_checkpoint_29epoch.pt"
)

output_dir = ROOT + "test/picture/embedding/"
os.makedirs(output_dir, exist_ok=True)


batch_size = 64

atom_dim = 72
bond_dim = 14
hidden_dim = 512
depth = 5

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Dataset
# =========================================================

graphs = torch.load(
    graph_path,
    weights_only=False
)

print("Number of molecules:", len(graphs))


# embedding에서는 SSLDataset 사용 X
# masking되지 않은 원본 graph를 넣는다.
loader = DataLoader(
    graphs,
    batch_size=batch_size,
    shuffle=False
)


# =========================================================
# Encoder
# =========================================================

encoder = DMPNN(
    atom_dim=atom_dim,
    bond_dim=bond_dim,
    hidden_dim=hidden_dim,
    depth=depth
).to(device)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

encoder.load_state_dict(
    checkpoint["encoder"]
)

encoder.eval()

print(
    "Loaded checkpoint epoch:",
    checkpoint["epoch"] + 1
)


# =========================================================
# Molecule embedding 추출
# =========================================================

embeddings = []

with torch.no_grad():

    for batch in loader:
        batch = batch.to(device)

        atom_h, bond_h = encoder(batch)

        # atom embedding -> molecule embedding
        mol_h = global_mean_pool(
            atom_h,
            batch.batch
        )

        embeddings.append(
            mol_h.cpu()
        )


embeddings = torch.cat(
    embeddings,
    dim=0
)

print(
    "Embedding shape:",
    embeddings.shape
)


# =========================================================
# PCA
# =========================================================

pca = PCA(
    n_components=2
)

embedding_2d = pca.fit_transform(
    embeddings.numpy()
)

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
)


# =========================================================
# descriptor용 molecule 복원
# =========================================================
#
# 중요한 점:
#
# 현재 .pt graph 안에 RDKit Mol이나 SMILES가 저장되어 있지 않다면
# descriptor를 계산할 수 없다.
#
# 아래에서는 graph에 smiles가 저장되어 있다고 가정.
#
# graph.smiles가 없다면 preprocess에서
#
# graph.smiles = Chem.MolToSmiles(mol)
#
# 을 추가해서 dataset을 다시 만들어야 한다.
# =========================================================

mols = []

for i, graph in enumerate(graphs):

    if not hasattr(graph, "smiles"):
        raise AttributeError(
            "MolGraph에 smiles가 없습니다.\n"
            "preprocess의 mol2graph()에서\n"
            "graph.smiles = Chem.MolToSmiles(mol)\n"
            "을 저장한 뒤 다시 preprocessing 하세요."
        )

    mol = Chem.MolFromSmiles(
        graph.smiles
    )

    if mol is None:
        raise ValueError(
            f"SMILES 변환 실패: index={i}"
        )

    mols.append(mol)


# =========================================================
# Molecular descriptors
# =========================================================

molecular_weight = []
heavy_atom_count = []
hetero_atom_count = []
aromatic_ratio = []
stereo_exists = []


for mol in mols:

    # -----------------------------------------------------
    # Molecular weight
    # -----------------------------------------------------

    molecular_weight.append(
        Descriptors.MolWt(mol)
    )


    # -----------------------------------------------------
    # Heavy atom count
    # -----------------------------------------------------

    heavy_atoms = mol.GetNumHeavyAtoms()

    heavy_atom_count.append(
        heavy_atoms
    )


    # -----------------------------------------------------
    # Hetero atom count
    #
    # C / H 제외
    # -----------------------------------------------------

    hetero = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in [1, 6]
    )

    hetero_atom_count.append(
        hetero
    )


    # -----------------------------------------------------
    # Aromatic atom ratio
    # -----------------------------------------------------

    num_atoms = mol.GetNumAtoms()

    num_aromatic = sum(
        atom.GetIsAromatic()
        for atom in mol.GetAtoms()
    )

    if num_atoms > 0:
        ratio = num_aromatic / num_atoms
    else:
        ratio = 0.0

    aromatic_ratio.append(
        ratio
    )


    # -----------------------------------------------------
    # Stereo 존재 여부
    #
    # atom chirality 또는 bond stereo
    # -----------------------------------------------------

    has_atom_stereo = any(
        atom.GetChiralTag()
        != Chem.rdchem.ChiralType.CHI_UNSPECIFIED

        for atom in mol.GetAtoms()
    )

    has_bond_stereo = any(
        bond.GetStereo()
        != Chem.rdchem.BondStereo.STEREONONE

        for bond in mol.GetBonds()
    )

    stereo_exists.append(
        int(
            has_atom_stereo
            or has_bond_stereo
        )
    )


# =========================================================
# plotting function
# =========================================================

def plot_continuous(
    values,
    title,
    colorbar_label,
    filename
):

    plt.figure(
        figsize=(8, 6)
    )

    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=values,
        s=8,
        alpha=0.6
    )

    plt.colorbar(
        scatter,
        label=colorbar_label
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(title)

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.show()

    print("saved:", save_path)



def plot_binary(
    values,
    title,
    labels,
    filename
):

    plt.figure(
        figsize=(8, 6)
    )

    values = torch.tensor(values)

    for value, label in labels.items():

        mask = values == value

        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=8,
            alpha=0.6,
            label=label
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.show()

    print("saved:", save_path)


# =========================================================
# 1. Molecular Weight
# =========================================================

plot_continuous(
    molecular_weight,
    title="PCA of Molecular Embeddings - Molecular Weight",
    colorbar_label="Molecular Weight",
    filename="pca_molecular_weight.png"
)


# =========================================================
# 2. Heavy Atom Count
# =========================================================

plot_continuous(
    heavy_atom_count,
    title="PCA of Molecular Embeddings - Heavy Atom Count",
    colorbar_label="Heavy Atom Count",
    filename="pca_heavy_atom_count.png"
)


# =========================================================
# 3. Hetero Atom Count
# =========================================================

plot_continuous(
    hetero_atom_count,
    title="PCA of Molecular Embeddings - Hetero Atom Count",
    colorbar_label="Hetero Atom Count",
    filename="pca_hetero_atom_count.png"
)


# =========================================================
# 4. Aromatic Atom Ratio
# =========================================================

plot_continuous(
    aromatic_ratio,
    title="PCA of Molecular Embeddings - Aromatic Ratio",
    colorbar_label="Aromatic Atom Ratio",
    filename="pca_aromatic_ratio.png"
)


# =========================================================
# 5. Stereo information
# =========================================================

plot_binary(
    stereo_exists,
    title="PCA of Molecular Embeddings - Stereochemistry",
    labels={
        0: "No stereo",
        1: "Stereo"
    },
    filename="pca_stereo.png"
)