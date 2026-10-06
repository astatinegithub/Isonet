import os

import torch
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT
from isonet.model.dmpnn import DMPNN


# =========================================================
# Setting
# =========================================================

graph_path = ROOT + "dataset/processed_data/for_test_1787700186.pt"
checkpoint_path = ROOT + "model/checkpoint/model_A/ssl_checkpoint_0epoch.pt"

output_dir = ROOT + "test/picture/embedding3/"
os.makedirs(output_dir, exist_ok=True)

batch_size = 64

atom_dim = 72
bond_dim = 14
hidden_dim = 512
depth = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Dataset
# =========================================================

graphs = torch.load(graph_path, weights_only=False)

print("Number of molecules:", len(graphs))

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

encoder.load_state_dict(checkpoint["encoder"])
encoder.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"] + 1)


# =========================================================
# Molecule embedding
# =========================================================

embeddings = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)

        atom_h, bond_h = encoder(batch)

        mol_h = global_mean_pool(
            atom_h,
            batch.batch
        )

        embeddings.append(mol_h.cpu())

embeddings = torch.cat(embeddings, dim=0)

print("Embedding shape:", embeddings.shape)


# =========================================================
# PCA
# =========================================================

pca_2d = PCA(n_components=2)
embedding_2d = pca_2d.fit_transform(embeddings.numpy())

print("\n===== 2D PCA =====")
print("Explained variance:", pca_2d.explained_variance_ratio_)
print("Total explained variance:", pca_2d.explained_variance_ratio_.sum())


pca_3d = PCA(n_components=3)
embedding_3d = pca_3d.fit_transform(embeddings.numpy())

print("\n===== 3D PCA =====")
print("Explained variance:", pca_3d.explained_variance_ratio_)
print("Total explained variance:", pca_3d.explained_variance_ratio_.sum())


pca_full = PCA()
pca_full.fit(embeddings.numpy())

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

print("\n===== Cumulative Explained Variance =====")

for n in [2, 3, 5, 10, 20, 50, 100]:
    if n <= embeddings.shape[1]:
        print(f"{n:3d} PCs: {cumulative_variance[n - 1]:.4f}")


# =========================================================
# Restore RDKit molecules
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

    mol = Chem.MolFromSmiles(graph.smiles)

    if mol is None:
        raise ValueError(f"SMILES 변환 실패: index={i}")

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
    molecular_weight.append(Descriptors.MolWt(mol))

    heavy_atoms = mol.GetNumHeavyAtoms()
    heavy_atom_count.append(heavy_atoms)

    hetero = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in [1, 6]
    )
    hetero_atom_count.append(hetero)

    num_atoms = mol.GetNumAtoms()
    num_aromatic = sum(
        atom.GetIsAromatic()
        for atom in mol.GetAtoms()
    )

    ratio = num_aromatic / num_atoms if num_atoms > 0 else 0.0
    aromatic_ratio.append(ratio)

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
        int(has_atom_stereo or has_bond_stereo)
    )


# =========================================================
# 2D plotting functions
# =========================================================

def plot_continuous(
    values,
    title,
    colorbar_label,
    filename,
    clip_percentile=None
):
    values = np.asarray(values)

    if clip_percentile is not None:
        vmax = np.percentile(values, clip_percentile)
    else:
        vmax = None

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=values,
        s=8,
        alpha=0.6,
        vmax=vmax
    )

    plt.colorbar(scatter, label=colorbar_label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("saved:", save_path)


def plot_binary(
    values,
    title,
    labels,
    filename
):
    values = np.asarray(values)

    plt.figure(figsize=(8, 6))

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

    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("saved:", save_path)


# =========================================================
# 3D plotting functions
# =========================================================

def plot_continuous_3d(
    values,
    title,
    colorbar_label,
    filename,
    clip_percentile=None
):
    values = np.asarray(values)

    if clip_percentile is not None:
        vmax = np.percentile(values, clip_percentile)
    else:
        vmax = None

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=values,
        s=5,
        alpha=0.5,
        vmax=vmax
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)

    fig.colorbar(
        scatter,
        ax=ax,
        label=colorbar_label,
        shrink=0.7
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("saved:", save_path)


def plot_binary_3d(
    values,
    title,
    labels,
    filename
):
    values = np.asarray(values)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for value, label in labels.items():
        mask = values == value

        ax.scatter(
            embedding_3d[mask, 0],
            embedding_3d[mask, 1],
            embedding_3d[mask, 2],
            s=5,
            alpha=0.5,
            label=label
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("saved:", save_path)


# =========================================================
# 2D Molecular Weight
# =========================================================

plot_continuous(
    molecular_weight,
    title="PCA of Molecular Embeddings - Molecular Weight",
    colorbar_label="Molecular Weight",
    filename="pca_molecular_weight.png",
    clip_percentile=99
)


# =========================================================
# 2D Heavy Atom Count
# =========================================================

plot_continuous(
    heavy_atom_count,
    title="PCA of Molecular Embeddings - Heavy Atom Count",
    colorbar_label="Heavy Atom Count",
    filename="pca_heavy_atom_count.png",
    clip_percentile=99
)


# =========================================================
# 2D Hetero Atom Count
# =========================================================

plot_continuous(
    hetero_atom_count,
    title="PCA of Molecular Embeddings - Hetero Atom Count",
    colorbar_label="Hetero Atom Count",
    filename="pca_hetero_atom_count.png",
    clip_percentile=99
)


# =========================================================
# 2D Aromatic Ratio
# =========================================================

plot_continuous(
    aromatic_ratio,
    title="PCA of Molecular Embeddings - Aromatic Ratio",
    colorbar_label="Aromatic Atom Ratio",
    filename="pca_aromatic_ratio.png"
)


# =========================================================
# 2D Stereo
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


# =========================================================
# 3D Molecular Weight
# =========================================================

plot_continuous_3d(
    molecular_weight,
    title="3D PCA - Molecular Weight",
    colorbar_label="Molecular Weight",
    filename="pca_3d_molecular_weight.png",
    clip_percentile=99
)


# =========================================================
# 3D Heavy Atom Count
# =========================================================

plot_continuous_3d(
    heavy_atom_count,
    title="3D PCA - Heavy Atom Count",
    colorbar_label="Heavy Atom Count",
    filename="pca_3d_heavy_atom_count.png",
    clip_percentile=99
)


# =========================================================
# 3D Hetero Atom Count
# =========================================================

plot_continuous_3d(
    hetero_atom_count,
    title="3D PCA - Hetero Atom Count",
    colorbar_label="Hetero Atom Count",
    filename="pca_3d_hetero_atom_count.png",
    clip_percentile=99
)


# =========================================================
# 3D Aromatic Ratio
# =========================================================

plot_continuous_3d(
    aromatic_ratio,
    title="3D PCA - Aromatic Ratio",
    colorbar_label="Aromatic Atom Ratio",
    filename="pca_3d_aromatic_ratio.png"
)


# =========================================================
# 3D Stereo
# =========================================================

plot_binary_3d(
    stereo_exists,
    title="3D PCA - Stereochemistry",
    labels={
        0: "No stereo",
        1: "Stereo"
    },
    filename="pca_3d_stereo.png"
)


# =========================================================
# PCA / Descriptor Correlation
# =========================================================

print("\n===== PCA / Descriptor Correlation =====")

for name, values in [
    ("Molecular Weight", molecular_weight),
    ("Heavy Atom Count", heavy_atom_count),
    ("Hetero Atom Count", hetero_atom_count),
    ("Aromatic Ratio", aromatic_ratio),
]:
    pc1_corr, _ = spearmanr(embedding_3d[:, 0], values)
    pc2_corr, _ = spearmanr(embedding_3d[:, 1], values)
    pc3_corr, _ = spearmanr(embedding_3d[:, 2], values)

    print(
        f"{name:20s} | "
        f"PC1: {pc1_corr:.3f} | "
        f"PC2: {pc2_corr:.3f} | "
        f"PC3: {pc3_corr:.3f}"
    )