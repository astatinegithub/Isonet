import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import umap.umap_ as umap
from rdkit import Chem
from rdkit.Chem import Descriptors
# from scipy.stats import pearsonr
# from collections import defaultdict

import torch 
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from isonet.config import ROOT
from isonet.model.dmpnn import DMPNN
from isonet.data.featurizer import AtomFeaturizer, BondFeaturizer



graph_path = ROOT + "dataset/processed_data/for_test_1787700186.pt"
checkpoint_path = ROOT + "model/checkpoint/model_A/ssl_checkpoint_29epoch.pt"
output_dir = ROOT + "eval/pretrain/model_A/"
os.makedirs(output_dir, exist_ok=True)

ISONET_CONFIG = {
    "atom_dim": 72,        # atom feature 차원
    "bond_dim": 14,        # bond feature 차원
    "hidden_dim": 512,
    "atom_types": AtomFeaturizer.model_A().atom_type_nums,
    "bond_types:": BondFeaturizer.model_A().bond_type_nums,
    "depth": 5
}
batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load(graph_path, weights_only=False)
print("Number of molecules:", len(graphs))
loader = DataLoader(
    graphs,
    batch_size=batch_size,
    shuffle=False
)


encoder = DMPNN(
    atom_dim=ISONET_CONFIG["atom_dim"],
    bond_dim=ISONET_CONFIG["bond_dim"],
    hidden_dim=ISONET_CONFIG["hidden_dim"],
    depth=ISONET_CONFIG["depth"]
).to(device)
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)
encoder.load_state_dict(checkpoint["encoder"])
encoder.eval()
print("Loaded checkpoint epoch:", checkpoint["epoch"] + 1)


embeddings = []

with torch.no_grad():
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        atom_h, _ = encoder(batch)
        mol_h = global_mean_pool(atom_h, batch.batch)
        mol_h = F.normalize(mol_h, p=2, dim=-1)
        embeddings.append(mol_h.cpu())

embeddings = torch.cat(embeddings, dim=0)
print("Embedding shape:", embeddings.shape)
embeddings = pd.DataFrame(embeddings.numpy())


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())

    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
        "HeteroAtoms": Descriptors.NumHeteroatoms(mol),
        "AromaticRatio": aromatic_atoms/num_atoms if num_atoms > 0 else 0
    }


descriptor_list = []
valid_indices = []

for i, graph in enumerate(graphs):
    smiles = graph.smiles # 테스트용에만 추가하는게 좋을듯
    desc = get_descriptors(smiles)
    descriptor_list.append(desc)
    valid_indices.append(i)

df = pd.DataFrame(descriptor_list)


columns_to_filter = ["MW", "LogP", "TPSA", "HeavyAtoms", "HeteroAtoms"]
is_valid = pd.Series(True, index=df.index)

for col in columns_to_filter:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    is_valid &= (df[col] >= Q1 - 3.0 * IQR) & (df[col] <= Q3 + 3.0 * IQR)

print(f"Original size: {len(df)} -> Cleaned size: {is_valid.sum()}")

# ----------------------------------------------------
df_clean = df[is_valid].reset_index(drop=True)
embeddings_clean = embeddings[is_valid].reset_index(drop=True) # 행을 똑같이 맞춰서 잘라냄


umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = umap_model.fit_transform(embeddings)

df_umap = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])

print(df.info())
print(df.head())

plt.figure(figsize=(18, 18))
plt.scatter(df_umap['UMAP-1'], df_umap['UMAP-2'], c=df["MW"], cmap='viridis', alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.title('UMAP(2D) embedding with MW')
plt.show()

plt.figure(figsize=(18, 18))
plt.scatter(df_umap['UMAP-1'], df_umap['UMAP-2'], c=df["LogP"], cmap='viridis', alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.title('UMAP(2D) embedding with LogP')
plt.show()

plt.figure(figsize=(18, 18))
plt.scatter(df_umap['UMAP-1'], df_umap['UMAP-2'], c=df["TPSA"], cmap='viridis', alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.title('UMAP(2D) embedding with TPSA')
plt.show()

plt.figure(figsize=(18, 18))
plt.scatter(df_umap['UMAP-1'], df_umap['UMAP-2'], c=df["HeavyAtoms"], cmap='viridis', alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.title('UMAP(2D) embedding with HeavyAtoms')
plt.show()

plt.figure(figsize=(18, 18))
plt.scatter(df_umap['UMAP-1'], df_umap['UMAP-2'], c=df["AromaticRatio"], cmap='viridis', alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.title('UMAP(2D) embedding with AromaticRatio')
plt.show()

# X = embeddings.numpy()z
# print("Embedding mean:", X.mean())
# print("Embedding std :", X.std())
# # 각 dimension의 표준편차
# dim_std = X.std(axis=0)
# print("Mean dimension std:", dim_std.mean())
# print("Min dimension std :", dim_std.min())
# print("Max dimension std :", dim_std.max())


# # PCA
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(X)
# print("\n===== PCA Explained Variance =====")

# for i, ratio in enumerate(pca.explained_variance_ratio_):
#     print(f"PC{i+1}: {ratio:.4f}")
# print("Total explained variance (10 PCs):", pca.explained_variance_ratio_.sum())


# plt.figure(figsize=(7, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.5)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Molecular Embedding PCA")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "embedding_pca.png"),dpi=300)

# plt.close()





# descriptor_names = descriptor_list[0].keys()
# X_valid = X[valid_indices]
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X_valid)
# print("\n===== PCA / Descriptor Correlation =====")


# for name in descriptor_names:
#     values = np.array([d[name] for d in descriptor_list])
#     correlations = []
#     for pc_idx in range(3):
#         r, p = pearsonr(X_pca[:, pc_idx], values)
#         correlations.append(r)
#     print(
#         f"{name:15s} | "
#         f"PC1: {correlations[0]: .3f} | "
#         f"PC2: {correlations[1]: .3f} | "
#         f"PC3: {correlations[2]: .3f}"
#     )


# def cosine_similarity(a, b):
#     a = torch.tensor(a).unsqueeze(0)
#     b = torch.tensor(b).unsqueeze(0)
#     return F.cosine_similarity(a, b).item()


# def remove_stereo(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     Chem.RemoveStereochemistry(mol)
#     return Chem.MolToSmiles(mol, canonical=True)


# stereo_groups = defaultdict(list)
# for i, graph in enumerate(graphs):
#     smiles = graph.smiles
#     base_smiles = remove_stereo(smiles)
#     if base_smiles is not None:
#         stereo_groups[base_smiles].append((i, smiles))


# candidate_groups = {
#     key: value
#     for key, value in stereo_groups.items()
#     if len(value) >= 2
# }

# print("Potential stereoisomer groups:", len(candidate_groups))

# stereo_similarities = []
# for group in candidate_groups.values():
#     for i in range(len(group)):
#         for j in range(i + 1, len(group)):
#             idx1, smiles1 = group[i]
#             idx2, smiles2 = group[j]
#             if smiles1 == smiles2: # 완전히 같은 SMILES면 제외
#                 continue

#             sim = cosine_similarity(X[idx1], X[idx2])
#             stereo_similarities.append(sim)

# stereo_similarities = np.array(stereo_similarities)
# print("\n===== Stereoisomer Similarity =====")
# print("N pairs:", len(stereo_similarities))
# print("Mean cosine similarity:", stereo_similarities.mean())
# print("Std:",stereo_similarities.std())




# random_similarities = []
# num_pairs = len(stereo_similarities)

# N = len(X)
# for _ in range(num_pairs):
#     i, j = random.sample(range(N), 2)
#     sim = cosine_similarity(X[i], X[j])
#     random_similarities.append(sim)

# random_similarities = np.array(random_similarities)

# print("\n===== Comparison =====")
# print("Stereo pairs mean:",stereo_similarities.mean())
# print("Random pairs mean:", random_similarities.mean())


# plt.figure(figsize=(7, 5))
# plt.hist(stereo_similarities, bins=40, alpha=0.6, label="Stereoisomer")
# plt.hist(random_similarities, bins=40, alpha=0.6, label="Random")
# plt.xlabel("Cosine Similarity")
# plt.ylabel("Count")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir,"stereo_vs_random_similarity.png"),dpi=300)
# plt.close()



# dead_dims = np.sum(X.std(axis=0) < 1e-6)
# print(f"Dead dimensions: {dead_dims} / {X.shape[1]}")

# stereo_l2 = []
# for group in candidate_groups.values():
#     for i in range(len(group)):
#         for j in range(i + 1, len(group)):
#             idx1, smiles1 = group[i]
#             idx2, smiles2 = group[j]

#             if smiles1 == smiles2:
#                 continue

#             dist = np.linalg.norm(X[idx1] - X[idx2])
#             stereo_l2.append(dist)

# stereo_l2 = np.array(stereo_l2)

# print("Stereo L2 mean:", stereo_l2.mean())
# print("Stereo L2 std :", stereo_l2.std())
# print("Stereo L2 max :", stereo_l2.max())


# # ==============================
# # Random Initialized DMPNN 비교
# # ==============================

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np


# # 1. 같은 구조의 DMPNN 생성
# # checkpoint를 load하지 않으므로 random initialization 상태
# random_encoder = DMPNN(
#     atom_dim=ISONET_CONFIG["atom_dim"],
#     bond_dim=ISONET_CONFIG["bond_dim"],
#     hidden_dim=ISONET_CONFIG["hidden_dim"],
#     depth=ISONET_CONFIG["depth"]
# ).to(device)

# random_encoder.eval()


# # 2. Random encoder embedding 추출
# random_embeddings = []

# with torch.no_grad():
#     for batch in loader:
#         batch = batch.to(device)
#         atom_h, _ = random_encoder(batch)

#         # pretrained와 완전히 동일한 pooling 사용
#         mol_h = global_mean_pool(atom_h, batch.batch)

#         random_embeddings.append(mol_h.cpu())

# random_embeddings = torch.cat(random_embeddings, dim=0)

# print("Random embedding shape:", random_embeddings.shape)


# # ==============================
# # 기본 통계 비교
# # ==============================

# X_pretrained = embeddings.numpy()
# X_random = random_embeddings.numpy()


# def print_embedding_statistics(name, X):
#     dim_std = X.std(axis=0)
#     print(f"\n===== {name} =====")
#     print("Embedding mean:", X.mean())
#     print("Embedding std:", X.std())
#     print("Mean dimension std:", dim_std.mean())
#     print("Dead dimensions:", np.sum(dim_std < 1e-6), "/", X.shape[1])

# print_embedding_statistics("Random DMPNN",X_random)
# print_embedding_statistics("Pretrained DMPNN",X_pretrained)


# # ==============================
# # 각각 PCA 수행
# # ==============================

# pca_random = PCA(n_components=10)
# random_pca = pca_random.fit_transform(X_random)
# pca_pretrained = PCA(n_components=10)
# pretrained_pca = pca_pretrained.fit_transform(X_pretrained)


# print("\n===== PCA Comparison =====")
# print("Random PC1:", pca_random.explained_variance_ratio_[0])
# print("Pretrained PC1:", pca_pretrained.explained_variance_ratio_[0])
# print("Random PC1~10:",pca_random.explained_variance_ratio_.sum())
# print("Pretrained PC1~10:", pca_pretrained.explained_variance_ratio_.sum())


# # ==============================
# # 중간보고용 PCA 그림
# # ==============================
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# # Random
# axes[0].scatter(random_pca[:, 0], random_pca[:, 1], s=5, alpha=0.35)
# axes[0].set_title("Random Initialized D-MPNN")
# axes[0].set_xlabel(f"PC1 ({pca_random.explained_variance_ratio_[0]*100:.1f}%)")
# axes[0].set_ylabel(f"PC2 ({pca_random.explained_variance_ratio_[1]*100:.1f}%)")


# # Pretrained
# axes[1].scatter(pretrained_pca[:, 0], pretrained_pca[:, 1], s=5,alpha=0.35)
# axes[1].set_title("SSL Pretrained D-MPNN")
# axes[1].set_xlabel(f"PC1 ({pca_pretrained.explained_variance_ratio_[0]*100:.1f}%)")
# axes[1].set_ylabel(f"PC2 ({pca_pretrained.explained_variance_ratio_[1]*100:.1f}%)")


# plt.suptitle("Molecular Representation Before vs After SSL Pretraining", fontsize=14)

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "random_vs_pretrained_pca.png"), dpi=300)

# plt.close()