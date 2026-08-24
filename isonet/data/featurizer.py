import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem.rdchem import Bond, Atom



class OneHotFeaturizer:
    def __init__(self):
        ...

    def one_hot_unknown(self, value, choices) -> Tensor:
        """choices + unknown slot"""
        out = torch.zeros(len(choices) + 1)
        try:
            idx = choices.index(value)
        except ValueError:
            idx = len(choices)
        out[idx] = 1.0
        return out


class AtomFeaturizer(OneHotFeaturizer):
    def __init__(self, atomic_nums, degrees, formal_charges,
                 chiral_tags, num_hs, hybeidizations, use_stereo):
        
        self.atomic_nums: list    = atomic_nums
        self.degrees: list        = degrees
        self.formal_charges: list = formal_charges
        self.chiral_tags: list    = chiral_tags
        self.num_hs: list         = num_hs
        self.hybeidizations: list = hybeidizations

        self.use_stereo: bool = use_stereo


    def atom_type(self, a: Atom):
        return self.atomic_nums.index(a.GetAtomicNum())


    @property
    def num_atom_classes(self):
        return len(self.atomic_nums)


    def __call__(self, a: Atom) -> Tensor:
        atomic_num = self.one_hot_unknown(a.GetAtomicNum(), self.atomic_nums)           # 1. atomic number: 37 + unknown = 38
        degree = self.one_hot_unknown(a.GetDegree(), self.degrees)                      # 2. degree: 6 + unknown = 
        charge = self.one_hot_unknown(a.GetFormalCharge(), self.formal_charges)         # 3. formal charge: 5 + unknown = 6

        if self.use_stereo:                                                             # 4. chirality: 4 + unknown = 5
            chirality = self.one_hot_unknown(int(a.GetChiralTag()), self.chiral_tags)
        else:
            chirality = torch.zeros(5) # 중요: 제거하지 말고 5차원 그대로 0
            
        num_h = self.one_hot_unknown(a.GetTotalNumHs(), self.num_hs)                    # 5. H count: 5 + unknown = 6
        hybridization = self.one_hot_unknown(a.GetHybridization(), self.hybeidizations) # 6. hybridization: 7 + unknown = 8
        mass = torch.tensor([a.GetMass() / 100.0])                                      # 8. mass: 1
        aromatic = torch.tensor([float(a.GetIsAromatic())])                             # 7. aromatic: 1 

        feature = torch.cat([
            atomic_num,       # 38
            degree,           # 7
            charge,           # 6
            chirality,        # 5
            num_h,            # 6
            hybridization,    # 8
            aromatic,         # 1
            mass,             # 1
        ])

        assert feature.shape[0] == 72
        return feature


    @classmethod
    def model_A(cls): # model A용 feature
        ATOM_NUMS = list(range(1, 37)) + [53]  # H ~ Kr + I
        DEGREES = [0, 1, 2, 3, 4, 5]
        FORMAL_CHARGES = [-2, -1, 0, 1, 2]
        CHIRAL_TAGS = [0, 1, 2, 3]
        NUM_HS = [0, 1, 2, 3, 4]
        HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP2D,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

        return cls(
            atomic_nums=ATOM_NUMS,
            degrees=DEGREES,
            formal_charges=FORMAL_CHARGES,
            chiral_tags=CHIRAL_TAGS,
            num_hs=NUM_HS,
            hybeidizations=HYBRIDIZATIONS,
            use_stereo=False
        )



class BondFeaturizer(OneHotFeaturizer):
    def __init__(self, bond_types, bond_stereos, use_stereo):
    
        self.bond_types: list = bond_types
        self.bond_stereos: list = bond_stereos
        self.use_stereo: bool = use_stereo


    def bond_type(self, bond: Bond) -> int:
        return self.bond_types.index(bond.GetBondType())


    @property
    def num_bond_classes(self):
        return len(self.bond_types)

    
    def __call__(self, bond: Bond) -> Tensor:
        null = torch.tensor([0.0])    # null bit
        # 4 bond types
        bond_type = torch.tensor([float(bond.GetBondType() == bt) for bt in self.bond_types])
        conjugated = torch.tensor([float(bond.GetIsConjugated())])
        ring = torch.tensor([float(bond.IsInRing())])
        # 6 stereo + unknown = 7
        if self.use_stereo:
            stereo = self.one_hot_unknown(int(bond.GetStereo()), self.bond_stereos)
        else:
            stereo = torch.zeros(7)

        feature = torch.cat([
            null,          # 1
            bond_type,     # 4
            conjugated,    # 1
            ring,          # 1
            stereo,        # 7
        ])

        assert feature.shape[0] == 14

        return feature


    @classmethod
    def model_A(cls):
        BOND_TYPES = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        BOND_STEREOS = [0, 1, 2, 3, 4, 5]


        return cls(
            bond_types=BOND_TYPES,
            bond_stereos=BOND_STEREOS,
            use_stereo=False
        )
