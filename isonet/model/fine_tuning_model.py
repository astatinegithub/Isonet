from copy import copy 
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import softmax

from isonet.data.dataset import MolGraph
from isonet.model.dmpnn import DMPNN



class IsonetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = DMPNN(
            cfg["atom_dim"],
            cfg["bond_dim"],
            cfg["hidden_dim"],
            cfg["depth"]
        )
        self.use_stereo_attention = cfg.get("use_stereo_attention", False)
        self.encoder_use_cip = cfg.get("encoder_use_cip", True)
        self.cip_slice = slice(51, 56) # 이거 나중에 바꿔야할듯 -> 명시보다는 변수로 입력받아야함
        self.pool = cfg.get('pooling', global_add_pool) # 임시pooling 방식
        self.heads = nn.ModuleDict({
            admet : nn.Sequential(
                nn.Linear(cfg['hidden_dim'], cfg['admet_hidden_dim']),
                nn.ReLU(),
                nn.Dropout(cfg['drop_rate']),
                nn.Linear(cfg['admet_hidden_dim'], 1)
            ) for admet in cfg['endpoints']
        })
        if self.use_stereo_attention:
            self.stereo_attention = StereoAttention(
                hidden_dim=cfg['hidden_dim'],
                bond_dim=cfg['bond_dim'],
                attn_dim=cfg['attn_dim'],
                dropout=cfg['attn_droprate'],
                cip_start=51,                
            )


    def forward(self, data: MolGraph):
        input_data = data
        if not self.encoder_use_cip:
            input_data = copy(data)
            input_data.x = data.x.clone()
            input_data.x[:, self.cip_slice] = 0

        h, _ = self.encoder(input_data)

        if self.use_stereo_attention:
            h, attention = self.stereo_attention(h, data)
        
        h = self.pool(h, data.batch)

        return {name: head(h) for name, head in self.heads.items()}



# 임시 나중에 맘에 안드는 코드 수정할꺼임
class StereoAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        bond_dim=14,
        attn_dim=64,
        dropout=0.1,
        cip_start=51,
        kqv_bias=False,
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.cip_start = cip_start

        # Attention 경로에만 적용.
        # 기존 residual 경로의 h는 그대로 유지한다.
        self.norm = nn.LayerNorm(hidden_dim)

        # 중심: 원자 표현 + CIP(R/S/r/s)
        self.W_query = nn.Linear(hidden_dim + 4, attn_dim, bias=kqv_bias)

        # 이웃: 원자 표현 + 중심과의 결합 feature
        self.W_key = nn.Linear(hidden_dim + bond_dim, attn_dim, bias=kqv_bias)
        self.W_value = nn.Linear(hidden_dim + bond_dim, attn_dim, bias=kqv_bias)
        self.W_output = nn.Linear(attn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 중심별 보정량 조절
        self.gate = nn.Linear(hidden_dim + 4, 1)

        # 초기 gate ≈ sigmoid(-2) ≈ 0.12
        # 보정값을 작게 반영하면서 학습을 시작한다.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)


    def forward(self, h: Tensor, data: MolGraph):
        """
        h: [batch 전체 원자 수, hidden_dim]

        반환:
            updated_h: 원자별 보완 표현
            attention: 선택된 방향성 결합과 가중치
        """
        cip = data.x[:, self.cip_start:self.cip_start + 4].to(dtype=h.dtype)

        # R/S/r/s 중 하나가 지정된 원자만 선택
        center_mask = cip.sum(dim=-1) > 0
        _, dst = data.edge_index

        # 이웃 src -> 입체중심 dst인 방향만 사용
        selected = center_mask[dst]

        edge_index = data.edge_index[:, selected]
        neighbor = edge_index[0]
        center = edge_index[1]

        # 대상 입체중심으로 들어가는 결합이 없는 경우
        if center.numel() == 0:
            return h, {
                "edge_index": edge_index.detach(),
                "alpha": h.new_empty(0),
            }

        # centers: 실제 대상 중심의 원자 인덱스
        # group: 각 결합이 centers의 몇 번째 중심에 속하는지
        centers, group = torch.unique(center, sorted=True, return_inverse=True)
        num_centers = centers.numel()

        h_norm = self.norm(h)

        center_input = torch.cat([h_norm[centers], cip[centers]], dim=-1,)
        neighbor_input = torch.cat([h_norm[neighbor], data.edge_attr[selected].to(dtype=h.dtype),], dim=-1,)

        # q: [입체중심 수, attn_dim]
        # k, v: [선택된 방향성 결합 수, attn_dim]
        q: Tensor = self.W_query(center_input)
        k: Tensor = self.W_key(neighbor_input)
        v: Tensor = self.W_value(neighbor_input)

        # 각 결합에 해당하는 중심의 q를 가져온다.
        attn_score = (q[group] * k).sum(dim=-1)
        attn_score = attn_score / self.attn_dim**0.5

        # 중심별 이웃끼리만 softmax
        alpha = softmax(attn_score, index=group, num_nodes=num_centers,)

        # 중심별 이웃 정보의 가중합
        messages = alpha.unsqueeze(-1) * v

        context = v.new_zeros((num_centers, self.attn_dim))
        context = context.index_add(0, group, messages)

        gate = torch.sigmoid(self.gate(center_input))
        update = gate * self.dropout(self.W_output(context))

        # 입체중심 위치에만 보정값을 더한다.
        # 나머지 원자의 표현은 변경하지 않는다.
        updated_h = h.index_add(0, centers, update)

        # 시각화용 값은 gradient에서 분리
        attention = {"edge_index": edge_index.detach(), "alpha": alpha.detach(),}

        return updated_h, attention
    