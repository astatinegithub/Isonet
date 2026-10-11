import torch
from torch import Tensor


__all__ = ["compute_metrics"]


@torch.no_grad()
def regression_metrics(y_true: Tensor, y_pred: Tensor) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mask = torch.isfinite(y_true) # 값이 inf/-inf/NaN인 (결측인)값 마스크
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.numel() == 0: # tenser안에 든게 없을때
        return {"n": 0, "rmse": None, "mae": None}

    if not torch.isfinite(y_pred).all(): # 모델에 결측이 있을때 
        raise ValueError("예측값에 NaN 또는 Inf가 있습니다.")

    error = y_true - y_pred

    return {
        "n": y_true.numel(),
        "rmse": error.square().mean().sqrt().item(),
        "mae": error.abs().mean().item()
    }


def binary_metrics():
    ...



def compute_metrics(y_true: Tensor, y_pred: Tensor, endpoints: dict) -> dict:
    """endpoints순서와 y_true, y_pred 내부 (batch, endpoint)의 endpoint순서가 동일해야 함"""
    metrics = dict()
    for i, endpoint in enumerate(endpoints.keys()):
        if endpoints[endpoint]["type"] == "regression":
            metrics[endpoint] = regression_metrics(y_true[:, i], y_pred[:, i])
        elif endpoints[endpoint]["type"] == "binary":
            metrics[endpoint] = binary_metrics(y_true[:, i], y_pred[:, i])
        else: 
            raise ValueError("알수없는 type의 endpoint입니다.")

    return metrics