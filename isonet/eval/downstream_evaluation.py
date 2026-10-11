import json
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from isonet.model.fine_tuning_model import IsonetModel
from isonet.eval.metrics import compute_metrics


__all__ = ["evaluate_model"]


@torch.no_grad()
def evaluate(model, loader, device, endpoints):
    model.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        pred = torch.cat([outputs[name] for name in endpoints], dim=1)
        true: Tensor = batch.y

        if true.shape != pred.shape: # 이게 필요한가?
            raise ValueError(f"정답/예측 shape 불일치: {true.shape} vs {pred.shape}")

        all_true.append(true.cpu())
        all_pred.append(pred.cpu())
    if not all_true:
        raise ValueError("평가 데이터가 비어 있습니다.")

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    metrics = compute_metrics(y_true, y_pred, endpoints)

    return metrics, y_true, y_pred


def evaluate_model(eval_model_path, test_graphs_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        eval_model_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["cfg"]
    endpoints = cfg["endpoints"]

    model = IsonetModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # 전처리된 test 그래프 목록
    test_graphs = torch.load(
        test_graphs_path,
        map_location="cpu",
        weights_only=False,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=64,
        shuffle=False,
    )

    metrics, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        endpoints=endpoints,
    )

    # 지표 출력 및 저장
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            metrics, f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    # 추후 분석을 위한 정답·예측 저장
    torch.save(
        {
            "endpoints": list(endpoints),
            "y_true": y_true,
            "y_pred": y_pred,
        },
        output_dir / "predictions.pt",
    )


if __name__ == "__main__":
    evaluate_model(
        eval_model_path=...,
        test_graphs_path=...,
        output_dir=...
    ) # 여기 입력하면 원하는 파일들 분석가능