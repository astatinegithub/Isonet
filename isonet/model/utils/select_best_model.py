import torch
import os
from pathlib import Path

def select_best_model(last_model_path, save_dir) -> str:
    save_dir = Path(save_dir)
    checkpoint = torch.load(last_model_path, weights_only=False)
    best_loss = torch.inf
    best_epoch = 0
    for epoch, loss in enumerate(checkpoint['loss_history']['val_loss']):
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

    print(f"best_epoch {best_epoch} | loss {best_loss}")
    print(save_dir)

    if os.path.exists(save_dir / f"finetune_epoch_{best_epoch+1}.pt"):
        return save_dir / f"finetune_epoch_{best_epoch+1}.pt"
    elif os.path.exists(save_dir / f"finish_train_epoch_{best_epoch+1}.pt"):
        return save_dir / f"finish_train_epoch_{best_epoch+1}.pt"
    elif os.path.exists(save_dir / f"finetune_val_best.pt"):
        print('aaaaa')
        return save_dir / f"finetune_val_best.pt"
    else:
        raise ValueError("best model파일이 경로에 존재하지 않습니다.")

    