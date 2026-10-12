"""
Training Script for Joint Self-Attention House Price Prediction Model.
Trains with MSE loss + scheduler, saves best checkpoint.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import joblib
import json

from utils.data_generator import generate_dataset
from utils.feature_engineering import load_and_prepare
from models.joint_attention_model import build_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 80
BATCH_SIZE = 128
LR = 3e-4
PATIENCE = 15


def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def train():
    print(f"[Trainer] Device: {DEVICE}")

    # 1. Generate / load data
    if not os.path.exists('data/house_data.csv'):
        generate_dataset(3000)

    print("[Trainer] Loading and preparing features...")
    pack = load_and_prepare('data/house_data.csv')

    X_num_tr  = to_tensor(pack['X_num_train'])
    X_cat_tr  = to_tensor(pack['X_cat_train'], dtype=torch.long)
    y_tr      = to_tensor(pack['y_train'])
    X_num_val = to_tensor(pack['X_num_val'])
    X_cat_val = to_tensor(pack['X_cat_val'], dtype=torch.long)
    y_val     = to_tensor(pack['y_val'])
    fe        = pack['fe']

    train_ds = TensorDataset(X_num_tr, X_cat_tr, y_tr)
    val_ds   = TensorDataset(X_num_val, X_cat_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    n_num = X_num_tr.shape[1]
    model = build_model(n_num_features=n_num, cat_dims=fe.cat_dims, device=DEVICE)
    print(f"[Trainer] Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xn, xc, yb in train_dl:
            optimizer.zero_grad()
            pred, _, _ = model(xn, xc)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        preds_all, y_all = [], []
        with torch.no_grad():
            for xn, xc, yb in val_dl:
                pred, _, _ = model(xn, xc)
                val_loss += criterion(pred, yb).item() * len(yb)
                preds_all.append(pred.cpu().numpy())
                y_all.append(yb.cpu().numpy())
        val_loss /= len(val_ds)

        # Convert scaled back to raw prices for MAE
        preds_raw = fe.inverse_transform_target(np.concatenate(preds_all))
        y_raw = fe.inverse_transform_target(np.concatenate(y_all))
        mae = np.mean(np.abs(preds_raw - y_raw))

        scheduler.step()
        history['train_loss'].append(round(train_loss, 6))
        history['val_loss'].append(round(val_loss, 6))
        history['val_mae'].append(round(float(mae), 2))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | MAE=${mae:,.0f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"[Trainer] Early stopping at epoch {epoch}")
                break

    # Save metadata
    meta = {
        'n_num_features': int(n_num),
        'cat_dims': {k: int(v) for k, v in fe.cat_dims.items()},
        'feature_names': fe.feature_names,
        'd_model': 64, 'n_heads': 4, 'n_self_attn_layers': 2,
        'best_val_loss': round(best_val_loss, 6),
        'final_val_mae': history['val_mae'][-1]
    }
    with open('models/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"[Trainer] Training complete. Best val_loss={best_val_loss:.5f}")
    print(f"[Trainer] Saved: models/best_model.pt, models/model_meta.json")
    return history


if __name__ == '__main__':
    train()
