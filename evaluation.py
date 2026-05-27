import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import training


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    """
    Compute R², MAE, MSE, RMSE and print a formatted table row.
    All values are in the ORIGINAL price scale (after inverse transform).
    """
    
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"  {label:<25} R²={r2:+.4f}  MAE={mae:.4f}  "
          f"MSE={mse:.4f}  RMSE={rmse:.4f}")

    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}


def predict(model, loader, device=training.DEVICE):
    """Collect all predictions and ground-truth from a DataLoader."""

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())
            trues.append(y_batch.numpy())

    return np.concatenate(preds).flatten(), np.concatenate(trues).flatten()


def ensemble_predictions(lstm_model, ann_model, meta_model,
                              loader, device=training.DEVICE):
    """Run all three models to get final ensemble predictions."""
    lstm_model.eval(); ann_model.eval(); meta_model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            lp = lstm_model(X_batch)
            ap = ann_model(X_batch)
            mp = meta_model(lp, ap).cpu().numpy()
            preds.append(mp)
            trues.append(y_batch.numpy())

    return np.concatenate(preds).flatten(), np.concatenate(trues).flatten()


def inverse_scale(values: np.ndarray, close_scaler):
    """
    MinMaxScaler was fit on the Close column only.
    We reshape to (N,1) for inverse_transform, then flatten back.
    """
    return close_scaler.inverse_transform(values.reshape(-1, 1)).flatten()