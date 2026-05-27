import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt


def training_curves(history_dict: dict, save_path: str = "evaluation/training_curves.png"):
    """
    history_dict = {
        "LSTM":  (train_losses, val_losses),
        "ANN":   (train_losses, val_losses),
        "Meta":  (train_losses, val_losses),
    }
    """
    fig, axes = plt.subplots(1, len(history_dict), figsize=(6 * len(history_dict), 4))
    if len(history_dict) == 1:
        axes = [axes]

    for ax, (name, (tr, vl)) in zip(axes, history_dict.items()):
        ax.plot(tr, label="Train", linewidth=1.5)
        ax.plot(vl, label="Val",   linewidth=1.5, linestyle="--")
        ax.set_title(f"{name} — Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def predictions(y_true_real: np.ndarray, predictions: dict,
                     ticker: str = "Stock",
                     save_path: str = "evaluation/predictions.png"):
    """
    predictions = {"LSTM": arr, "ANN": arr, "LSTM+ANN": arr}
    All arrays should be in real dollar values.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_real, label="Actual", color="black", linewidth=2, alpha=0.8)

    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    styles = ["--", "-.", "-"]
    for (name, pred), color, style in zip(predictions.items(), colors, styles):
        plt.plot(pred, label=name, color=color, linestyle=style,
                 linewidth=1.5, alpha=0.85)

    plt.title(f"{ticker} — Actual vs. Predicted Close Price (Test Set)")
    plt.xlabel("Trading Days (Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def metrict_comparison(results: dict, save_path: str = "evaluation/metrics_comparison.png"):
    """
    results = {
        "LSTM":     {"R2": ..., "MAE": ..., "MSE": ..., "RMSE": ...},
        "ANN":      {...},
        "LSTM+ANN": {...},
    }
    """
    metrics = ["R2", "MAE", "RMSE"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#1f77b4", "#ff7f0e", "#d62728"]

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.8)
        # Annotate values on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title("Model Performance Comparison (Test Set — Real $ Scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")