import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def compute_classification_metrics(y_true: list, y_pred: list, y_probs: list = None) -> dict:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    fpr, tpr = None, None
    if y_probs is not None:
        try:
            # y_probs contains positive class probability
            pos_probs = [p[1] if isinstance(p, (list, np.ndarray)) else p for p in y_probs]
            auc = roc_auc_score(y_true, pos_probs)
            fpr, tpr, _ = roc_curve(y_true, pos_probs)
            fpr = fpr.tolist()
            tpr = tpr.tolist()
        except Exception:
            auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr
    }

def benchmark_inference_latency(predict_fn, sample_texts: list[str], runs: int = 3) -> float:
    """
    Measures mean inference latency per sample in milliseconds.
    """
    latencies = []
    for _ in range(runs):
        start_time = time.perf_counter()
        _ = predict_fn(sample_texts)
        end_time = time.perf_counter()
        elapsed_ms = ((end_time - start_time) / len(sample_texts)) * 1000.0
        latencies.append(elapsed_ms)
    return round(float(np.mean(latencies)), 2)

def count_model_parameters(model) -> dict:
    """Returns total trainable and non-trainable parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }

def get_directory_size_mb(path: str) -> float:
    """Calculates size of model file or directory in Megabytes."""
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)
