import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from . import CLASS_NAMES

def evaluate_predictions(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    result = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }
    try:
        result["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        result["macro_roc_auc_ovr"] = None
    return result

def save_metrics(metrics, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
