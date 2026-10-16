import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from .data import ARTIFACTS, GROUPS, load, view
from .models import probabilities, shared_model


def metrics(y, p):
    p = np.asarray(p, dtype=np.float64)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auroc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
        "binary_cross_entropy": float(log_loss(y, p, labels=[0, 1])),
    }


def evaluate_shared(model, data, split):
    predictions = []
    result = {}
    for client in GROUPS:
        x, y = view(data, client, split, shared=True)
        p = probabilities(model, x)
        predictions.append(p)
        result[client] = metrics(y, p)
    result["mean_probability_ensemble"] = metrics(y, np.mean(predictions, axis=0))
    return result


if __name__ == "__main__":
    model = shared_model()
    model.load_state_dict(torch.load(ARTIFACTS / "shared_final.pt", weights_only=True))
    print(json.dumps(evaluate_shared(model, load(), "test"), indent=2))
