import json

import torch

from .data import ARTIFACTS, GROUPS, load, view
from .evaluate import metrics
from .models import Classifier, probabilities, train


def run():
    data = load()
    results = {}
    for i, (client, features) in enumerate(GROUPS.items()):
        torch.manual_seed(42 + i)
        model = Classifier(len(features))
        x, y = view(data, client, "train")
        loss = train(model, x, y, epochs=30, seed=42 + i)
        directory = ARTIFACTS / f"client_{client}"
        directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), directory / "specialist.pt")
        results[client] = {"features": features, "last_batch_loss": loss}
        for split in ("val", "test"):
            x, y = view(data, client, split)
            results[client][split] = metrics(y, probabilities(model, x))
        print(client, json.dumps(results[client]))
    (ARTIFACTS / "specialists_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
