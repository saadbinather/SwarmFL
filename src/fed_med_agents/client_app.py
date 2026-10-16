import json

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .data import ARTIFACTS, GROUPS, load, view
from .evaluate import metrics
from .models import probabilities, shared_model, train

app = ClientApp()


@app.train()
def train_client(msg: Message, context: Context):
    client = list(GROUPS)[int(context.node_config["partition-id"])]
    round_id = int(msg.content["config"]["server-round"])
    torch.manual_seed(42 + round_id)
    model = shared_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    data = load()
    x, y = view(data, client, "train", shared=True)
    loss = train(model, x, y, seed=42 + round_id)
    delta = sum(float(torch.sum((v - initial[k]) ** 2)) for k, v in model.state_dict().items()) ** 0.5
    vx, vy = view(data, client, "val", shared=True)
    scores = metrics(vy, probabilities(model, vx))
    audit = {
        "client": client,
        "round": round_id,
        "num_examples": len(y),
        "features": GROUPS[client],
        "last_batch_loss": loss,
        "shared_weight_l2_change": delta,
        "validation": scores,
    }
    directory = ARTIFACTS / f"client_{client}"
    directory.mkdir(exist_ok=True)
    (directory / f"round_{round_id}.json").write_text(json.dumps(audit, indent=2))
    torch.save(model.state_dict(), directory / "shared_last.pt")
    print(json.dumps(audit), flush=True)
    # Only the shared state_dict and aggregate scalar metrics enter Flower messages.
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "num-examples": len(y),
                        "client-index": int(context.node_config["partition-id"]),
                        "round": round_id,
                        "weight-change": delta,
                        **scores,
                    }
                ),
            }
        ),
        reply_to=msg,
    )
