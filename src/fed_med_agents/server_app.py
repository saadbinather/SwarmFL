import json
import platform
from importlib.metadata import version

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .data import ARTIFACTS, GROUPS, load
from .evaluate import evaluate_shared
from .models import shared_model

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rounds = int(context.run_config.get("num-server-rounds", 5))
    if rounds != 5:
        raise ValueError("Milestone 1 requires exactly five rounds")
    model = shared_model()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, ARTIFACTS / "shared_initial.pt")
    history = {}
    received = {c: [] for c in GROUPS}
    data = load()

    def evaluate(round_id, arrays):
        model.load_state_dict(arrays.to_torch_state_dict())
        scores = evaluate_shared(model, data, "val")
        history[str(round_id)] = scores
        return MetricRecord(scores["mean_probability_ensemble"])

    def aggregate(records, weight_key):
        reports = [record["metrics"] for record in records]
        clients = [int(report["client-index"]) for report in reports]
        if sorted(clients) != [0, 1, 2] or any(report["weight-change"] <= 0 for report in reports):
            raise RuntimeError("Missing, duplicate, or unchanged client update")
        for report in reports:
            received[list(GROUPS)[int(report["client-index"])]].append(int(report["round"]))
        total = sum(report[weight_key] for report in reports)
        return MetricRecord(
            {
                key: sum(report[key] * report[weight_key] for report in reports) / total
                for key in ("accuracy", "auroc", "f1", "binary_cross_entropy")
            }
        )

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=3,
        min_available_nodes=3,
        train_metrics_aggr_fn=aggregate,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        num_rounds=5,
        train_config=ConfigRecord({}),
        evaluate_fn=evaluate,
    )
    final = result.arrays.to_torch_state_dict()
    model.load_state_dict(final)
    torch.save(final, ARTIFACTS / "shared_final.pt")
    delta = sum(float(torch.sum((final[k] - initial[k]) ** 2)) for k in initial) ** 0.5
    participation = {
        c: sorted(int(p.stem.split("_")[1]) for p in (ARTIFACTS / f"client_{c}").glob("round_*.json"))
        for c in GROUPS
    }
    if (
        any(rounds != [1, 2, 3, 4, 5] for rounds in participation.values())
        or participation != received
        or delta <= 0
    ):
        raise RuntimeError(f"Verification failed: {participation}, delta={delta}")
    output = {
        "validation_history": history,
        "aggregated_client_validation_metrics": {
            str(k): dict(v) for k, v in result.train_metrics_clientapp.items()
        },
        "test": evaluate_shared(model, data, "test"),
        "verification": {
            "client_rounds": participation,
            "received_client_rounds": received,
            "shared_weight_l2_change": delta,
            "flower_version": version("flwr"),
            "torch_version": version("torch"),
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "device": "cpu",
        },
    }
    (ARTIFACTS / "federated_metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    from flwr.simulation import run_simulation

    from .client_app import app as client_app

    run_simulation(
        server_app=app,
        client_app=client_app,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
            "init_args": {"num_cpus": 3, "include_dashboard": False},
        },
    )
