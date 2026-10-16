"""Real Flower inference workflow for SLM and matched routing controls."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from ..devices import select_device
from ..evaluate import metrics
from ..feedback.versions import PolicyRegistry
from ..fusion import CollaborativeModel
from .client_app import configuration
from .coordinator import AgentTransport, CoordinationSession, coordinate
from .provider import OllamaProvider
from .schemas import Capability, PatientRequest

app = ServerApp()


def evaluate(results: list[dict], labels: dict[str, int]) -> dict:
    valid = [result for result in results if result["probability"] is not None]
    y = np.asarray([labels[result["patient_id"]] for result in valid]) if labels else np.array([])
    p = np.asarray([result["probability"] for result in valid])
    scores = (
        metrics(y, p)
        if len(np.unique(y)) == 2
        else {key: None for key in ("accuracy", "auroc", "f1", "binary_cross_entropy")}
    )
    times = [result["latency_seconds"] for result in results]
    return scores | {
        "requests": len(results),
        "successful_predictions": len(valid),
        "prediction_coverage": len(valid) / len(results),
        "routing_validity": sum(result["routing_valid"] for result in results) / len(results),
        "average_consultations": float(np.mean([len(result["consulted"]) for result in results])),
        "tool_failures": sum(result["tool_failures"] for result in results),
        "single_patient_latency_seconds": {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "max": max(times),
        },
        "failed_predictions_counted_wrong_accuracy": (
            sum((result["probability"] >= 0.5) == labels[result["patient_id"]] for result in valid)
            / len(results)
        )
        if labels
        else None,
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = configuration()
    output = Path(os.environ["FED_MED_M3_DIR"])
    provider = OllamaProvider(**config["provider"])
    device = select_device(os.environ.get("FED_MED_DEVICE", "auto"))
    torch.set_num_threads(1)
    checkpoint = Path(config["checkpoint_dir"]) / "shared_best.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = CollaborativeModel(len(config["clients"]), config["budget"]).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    model.requires_grad_(False)
    nodes, capabilities = {}, []
    deadline = time.perf_counter() + 60
    while len(list(grid.get_node_ids())) < len(config["clients"]):
        if time.perf_counter() > deadline:
            raise RuntimeError("Agent clients did not connect")
        time.sleep(0.1)
    for node in grid.get_node_ids():
        message = Message(
            RecordDict(), dst_node_id=node, message_type="query.agent_capabilities", group_id="agents"
        )
        replies = list(grid.send_and_receive([message], timeout=60))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Agent capability handshake failed")
        cap = Capability.model_validate_json(replies[0].content["capability"]["json"])
        if cap.specialist in nodes:
            raise ValueError("Duplicate specialist")
        nodes[cap.specialist] = node
        capabilities.append(cap)
    capabilities.sort(key=lambda cap: cap.index)
    if [cap.index for cap in capabilities] != list(range(len(config["clients"]))):
        raise ValueError("Noncontiguous client identity mapping")
    manifest = json.loads((output / "requests.json").read_text())
    # Evaluation labels remain outside all model prompts and tool inputs.
    labels = manifest["labels"]
    results, traces = [], []
    realized_budgets = {}
    registry = PolicyRegistry(Path(config["policy_registry"])) if config.get("policy_registry") else None
    transport = AgentTransport(grid, nodes)
    for item in manifest["requests"]:
        request = PatientRequest.model_validate(item["request"])
        policy = item["policy"]
        key = (request.patient_id, request.mode)
        if config.get("match_realized_budget", False) and policy != "slm":
            request = request.model_copy(update={"budget": realized_budgets[key]})
        trace_offset = len(transport.trace)
        active_version = registry.current() if registry else None
        active = registry.load(active_version, device) if registry else model
        active.requires_grad_(False)
        session = CoordinationSession(request, capabilities, transport, active, device, policy, registry)
        session.policy_version = active_version
        agent_trace = coordinate(session, provider, config["max_agent_steps"])
        result = session.finished.model_dump()
        if policy == "slm":
            realized_budgets[key] = len(result["consulted"])
        results.append(result)
        traces.append(
            {
                "request_id": request.request_id,
                "patient_id": request.patient_id,
                "mode": request.mode,
                "policy": policy,
                "approved_policy_version": session.policy_version,
                "coordinator_trace": agent_trace,
                "communication_trace": transport.trace[trace_offset:],
                "final": result,
            }
        )
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        with (output / "traces.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(traces[-1]) + "\n")
        print(
            f"M3 request={request.request_id} mode={request.mode} policy={policy} status={result['status']} "
            f"consulted={result['consulted']} probability={result['probability']} latency={result['latency_seconds']:.3f}s",
            flush=True,
        )
    tables = {}
    for mode in sorted({item["request"]["mode"] for item in manifest["requests"]}):
        tables[mode] = {}
        for policy in sorted({item["policy"] for item in manifest["requests"]}):
            selected = [
                result
                for result, item in zip(results, manifest["requests"], strict=True)
                if item["policy"] == policy and item["request"]["mode"] == mode
            ]
            tables[mode][policy] = evaluate(selected, labels)
    unchanged = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    if not unchanged:
        raise RuntimeError("Fusion checkpoint was modified")
    output_metrics = {
        "flower_run_id": context.run_id,
        "source_split": manifest["source_split"],
        "patients": len({item["request"]["patient_id"] for item in manifest["requests"]}),
        "metrics": tables,
        "fusion_checkpoint_sha256": before,
        "fusion_checkpoint_reference": "historical-M2-source" if registry else "active-fusion",
        "fusion_unchanged": unchanged,
        "policy_registry": str(registry.path) if registry else None,
        "tabular_device": str(device),
        "provider": provider.audit(),
        "capabilities": [cap.model_dump() for cap in capabilities],
        "prompt_config": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "max_steps": config["max_agent_steps"],
        },
        "model_training": "frozen pretrained Qwen; inference only; no SLM weight updates",
        "match_realized_budget": config.get("match_realized_budget", False),
    }
    (output / "metrics.json").write_text(json.dumps(output_metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"flower_run_id": context.run_id, "metrics": tables, "fusion_unchanged": unchanged}, indent=2
        ),
        flush=True,
    )
