"""Explicit CPU/CUDA selection, synchronization and measured device audits."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def select_device(request: str = "auto") -> torch.device:
    """Resolve auto, or fail explicitly when requested CUDA is unavailable."""
    if request not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device {request}")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; inspect torch.version.cuda and nvidia-smi")
    return torch.device("cuda" if request != "cpu" and torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


def audit(path: Path | None = None) -> dict[str, Any]:
    """Run a real CUDA matrix operation, not just a device availability check."""
    result: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi"], text=True, encoding="utf-8", errors="replace", timeout=20
        )
    except (OSError, subprocess.SubprocessError) as error:
        result["nvidia_smi_error"] = str(error)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        x = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
        output = x @ x.T
        synchronize(device)
        result.update(
            device_name=properties.name,
            vram_bytes=properties.total_memory,
            tensor_operation_checksum=float(output.sum().cpu()),
            peak_allocated_bytes=peak_memory(device),
        )
        assert torch.allclose(output.cpu(), x.cpu() @ x.cpu().T)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
