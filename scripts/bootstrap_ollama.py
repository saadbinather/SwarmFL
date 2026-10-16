"""Download a portable Windows Ollama runtime into gitignored artifacts; no OS install."""

from __future__ import annotations

import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts" / "ollama"
URL = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"


def main() -> None:
    if os.name != "nt":
        raise SystemExit("Use an existing Ollama install on Linux; this bootstrap is Windows portable only")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    executable = RUNTIME / "ollama.exe"
    if not executable.exists():
        archive = RUNTIME / "runtime.zip"
        with requests.get(URL, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            (RUNTIME / "download.json").write_text(
                json.dumps(
                    {"url": response.url.split("?")[0], "bytes": response.headers.get("content-length")},
                    indent=2,
                )
            )
            with archive.open("wb") as file:
                for chunk in response.iter_content(2**20):
                    file.write(chunk)
        root = RUNTIME.resolve()
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError("Unsafe archive member")
            package.extractall(root)
        print("Portable runtime extracted", flush=True)
    environment = os.environ | {
        "OLLAMA_HOST": "127.0.0.1:11435",
        "OLLAMA_MODELS": str(RUNTIME / "models"),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    try:
        requests.get("http://127.0.0.1:11435/api/version", timeout=2).raise_for_status()
    except requests.RequestException:
        # CREATE_NO_WINDOW avoids a visible helper window; logs are persisted.
        with (RUNTIME / "serve.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(executable), "serve"],
                env=environment,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            (RUNTIME / "service_pid.txt").write_text(str(process.pid))
        for _ in range(60):
            try:
                requests.get("http://127.0.0.1:11435/api/version", timeout=2).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Ollama did not start; inspect artifacts/ollama/serve.log")
    with (RUNTIME / "pull.log").open("a", encoding="utf-8") as pull_log:
        subprocess.run(
            [str(executable), "pull", "qwen2.5:1.5b-instruct"],
            env=environment,
            stdout=pull_log,
            stderr=pull_log,
            check=True,
        )
    print(requests.get("http://127.0.0.1:11435/api/tags", timeout=10).json())


if __name__ == "__main__":
    main()
