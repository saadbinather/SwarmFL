"""Run the small SwarmFL Flower demo from the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SwarmFL synthetic Flower demo")
    parser.add_argument("--output", type=Path, default=Path("swarmfl-demo"))
    parser.add_argument("--feedback", action="store_true")
    args = parser.parse_args()

    command = shutil.which("swarmfl-quickstart")
    if command is None:
        raise SystemExit("Install first: pip install swarmfl==0.1.0")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose another --output")

    cmd = [command, "--output", str(args.output)]
    if args.feedback:
        cmd.append("--feedback")
    subprocess.run(cmd, check=True)
    result = json.loads((args.output / "result.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "flower_clients": result.get("flower_clients"),
        "predictions": len(result.get("predictions", [])),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
