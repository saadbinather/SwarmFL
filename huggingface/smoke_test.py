"""Manual smoke test for the public Hugging Face demo.

This file is intentionally not run as part of the package test suite. Execute it
from a clean environment only when checking the public Hub release.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="swarmfl-hf-smoke-") as directory:
        output = Path(directory) / "demo"
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("demo.py")), "--output", str(output), "--feedback"],
            check=True,
        )
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert result["flower_clients"] == 3
        assert len(result["predictions"]) == 3
        assert result["feedback"]["authenticated"] is True
        print("PASS: public Hugging Face demo smoke test")


if __name__ == "__main__":
    main()
