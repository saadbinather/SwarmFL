"""Reject internal or private material from SwarmFL distribution archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PROHIBITED_PREFIXES = (
    "fed_med_agents/experiments/",
    "src/fed_med_agents/experiments/",
    "scripts/",
    "tests/",
    "docs/devlog/",
    "artifacts/",
    "data/",
    "datasets/",
    "checkpoints/",
    "models/",
    "outputs/",
    "research/runs/",
    "research/artifacts/",
    "research/checkpoints/",
    "research/logs/",
)
PROHIBITED_PATHS = {
    "research/results.md",
}
PROHIBITED_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".joblib",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".sqlite",
    ".sqlite3",
    ".safetensors",
    ".h5",
    ".hdf5",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".log",
)
REQUIRED_WHEEL = {
    "fed_med_agents/__init__.py",
    "fed_med_agents/quickstart.py",
    "fed_med_agents/collaboration/registry.py",
}
REQUIRED_SDIST = {
    "pyproject.toml",
    "README.md",
    "examples/custom_specialist.py",
    "research/README.md",
}


def normalize(name: str) -> str:
    """Remove an sdist root directory from an archive member name."""
    parts = PurePosixPath(name).parts
    if parts and parts[0].startswith("swarmfl-"):
        parts = parts[1:]
    return "/".join(parts)


def is_prohibited(name: str) -> bool:
    """Return whether an archive member violates the public package boundary."""
    path = normalize(name)
    filename = PurePosixPath(path).name
    return (
        path in PROHIBITED_PATHS
        or any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in PROHIBITED_PREFIXES)
        or path.lower().endswith(PROHIBITED_SUFFIXES)
        or filename == ".env"
        or (filename.startswith(".env.") and filename != ".env.example")
    )


def audit(directory: Path) -> tuple[int, int]:
    """Audit the single wheel and sdist in *directory* and return member counts."""
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"Expected one wheel and one sdist, found {wheels!r} and {sdists!r}")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = archive.getnames()

    for archive, names in ((wheels[0], wheel_names), (sdists[0], sdist_names)):
        rejected = sorted(name for name in names if is_prohibited(name))
        if rejected:
            raise ValueError(f"Prohibited content in {archive}: {rejected}")

    normalized_wheel = {normalize(name) for name in wheel_names}
    normalized_sdist = {normalize(name) for name in sdist_names}
    if missing := REQUIRED_WHEEL - normalized_wheel:
        raise ValueError(f"Required wheel content missing: {sorted(missing)}")
    if missing := REQUIRED_SDIST - normalized_sdist:
        raise ValueError(f"Required sdist content missing: {sorted(missing)}")
    return len(wheel_names), len(sdist_names)


def main() -> None:
    """Run the distribution archive audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing one wheel and one sdist")
    args = parser.parse_args()
    wheel_count, sdist_count = audit(args.directory)
    print(f"Audited {wheel_count} wheel entries and {sdist_count} sdist entries")


if __name__ == "__main__":
    main()
