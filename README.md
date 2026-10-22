 <div align="center">
  <img src="resources/swarmfl-logo-transparent.png" alt="SwarmFL" width="600">
</div>

[![CI](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml/badge.svg)](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/swarmfl)](https://pypi.org/project/swarmfl/)
[![Python](https://img.shields.io/pypi/pyversions/swarmfl)](https://pypi.org/project/swarmfl/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-Read%20the%20Docs-7B42F6?logo=readthedocs&logoColor=white)](https://swarmfl.readthedocs.io/en/latest/)

**SwarmFL** is an experimental Python framework for federated multi-agent learning. It enables independently managed specialists to collaborate through Flower, with cost-aware consultation, dynamic prediction fusion, feedback-driven updates, and version control.

## Features

- **Specialist Registry:** Register heterogeneous specialists with explicit compatibility and authorization contracts.
- **Intelligent Routing:** Select eligible specialists using fixed, learned, or graph-inspired routing policies under consultation costs.
- **Federated Collaboration:** Execute selected-only consultations using Flower.
- **Dynamic Fusion:** Combine predictions through mean, reliability-weighted, or DeepSets fusion.
- **Feedback Learning:** Support authenticated delayed feedback and federated shared-parameter updates.
- **Version Control:** Validate, activate, reject, and roll back learned policies.

## Installation

Install from PyPI:

```bash
pip install swarmfl
```

SwarmFL v0.1.0 is available from PyPI.

For source installation:

```bash
git clone https://github.com/saadbinather/SwarmFL.git
cd SwarmFL
pip install -e .
```

Python 3.11–3.13 is supported. A virtual environment is recommended.

## Quickstart

Run the synthetic CPU demonstration:

```bash
swarmfl-quickstart --output demo --feedback
```

The example runs three Flower clients and demonstrates selected-only consultation, prediction fusion, feedback verification, version activation and rollback.

Results are saved to `demo/result.json`.

## Documentation

📚 **[Explore the SwarmFL Documentation](https://swarmfl.readthedocs.io/en/latest/)**

- [Getting Started](https://swarmfl.readthedocs.io/en/latest/start/index.html)
- [Installation](https://swarmfl.readthedocs.io/en/latest/installation.html)
- [Quickstart](https://swarmfl.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://swarmfl.readthedocs.io/en/latest/api.html)
- [Research](https://swarmfl.readthedocs.io/en/latest/research/index.html)

## Research

SwarmFL includes research implementations and experimental results for federated specialist collaboration.

Some datasets, model checkpoints, and internal scripts are intentionally excluded from public distributions.

SwarmFL is research software and is not validated for clinical deployment or formal privacy guarantees.


## Authors

Maintained by [Saad](https://github.com/saadbinather), [Ali Hassan](https://github.com/Alihassan0155), and [Saif](https://github.com/rajasaifr).

## License

Licensed under the [Apache License 2.0](LICENSE).
