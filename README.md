 <div align="center">
  <img src="resources/swarmfl-logo-transparent.png" alt="SwarmFL" width="600">
</div>

[![CI](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml/badge.svg)](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/saadbinather/SwarmFL?color=blue&style=flat)](https://github.com/saadbinather/SwarmFL/releases)
[![PyPI](https://img.shields.io/pypi/v/swarmfl?color=blue&style=flat)](https://pypi.org/project/swarmfl/)
[![Python](https://img.shields.io/pypi/pyversions/swarmfl?color=blue&style=flat)](https://pypi.org/project/swarmfl/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)](LICENSE)
[![Docs](https://img.shields.io/badge/view-docs-blue?style=flat)](https://swarmfl.readthedocs.io/en/latest/)
**SwarmFL** is an experimental Python framework for federated multi-agent learning. It enables independently managed specialists to collaborate through Flower, with cost-aware consultation, dynamic prediction fusion, feedback-driven updates, and version control.


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


## Authors

Maintained by [Saad](https://github.com/saadbinather), [Ali Hassan](https://github.com/Alihassan0155), and [Saif](https://github.com/rajasaifr).

## License

Licensed under the [Apache License 2.0](LICENSE).
