# SwarmFL

SwarmFL is a Python research framework for federated specialist systems. It
provides typed specialist registration, cost-aware routing, selected-only Flower
consultations, dynamic prediction fusion, signed delayed feedback, federated
router and fusion updates, and versioned activation and rollback.

## Badges

[![CI](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml/badge.svg)](https://github.com/saadbinather/SwarmFL/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/swarmfl.svg)](https://pypi.org/project/swarmfl/)
[![PyPI](https://img.shields.io/pypi/v/swarmfl.svg)](https://pypi.org/project/swarmfl/)

The PyPI badge shows the currently published release. Version `0.1.0` is prepared
in this repository but remains unreleased until its release tag is approved and
published.

## Features

- Register heterogeneous specialists with explicit task, output, feature,
  checkpoint, capability, availability, authorization, and cost contracts.
- Route each request to STOP or at most one eligible peer using fixed,
  contextual MLP, or graph-inspired policies.
- Exchange only selected consultation requests through genuine Flower messages.
- Fuse changing evidence sets with mean, reliability-weighted, or
  permutation-invariant DeepSets fusion.
- Separate signed predictions from delayed, authenticated outcomes.
- Federate compatible shared router and fusion parameters without transmitting
  private specialist weights or training rows.
- Validate, activate, recover, and exactly roll back immutable policy versions.
- Run a self-contained three-client CPU demonstration with synthetic data.

## Installation

Install the public package:

```bash
pip install swarmfl
```

Until `0.1.0` is published, this command installs the latest public version shown
by the PyPI badge. To work from the current source:

```bash
git clone https://github.com/saadbinather/SwarmFL.git
cd SwarmFL
pip install -e .
```

SwarmFL supports Python 3.11–3.13. Research and contributor dependencies are
available through the `research`, `qwen`, and `dev` optional dependency groups.

## Quickstart

Run the verified synthetic demonstration from an isolated working directory:

```bash
swarmfl-quickstart --output demo --feedback
```

The command starts three CPU Flower clients, registers independent synthetic
specialists, performs selected-only consultations, fuses predictions, verifies a
signed delayed outcome, activates a candidate version, and rolls back. Inspect
`demo/result.json` for node identifiers, routing decisions, predictions, feedback,
and version transitions. The generated task has no clinical meaning.

To integrate an application-owned model, declare its contract and keep execution
behind your adapter. Registration does not load a checkpoint or dynamically add a
physical Flower client. See the
[complete custom-specialist example](examples/custom_specialist.py).

## Documentation

- [Architecture](docs/architecture.md)
- [Custom specialist example](examples/custom_specialist.py)
- [Research](research/README.md)
- [Verified results](research/results.md)
- [Limitations](docs/limitations.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Research

Public reports cover CDC diabetes development experiments, routing and fusion
comparisons, continual federated feedback cycles, and synthetic external-specialist
validation. Detailed results and provenance are in
[`research/results.md`](research/results.md). Private datasets, checkpoints,
frozen evaluation material, and generated artifacts are not distributed.

Installable distributions contain the reusable framework and public examples,
not repository-only experiment runners, tests, private evidence, or development
artifacts. Historical research implementations remain in the source repository.

The evidence does not establish clinical utility, learned-routing superiority,
confidentiality, differential privacy, or secure multi-hospital deployment. Read
the [limitations](docs/limitations.md) before interpreting results.

## Authors / Maintainers

SwarmFL is maintained by [Saad Bin Ather](https://github.com/saadbinather).
Contributions are welcome under the repository's [contribution guidelines](CONTRIBUTING.md).

## License

SwarmFL is licensed under the [Apache License 2.0](LICENSE). Dependencies,
datasets, and separately obtained model weights remain subject to their own terms;
see [NOTICE](NOTICE).
