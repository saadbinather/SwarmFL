# SwarmFL

SwarmFL is an experimental federated multi-agent learning research framework for
specialist registration, selected-only Flower consultations, cost-aware routing,
dynamic fusion, signed delayed feedback, and version-controlled federated updates.

> **Alpha research software:** SwarmFL is not a medical device, clinical decision
> system, privacy guarantee, or secure multi-hospital deployment.

Distribution: `swarmfl` · Python package: `fed_med_agents` · Version: `0.1.0a1`

## Architecture overview

```text
origin specialist
      │
      ├─ registry + authorization + availability/cost contracts
      ├─ router ──> STOP
      │       └──> one selected peer ── Flower request/reply
      ├─ dynamic fusion of actually observed predictions
      └─ signed prediction ── delayed verified outcome
                              └─ shared router/fusion update
                                  └─ validate → activate/reject → rollback
```

Private specialists retain their own preprocessing and checkpoints. Consultations
do not send private training rows or specialist weights. Shared Flower updates
optimize compatible router/fusion parameters, not private specialist models.
Logical registry membership is distinct from physical Flower federation membership.
Read the full [technical architecture](docs/architecture.md).

## Key capabilities

- Typed external specialist declarations and compatibility checks.
- Logical registration, availability, authorization, and consultation budgets.
- Fixed, contextual MLP, and graph-inspired cost-aware routing.
- Genuine selected-only Flower communication with STOP support.
- Dynamic mean, reliability-weighted, and DeepSets fusion.
- Signed prediction/outcome separation and delayed verified feedback.
- Federated shared-parameter optimization with immutable versions and rollback.
- CPU-only synthetic onboarding without CDC data, GPU, Ollama, or private weights.

## Installation

From a source checkout:

```bash
python -m pip install .
```

From a built wheel:

```bash
python -m pip install dist/swarmfl-0.1.0a1-py3-none-any.whl
```

After the package is actually published to PyPI, installation will be:

```bash
python -m pip install swarmfl
```

The PyPI command is a post-publication instruction; `swarmfl` has not yet been
uploaded. Supported Python versions are 3.11–3.13. Optional historical research
dependencies are available through `.[research]`; Qwen HTTP integration is
`.[qwen]`; contributor tooling is `.[dev]`.

## Five-minute CPU quickstart

Run from a directory outside the source checkout:

```bash
swarmfl-quickstart --output demo --feedback
```

The command generates synthetic records, creates three independent NumPy
specialists, starts three real CPU Flower clients, registers their contracts,
selects at most one peer per case, transfers only authorized synthetic fields,
fuses observed predictions, verifies one signed delayed outcome, activates a
candidate version, and rolls back. Results are written to `demo/result.json`.
The synthetic task has no clinical meaning.

Flower warns that its Python `run_simulation` launcher is deprecated. SwarmFL pins
the tested Flower 1.40.0 path while a CLI migration remains separate work. Ray on
Windows is experimental. Ubuntu GitHub Actions passed tests, build, Twine checks,
and installed-wheel smoke validation on 2026-10-16.

## Custom specialist integration

```python
from fed_med_agents import SpecialistRegistry
from fed_med_agents.collaboration.registry import Specialist

registry = SpecialistRegistry()
specialist: Specialist = build_your_declaration()  # application-defined
registry.register(specialist)
```

A declaration binds task, output, feature, adapter, checkpoint, capability, cost,
and availability metadata. Registration does **not** physically join Flower. The
application must separately run a compatible `ClientApp`, discover its node ID,
and bind that transport identity.

See [`examples/custom_specialist.py`](examples/custom_specialist.py) for a complete
declaration and [`fed_med_agents.quickstart`](src/fed_med_agents/quickstart.py) for
an end-to-end synthetic transport implementation.

## Research results

The consolidated [verified results](research/results.md) report covers CDC diabetes
development research, routing and fusion comparisons, continual feedback cycles,
Step 8A external specialists, and installed-wheel validation. The
[research index](research/README.md) explains what is public and what remains
intentionally private.

Key findings remain narrow. At 160,000 CDC development routing records,
fixed/MLP/graph objectives were 0.350173/0.350110/0.350173, while MLP utility R²
was 0.000003; learned-routing superiority was not established. Dynamic fusion
results are promising development diagnostics, not matched-protocol clinical
evidence. Sequential validation populations differ and do not demonstrate
sustained continual improvement.

## Documentation

| Resource | Purpose |
|---|---|
| [Architecture](docs/architecture.md) | Technical components, contracts, and data flow |
| [Limitations](docs/limitations.md) | Scientific, privacy, security, and operational boundaries |
| [Feedback learning](docs/feedback-learning.md) | Delayed-outcome and shared-update semantics |
| [Current status](docs/status.md) | Verified milestone and release state |
| [Research index](research/README.md) | Public research organization and evidence policy |
| [Verified results](research/results.md) | Numerical findings and provenance |
| [Milestone reports](docs/experiments/) | Detailed historical methodology |
| [Contributing](CONTRIBUTING.md) | Development and evidence requirements |
| [Security](SECURITY.md) | Reporting route and known boundaries |

Generated artifacts, CDC datasets, private checkpoints, frozen evaluation files,
Qwen weights, caches, and local run logs are excluded from Git and distributions.
Historical artifact paths provide provenance, not a promise that private evidence
is downloadable.

## Security and limitations

SwarmFL currently operates as a trusted local simulation. Predictions, identifiers,
timing, metadata, logs, and shared updates can disclose information. It provides no
formal differential privacy, secure aggregation, malicious-client resistance,
transport isolation, HIPAA/GDPR compliance, or confidentiality guarantee.

Flower 1.40.0's simulation extra pins Ray 2.55.1, which has a high-severity unsafe
deserialization advisory in `ray.data.read_webdataset`. SwarmFL does not use Ray
Data and accepts no WebDataset/TAR input. The owner accepts this residual dependency
risk for this trusted-local alpha only. See [SECURITY.md](SECURITY.md).

The frozen CDC test container and ID file have not been opened, loaded, hashed,
modified, or evaluated in the relevant research sequence. There is no independent
multi-hospital validation, established clinical utility, general predictive
superiority, or physical dynamic Flower membership. Read
[limitations](docs/limitations.md) before interpreting results.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Do not describe an
unarchived alpha as a peer-reviewed publication or assign it a DOI unless one is
actually issued.

SwarmFL is licensed under the [Apache License 2.0](LICENSE). Dependencies, datasets,
and separately obtained model weights remain subject to their own terms. See
[`NOTICE`](NOTICE).
