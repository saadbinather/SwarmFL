# Research @ SwarmFL

This directory summarizes the research, experiments, and validation supporting
SwarmFL, an experimental federated multi-agent learning framework.

The public repository contains verified methodological summaries and
reproducibility guidance. Some datasets, private specialist checkpoints, and
detailed experiment artifacts are intentionally excluded because of size,
licensing, privacy, or reproducibility constraints. Their absence is deliberate:
public summaries do not turn private historical evidence into distributable data.

## Research map

- [Verified results](results.md) — consolidated numerical results, provenance,
  engineering validations, and comparability boundaries.
- [Architecture](../docs/architecture.md) — specialist, registry, consultation,
  routing, fusion, feedback, federation, and version-control design.
- [Milestone methodology](../docs/experiments/milestone-05.md) — public index of
  the latest experimental sequence and historical evidence boundaries.
- [Limitations](../docs/limitations.md) — scientific, privacy, security, and
  operational constraints.
- [Synthetic quickstart](../README.md#five-minute-cpu-quickstart) — the publicly
  supported reproducible demonstration requiring no private data or checkpoint.

Detailed historical reports remain under [`docs/experiments/`](../docs/experiments/).
Generated `artifacts/`, CDC data, private model checkpoints, frozen evaluation
material, Qwen weights, and local run logs are not part of the public release.

## Evidence policy

Every number in `results.md` cites a repository report or an ignored
machine-readable artifact path. Artifact paths document provenance for the local
research workspace; they do not imply that those artifacts are distributed.
Results are labeled as development, retrospective, synthetic, or engineering
evidence. No result is presented as clinical validation or a privacy guarantee.
