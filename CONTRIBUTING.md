# Contributing

Use Python 3.11–3.13, create an isolated environment, then install
`.[dev,research,qwen]`. Run `pytest`, `ruff check .`, `ruff format --check .`, and
`python -m build` before proposing a change. Keep clinical/dataset adapters out of
reusable routing, fusion, federation, and consultation modules.

Do not commit datasets, patient-level records, credentials, signing keys, model
weights, checkpoints, caches, or generated `artifacts/`. Preserve patient/group
alignment and train/validation/test boundaries. New model selection must finish
before test evaluation. Flower communication must remain genuine and selected-
only; consultations must not transmit private training rows or specialist weights.

Changes need typed public APIs, targeted tests, documentation of limitations, and
a dated execution report. Contributors must have the right to license submissions.
Use the repository issue tracker for non-sensitive reports:
`https://github.com/saadbinather/SwarmFL/issues`. Follow `SECURITY.md` for private
vulnerability reporting.
