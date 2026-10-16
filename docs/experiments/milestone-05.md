# Milestone 5 release-candidate summary

Milestone 5 Steps 1–8A remain historical evidence. Step 8B packages reusable
components without changing results or promoting a research candidate. It adds a
synthetic CPU quickstart and does not access, evaluate, hash, or redistribute the
frozen CDC test, CDC records, private checkpoints, or Qwen weights.

Step 8A run `11798885370385872129` remains the comprehensive validation evidence:
three CPU Flower clients, selected-only consultation, signed feedback, federated
updates, approval/rejection/rollback, 195 passed/3 skipped, and 12,881 preserved
hashes. Those are not Step 8B rerun results. Step 8B evidence belongs in its dated
devlog and `artifacts/m5-step08b-v1/`.

Claims remain narrow: integration/lifecycle behavior in trusted local simulations.
Routing superiority, continual improvement, clinical utility, secure deployment,
differential privacy, secure aggregation, and confidentiality remain unproven.

## Step 8B local release candidate

The `swarmfl` 0.1.0a1 wheel and sdist build and pass Twine validation. A wheel
installed into a fresh venv was invoked outside the repository: three actual CPU
Flower clients returned selected-only synthetic predictions, signed feedback was
verified, and version rollback completed. QA is 196 passed/4 explicitly skipped;
Ruff and `pip check` pass. Distribution inspection found no private research assets.

This is a **CONDITIONAL GO** for an alpha, not a publication authorization. License
ownership/full text, repository and security URLs, Git history, dependency advisory
risk, final PyPI availability, and Linux CI remain pre-publication conditions. The
Hugging Face material is a synthetic demonstration card, not a released CDC model.
Exact commands, hashes, failures, and limitations are in the dated Step 8B devlog.

## Step 8C release blockers

Release metadata now points to the canonical GitHub repository, local Git is
initialized on `main` with the correct `origin`, and an explicit 137-file public
manifest is audited but unstaged. Flower 1.40.0 upgrades cryptography to 50.0.2;
isolated builds use setuptools 84. The remaining audit finding is Flower-pinned
Ray 2.55.1/PYSEC-2026-2273 in `ray.data.read_webdataset`, which SwarmFL does not
call or expose. This is documented residual package risk, not a claimed fix.

The rebuilt wheel passes a fully isolated Python 3.12 install and genuine
three-client CPU Flower quickstart outside the repository. Regression remains
196 passed/4 skipped. Apache-2.0 and PEP 639 metadata remain pending explicit
ownership confirmation, and actual GitHub-hosted Linux CI remains unverified.
No research model or CDC experiment was opened or rerun.
