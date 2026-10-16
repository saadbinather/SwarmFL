# CDC three-specialist amendment — benchmark

Executed 2026-10-14. Exactly the user-specified A/B/C domains, selected before
amendment outcomes. This supersedes the six-agent CDC benchmark as the current
CDC experiment, while preserving all its evidence and entry points. No Step4,
registry, Qwen training, UCI529 model/default change or publication.

Read [amendment protocol](m5-step035-three-agent-protocol.md),
[routing feasibility](m5-step035-three-agent-feasibility.md) and
[execution log](../devlog/2026-10-14-m5-step035-three-agent.md).
Historical six-agent [benchmark](m5-step035-benchmark.md),
[feasibility](m5-step035-routing-feasibility.md) and original frozen protocol remain unchanged.

## Fixed domains and compatible reuse

| Agent | Domain | Required private fields | Old slot |
| --- | --- | --- | --- |
| A | metabolic | BMI, HighBP, HighChol, CholCheck | 1 |
| B | lifestyle | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump | 3 |
| C | general_health | GenHlth, MentHlth, PhysHlth, DiffWalk | 4 |

**576 byte-identical private model/calibrator files** reused, covering ALL12
small/pilot inner/outer contexts, four upstream blocks and four fits/client/block.
No specialist needed retraining for the experiment. Independent verification
does perform12 ephemeral refits without replacing them. Files/provenance/seeds,
features and patient probability association are checked; source/destination
hashes are in reuse.json. Only probabilities/reliability columns are remapped.
No six-agent shared weights are copied or loaded. All96 trainable target/final
fusion states and every ridge/MLP/graph router are fitted from fresh initialization;
48 arithmetic-mean snapshots remain stateless. Initial fusion states are saved
and independently verified against their seeds, then all96 fits are reproduced.

## Boundaries, schedules and unchanged gates

The exact frozen **50,861-row CDC test file**,202,819-row development file,
15k pilot IDs, grouped boundaries and12 context definitions are copied byte-for-byte.
No new test or historical validation/test prediction/metric. Full pilot outer
fitting sizes10000/9999/10001, held5000/5001/4999; small fitting2500 uses those SAME
assessments. Inner grouped80/20 and whole-upstream target cross-fitting are unchanged.
Copied contexts.json retains historical preparation timings; new runtime below
measures reuse/new fitting, not original specialist training.

Feature duplicate groups still use the ORIGINAL21 fields, preserving the boundary;
they do not assert unique patients or exclusion of every13-field lookalike after
removing domains. All fitting/calibration/target group exclusions remain intact.
Development outcomes have been used in earlier CDC analyses: this is an explicit
requested development amendment, not a newly blinded independent study.

Original residual fusion now12→24→1 with own-logit residual; logistic6→1; mean
same masked evidence. Nine valid STOP/one-peer masks replace36; keep15 epochs,
Adam.003/batch4096/clip5 and seeds. Fixed epoch count now implies fewer updates;
no compute-matched causal agent-count claim. Routers: ridgealpha100 or32-Tanh MLP,
base12/enriched37 inputs,20 epochs Adam.003/weight_decay.01; graph3×3 edges and
11→16Tanh→4 context,15 epochs on frozen action risks. No hyperparameter search.

Static and variable services retain original origin AND candidate slot1/3/4 random
schedules. Recipient multipliers are mapped to[1,1,1.5], with the same congestion.
Do not reseed by new client indices. Public21-field coverage is metadata, not
clinical values. Missing required fields make a peer unavailable; no expert is
weakened. Costs .02/.05/.10, maximum one peer. Every policy sees matched eligibility,
prices and evidence constraints. Only own calibrated p and public/train-derived
metadata enter inference. No labels/hidden peer predictions/unauthorized raw rows.

All numerical acceptance gates remain EXACTLY unchanged, including >=.005 gain
every fold, R²>=.05 every fold, accuracy drop<=.005, no origin BCE degradation>.01,
positive static gain and >=3 varying origins. With three agents the last criterion
requires ALL THREE; it is not relaxed. Inner-only selections precede outer policy
metrics. No50k expansion or default promotion is part of this amendment.

## Specialist metrics (unchanged checkpoints)

| Size | Agent | Acc | AUROC | AP | BCE | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| small | A | 0.85867 | 0.74382 | 0.30512 | 0.35865 | 0.01982 |
| small | B | 0.86133 | 0.61212 | 0.18687 | 0.39603 | 0.01950 |
| small | C | 0.85940 | 0.71935 | 0.26823 | 0.36899 | 0.02269 |
| pilot | A | 0.86033 | 0.76457 | 0.32496 | 0.34859 | 0.01263 |
| pilot | B | 0.86133 | 0.62354 | 0.19431 | 0.39285 | 0.00763 |
| pilot | C | 0.86053 | 0.74403 | 0.29139 | 0.35651 | 0.01293 |

## Commands and actual execution

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_three --device cuda --output artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_flower --output artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe scripts/verify_m5_step035.py artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe scripts/report_cdc_three.py artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

Use a fresh output and the preserved six-agent source. Neural --device auto/cuda/cpu;
actual **cuda**, RTX A4000 audit/environment recorded. Reuse/fitting
**96.076s**, peak torch allocation **29491712 bytes**,
excluding CUDA contexts/reserves/board usage. No speedup/saturation claim; private
fits were reused, not GPU-retrained. Flower uses one CPU worker and fixed3 clients.
No OS/driver/WSL/Docker installation. CPU fusion/routing paths are exercised by tests
and actual CPU federation, not a new full CPU nested reproduction.

Verification: `{"private_models": 576, "private_models_refitted": 12, "fusion_states": 144, "assessment_metric_fields": 154872, "aggregations": 3, "client_updates_replayed": 9, "fusion_optimizations_refitted": 96, "byte_identical_reused_models": 576, "preserved_files": 6776, "changed": [], "frozen_test_evaluations": 0, "frozen_test_hash_unchanged": true, "success": true}`. Full CSV/actions/checkpoints/provenance,
fitting-loss curves, initial states, reuse/preservation hashes, device evidence,
and Flower records stay under the new gitignored output. Six-agent results and
frozen test bytes are preserved. Exact changed files/QA/failures are in the devlog.
Fresh regression: **105 passed, 2 skipped, 2 warnings in 9.31s**; Ruff lint/format and pip check are recorded in qa/.
