# M5 Step 3.5 — CDC benchmark

Executed2026-10-14. This is a NEW fixed-six development benchmark, not a registry
or replacement for the preserved UCI529 system. Read [frozen protocol](m5-step035-protocol.md),
[routing feasibility](m5-step035-routing-feasibility.md) and
[execution log](../devlog/2026-10-14-m5-step035.md).

## Dataset and frozen boundary

Official [UCI891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
fetched via ucimlrepo: **253680 rows,21 features**, target Diabetes_binary,
counts **[218334, 35346]** for0/1. Its variable table says0=neither,
1=prediabetes OR diabetes; the page narrative mentions three classes, but that
is not the delivered binary target. This is cross-sectional self-reported survey
classification, not prospectively validated incident diabetes prediction.

Feature-value hash grouping (all21 values,not target) freezes **50861
test rows** by hash modulo5. **202819 development rows** are
stored separately; pilot15000 sampled only there. **No frozen-test prediction,
metric or selection**, and no historical validation/test evaluation. Hash files
are audit evidence, not privacy or patient identity. **227908
unique feature groups** in the full source: identical features remain in one
partition/fold even when outcomes conflict. This cannot detect repeated people
with different responses. Test-file byte hash is verified without parsing labels.

Three grouped outer folds assess about5000 pilot rows each. Full fitting sets
have about10000 rows; small2500-row fitting subsets use the SAME outer assessments
and separately refit nested pipelines. Inner grouped80/20 selection; target3fold
cross-fitting rebuilds the WHOLE specialist→calibration→fusion stack inside each
target-fold complement. Even fusion-training specialist OOF excludes target H.
No old dataset/model initializes these experts. Small/full comparisons isolate
fitting-size sensitivity within this new cohort, not a causal comparison to529.

## Specialist domains

| Specialist | Private fields |
| --- | --- |
| demographics | Age, Sex, Education, Income |
| metabolic | BMI, HighBP, HighChol, CholCheck |
| cardiovascular | Stroke, HeartDiseaseorAttack, HighBP, Age |
| lifestyle | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump |
| general_health | GenHlth, MentHlth, PhysHlth, DiffWalk |
| access | AnyHealthcare, NoDocbcCost, Education, Income, Age |

All experts use the same legitimate HistGradientBoosting capacity/schedule,
100iterations/15leaves/minleaf30/l2.1/lr.1, no early stopping/class rebalance.
Independent group-held20% sigmoid calibration; original fit/calibration model
weights are retained rather than refitting through calibration labels. Overlap
between domains is intentional. Every private model, including OOF, is saved.
This is simulated feature heterogeneity with aligned survey rows/labels available
at origins, not independent hospitals, true distributed vertical FL or deployment.

Frozen fusion conditions: existing original residual PredictionFusion(6),
six-client masked-logit linear12→1, arithmetic mean. Neural/logistic15epochs
Adam.003 batch4096 uniformly over36 valid STOP/one-peer masks, trained on specialist
OOF only. Final checkpoints frozen during routing; original3-client weights never
loaded. Mean gets no additional evidence. Learned losses/schedules are fixed,
not tuned to make a fusion/router win. Training loss curves are retained.

## Specialist predictive performance

Means across outer folds, shared assessment patients. Natural class imbalance is
retained. Average precision is the reported PR-AUC convention, not trapezoidal
interpolation. Accuracy alone is misleading near a14% positive prevalence.

| Size | Specialist | Acc | AUROC | PR-AUC(AP) | BCE | ECE10 |
| --- | --- | --- | --- | --- | --- | --- |
| small | demographics | 0.86140 | 0.66944 | 0.22369 | 0.37897 | 0.01071 |
| small | metabolic | 0.85867 | 0.74382 | 0.30512 | 0.35865 | 0.01982 |
| small | cardiovascular | 0.85947 | 0.72658 | 0.26280 | 0.36349 | 0.01485 |
| small | lifestyle | 0.86133 | 0.61212 | 0.18687 | 0.39603 | 0.01950 |
| small | general_health | 0.85940 | 0.71935 | 0.26823 | 0.36899 | 0.02269 |
| small | access | 0.86127 | 0.66457 | 0.21372 | 0.38164 | 0.01918 |
| pilot | demographics | 0.86020 | 0.69834 | 0.24707 | 0.37234 | 0.00813 |
| pilot | metabolic | 0.86033 | 0.76457 | 0.32496 | 0.34859 | 0.01263 |
| pilot | cardiovascular | 0.86120 | 0.73805 | 0.27391 | 0.35711 | 0.01009 |
| pilot | lifestyle | 0.86133 | 0.62354 | 0.19431 | 0.39285 | 0.00763 |
| pilot | general_health | 0.86053 | 0.74403 | 0.29139 | 0.35651 | 0.01293 |
| pilot | access | 0.86053 | 0.68980 | 0.23840 | 0.37452 | 0.00909 |

## Devices, timing, execution and reproducibility

Actual shared fitting device **cuda**, pipeline **500.384s**
(excludes fetch/hash/audit). Histogram experts run CPU, one native thread.
Torch peak allocation **74117120 bytes** excludes CUDA
contexts/reserves/board activity. environment.json records actual RTX A4000 CUDA
operation/checksum, CUDA build and nvidia-smi; no speedup/saturation claim.
CPU fallback is supported. At most one CPU Flower actor is used in the integration.

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.m5_step035 --device cuda --output artifacts/m5-step035-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_flower --output artifacts/m5-step035-v1
.venv/Scripts/python.exe scripts/verify_m5_step035.py artifacts/m5-step035-v1
.venv/Scripts/python.exe scripts/report_m5_step035.py artifacts/m5-step035-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

Use a fresh output, existing environment, --device auto/cuda/cpu. Cold fetch uses
ucimlrepo; the executed initial fetch is preserved under artifacts/m5-step035-fetch.
The new dataset does not reuse or overwrite artifacts/dataset.npz. Prepared test
and development are separate files. Inner selections precede outer routing metrics.
No test-set evaluation command is provided. The Flower launcher is a compatibility
check for THIS fixed-six adapter, not a generalized configurable service.

All artifacts gitignored: source metadata, frozen partition/hash, pilot IDs, every
context/target/OOF/calibration model/provenance, frozen fusion/router checkpoints,
metrics/actions/selection/gates, diversity/group records, full per-origin/cost CSV,
Flower probabilities/updates/consultations, verification and QA logs. Source and
documentation changes are listed in the execution log. Original defaults untouched.

## Limitations

No survey weights or population representativeness correction; no temporal/external
clinical validation. Ordinal survey codes are modeled numerically; Age is a band,
not age in years. Domains overlap and small capacity choices may limit attainable
benefits; all are trained honestly without designed degradation. Full-information
OOF utility supervision is more expensive/disclosive than serving one selected
peer. Prepared local files are accessible to a trusted operator. No isolation,
formal privacy or secure aggregation claim. Frozen calibration is estimated and
can shift between target-fold and final fitting sizes. Correlated errors and oracle
variation are descriptive associations, not causal diversity or achievable routing.
Outer folds share fitting rows and origin views share assessment patients; no
statistical superiority or independent sample pooling. No registry/Qwen/GAT/
bandit/ranker, production-default change or publication. Stop after Step3.5.
