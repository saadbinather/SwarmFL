# CDC three-specialist amendment protocol — 2026-10-14

Frozen before amendment fitting/results. Inherit all dataset, calibration,
cross-fitting, selection, costs, training schedules and numeric acceptance gates
from m5-step035-protocol.md; do not modify that historical protocol.

Exactly A metabolic(BMI,HighBP,HighChol,CholCheck), B lifestyle(Smoker,PhysActivity,
Fruits,Veggies,HvyAlcoholConsump), C general health(GenHlth,MentHlth,PhysHlth,DiffWalk).
Map original CDC client slots1/3/4 to0/1/2. Reuse byte-identical private calibrated
OOF/final checkpoints ONLY after verifying columns, IDs, upstream group exclusion
and probability alignment. Preserve original specialist seeds/provenance. If any
checkpoint is incompatible, refit only its declared development fitting/calibration
rows using original settings; record failure/reason. Do not reuse trained six-agent
fusion/router parameters or slice their weights.

Copy the original development/pilot/test-ID metadata and frozen test BYTES, never
parse its labels. Reuse EXACT 12 small/pilot inner/outer contexts and target folds.
Only upstream specialist probabilities/reliability columns are remapped. Train
new three-agent fusion in every target-fold complement and final context, then
new ridge/MLP/graph routing from scratch. Original residual uses3clients; logistic
6inputs; mean uses identical STOP/one-peer evidence. Uniform9valid masks instead
of36 is the required dimensional adaptation; retain15epochs/Adam.003/batch4096.
Keep router widths/schedules, seeds, costs .02/.05/.10 and selection rules unchanged.
No outer outcome influences specialist membership or candidate configuration.

Preserve exogenous service realizations/prices by mapping original ORIGIN and
candidate slots1/3/4, not reseeding their new indices. Keep public21-field coverage
metadata and full21-feature duplicate groups. Static/variable scenarios and
cost-aware/always-best/random/confidence/graph/value/matched-random controls remain.
Utility R² now measures the two potential non-self peers, never hidden inference
inputs. No raw features/private weights in update or consultation messages.

Inner-only selections precede outer policy assessment, separately for each size
and original/logistic/mean condition. Retain EVERY numeric gate, including >=3
varying origins: now ALL THREE must satisfy it. No threshold relaxation. No50k
expansion or Step4 in this amendment, even if a gate passes; report and stop.

Three actual Flower logical clients, one CPU worker; same three-round/two-local-
epoch routing-only check on pilot fold0/original fusion, paired six-epoch central
reference. Compatible shared MLP now37→32→1(enriched), base12; no old37/43-dimensional
router weights loaded. Save/replay all nine updates, aggregations, real aligned-ID
selected consultations and frozen model hashes. Device auto/cuda/cpu; actual CUDA
audit before use. Historical six-agent source entry points/results must remain
reproducible; fixed-three options are explicit, not a registry expansion.

Run pytest/Ruff/pip and independent exclusion/prediction/model-update verification.
Preserve earlier experiment reports/checkpoints and new frozen test; write new
amendment reports, update current docs. Stop after the three-specialist benchmark.
