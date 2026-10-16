# M5 Step 3.5 frozen pilot protocol — 2026-10-14

Written before data acquisition/model results. Outputs artifacts/m5-step035-v1.
Official ucimlrepo ID891; require253680 rows,21 features, Diabetes_binary in{0,1}.
Positive means prediabetes OR diabetes, not future incident diabetes. Preserve
source metadata and file hashes. No historical UCI529 partition is evaluated.

Test boundary: hash the21 feature values (not labels), first64bits modulo5==0
are frozen test. Identical-feature groups stay together even with conflicting
labels; no new test performance or model selection. Store development and test
separately. Sample15000 development rows stratified seed53500. Stratified GROUP
outer3fold seed53501 (feature hashes as groups), inner grouped80/20 seed53510+fold.
All additional specialist calibration/OOF/utility splits respect those groups.
Preserve source IDs and label/provenance exclusion at every fit.

Six independently fitted legitimate subsets (overlap allowed): demographics
Age/Sex/Education/Income; metabolic BMI/HighBP/HighChol/CholCheck; cardiovascular
Stroke/HeartDiseaseorAttack/HighBP/Age; lifestyle Smoker/PhysActivity/Fruits/Veggies/
HvyAlcoholConsump; general health GenHlth/MentHlth/PhysHlth/DiffWalk; access
AnyHealthcare/NoDocbcCost/Education/Income/Age. No full-feature privileged oracle,
no artificial expert degradation. HistGradientBoostingClassifier100iterations,
15leaves,minleaf30,l2=.1,lr=.1,early_stopping=False, fixedseed; no class balancing.
Each fit uses independent grouped20% sigmoid-calibration holdout (logit→logistic
C=1), retaining fit/calibrator separately. No final-on-fit probabilities in fusion.
Histogram boundaries are fitted on model-fitting rows only; no other fitted scaling.

Every inner/outer T: grouped3-fold specialists OOF for final frozen fusion and
final calibrated experts for assessment. For utility targets H, rebuild specialist
OOF/final/calibration AND fusion solely on T\H, then assess H. Uniformly train
fusion on STOP and one-peer masks, same original PredictionFusion(clients=6),
fixed experimental masked-logit logistic(12→1), or arithmetic mean. Neural and
logistic15epochs Adam.003,batch4096,clip5; weights frozen for routing comparisons.
All eligible origin-peer training examples; gross BCEgain untruncated. Same
information/evidence for all paired comparisons, budgets0/1 only. No joint fusion/
router tuning and no new generalized fusion framework.

Data-size control: nested small condition uses at most2500 stratified grouped
rows from each outer fitting set and its own inner splits; same outer assessments
as full pilot outer fits(~10k). Train all specialists/fusers from scratch in both.
No comparison against old529 metrics as a causal dataset-size effect. Full15000
pilot and small control fixed upfront; no50k expansion unless gate below passes.

Contexts: (1) all available,full public coverage,uniform cost; (2) seeded public
feature-domain outages10% lifestyle and10% general-health, independent service
availability80%, origin required fields always present. A specialist is eligible
ONLY if ALL its fields are covered; no masked/imputed input or weakened classifier.
Cost multipliers by specialist [.75,1,1.25,1,1.5,.75] × independent [.8,1.2]
congestion. Schedules derive from source row IDs and origin/seed, never outcomes
or predictions. Actual classifier inputs/weights do not change across scenarios.
Tariffs .02/.05/.10, apply actual recipient multiplier. Origins always available.

Pre-query base: own calibrated p/confidence, origin/candidate identities, public
domain width/overlap/availability/budget. Enriched adds own entropy/variance,
training-only own/candidate OOF BCE and public21-field coverage flags. No raw
clinical values, hidden predictions or labels in routing API. No patient IDs as
model inputs. Routing training metadata can use IDs only to construct the fixed
exogenous schedules. Reliability for held utility H uses T\H labels only.

Compare STOP; best training-selected peer with availability fallback ranked by
training mean gains; positive-net fixed-value policy; random max; confidence<.6
then best available peer; original ContextualRouter(6,1) with frozen cross-fitted
action risks; ridge alpha100+fit-only standardization; MLP32Tanh→1,20epochs,
Adam.003 weight_decay.01,batch4096,clip5. Ridge/MLP base/enriched. Graph15epochs,
cost-aware soft expected loss using variable training schedules; inference subtracts
the known per-peer cost from action scores. Existing graph architecture accepts
six clients; the old three-client checkpoints/adapters are never loaded.
Inner selection chooses four ridge/MLP feature candidates by variable-scenario
macro-origin BCE+actual cost at.02, separately per fusion and size. Report all
outer outcomes, no outer hyperparameter/model/round selection. Confidence/graph
are controls, not additional selection candidates. Exact-call-mask random controls.

Metrics: accuracy/AUROC/average precision(PR-AUC convention)/F1/BCE/Brier/ECE10,
cost objective/calls/STOP/action frequencies, utility MSE/R2 against training
origin/peer mean, benefit correlations, oracle regret (retrospective only), selected
call harm, prediction/error correlations/disagreement and best-peer variation.
Report by size/fold/origin/fusion/scenario/cost, group by public age/sex/health
metadata only in retrospective diagnostics (these values never enter router).

Scale gate for15000 pilot, no automatic promotion: a preselected pipeline must
improve fixed-best-peer objective >=.005 EVERY outer fold at.02, no origin mean
BCE degradation>.01, accuracy drop<=.005, macro utility R2>=.05 EVERY fold,
and vary selected peer (not just STOP) on >=5% of calls in >=3origins per fold.
Require gain in STATIC scenario too, to distinguish service-mask adaptability
from clinical personalization. Require all verification checks. If any frozen
fusion condition passes all gates, execute a separate fresh50000-development
sample nested reproduction with identical settings; otherwise STOP scaling.
These are conservative development gates, not significance/clinical safety.

Genuine Flower compatibility check after pilot: six logical ClientApps, one CPU
Ray worker, same shared32-unit numeric routing MLP for each origin, three FedAvg
rounds/two local epochs on the first full outer context/original fusion. Only
compatible router arrays/scalars; save/replay every client/global state and make
real selected-only aligned-ID consultations. It is a fixed-six adapter check,
NOT proof of generalized federation or federated performance superiority. Compare
central same architecture and matched six-epoch pooled schedule. All preparation
and central diagnosis are trusted-local/full-information, explicitly not private FL.

Use CPU for histogram experts; auto/cuda/cpu for neural fitting after actual CUDA
audit, at most one GPU worker if used. Record runtime/torch allocation, no saturation
claim. Preserve all old artifact/report hashes; pytest/Ruff/pipcheck. No advanced
GAT/bandit/ranker, registry, Qwen training or default changes. STOP after3.5.
