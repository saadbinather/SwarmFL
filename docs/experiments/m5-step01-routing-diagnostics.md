# M5 Step 1 — Routing and fusion degradation diagnostics

Executed 2026-10-14 on Windows. **Step 1 only is complete.** M1–M4 defaults,
specialists, checkpoints and historical results are preserved. No historical-test
inference, tuning, new SLM, Qwen fine-tuning, new specialist registry or flexible
fusion architecture was introduced. No clinical validity or privacy guarantee is
claimed. This report distinguishes fresh training/validation diagnostics from
historical evidence.

## Findings for architecture review

1. **Origin-local objective conflict is supported.** Several simultaneous origin
   gradients and local fusion updates oppose one another. Both centralized
   references have lower validation BCE in all three seeds, including the control
   with matched global optimizer steps. This implicates the local-training /
   averaging protocol; it does not isolate arithmetic averaging from optimizer
   state resets, local drift or batching as a unique cause.
2. **Joint optimization is not established as the main problem.** Joint training
   has lower validation BCE than either component-freeze ablation in every seed.
   The router and fusion have disjoint parameter sets: there is no direct
   competing-gradient dot product on shared parameters between these modules.
   Coupling occurs through policy-weighted fusion losses and changing action
   values. Freezing at random initialization is a substantial restriction, not a
   complete test of alternating training or a good pretrained fixed fusion.
3. **Always-consult routing is confirmed; fixed-peer collapse is common.** At
   the final round every origin consults on every validation patient. Eight of
   nine seed/origin views use one fixed peer; seed 44 origin A selects B on 50
   cases and C on 28. This confirms always-consult behavior, but does
   not prove statistical overfitting. Validation BCE is below OOF-training BCE,
   and the cohorts and prediction-generating models differ.
4. **Soft/hard mismatch exists but is localized.** Final-round hard-minus-soft
   BCE is largest for origin A; B/C policies are almost deterministic. It cannot
   by itself explain all degradation. M4 remains nearly uniform stochastically
   while its argmax actions scarcely change, so streaming and deterministic
   validation policies must not be conflated.
5. **Calibration/distribution shift is measured, causality unresolved.** Fold
   and final specialist predictions differ on the same training patients, and
   validation calibration is imperfect. Final-on-training values are optimistic
   resubstitution diagnostics, never an unbiased comparator. No recalibration was
   fitted on validation or test, and no calibration fix is claimed.
6. **Fusion quality depends on the metric and evidence.** Identical-evidence
   arithmetic means can improve threshold accuracy while learned fusion improves
   BCE. Random/cyclic controls below match maximum and exact per-case call
   budgets. Two-peer full ensembles are not used to claim a one-peer deficit.
   Some selected calls harm realized BCE; this is not clinical necessity.
   Learned routing's macro accuracy (.93162) is slightly below seeded random
   (.93305), while its BCE (.19573) is slightly lower than random (.19798).
   These small reused-validation differences do not establish a policy winner.
7. **M4 is sparse and changes little.** Authentic selected-only replay has
   19–122 records/origin/update. Default cap 2 clips no inspected parent-policy
   importance weights; effective sample size stays close to sample count. Thus
   active clipping is not supported as the explanation here. Older reward
   versions are common but their measured risk/loss changes are small. Selection
   bias remains an off-policy modeling risk, not a demonstrated failure from
   observing unqueried outcomes.

## Architecture and learning audit

The current `CollaborativeModel` contains a three-client contextual router and
masked residual fusion. Router: a 3×3 edge-logit parameter matrix; context inputs
are origin one-hot, own probability/confidence, domain flags and availability
(11 inputs), then Linear(11,16), Tanh, Linear(16,4). Budget one enumerates
empty/A/B/C actions; self/unavailable actions are masked. Fusion: each specialist's
masked logit, confidence and presence plus origin identity (12 inputs), then
Linear(12,24), ReLU, Linear(24,1), with origin-logit residual. Specialists are
separate Linear(local_width,24), ReLU, Linear(24,1) classifiers.
M1's preserved workflow instead trains a shared masked-input classifier with
32 inputs through Flower. M3 performs frozen-Qwen tool dispatch and numerical
inference with saved specialists/shared weights; it performs no optimization.

M2 jointly minimizes soft expected action BCE plus .02 per call using complete
training-only nested OOF evidence. Inference selects argmax before consultation.
Each origin trains from common shared arrays with its own labels/evidence cache;
Flower 1.39 `FedAvg` aggregates only compatible shared ArrayRecords, weighted by
`num-examples`. Here each origin has 364 aligned training patients, so weighting
is equal. Specialists, temperatures and preprocessing are not in shared updates.
These are dependent views of one cohort, not independent hospitals.

M4 starts adaptive/frozen shared policies at label-independent random weights,
not the already-supervised M2 checkpoint. Signed prediction and verified outcome
ledgers are separate; delayed true training labels are joined locally after
decisions. Behavior is .8×policy + .2×uniform over valid actions. Shared gradients
use selected evidence only: detached clipped IPS-weighted fusion BCE plus .1×
cost policy surrogate, with logged fusion reward, fixed .5 baseline, Adam .0003,
two full replay steps, norm clip .5 and latest 128 verified events per origin.
Three clients send shared arrays and scalar metrics over real Flower; approval
uses validation and immutable retained versions. Private specialists and Qwen are
frozen throughout M4. Logs/acknowledgements are not parameter learning or labels.
Prediction, identifier and model-update disclosure is not proven privacy.
Training feature rows and private weights never enter consultation messages;
M3's separately authorized new-patient feature-subset path is unchanged.

## Protocol and controls

Existing fixed split: 364 training, 78 validation, 78 untouched historical-test
patients. Seeds 42/43/44 change specialist/fold/model fits, not the patient split.
No variant or hyperparameter was selected using these diagnostic results. Report
round 5 for every variant; complete rounds 0–5 remain in JSON (central 1–5).
Common initial checkpoint within seed; Adam .01; 10 local epochs × 5 rounds;
batch 64, norm clip 5, FP32, cost .02, budget one. Frozen components retain common
random initial weights. All three main variants use actual Flower train/reply
messages and FedAvg. The equal-total-step central reference interleaves origins
for 180 optimizer steps/round, versus 60/client. The follow-up pooled central
reference averages three origin losses per matched batch for 60 global steps,
matching the 180 origin-loss computations of three clients. Optimizers reset
each round in all schedules. Pooled control was specified in the execution log
before running it, to resolve the first reference's update-count confound.

OOF training, final-on-training and validation probability caches were acquired
through genuine Flower specialist queries. Diagnostic evaluation replays masked
subsets from these full-information caches; it is **not** a selected-only live
consultation benchmark. Routing uses only own evidence and public metadata.
The central reference deliberately has training labels/probabilities, never
private feature rows. Historical test IDs are used only for exclusion checks;
no historical test label is indexed for a new metric. Final-on-training
predictions are used only to describe fold/final shift, never for model fitting.

Controls use the same fusion checkpoint and budget one. `random_max` chooses a
seeded peer; `cyclic_max` chooses the next origin index, independent of labels or
peer values. `_matched` variants reuse the learned per-patient call/no-call mask
and change only the peer. Arithmetic mean uses the identical selected evidence;
stop+mean is the unmodified origin classifier, whereas stop+learned fusion still
applies its learned residual. All origins share patient IDs and are not
independent samples. Tables macro-average origin metrics, then seeds where
specified; AUROC/F1 are means, not pooled metrics or confidence intervals. ECE is
10 equal-width probability bins; accuracy/F1 use threshold .5.

## Fresh controlled ablations

Final-round training is OOF; validation uses final frozen specialists. All metrics
below are development results. No winner is promoted to a default.

| Seed | Variant | Train BCE | Val accuracy | Val AUROC | Val BCE | Val F1 | Val ECE | Val Brier | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | joint | 0.30369 | 0.92308 | 0.96076 | 0.21437 | 0.93599 | 0.06958 | 0.05738 | 1.00000 |
| 42 | frozen_router | 0.42145 | 0.83761 | 0.91632 | 0.31727 | 0.84682 | 0.08226 | 0.09735 | 0.19231 |
| 42 | frozen_fusion | 0.39055 | 0.90171 | 0.93970 | 0.32016 | 0.92022 | 0.11791 | 0.09957 | 0.66667 |
| 42 | central_joint | 0.23287 | 0.93162 | 0.97072 | 0.18066 | 0.94383 | 0.06418 | 0.04813 | 1.00000 |
| 42 | central_pooled | 0.24233 | 0.94444 | 0.97025 | 0.18275 | 0.95393 | 0.06819 | 0.04717 | 1.00000 |
| 43 | joint | 0.23720 | 0.94017 | 0.96678 | 0.17952 | 0.95043 | 0.03877 | 0.04781 | 1.00000 |
| 43 | frozen_router | 0.29849 | 0.93162 | 0.96701 | 0.20507 | 0.94378 | 0.07088 | 0.05405 | 1.00000 |
| 43 | frozen_fusion | 0.39690 | 0.85470 | 0.91181 | 0.32692 | 0.86388 | 0.09561 | 0.10317 | 0.00000 |
| 43 | central_joint | 0.18000 | 0.97009 | 0.97870 | 0.13166 | 0.97506 | 0.03669 | 0.03079 | 1.00000 |
| 43 | central_pooled | 0.17825 | 0.97009 | 0.97731 | 0.13193 | 0.97506 | 0.03970 | 0.03025 | 1.00000 |
| 44 | joint | 0.29970 | 0.93162 | 0.96308 | 0.19329 | 0.94326 | 0.03866 | 0.05340 | 1.00000 |
| 44 | frozen_router | 0.37094 | 0.91026 | 0.94479 | 0.25387 | 0.92434 | 0.05437 | 0.07113 | 0.54701 |
| 44 | frozen_fusion | 0.43324 | 0.85897 | 0.91019 | 0.32948 | 0.89134 | 0.08052 | 0.10460 | 0.00000 |
| 44 | central_joint | 0.20386 | 0.97436 | 0.97917 | 0.12979 | 0.97872 | 0.04492 | 0.03283 | 1.00000 |
| 44 | central_pooled | 0.21462 | 0.96581 | 0.98102 | 0.12672 | 0.97140 | 0.03760 | 0.03198 | 1.00000 |


### Joint-trained policy and fusion controls

Means across three seeds and three origins. All non-stop joint controls actually
use exactly one call per patient here; matched and maximum rows coincide in call
count, not necessarily predictions.

| Policy | Fusion | Accuracy | AUROC | BCE | F1 | ECE | Brier | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learned | learned_fusion | 0.93162 | 0.96354 | 0.19573 | 0.94323 | 0.04900 | 0.05286 | 1 |
| learned | mean_fusion | 0.95014 | 0.96601 | 0.26517 | 0.95966 | 0.14599 | 0.07317 | 1 |
| stop | learned_fusion | 0.84900 | 0.91096 | 0.33965 | 0.85806 | 0.08052 | 0.10541 | 0 |
| stop | mean_fusion | 0.85755 | 0.91096 | 0.33063 | 0.87957 | 0.08200 | 0.10485 | 0 |
| random_max | learned_fusion | 0.93305 | 0.96570 | 0.19798 | 0.94455 | 0.06035 | 0.05335 | 1 |
| random_max | mean_fusion | 0.94160 | 0.96779 | 0.29660 | 0.95267 | 0.16999 | 0.08235 | 1 |
| cyclic_max | learned_fusion | 0.92735 | 0.96308 | 0.19884 | 0.93964 | 0.05462 | 0.05550 | 1 |
| cyclic_max | mean_fusion | 0.94444 | 0.96678 | 0.29382 | 0.95517 | 0.16890 | 0.08108 | 1 |
| random_matched | learned_fusion | 0.93305 | 0.96570 | 0.19798 | 0.94455 | 0.06035 | 0.05335 | 1 |
| random_matched | mean_fusion | 0.94160 | 0.96779 | 0.29660 | 0.95267 | 0.16999 | 0.08235 | 1 |
| cyclic_matched | learned_fusion | 0.92735 | 0.96308 | 0.19884 | 0.93964 | 0.05462 | 0.05550 | 1 |
| cyclic_matched | mean_fusion | 0.94444 | 0.96678 | 0.29382 | 0.95517 | 0.16890 | 0.08108 | 1 |


### Per-origin joint diagnostics

Action frequencies are [stop,A,B,C] counts over 78 validation patients. Harm is
the fraction of selected calls whose realized BCE is no better than the same
model's stop-fusion BCE. Oracle regret uses the true validation outcome only
after predictions and is an unattainable full-information diagnostic, not a
policy or a selection criterion.

| Seed | Origin | Accuracy | AUROC | BCE | F1 | Actions | Harm/calls | Call BCE delta | Hard−soft BCE | Entropy | Oracle cost regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 0.89744 | 0.93438 | 0.31740 | 0.91667 | [0, 0, 78, 0] | 0.14103 | -0.17888 | 0.00027 | 0.01647 | 0.18512 |
| 42 | 1 | 0.92308 | 0.96944 | 0.17541 | 0.93478 | [0, 0, 0, 78] | 0.05128 | -0.15607 | -0.00000 | 0.00020 | 0.04318 |
| 42 | 2 | 0.94872 | 0.97847 | 0.15029 | 0.95652 | [0, 0, 78, 0] | 0.29487 | -0.03409 | -0.00000 | 0.00000 | 0.06014 |
| 43 | 0 | 0.91026 | 0.94896 | 0.24737 | 0.92632 | [0, 0, 78, 0] | 0.11538 | -0.28617 | 0.01489 | 0.42925 | 0.17087 |
| 43 | 1 | 0.93590 | 0.97292 | 0.17710 | 0.94624 | [0, 0, 0, 78] | 0.19231 | -0.18142 | -0.00003 | 0.00199 | 0.02753 |
| 43 | 2 | 0.97436 | 0.97847 | 0.11408 | 0.97872 | [0, 0, 78, 0] | 0.14103 | -0.02711 | -0.00000 | 0.00021 | 0.04214 |
| 44 | 0 | 0.89744 | 0.94340 | 0.27852 | 0.91489 | [0, 0, 50, 28] | 0.08974 | -0.24259 | 0.00560 | 0.40710 | 0.13531 |
| 44 | 1 | 0.92308 | 0.96875 | 0.18287 | 0.93617 | [0, 0, 0, 78] | 0.11538 | -0.16140 | -0.00000 | 0.00043 | 0.03559 |
| 44 | 2 | 0.97436 | 0.97708 | 0.11849 | 0.97872 | [0, 0, 78, 0] | 0.14103 | -0.02755 | -0.00000 | 0.00037 | 0.06293 |


### Cost and redundancy

Fraction of validation cases for which at least one valid peer lowers realized
fusion BCE by more than the stated cost. This is retrospective and label-aware;
it does not show that a router can identify those cases prospectively. Costs
were not retuned and these are fixed-model diagnostics, not cost-trained variants.

| Origin | Cost 0 | Cost .02 | Cost .1 | Cost .5 |
| --- | --- | --- | --- | --- |
| 0 | 0.97436 | 0.86325 | 0.65385 | 0.31624 |
| 1 | 0.97436 | 0.41880 | 0.20940 | 0.11538 |
| 2 | 0.92308 | 0.35470 | 0.16239 | 0.03419 |


Signed probability-error correlations describe redundancy, not conditional
independence or a causal value of information.

| Seed | A/B error corr | A/C error corr | B/C error corr | B/C disagreement |
| --- | --- | --- | --- | --- |
| 42 | 0.64753 | 0.57290 | 0.60452 | 0.11538 |
| 43 | 0.62512 | 0.55921 | 0.58832 | 0.10256 |
| 44 | 0.61996 | 0.51350 | 0.57755 | 0.14103 |


## Gradients, local drift and calibration

Pair order is A/B, A/C, B/C. Gradients are simultaneous full-OOF-training
gradients at final shared weights, before clipping. Updates are the final
round's independently trained local state minus its common parent. Negative
cosine supports objective/update conflict, not an aggregation implementation bug.

| Seed | Component | Origin gradient cosines | Local update cosines |
| --- | --- | --- | --- |
| 42 | router. | -0.4771, 0.6580, 0.0732 | 0.0357, 0.2984, 0.3822 |
| 42 | fusion. | -0.6421, 0.6692, -0.5271 | -0.2115, -0.2566, -0.2524 |
| 43 | router. | -0.5354, 0.5268, 0.2206 | 0.0728, 0.1757, 0.2745 |
| 43 | fusion. | -0.1818, 0.0361, -0.6555 | -0.1689, -0.1260, -0.0487 |
| 44 | router. | -0.6647, 0.6477, -0.1352 | 0.0843, -0.0167, 0.1076 |
| 44 | fusion. | -0.0074, -0.2279, -0.4712 | -0.3916, -0.2586, -0.1419 |


Initial-to-final parameter L2 (buffers included, constant action buffer has zero
delta); mean local preclip gradient norms across saved update reports. Tiny
frozen server drift can arise solely from Flower floating point averaging.

| Seed | Variant | Router L2 | Fusion L2 | Router gradient norm | Fusion gradient norm |
| --- | --- | --- | --- | --- | --- |
| 42 | joint | 6.54429 | 3.41283 | 0.00465 | 0.14538 |
| 42 | frozen_router | 0.00000 | 3.58307 | 0.00000 | 0.09879 |
| 42 | frozen_fusion | 5.57465 | 0.00000 | 0.00425 | 0.00000 |
| 42 | central_joint | 8.96876 | 8.40251 | 0.00216 | 0.17715 |
| 43 | joint | 5.96026 | 3.77524 | 0.02153 | 0.12727 |
| 43 | frozen_router | 0.00000 | 3.76531 | 0.00000 | 0.11293 |
| 43 | frozen_fusion | 5.70746 | 0.00000 | 0.00147 | 0.00000 |
| 43 | central_joint | 13.38032 | 7.13855 | 0.02848 | 0.15065 |
| 44 | joint | 6.03661 | 3.50650 | 0.02845 | 0.17437 |
| 44 | frozen_router | 0.00000 | 3.94997 | 0.00000 | 0.13829 |
| 44 | frozen_fusion | 5.53247 | 0.00000 | 0.00069 | 0.00000 |
| 44 | central_joint | 9.06151 | 8.11649 | 0.00268 | 0.23904 |


Same-training-patient OOF/final absolute probability shift, followed by raw
specialist calibration/BCE. Final-training BCE is optimistic resubstitution.
Differences combine model fitting, fold sample size and temperature estimation;
these data do not isolate calibration as the cause.

| Seed | Origin | Mean abs(OOF−final) | OOF ECE | Train-final ECE | Val ECE | OOF BCE | Train-final BCE | Val BCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 0.03986 | 0.06099 | 0.05538 | 0.07174 | 0.54237 | 0.50871 | 0.49402 |
| 42 | 1 | 0.05150 | 0.05757 | 0.03289 | 0.08213 | 0.36182 | 0.30271 | 0.31387 |
| 42 | 2 | 0.08322 | 0.05793 | 0.04706 | 0.10359 | 0.34742 | 0.23592 | 0.21481 |
| 43 | 0 | 0.07410 | 0.09784 | 0.03748 | 0.08299 | 0.57015 | 0.51271 | 0.49731 |
| 43 | 1 | 0.03927 | 0.03866 | 0.03893 | 0.07340 | 0.31675 | 0.29131 | 0.30300 |
| 43 | 2 | 0.06641 | 0.03534 | 0.03784 | 0.09484 | 0.29106 | 0.21259 | 0.17163 |
| 44 | 0 | 0.04936 | 0.04369 | 0.04242 | 0.05651 | 0.57368 | 0.51031 | 0.48992 |
| 44 | 1 | 0.05515 | 0.05737 | 0.03793 | 0.09995 | 0.34946 | 0.29179 | 0.31086 |
| 44 | 2 | 0.08051 | 0.06477 | 0.03141 | 0.07285 | 0.37987 | 0.21729 | 0.18023 |


## M4 feedback investigation and reproduction

Read-only independent verification of the historical run passed all 1,092
events, authentication/source/delay/partial-evidence checks and seven shared
FedAvg candidates. Replay diagnostics below use only authenticated observed
evidence and outcomes at each saved training cutoff, evaluated at the parent
policy before local updates. They do not recover unselected peer outcomes.

| Update | Origin | Replay n | Actions | Older reward version fraction | Clipped at 2 | ESS | Mean abs(risk change) | Current−logged BCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v001 | 0 | 19 | [8, 0, 5, 6] | 0.00000 | 0.00000 | 18.97909 | 0.00000 | -0.00000 |
| v001 | 1 | 19 | [5, 7, 0, 7] | 0.00000 | 0.00000 | 18.99673 | 0.00000 | 0.00000 |
| v001 | 2 | 19 | [3, 10, 6, 0] | 0.00000 | 0.00000 | 18.99576 | 0.00000 | 0.00000 |
| v002 | 0 | 39 | [14, 0, 13, 12] | 0.51282 | 0.00000 | 38.96327 | 0.00027 | -0.00054 |
| v002 | 1 | 39 | [10, 12, 0, 17] | 0.51282 | 0.00000 | 38.99185 | 0.00020 | -0.00010 |
| v002 | 2 | 39 | [9, 15, 15, 0] | 0.51282 | 0.00000 | 38.98891 | 0.00012 | -0.00009 |
| v003 | 0 | 59 | [19, 0, 20, 20] | 0.67797 | 0.00000 | 58.95402 | 0.00053 | -0.00084 |
| v003 | 1 | 59 | [17, 15, 0, 27] | 0.67797 | 0.00000 | 58.98954 | 0.00043 | -0.00013 |
| v003 | 2 | 59 | [14, 24, 21, 0] | 0.67797 | 0.00000 | 58.98821 | 0.00029 | -0.00024 |
| v004 | 0 | 79 | [24, 0, 26, 29] | 0.75949 | 0.00000 | 78.94953 | 0.00073 | -0.00106 |
| v004 | 1 | 79 | [23, 24, 0, 32] | 0.75949 | 0.00000 | 78.98882 | 0.00056 | -0.00013 |
| v004 | 2 | 79 | [19, 35, 25, 0] | 0.75949 | 0.00000 | 78.98887 | 0.00041 | -0.00037 |
| v005 | 0 | 99 | [29, 0, 35, 35] | 0.80808 | 0.00000 | 98.93642 | 0.00083 | -0.00120 |
| v005 | 1 | 99 | [29, 29, 0, 41] | 0.80808 | 0.00000 | 98.98685 | 0.00055 | -0.00011 |
| v005 | 2 | 99 | [27, 41, 31, 0] | 0.80808 | 0.00000 | 98.98232 | 0.00049 | -0.00045 |
| v006 | 0 | 119 | [40, 0, 39, 40] | 0.84034 | 0.00000 | 118.92097 | 0.00102 | -0.00159 |
| v006 | 1 | 119 | [42, 31, 0, 46] | 0.84034 | 0.00000 | 118.98631 | 0.00056 | -0.00019 |
| v006 | 2 | 119 | [31, 51, 37, 0] | 0.84034 | 0.00000 | 118.98180 | 0.00063 | -0.00055 |
| v007 | 0 | 122 | [43, 0, 39, 40] | 0.98361 | 0.00000 | 121.92053 | 0.00138 | -0.00217 |
| v007 | 1 | 121 | [43, 32, 0, 46] | 0.99174 | 0.00000 | 120.98750 | 0.00066 | -0.00029 |
| v007 | 2 | 121 | [32, 51, 38, 0] | 0.99174 | 0.00000 | 120.98172 | 0.00088 | -0.00077 |


Each patient supplies one observed action only (three valid actions per origin);
full action counterfactuals are unavailable in M4 replay. Cap-sensitivity audits
(.5, 1, 2, 10) retain norms and ESS in `feedback.json`. A .5 cap clips every
inspected row, while the deployed cap 2 clips none at the parent state. This does
not prove clipping is absent after local optimizer steps or in other streams.
Late replay includes mostly older versions, but absolute risk changes are small;
no dominant stale-reward failure is demonstrated. Historical adaptive/frozen
stream BCE was .449557/.450215 with equal .788462 accuracy. These were stochastic
stream policies, not the hard validation policy in the next table.

| Version | Origin | Accuracy | AUROC | BCE | F1 | Hard actions | Policy entropy | Soft expected calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v000 | 0 | 0.71795 | 0.83160 | 0.48776 | 0.78000 | [78, 0, 0, 0] | 1.08491 | 0.58907 |
| v000 | 1 | 0.88462 | 0.92188 | 0.30751 | 0.90526 | [33, 45, 0, 0] | 1.09629 | 0.65738 |
| v000 | 2 | 0.94872 | 0.97326 | 0.22139 | 0.95745 | [78, 0, 0, 0] | 1.09445 | 0.62673 |
| v007 | 0 | 0.71795 | 0.83160 | 0.48768 | 0.78000 | [78, 0, 0, 0] | 1.08692 | 0.59509 |
| v007 | 1 | 0.88462 | 0.92535 | 0.30699 | 0.90526 | [33, 45, 0, 0] | 1.09688 | 0.65956 |
| v007 | 2 | 0.94872 | 0.97326 | 0.22017 | 0.95745 | [78, 0, 0, 0] | 1.09570 | 0.63212 |


Fresh full M4 rerun: Flower run `13948513765990979272`, device `cuda`, 477.536 seconds, approved `v007`. Comparison below is a reproduction, not a newly selected experiment.

| Run | Policy | Accuracy | AUROC | BCE | F1 |
| --- | --- | --- | --- | --- | --- |
| historical | adaptive | 0.78846 | 0.85831 | 0.44956 | 0.82697 |
| fresh | adaptive | 0.78846 | 0.85831 | 0.44956 | 0.82697 |
| historical | frozen | 0.78846 | 0.85794 | 0.45021 | 0.82697 |
| fresh | frozen | 0.78846 | 0.85794 | 0.45021 | 0.82697 |
| historical | offline | 0.88462 | 0.93542 | 0.32853 | 0.90367 |
| fresh | offline | 0.88462 | 0.93542 | 0.32853 | 0.90367 |


Adaptive BCE absolute reproduction difference: 2.563901036e-09; maximum final checkpoint element difference: 8.940696716e-08. Accuracy, macro-origin AUROC, F1 and call counts reproduce exactly.


Fresh saved-evidence verification:

```json
{
  "verified": true,
  "events": 1092,
  "unique_training_patients": 364,
  "updates": 7,
  "flower_run_id": 13948513765990979272,
  "approved_version": "v007"
}
```


## Reproducibility and preservation

Fresh M2 joint final weights reproduce the historical final shared checkpoint
**exactly** in all three seeds (maximum elementwise difference zero); only
training/validation predictions were rerun. Historical M2 checkpoint selection
was not repeated. Frozen local components are bitwise unchanged; server arrays
are checked within 1e-6 to allow FedAvg rounding. Independent M5 verification:

```json
{
  "verified": true,
  "states": 84,
  "aggregations": 45,
  "replayed_policy_metrics": 18144,
  "trace_rows": 7254
}
```


| Seed | Flower run ID | Post-cache server compute seconds |
| --- | --- | --- |
| 42 | 10221705930338539825 | 50.06899 |
| 43 | 13443808006279245134 | 47.00647 |
| 44 | 7155293317803481445 | 34.34858 |


Device audit: NVIDIA RTX A4000, PyTorch 2.11.0+cu128, CUDA build 12.8, actual tensor checksum 3680.0. Main ablations used CUDA with at most one full-GPU Ray client worker serving all three logical clients. Maximum saved torch allocation across new main training reports: 18664448 bytes. This excludes contexts, reserved memory and total board memory; no GPU saturation or speedup is claimed. Pooled central control used CPU and its own saved runtime/memory reports. Server compute times exclude startup and cache acquisition.


Preservation audit: 628 pre-existing artifact files hashed, changed list `[]`. Ollama runtime/model files were excluded from the manifest. This workspace has no `.git` repository; `git status --short` fails, so no VCS cleanliness claim is possible. New artifacts remain under gitignored `artifacts/m5-step01*`.


## Commands and validation

Run from the repository root with the active environment's Python executable.
Fresh output paths are required; these commands never overwrite M1–M4 runs.

```powershell
python -u -m fed_med_agents.experiments.m5_step01 --device cuda --output artifacts/m5-step01 --seeds 42 43 44
python scripts/verify_m5_step01.py artifacts/m5-step01
python scripts/extend_m5_controls.py --device cpu
python scripts/audit_m5_feedback.py
python -u -m fed_med_agents.experiments.milestone04 --device cuda --output artifacts/m5-step01-m4-rerun
python -m fed_med_agents.feedback.verify artifacts/m5-step01-m4-rerun
python -m pytest -q
python -m ruff check --no-cache src tests scripts
python -m ruff format --no-cache --check src tests scripts
python -m pip check
python scripts/report_m5_step01.py
```

The original 52-test claim includes two opt-in runtime integrations. Ordinary
fresh QA after four new diagnostic tests passed **54, skipped 2** in 12.05 s;
two upstream Typer deprecation warnings remain. Opt-in integrations must use
new M3/M4 output paths to preserve original evidence. Saved QA logs below record
the final executed checks rather than assuming an integration passed.


`artifacts/m5-step01-audit/pytest-full.log`:

```text
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
56 passed, 2 warnings in 88.87s (0:01:28)
```


`artifacts/m5-step01-audit/ruff-check.log`:

```text
All checks passed!
```


`artifacts/m5-step01-audit/ruff-format.log`:

```text
58 files already formatted
```


`artifacts/m5-step01-audit/pip-check.log`:

```text
No broken requirements found.
```


## Independent continuation review

The completed diagnostic workflow/artifacts were already present when this
continuation began. They were audited rather than replacing their training runs.
Fresh read-only verification passed 84 states, 45 Flower aggregations, 18,144
policy metrics and 7,254 trace rows; historical and reproduced M4 runs both passed
all 1,092 events and seven aggregations. Separate tensor comparisons confirmed
bitwise equality of historical M2 and diagnostic joint final weights in all three
seeds. All 628 preserved artifact hashes still match.

Corrected the fixed-peer claim: eight of nine final seed/origin views choose one
peer; seed 44 origin A chooses B on 50 validation cases and C on 28. All nine
views always consult. This exception is evidence against universal fixed-peer
collapse. Corrected two Markdown absolute-difference table headings.

One existing M5 derived verification JSON contained null bytes. The underlying
checkpoints, raw metrics and ledgers verified successfully; fresh summaries are
saved in the new review directory. Historical artifacts were not repaired or
overwritten. This report uses the independently regenerated M4 verification.

Automatic approval review rejected the combined Ollama-start/background-QA
command with reason "blocked by policy". No service start was executed. Safe
direct replay and ordinary pytest/Ruff/dependency checks completed instead.
The existing 56-test full-runtime result above is preserved evidence; the fresh
review below skips the two opt-in runtime integrations and must not be described
as a new 56-test run. No models were retrained in this review.


`artifacts/m5-step01-review-audit/pytest-full.log`:

```text
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
54 passed, 2 skipped, 2 warnings in 11.65s
```


`artifacts/m5-step01-review-audit/ruff-check.log`:

```text
All checks passed!
```


`artifacts/m5-step01-review-audit/ruff-format.log`:

```text
58 files already formatted
```


`artifacts/m5-step01-review-audit/pip-check.log`:

```text
No broken requirements found.
```


## Failures, caveats and unresolved hypotheses

- Existing Ruff cache panicked with `wrong package cache for file`; no-cache
  lint/format works. Initial replay on CPU failed a CUDA AUROC comparison
  (.93125 versus .927778) because tiny probability/tie changes affect ranks.
  Recorded-device replay passes tight checks. This is a reproducibility limit,
  not evidence of predictive improvement. Exact frozen-server equality also
  failed due to FedAvg rounding; local freezing remains exact and server
  numerical preservation passes 1e-6.
- The local Ollama endpoint was initially offline. Tabular diagnostics need no
  Qwen. Any restarted regression uses the existing frozen model, not a new SLM.
- Windows Ray shutdown/access-violation diagnostics and deprecated simulation
  launcher warnings remain; completed runs are checked by saved results, all
  three returned updates and independent numerical aggregation, not log silence.
- No independent split, future cohort, prospective outcome collection or formal
  significance test. Same 78 validation patients were previously reused in M2–M4.
  Seeds are not independent cohorts and origins are not independent patients.
- Calibration mismatch and policy/fusion coupling are plausible mechanisms but
  are not isolated causal explanations. Component freezing at random weights
  is limited; alternating updates, pretrained fixed fusion and nested calibration
  variants remain untested. No claim that centralization is deployable or private.
- Per-patient harm and label-aware oracle gains are retrospective loss summaries,
  not clinical harms or an unbiased off-policy estimate. Random routing has one
  fixed draw per seed, not a large repeated-policy study. Cost sensitivity holds
  models fixed; it does not retrain an optimal cost-aware policy.
- M4 sparse feedback, model-version staleness and counterfactual missingness
  remain structural limitations. Default clipping was inactive in the inspected
  parent-state replay, so it must not be blamed solely from its theoretical bias.

## Recommendations for Step 2 — proposals only

1. Prioritize a preregistered training/validation-only experiment separating
   origin-local gradient conflict, optimizer resets and local epoch count. Test
   fewer local steps or an explicitly separated router/fusion schedule against
   both matched centralized references; preserve the current default.
2. Retain exact-evidence mean and stop baselines and report BCE, calibration and
   threshold metrics together. Test pretrained fixed fusion and a train-only
   deterministic peer choice before expanding architecture.
3. Diagnose fold/final shift with nested train-only calibration and robust-logit
   experimental variants, without selecting on historical test performance.
4. Reconsider the action objective/cost protocol only after documenting a
   train-only decision rule for stop utility. Do not treat label-aware hindsight
   oracle choices as achievable routing or medical necessity.
5. For M4, preregister more feedback streams and measure coverage, reward age,
   ESS and clipped fractions at actual local steps. Evaluate independent future
   outcomes before making adaptation or clinical improvement claims.

No Step 2 implementation was performed. The specialist registry, flexible fusion
architecture, serving/security redesign and Qwen fine-tuning remain outside scope.

## Exact changed files

New:

- `src/fed_med_agents/experiments/routing_diagnostics.py`
- `src/fed_med_agents/experiments/m5_step01.py`
- `scripts/audit_m5_feedback.py`
- `scripts/extend_m5_controls.py`
- `scripts/verify_m5_step01.py`
- `scripts/report_m5_step01.py`
- `tests/test_routing_diagnostics.py`
- `docs/experiments/m5-step01-routing-diagnostics.md`
- `docs/devlog/2026-10-14-m5-step01.md`

Updated: `README.md`, `docs/status.md`, `docs/architecture.md`,
`docs/limitations.md`, `docs/next-steps.md`. Original M1–M4 reports remain historical
records and were not regenerated. The diagnostic pipeline is separate from the
reusable production routing/fusion/federation modules.

Raw evidence: `artifacts/m5-step01/seed_*/{results.json,evidence.npz,consultations.jsonl}`,
per-variant checkpoints/audits/histories, `artifacts/m5-step01/verification.json`,
`artifacts/m5-step01-central-pooled/seed_*/results.json`,
`artifacts/m5-step01-audit/{feedback.json,preservation.json,preserved-before.json}`,
fresh M4 output `artifacts/m5-step01-m4-rerun`, and QA/driver logs in the audit
directory. Full round-specific values and gradient weighting cosines remain in
these artifacts. [Execution log](../devlog/2026-10-14-m5-step01.md).
