# Milestone 2 experiment — 2026-10-14

Implemented and executed learned directed routing, shared prediction fusion, three-client Flower FedAvg, and genuine server-mediated consultations. This establishes working mechanics; it does not establish superiority or privacy.

## Reconstructing the experiment

Read [architecture](../architecture.md) for equations, data provenance, payloads and exact implemented modules, [limitations](../limitations.md) for caveats, and [execution log](../devlog/2026-10-14-milestone-02.md) for commands/failures. Frozen configuration: seeds 42/43/44; immutable split seed 42; 364 training, 78 validation, 78 test; 4 OOF folds with nested 80/20 fit/calibration; specialists 30 epochs, batch 32, Adam 0.01; temperature 80 Adam steps, learning rate 0.05, bounded exp(logT) with logT in [-2.3,2.3]; Flower 5 rounds, 10 local shared epochs, batch 64, Adam 0.01, clip norm 5; maximum 1 consultation; cost 0.02 BCE units per call; FP32. Select the trained round with smallest mean-origin validation BCE, then freeze and evaluate test. No hyperparameter, seed, cost, or threshold was selected using test outcomes.

The objective is `mean_i sum_a softmax(score_i)[a] * (BCE(fusion_i,a, y_i) + 0.02 * |a|)`. The router's inputs contain no unconsulted probability. At deployment it selects argmax action, obtains only selected peer predictions via Flower, and applies masked fusion. Training exploration obtains all OOF outcomes, so deployment call counts below do not describe training traffic. Final models and test predictions are saved for all origins/policies.

## Mean test metrics across three configured seeds

| Model/policy | Accuracy | AUROC | F1 | BCE | Calls/patient | Collaboration % | Amortized ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 calibrated specialist A | 0.6197 | 0.6807 | 0.6675 | 0.6382 | 0.0000 | 0.0000 | 1.4082 |
| M2 calibrated specialist B | 0.9103 | 0.9289 | 0.9263 | 0.2921 | 0.0000 | 0.0000 | 1.7933 |
| M2 calibrated specialist C | 0.9060 | 0.9242 | 0.9208 | 0.3132 | 0.0000 | 0.0000 | 1.8251 |
| M2 fixed specialist mean (all evidence) | 0.9316 | 0.9593 | 0.9436 | 0.3379 | 2.0000 | 100.0000 | 4.6434 |
| Fixed all-peer consultation (same mean) | 0.9316 | 0.9593 | 0.9436 | 0.3379 | 2.0000 | 100.0000 | 6.3295 |
| Random routing, max budget 1 | 0.8205 | 0.8912 | 0.8466 | 0.4170 | 1.0000 | 100.0000 | 7.3478 |
| Confidence routing, max budget 1 | 0.8077 | 0.8394 | 0.8359 | 0.5090 | 0.6581 | 65.8120 | 2.7659 |
| Learned routing + fusion, origin A | 0.8162 | 0.8792 | 0.8428 | 0.4277 | 1.0000 | 100.0000 | 3.5438 |

Sample standard deviations (ddof=1), same fixed patient split:

| Model/policy | Accuracy | AUROC | F1 | BCE | Calls/patient | Collaboration % | Amortized ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 fixed specialist mean (all evidence) | 0.0074 | 0.0004 | 0.0065 | 0.0076 | 0.0000 | 0.0000 | 0.6733 |
| Fixed all-peer consultation (same mean) | 0.0074 | 0.0004 | 0.0065 | 0.0076 | 0.0000 | 0.0000 | 0.7479 |
| Random routing, max budget 1 | 0.0588 | 0.0542 | 0.0577 | 0.1182 | 0.0000 | 0.0000 | 0.7365 |
| Confidence routing, max budget 1 | 0.0222 | 0.0124 | 0.0238 | 0.0595 | 0.0267 | 2.6688 | 0.0752 |
| Learned routing + fusion, origin A | 0.0323 | 0.0170 | 0.0330 | 0.0504 | 0.0000 | 0.0000 | 1.5030 |

The primary learned row originates at A. Specialist B/C rows are their own local inference, not consultations from A. Fixed mean/all receive all three calibrated specialist probabilities and require two peer consultations. Random and learned have the same maximum budget of one, use the same selected fusion checkpoint, and use one call per A-origin patient in the observed runs; confidence can abstain. Fixed mean and fixed all are deliberately the same predictor, not two independent algorithms. No learned all-evidence fusion ablation is claimed.

Learned A-origin accuracy change: +0.1966 versus A alone; -0.0043 versus matched-maximum-budget random; -0.1154 versus full-evidence fixed mean; -0.1068 versus historical M1 shared-model mean. Comparisons against full evidence and historical M1 are not evidence-matched causal claims. Individual seeds can favor different policies. ECE is saved in per-seed metrics; temperature calibration does not guarantee correctness confidence.

## Historical masked-model baseline on identical held-out IDs

| M1 view | Accuracy | AUROC | F1 | BCE |
|---|---:|---:|---:|---:|
| A | 0.6154 | 0.6753 | 0.6250 | 0.6425 |
| B | 0.9103 | 0.9285 | 0.9263 | 0.3080 |
| C | 0.8462 | 0.8719 | 0.8696 | 0.4119 |
| mean_probability_ensemble | 0.9231 | 0.9347 | 0.9375 | 0.3919 |

These values were verified by loading the preserved M1 checkpoint. M1 trained a masked-input shared classifier, not the new calibrated specialists/router/fusion. Its original runtime was torch 2.14.1+cpu. No new five-round M1 training run is claimed during M2.

## Learned adjacency and transparent decisions

Seed 42, selected round 5 (rows origin, columns destination A/B/C):

```text
[0.000000, 0.585629, 0.414371]
[0.368445, 0.000000, 0.631555]
[0.373372, 0.626628, 0.000000]
```

Seed 43, selected round 2 (rows origin, columns destination A/B/C):

```text
[0.000000, 0.492384, 0.507616]
[0.445343, 0.000000, 0.554657]
[0.435407, 0.564593, 0.000000]
```

Seed 44, selected round 5 (rows origin, columns destination A/B/C):

```text
[0.000000, 0.501896, 0.498104]
[0.381933, 0.000000, 0.618067]
[0.376293, 0.623707, 0.000000]
```

The matrices are normalized edge priors, not observed consultation frequencies. Actual scores include contextual network output. Example from seed 42 (not a verbal rationale):

```json
{
  "patient_id": "case-0507",
  "origin": 0,
  "origin_probability": 0.5784837007522583,
  "calibrated_probability_confidence": 0.1569674015045166,
  "actions": [
    [
      0.0,
      0.0,
      0.0
    ],
    [
      1.0,
      0.0,
      0.0
    ],
    [
      0.0,
      1.0,
      0.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "scores": [
    -12.39394474029541,
    null,
    8.133195877075195,
    2.0774407386779785
  ],
  "selected": [
    1
  ]
}
```

Action ordering is empty/A/B/C. Self-action A is masked (null represents −∞). The selected B score exceeds C and the stop score using only origin probability/confidence and public availability. Other seeds can select C. No private candidate probability or true label was available to this decision.

## Real Flower evidence

| Seed | Device | In-memory Flower run ID | Selected round | Verified client updates | Consultation events |
|---|---|---|---:|---:|---:|
| 42 | cuda | 6004431366139231 | 5 | 15 | 8820 |
| 43 | cuda | 11776220363628869820 | 2 | 15 | 8992 |
| 44 | cuda | 14463032291229267609 | 5 | 15 | 8891 |
| 42 | cpu | 18372766845876811423 | 5 | 15 | 8820 |

Each run received all three shared updates in each of five rounds (15 updates), with zero training failures. Router and fusion parameter changes were independently verified. Private checkpoint hashes remained unchanged. The checker verifies every OOF patient's excluded fitting/calibration IDs, train-only preprocessing, client feature width, lack of val/test labels in client files, training-only exploration IDs, request association and test occurring after all validation queries. In-memory run IDs identify the executed simulations; they are not persistent CLI SuperLink run histories. Full traces/logs: `artifacts/milestone02/seed_{42,43,44}/consultations.jsonl`, `flower.log`, `metrics.json`, and `client_*/round_*.json`.

## GPU, timing, and CPU fallback

Device: NVIDIA RTX A4000; physical VRAM 15.992 GiB; Python 3.11.9; PyTorch 2.11.0+cu128; CUDA wheel runtime 12.8; CUDA available True; real tensor checksum 3680.0. Original CPU-only build was replaced only inside the venv. Driver/OS were unchanged. One full-GPU Ray actor sequentially executes the three logical clients; no concurrent GPU clients were enabled. Shared training supports opt-in AMP, but experiments used FP32. CUDA AMP gradients passed a numerical smoke test.

| Seed | Device | Specialist/OOF prep seconds | Server workflow seconds | Sum local shared train seconds | Max torch allocated MiB |
|---|---|---:|---:|---:|---:|
| 42 | cuda | 6.899 | 22.466 | 3.380 | 17.377 |
| 43 | cuda | 6.136 | 22.347 | 3.087 | 17.377 |
| 44 | cuda | 6.649 | 23.780 | 3.307 | 17.377 |
| 42 | cpu | 4.543 | 18.473 | 1.942 | 0.000 |

Preparation includes dataset fetching, fitting and calibration. Server workflow includes Flower setup queries, exploration, shared training, validation, and all test policies; it is not kernel-only time. The latest GPU rerun reused prepared OOF files to validate the final communication changes. 79 nvidia-smi samples: board GPU utilization mean 8.16%, range 0–51%. Board memory range 1098–1570 MiB. These include desktop/background processes.

CPU seed 42 completed all five rounds and all consultations; learned A-origin accuracy 0.7821, AUROC 0.8597, F1 0.8090, BCE 0.4841. The tiny dataset and measured preparation/round-trip overhead provide no reason to claim GPU saturation or throughput advantage. Torch allocated memory excludes CUDA contexts/reserved memory. Inference times are amortized batch wall times, not single-patient deployment latency.

## Tests and checks

```text
..................                                                       [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\typer\__init__.py:24
  <venv>/site-packages/typer/__init__.py:24: DeprecationWarning: 'click.utils.get_binary_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_binary_stream as get_binary_stream

.venv\Lib\site-packages\typer\__init__.py:25
  <venv>/site-packages/typer/__init__.py:25: DeprecationWarning: 'click.utils.get_text_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
18 passed, 2 warnings in 3.56s
```

Ruff check, Ruff format check, and pip check exited 0. Saved evidence verification passed all four runs. Tests cover general graph dimensions/budget/self mask, unavailable domains/nodes, router and fusion gradients on CPU/CUDA, specialist isolation, checkpoint/ArrayRecord roundtrip, unavailable/incorrect patient and destination replies, real Message transport adapter, heldout-cache rejection, training-only OOF preprocessing/calibration and label perturbation, unconsulted evidence exclusion, CUDA AMP finite gradients, explicit CUDA failure, and actual three-client participation/provenance. On machines without CUDA, CUDA-specific tests skip explicitly. Saved-artifact verification test skips only when experiments have not been run.

## Outstanding limitations and next work

Learned routing/fusion did not outperform fixed full evidence. The A-origin stop action was never selected: observed deployment collaboration rate is 100%. Budget/cost sensitivity, independent split variability, validation-driven improvements, an all-evidence learned fusion control, authenticated transport, privacy defenses, and persistent standalone serving remain undone. Windows Ray shutdown diagnostics and deprecated Python launcher remain documented. No SLMs, multi-hop routing, package publication, GitHub push or OS installation occurred.

## Per-seed per-origin metrics

| Model/policy | Accuracy | AUROC | F1 | BCE | Calls/patient | Collaboration % | Amortized ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42, none, origin A | 0.6282 | 0.6788 | 0.6813 | 0.6349 | 0.0000 | 0.0000 | 1.4854 |
| seed 42, none, origin B | 0.9103 | 0.9271 | 0.9263 | 0.3014 | 0.0000 | 0.0000 | 1.3109 |
| seed 42, none, origin C | 0.9103 | 0.9267 | 0.9247 | 0.3033 | 0.0000 | 0.0000 | 1.3554 |
| seed 42, fixed_mean, origin A | 0.9359 | 0.9590 | 0.9474 | 0.3466 | 2.0000 | 100.0000 | 4.1601 |
| seed 42, fixed_mean, origin B | 0.9359 | 0.9590 | 0.9474 | 0.3466 | 2.0000 | 100.0000 | 4.1500 |
| seed 42, fixed_mean, origin C | 0.9359 | 0.9590 | 0.9474 | 0.3466 | 2.0000 | 100.0000 | 4.0949 |
| seed 42, random, origin A | 0.8077 | 0.8840 | 0.8352 | 0.4566 | 1.0000 | 100.0000 | 8.1979 |
| seed 42, random, origin B | 0.9231 | 0.9708 | 0.9362 | 0.1945 | 1.0000 | 100.0000 | 7.4636 |
| seed 42, random, origin C | 0.9103 | 0.9625 | 0.9247 | 0.2453 | 1.0000 | 100.0000 | 7.0158 |
| seed 42, confidence, origin A | 0.7821 | 0.8250 | 0.8090 | 0.5775 | 0.6795 | 67.9487 | 2.8384 |
| seed 42, confidence, origin B | 0.9103 | 0.9549 | 0.9247 | 0.2354 | 0.4744 | 47.4359 | 5.3173 |
| seed 42, confidence, origin C | 0.8974 | 0.9260 | 0.9130 | 0.3231 | 0.3462 | 34.6154 | 2.7889 |
| seed 42, learned, origin A | 0.7821 | 0.8597 | 0.8090 | 0.4841 | 1.0000 | 100.0000 | 2.6542 |
| seed 42, learned, origin B | 0.9103 | 0.9729 | 0.9247 | 0.1915 | 1.0000 | 100.0000 | 2.6716 |
| seed 42, learned, origin C | 0.9103 | 0.9632 | 0.9247 | 0.2445 | 1.0000 | 100.0000 | 2.7599 |
| seed 43, none, origin A | 0.6026 | 0.6747 | 0.6265 | 0.6488 | 0.0000 | 0.0000 | 1.3820 |
| seed 43, none, origin B | 0.9103 | 0.9306 | 0.9263 | 0.2918 | 0.0000 | 0.0000 | 2.6901 |
| seed 43, none, origin C | 0.9103 | 0.9302 | 0.9247 | 0.2846 | 0.0000 | 0.0000 | 2.7224 |
| seed 43, fixed_mean, origin A | 0.9231 | 0.9590 | 0.9362 | 0.3329 | 2.0000 | 100.0000 | 5.4124 |
| seed 43, fixed_mean, origin B | 0.9231 | 0.9590 | 0.9362 | 0.3329 | 2.0000 | 100.0000 | 6.6703 |
| seed 43, fixed_mean, origin C | 0.9231 | 0.9590 | 0.9362 | 0.3329 | 2.0000 | 100.0000 | 6.7806 |
| seed 43, random, origin A | 0.8846 | 0.9486 | 0.9091 | 0.2842 | 1.0000 | 100.0000 | 6.9422 |
| seed 43, random, origin B | 0.9103 | 0.9382 | 0.9263 | 0.3405 | 1.0000 | 100.0000 | 4.8463 |
| seed 43, random, origin C | 0.9487 | 0.9625 | 0.9583 | 0.2182 | 1.0000 | 100.0000 | 4.1220 |
| seed 43, confidence, origin A | 0.8205 | 0.8458 | 0.8542 | 0.4710 | 0.6282 | 62.8205 | 2.6883 |
| seed 43, confidence, origin B | 0.9103 | 0.9549 | 0.9247 | 0.2779 | 0.3718 | 37.1795 | 2.6742 |
| seed 43, confidence, origin C | 0.9103 | 0.9253 | 0.9247 | 0.2972 | 0.2821 | 28.2051 | 3.9370 |
| seed 43, learned, origin A | 0.8462 | 0.8910 | 0.8750 | 0.3871 | 1.0000 | 100.0000 | 2.6981 |
| seed 43, learned, origin B | 0.9231 | 0.9722 | 0.9362 | 0.2351 | 1.0000 | 100.0000 | 4.0524 |
| seed 43, learned, origin C | 0.9487 | 0.9597 | 0.9583 | 0.2240 | 1.0000 | 100.0000 | 4.0111 |
| seed 44, none, origin A | 0.6282 | 0.6885 | 0.6947 | 0.6309 | 0.0000 | 0.0000 | 1.3573 |
| seed 44, none, origin B | 0.9103 | 0.9292 | 0.9263 | 0.2832 | 0.0000 | 0.0000 | 1.3788 |
| seed 44, none, origin C | 0.8974 | 0.9156 | 0.9130 | 0.3518 | 0.0000 | 0.0000 | 1.3977 |
| seed 44, fixed_mean, origin A | 0.9359 | 0.9597 | 0.9474 | 0.3341 | 2.0000 | 100.0000 | 4.3577 |
| seed 44, fixed_mean, origin B | 0.9359 | 0.9597 | 0.9474 | 0.3341 | 2.0000 | 100.0000 | 4.2309 |
| seed 44, fixed_mean, origin C | 0.9359 | 0.9597 | 0.9474 | 0.3341 | 2.0000 | 100.0000 | 6.8301 |
| seed 44, random, origin A | 0.7692 | 0.8410 | 0.7955 | 0.5104 | 1.0000 | 100.0000 | 6.9031 |
| seed 44, random, origin B | 0.9103 | 0.9542 | 0.9247 | 0.2284 | 1.0000 | 100.0000 | 5.5642 |
| seed 44, random, origin C | 0.8974 | 0.9250 | 0.9130 | 0.3624 | 1.0000 | 100.0000 | 6.0031 |
| seed 44, confidence, origin A | 0.8205 | 0.8472 | 0.8444 | 0.4784 | 0.6667 | 66.6667 | 2.7711 |
| seed 44, confidence, origin B | 0.9103 | 0.9507 | 0.9247 | 0.2593 | 0.3846 | 38.4615 | 5.3731 |
| seed 44, confidence, origin C | 0.8846 | 0.9167 | 0.9032 | 0.3729 | 0.2949 | 29.4872 | 2.7793 |
| seed 44, learned, origin A | 0.8205 | 0.8868 | 0.8444 | 0.4119 | 1.0000 | 100.0000 | 5.2792 |
| seed 44, learned, origin B | 0.9103 | 0.9715 | 0.9247 | 0.2144 | 1.0000 | 100.0000 | 2.7360 |
| seed 44, learned, origin C | 0.9103 | 0.9597 | 0.9247 | 0.2703 | 1.0000 | 100.0000 | 2.8181 |

All Markdown report paths are relative to the repository root. Raw numeric source artifacts remain gitignored; this report embeds the main quantitative evidence so an independent reviewer does not require conversation history.
