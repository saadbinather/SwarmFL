# Milestone 1 retrospective — verified 2026-10-14

Evidence source: preserved `artifacts/milestone01/` copied before M2 changes. Existing recorded CLI run `9761394553627465436`; Flower 1.39.0; original torch 2.14.1 CPU; Python 3.11.9; Windows. Original logs record five rounds with three client replies and zero failures. Received participation and saved audits show 15 updates. Initial/final shared L2 change 5.00756439. Checkpoint re-evaluation at the start of M2 reproduced mean accuracy 0.9230769, AUROC 0.9347222, F1 0.9375 and BCE 0.3919495. This is a historical evidence audit and checkpoint replay, not a claimed new M1 training run.

UCI 529 was fetched via ucimlrepo, with 520 rows, 16 expected columns, 320 positive/200 negative labels. Stratified 70/15/15 split seed 42: 364/78/78. Train-fitted age scaling and fixed binary mappings. A gets age/gender/obesity, B gets five primary symptoms, C gets eight remaining symptoms; all share aligned patient IDs and labels. Three independent specialists (`feature width→24→1`) train 30 epochs on CPU. Only the separate shared `32→24→1` model (16 zero-masked values plus 16 availability masks) is federated, using Flower's current ServerApp/ClientApp, ArrayRecord and FedAvg for five rounds, 10 local epochs each. Adam 0.01, batch 32, BCEWithLogitsLoss, fixed seeds, no optimizer aggregation. Mean-probability fusion is untrained.

| Model | Accuracy | AUROC | F1 | BCE |
|---|---:|---:|---:|---:|
| Private specialist A | 0.6026 | 0.6760 | 0.6437 | 0.6487 |
| Private specialist B | 0.9103 | 0.9271 | 0.9263 | 0.2980 |
| Private specialist C | 0.9103 | 0.9323 | 0.9247 | 0.3028 |
| Shared A | 0.6154 | 0.6753 | 0.6250 | 0.6425 |
| Shared B | 0.9103 | 0.9285 | 0.9263 | 0.3080 |
| Shared C | 0.8462 | 0.8719 | 0.8696 | 0.4119 |
| Shared mean_probability_ensemble | 0.9231 | 0.9347 | 0.9375 | 0.3919 |

Four initial tests passed before M2, covering split/statistic leakage, feature isolation/alignment, compatible shared shapes/private parameter separation, and reply payload content. CPU inference replay and dependency checks succeeded. No CUDA operation was possible in the original CPU-only torch wheel; nvidia-smi nevertheless detected the A4000. Existing logs included experimental Windows Ray shutdown diagnostics. Ubuntu/WSL was not installed or verified.

The simulator's shared dataset file was readable by all clients; raw training rows were absent from update messages, but that is not a privacy guarantee. The cohort is not three hospitals, AUROC/accuracy are not clinical validation, and the shared C view was worse than private C. Original M1 scripts remain runnable. M2 source formatting/import cleanup was nonbehavioral; the torch requirement was widened to permit the tested CUDA build, so new M1 reruns may differ slightly from historical values.
