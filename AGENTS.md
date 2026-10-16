# Persistent project instructions

For every implementation task, inspect the current source, README, status, architecture, limitations, and relevant experiment reports before editing. Preserve working milestones and their evidence. Do not rename the package for aesthetics.

Create a dated `docs/devlog/YYYY-MM-DD-task-name.md` execution report at the start and finish it before reporting completion. Record implementation decisions, pipeline, changed files, actual commands, execution results, tests, failed attempts, limitations, unresolved issues, and prioritized next tasks. Never invent evidence or mark planned work as implemented.

Maintain `docs/status.md`, `docs/architecture.md`, `docs/limitations.md`, `docs/next-steps.md`, and relevant `docs/experiments/milestone-XX.md` reports. Update README instructions when behavior changes. Reports must be understandable independently of chat history and must distinguish historical artifacts from reruns.

Keep reusable routing, fusion, federation, and consultation logic separate from disease-specific configuration. Add type hints and document public APIs. Preserve patient alignment; avoid preprocessing, calibration, stacking, validation, and test leakage. Test only after model selection is frozen.

Use genuine Flower communication for federation and consultations. Never send private training feature rows or specialist weights in consultations. New-patient inference may forward only the explicitly authorized feature subset required by a selected specialist, as authorized in Milestone 3; preserve this distinction in documentation and tests. Document that prediction, identifier, and model-update disclosure is not a proven privacy guarantee. SLM agents were authorized in Milestone 3; their initial pretrained weights are frozen, and conversation logs are not parameter learning. Do not push or publish without explicit instructions.

Audit GPU capability and actual CUDA operations before claiming GPU use. Support auto/cuda/cpu. Start Ray with at most one GPU client worker. Do not modify drivers, OS, WSL, or Docker. Record measured runtime and memory; do not infer GPU saturation from device availability.

Run applicable tests, Ruff, and end-to-end experiments. Document failures and limitations candidly. Keep artifacts gitignored, retain earlier milestone outputs, and link exact report paths in the final response.
