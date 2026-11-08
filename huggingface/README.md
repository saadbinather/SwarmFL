---
license: apache-2.0
library_name: swarmfl
tags:
  - federated-learning
  - flower
  - multi-agent-systems
  - tabular-classification
  - research
---

# SwarmFL

Small Flower demo for authorized specialist consultation. Synthetic only; not a
benchmark or clinical model.

## Quickstart

```bash
pip install swarmfl==0.1.0
swarmfl-quickstart --output demo --feedback
python -c "import json; print(json.load(open('demo/result.json')))"
```

The command starts three local Flower nodes, registers their declarations, performs
a selected one-peer consultation on a new case, and writes the fused probability
and audit artifacts under `demo/`. The synthetic specialists use the ordered input
views `(x0,x1)`, `(x2,x3)`, and `(x4,x5)`.

The demo keeps specialist views separate, makes one selected peer request, and
fuses the observed predictions.

```bash
python demo.py --feedback
```

If needed, install the package first:

```bash
pip install swarmfl==0.1.0
```

The public Hub release contains no research datasets, frozen tests, private
checkpoints, or experiment outputs. See the [repository](https://github.com/saadbinather/SwarmFL)
for source and research context.
