# CPU quickstart

Install SwarmFL and run the synthetic demonstration with delayed feedback:

```console
$ python -m pip install swarmfl
$ swarmfl-quickstart --output demo --feedback
```

The command runs three real CPU Flower clients using synthetic records. It
demonstrates selected-only consultation, dynamic fusion, a verified feedback
event, candidate activation, and rollback. Results are written to
`demo/result.json` and have no clinical meaning.

See the maintained [quickstart](../quickstart.md) for the concise project
instructions and [limitations](../limitations.md) before interpreting the demo.
