"""Minimal external specialist declaration; no core edits are required."""

import hashlib

from fed_med_agents import FeatureSchema, ModelAdapter, OutputSchema, Specialist, TaskSchema

task = TaskSchema(
    name="example-risk",
    labels=("negative", "positive"),
    label_definition="Application-defined binary outcome",
)
specialist = Specialist(
    specialist_id="my-external-model",
    task=task,
    output_schema=OutputSchema(positive_label="positive"),
    model_adapter=ModelAdapter(
        adapter_id="my-package:predict_proba:v1",
        model_family="external-example",
        checkpoint_sha256=hashlib.sha256(b"replace-with-real-checkpoint-bytes").hexdigest(),
        feature_names=("feature_a", "feature_b"),
        input_dim=2,
        alignment_namespace="my-application-v1",
    ),
    feature_schema=(FeatureSchema(name="feature_a"), FeatureSchema(name="feature_b")),
    capabilities=frozenset({"new_patient_prediction"}),
)
print(specialist.model_dump_json(indent=2))
