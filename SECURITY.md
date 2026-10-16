# Security policy

SwarmFL is research software and has no supported production-security release.
Report suspected vulnerabilities through the repository's GitHub private
vulnerability-reporting route:
`https://github.com/saadbinather/SwarmFL/security/advisories/new`. If that route is
not enabled, ask the maintainer to enable private reporting through a minimal
public issue that contains no vulnerability details. Do not include patient data,
credentials, signing keys, or exploitable deployment details in a public issue.

The current system is a trusted local simulation. Authorization metadata,
signatures, audit logs, opaque identifiers, and selected-only messages reduce
accidental disclosure but do not prove confidentiality. Prediction values,
identifiers, timing, logs, and shared model updates can disclose information.
SwarmFL does not claim HIPAA/GDPR compliance, differential privacy, secure
aggregation, transport isolation, authenticated remote services, or resistance
to malicious clients. Add deployment-appropriate authentication, encryption,
key management, isolation, retention, and incident response before non-local use.

Never use demonstration credentials or synthetic signing material in a deployment.
No remote unauthenticated service is started by package import.

## Known dependency boundaries

The supported quickstart uses Flower's local simulation extra, which currently
pins Ray 2.55.1. That Ray version is affected by PYSEC-2026-2273 in the default
decoder for attacker-controlled WebDataset TAR members. SwarmFL does not use Ray
Data, does not call `read_webdataset`, and accepts no WebDataset input. Do not add
untrusted Ray Data ingestion. Upgrade when Flower supports a patched Ray release.
The owner accepts this residual risk only for the trusted-local 0.1.0a1 research
release; it is not accepted for remote or adversarial deployment.

PyTorch constrains the installed setuptools version below the fix for
PYSEC-2026-3447. SwarmFL builds distributions in PEP 517 isolation with
setuptools >=83 and separately scans final wheel/sdist contents. Do not build a
release with the runtime setuptools copy or publish an uninspected archive.
