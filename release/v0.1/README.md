# Magnetar v0.1.0 Release Evidence

Date: 2026-08-27
Status: release-candidate evidence bundle for the `codex/build-issues` PR.

This directory materializes the v0.1.0 release-policy evidence into repository
artifacts. It is intentionally conservative: unsupported, deferred, unsigned,
or not-generated release gates are recorded as limitations rather than implied.

Artifacts:

- `compatibility-matrix.json`
- `conformance-report.json`
- `security-report.json`
- `provenance.json`
- `release-statement.md`
- `checksums.sha256`

The v0.1.0 statement is a CPU-local inference runtime baseline. CUDA, Metal,
OpenVINO, QNN, WebGPU, production server APIs, model hub downloads, and
agent/tool Runtime capabilities are not included in the stable baseline.
