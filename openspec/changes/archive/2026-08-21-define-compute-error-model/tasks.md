# Tasks

## Error Types

- [x] Define ComputeError
- [x] Define ComputeErrorCode
- [x] Define ComputeErrorPhase
- [x] Define ComputeErrorSeverity
- [x] Define ComputeDiagnostic
- [x] Define RecoveryHint

## Validation Errors

- [x] Define invalid tensor descriptor error
- [x] Define invalid shape error
- [x] Define invalid dtype error
- [x] Define invalid layout error
- [x] Define size overflow error
- [x] Define invalid graph error
- [x] Define cyclic graph error
- [x] Define missing input error
- [x] Define missing output error

## Capability and Provider Errors

- [x] Define no compatible Provider error
- [x] Define policy rejected Provider error
- [x] Define Provider unavailable error
- [x] Define Device unavailable error
- [x] Define unsupported operation error
- [x] Define unsupported dtype error
- [x] Define unsupported layout error
- [x] Define unsupported data movement error

## Resource Affinity Errors

- [x] Define incompatible resource affinity error
- [x] Define Provider-pinned resource error
- [x] Define Device-bound resource error
- [x] Define artifact fingerprint mismatch error
- [x] Define affinity group mismatch error

## Execution Errors

- [x] Define execution failed error
- [x] Define execution interrupted error
- [x] Define execution cancelled error
- [x] Define operation timeout error
- [x] Define out of memory error
- [x] Define resource exhausted error

## Diagnostics

- [x] Allow optional Provider diagnostics
- [x] Allow optional Device diagnostics
- [x] Allow optional backend diagnostic strings
- [x] Prevent diagnostics from defining stable contract behavior
- [x] Redact native handles and sensitive paths from diagnostics

## Recovery Hints

- [x] Define not-retryable hint
- [x] Define retry-before-state hint
- [x] Define restartable-with-replay hint
- [x] Define explicit-transfer-required hint
- [x] Define explicit-materialization-required hint
- [x] Define Provider-pinned hint

## Documentation

- [x] Document stable error categories
- [x] Document backend diagnostic rules
- [x] Document recovery hint semantics
- [x] Document relationship with Resource Affinity
- [x] Document relationship with Resolution Policy
