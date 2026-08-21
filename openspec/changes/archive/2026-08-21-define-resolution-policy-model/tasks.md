# Tasks

## Policy Types

- [x] Define ResolutionPolicy
- [x] Define ResolutionPolicyId
- [x] Define ResolutionContext
- [x] Define ResolutionCandidate
- [x] Define ResolutionDecision

## Built-in Policies

- [x] Define deterministic policy
- [x] Define priority policy
- [x] Define availability policy
- [x] Define performance-preferred policy placeholder
- [x] Define energy-preferred policy placeholder
- [x] Define memory-constrained policy placeholder

## Candidate Evaluation

- [x] Evaluate Capability compatibility
- [x] Evaluate Provider health
- [x] Evaluate Device availability
- [x] Evaluate Resource Affinity compatibility
- [x] Evaluate fallback classification
- [x] Evaluate execution phase

## Runtime Integration

- [x] Apply ResolutionPolicy during Component import resolution
- [x] Apply ResolutionPolicy during dependent resource validation
- [x] Preserve existing Resource Affinity for Provider-pinned resources
- [x] Return structured no-compatible-provider errors
- [x] Return structured policy-rejected-provider errors

## Fallback Semantics

- [x] Allow transparent re-resolution before observable work
- [x] Allow restartable re-resolution only with replayable inputs
- [x] Reject implicit re-resolution of Provider-pinned live state
- [x] Document that this change does not migrate live state

## Observability

- [x] Record selected Provider
- [x] Record selected Device when available
- [x] Record selected Capability version
- [x] Record policy decision reason
- [x] Expose diagnostics without making backend strings part of the stable contract

## Documentation

- [x] Document resolution lifecycle
- [x] Document built-in policies
- [x] Document examples for CPU/GPU selection
- [x] Document examples for generation-session pinning
- [x] Document examples for transparent versus restartable selection
