# Tasks

## Health Types

- [x] Define ProviderHealth
- [x] Define DeviceHealth
- [x] Define CapabilityHealth
- [x] Define HealthState
- [x] Define HealthScope
- [x] Define HealthReport
- [x] Define HealthDiagnostic
- [x] Define HealthTimestamp
- [x] Define HealthTimeToLive

## Health States

- [x] Define unknown state
- [x] Define initializing state
- [x] Define available state
- [x] Define degraded state
- [x] Define saturated state
- [x] Define draining state
- [x] Define unavailable state
- [x] Define interrupted state

## Provider Reporting

- [x] Allow Providers to report Provider-level health
- [x] Allow Providers to report Device-level health
- [x] Allow Providers to report Capability-level health
- [x] Allow Providers to report capacity hints
- [x] Allow Providers to report diagnostic metadata

## Runtime Integration

- [x] Use health during Resolution Policy evaluation
- [x] Use health during Execution Planning
- [x] Use health during Scheduler admission
- [x] Use health before Provider Execution API submission
- [x] Preserve Resource Affinity when health changes
- [x] Reject implicit Provider migration

## Scheduler Integration

- [x] Reject scheduling to unavailable Providers
- [x] Reject scheduling to unavailable Devices
- [x] Allow degraded Providers when policy permits
- [x] Allow saturated Providers to apply backpressure
- [x] Report interruption when running work cannot continue

## Diagnostics

- [x] Expose stable health diagnostics
- [x] Redact backend-private details
- [x] Redact native handles
- [x] Redact credentials and filesystem paths
- [x] Attach trace identifiers when available

## Errors

- [x] Define Provider unavailable error
- [x] Define Device unavailable error
- [x] Define Provider degraded rejection error
- [x] Define Provider saturated error
- [x] Define Provider interrupted error
- [x] Define stale health report error

## Documentation

- [x] Document Provider Health lifecycle
- [x] Document Device Health lifecycle
- [x] Document relationship with Resolution Policy
- [x] Document relationship with Scheduler
- [x] Document relationship with Provider Execution API
- [x] Document no automatic failover guarantee
