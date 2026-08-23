# Provider Status

Provider status is split into separate Runtime-owned dimensions.

Lifecycle describes Runtime management stage: registered, loading,
initializing, ready, draining, stopped, failed, or removed. Lifecycle
transitions are explicit and do not by themselves grant execution admission.

Health describes whether the Provider appears internally functional: unknown,
healthy, degraded, unhealthy, or failed. Health is not readiness. A Provider
can be healthy while warming, stale, draining, or saturated.

Readiness describes whether new work may be admitted: not-ready, ready,
read-only, or draining. Resolution and Scheduler admission require readiness in
addition to health.

Pressure describes load and capacity: unknown, low, moderate, high, or
saturated. Pressure can be derived from active operation count, queue depth,
memory pressure, device memory pressure, estimated queue delay, utilization, or
Provider-specific admission limits. Saturation rejects ordinary new work by
default, but it is not a Provider failure.

Admission is the status-derived decision for a scope: admit, prefer-not, or
reject. Admission may be scoped to Provider, Device, Capability, or operation
family. High pressure or degraded health can produce `prefer-not`; not-ready,
draining, unhealthy, failed, stale, or saturated status rejects ordinary new
work unless policy explicitly allows a narrower case.

Provider status snapshots are immutable decision inputs. A snapshot records
lifecycle, health, readiness, pressure, admission, severity, reason, freshness
metadata, Device status, Capability status, optional operation-family status,
and drain progress. If a status report has a TTL and the TTL expires, Runtime
policy treats the report as stale; stale status is not fully ready by default.

Device-level status allows one Device to be unavailable while the Provider
itself remains healthy. Capability-level status allows a Provider to be ready
for Compute while another Capability is unavailable. Operation-family status is
optional; when absent, Runtime falls back to Capability-level status.

Draining stops ordinary new unpinned work while allowing policy-controlled
handling of in-flight work and existing Provider-pinned resources. Drain does
not imply migration. Provider-pinned and Device-bound resources remain governed
by Resource Affinity; moving them requires explicit data movement.

Interruptions such as device reset, driver loss, device removal, allocator
failure, OOM recovery, thermal throttling, and administrative drain map into
health, readiness, and admission. Refusal due to readiness, pressure, draining,
or staleness is reported separately from execution failure after submission.

Resolution records status-based candidate decisions and preserves Resource
Affinity precedence. Scheduler rechecks the selected Provider and Device
before submission because status may change after planning.

Provider status observations are non-authoritative. They report lifecycle,
health, readiness, pressure, admission, stale status, drain start/completion,
Device status, and Capability status, but they never control Runtime state.
