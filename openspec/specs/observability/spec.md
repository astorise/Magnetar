# observability Specification

## Purpose
This specification defines runtime observability events, redaction, exporters, dynamic debug policy, sink authorization, and report mapping.
## Requirements
### Requirement: Observability Component Model

Magnetar SHALL support observability integrations implemented as WASM
Components.

Observability Components SHALL consume stable Magnetar observability contracts.

Observability Components SHALL NOT receive Provider-native implementation
objects.

#### Scenario: Load observability Component

Given a compatible Observability Component is available

When Runtime policy allows the Component

Then the Runtime may load and instantiate it as an observability integration.

---

### Requirement: Observability Components Are Not Providers

Observability Components SHALL NOT be treated as Providers.

They SHALL NOT participate in:

- Capability Provider resolution
- Device selection
- Compute Execution Planning
- Memory Planning
- Provider execution selection

#### Scenario: Resolve compute Capability

Given an OpenTelemetry exporter Component is active

When the Runtime resolves `magnetar:compute/run`

Then the exporter is not considered as a Provider candidate.

---

### Requirement: Observability Consumption Models

Magnetar SHALL support both stream-based and snapshot-based observability
consumption.

Stream-based consumption SHALL be available for event-oriented telemetry.

Snapshot-based consumption SHALL be available for aggregated Runtime state.

#### Scenario: Different consumers use different models

Given an OpenTelemetry exporter and a Prometheus exposition Component are active

When telemetry is consumed

Then OpenTelemetry may consume the observation stream

And Prometheus may consume Runtime metric snapshots.

---

### Requirement: Observability Emit Capability

Magnetar SHALL define a portable `magnetar:observability/emit` Capability.

The Capability SHALL allow authorized Components to emit custom observations.

Custom observations MAY include:

- metrics
- structured logs
- structured events
- diagnostics

#### Scenario: Component emits metric

Given a Component imports `magnetar:observability/emit`

When the Component submits a valid custom metric

Then the Runtime validates and records that metric in the observability plane.

---

### Requirement: Custom Metric Model

Custom metrics SHALL use portable values.

A custom metric SHALL include:

- metric name
- metric kind
- numeric value
- optional tags

Metric kinds SHALL include at least:

- counter
- gauge
- histogram

#### Scenario: Emit counter

Given a Component submits a counter metric

When the Runtime accepts it

Then the Runtime records the metric under the Component's authorized namespace.

---

### Requirement: Metric Namespace Scoping

Custom metric names SHALL be subject to Runtime policy.

A Component SHALL NOT emit metrics outside its authorized metric namespace.

#### Scenario: Unauthorized metric namespace

Given a Component is authorized for `component.foo.*`

When it attempts to emit `system.scheduler.queue_depth`

Then the Runtime rejects the metric with an access-denied error.

---

### Requirement: Observability Reader Capability

Magnetar SHALL define a portable `magnetar:observability/reader` Capability.

The reader SHALL expose aggregated Runtime observability snapshots.

The reader SHOULD be restricted to privileged observability or system Components.

#### Scenario: Read Runtime metrics

Given an authorized Prometheus Component imports
`magnetar:observability/reader`

When it requests a metrics snapshot

Then the Runtime returns the current aggregated portable snapshot.

---

### Requirement: Runtime Metrics Snapshot

The Runtime Metrics Snapshot SHALL be a stable portable snapshot schema.\r\n\r\nThe Runtime Metrics Snapshot MAY include:

- submitted operations
- running operations
- completed operations
- failed operations
- cancelled operations
- interrupted operations
- queue depth
- Provider count
- Device count
- available Provider count
- available Device count
- estimated memory pressure
- observation queue depth
- dropped observation count

#### Scenario: Empty Runtime snapshot

Given no compute operations have executed

When a metrics snapshot is requested

Then valid counters and gauges are returned with zero values where applicable.

---

### Requirement: Snapshot Independence

Aggregated Runtime metrics SHALL exist independently of external exporters.

Disabling all exporters SHALL NOT disable Runtime metrics aggregation when the
observability subsystem itself is enabled.

#### Scenario: No exporter configured

Given no Observability Component is active

When the Runtime executes compute work

Then Runtime metrics may still be aggregated for local diagnostics and future
readers.

---

### Requirement: Observability Stream Capability

Magnetar SHALL define a portable `magnetar:observability/stream` Capability.

The Capability SHALL expose typed Runtime observations to authorized Components.

#### Scenario: Subscribe to observations

Given an authorized exporter imports `magnetar:observability/stream`

When it subscribes to execution observations

Then the Runtime returns an Observation Stream resource.

---

### Requirement: Observation Stream Resource

Observation streaming SHALL define a stable stream resource contract.\r\n\r\nObservation streaming SHOULD use an opaque WIT resource.

The stream SHALL support bounded pull-oriented consumption.

The stream MAY expose operations equivalent to:

- `next`
- `close`

#### Scenario: Pull observation batch

Given an Observation Stream is active

When the exporter requests up to 128 observations

Then the Runtime returns no more than 128 portable Observation Records.

---

### Requirement: Observation Filters

Observation filters SHALL define portable subscription selection criteria.\r\n\r\nAn observation subscription MAY include a filter.

Filters MAY select by:

- observation category
- severity
- Provider
- Device
- Component
- Runtime subsystem

Filters SHALL be validated against Runtime policy.

#### Scenario: Subscribe to execution failures

Given an exporter subscribes only to execution failures

When successful operations occur

Then those successful observations need not be delivered to that stream.

---

### Requirement: Stable Observation Record

Observations delivered to Components SHALL use stable portable schemas.

Observation records MAY represent:

- Runtime Event
- Runtime Metric
- Runtime Trace
- Runtime Diagnostic
- Provider Health
- Device Health
- Scheduler event
- execution lifecycle event
- Memory Planning event
- Data Movement event

#### Scenario: Provider execution starts

Given a Scheduled Operation starts Provider execution

When observability is enabled

Then the Runtime may emit a typed execution-start Observation Record.

---

### Requirement: Correlation Identifiers

Observation Records SHALL define stable correlation identifier fields where correlation is provided.\r\n\r\nObservation Records MAY include stable correlation identifiers.

Correlation identifiers MAY include:

- TraceId
- SpanId
- CorrelationId
- ScheduledOperationId
- ComputeExecutionPlanId
- ProviderId
- DeviceId
- ComponentId

#### Scenario: Reconstruct execution trace

Given multiple observations belong to one execution

When an exporter consumes them

Then the exporter can correlate the observations using stable identifiers.

---

### Requirement: Observability Outside Compute Critical Path

Observability processing SHALL NOT be required for successful compute
execution.

Runtime compute execution SHALL NOT synchronously wait for an exporter to
process observations.

#### Scenario: Exporter is slow

Given compute work is executing

And an exporter becomes slow

When the Runtime produces observations

Then compute execution continues independently of exporter processing.

---

### Requirement: Bounded Observation Bus

The Runtime SHALL use bounded observability buffering.

The internal observation queue SHALL have a defined capacity.

#### Scenario: Observation queue reaches capacity

Given the internal observation queue is full

When a Runtime execution path produces another observation

Then the Runtime applies its configured observation overflow policy without
blocking compute execution by default.

---

### Requirement: Non-Blocking Publication

Runtime execution paths SHALL define publication semantics.\r\n\r\nRuntime execution paths SHOULD publish observations using non-blocking
semantics.

The initial default SHALL prefer dropping observations over blocking critical
compute execution.

#### Scenario: Non-blocking publication fails

Given the observation queue is full

When a critical execution path attempts to emit an observation

Then the observation may be dropped

And the compute operation continues.

---

### Requirement: Dropped Observation Accounting

The Runtime SHALL count observations dropped because of observability pressure.

The dropped observation count SHALL itself be available through Runtime
observability snapshots.

#### Scenario: Observation dropped

Given one observation is dropped

When the next Runtime Metrics Snapshot is read

Then the dropped observation count reflects the loss.

---

### Requirement: Stream Exporter Component

A Stream Exporter Component SHALL consume observations through
`magnetar:observability/stream` or an equivalent stable stream contract.

Stream Exporters MAY translate Magnetar observations to external telemetry
formats.

#### Scenario: OpenTelemetry exporter consumes stream

Given an OpenTelemetry exporter Component is active

When trace observations are available

Then the exporter consumes them through the typed observability stream.

---

### Requirement: OpenTelemetry Exporter Component

An OpenTelemetry exporter Component SHALL remain optional and separate from Runtime core.\r\n\r\nMagnetar MAY provide an OpenTelemetry exporter as a WASM Component.

The exporter MAY transform Magnetar:

- traces
- metrics
- logs
- Runtime events

into OpenTelemetry-compatible representations.

OpenTelemetry dependencies SHALL NOT be required by the Runtime core.

#### Scenario: Export trace using OTLP

Given an OpenTelemetry exporter receives a trace Observation Record

When it transforms the observation

Then it may send OTLP-compatible telemetry through an explicitly granted
outbound sink Capability.

---

### Requirement: Jaeger Exporter Component

A Jaeger exporter Component SHALL consume stable trace observation contracts when provided.\r\n\r\nMagnetar MAY provide a Jaeger exporter as a WASM Component.

The Jaeger exporter SHALL consume stable Magnetar trace observations.

#### Scenario: Export Jaeger trace

Given a trace is available

When the Jaeger exporter consumes it

Then it may transform the trace to a Jaeger-compatible representation without
requiring Jaeger support in the Runtime core.

---

### Requirement: Custom Stream Exporter

Magnetar SHALL permit custom Stream Exporter Components when allowed by policy.

Custom exporters SHALL use the same stable observability contracts as built-in
exporters.

#### Scenario: Enterprise exporter

Given an organization provides a compatible custom exporter

When Runtime policy permits it

Then the Component may consume authorized observations and send them to an
authorized sink.

---

### Requirement: Snapshot Exposer Component

A Snapshot Exposer Component SHALL consume aggregated Runtime observability
state through `magnetar:observability/reader`.

Snapshot Exposers SHALL NOT require access to the raw observation stream unless
explicitly needed.

#### Scenario: Prometheus Component

Given a Prometheus exposition Component is active

When a scrape request is processed

Then the Component reads the current Runtime Metrics Snapshot and renders the
appropriate Prometheus exposition.

---

### Requirement: Prometheus Is Not Runtime Core

Prometheus-specific metric types, exposition syntax and libraries SHALL NOT be
required by the Magnetar Runtime core.

#### Scenario: Run Magnetar without Prometheus

Given no Prometheus Component is installed

When Magnetar executes compute workloads

Then no Prometheus dependency is required by the Runtime.

---

### Requirement: Prometheus Snapshot Mapping

Prometheus snapshot mapping SHALL occur inside the Prometheus Component.\r\n\r\nA Prometheus Component MAY map:

- Runtime counters to Prometheus counters
- Runtime gauges to Prometheus gauges
- Runtime histogram summaries to Prometheus-compatible histograms

The mapping SHALL occur inside the Component.

#### Scenario: Queue depth gauge

Given the Runtime Metrics Snapshot contains Scheduler queue depth

When the Prometheus Component renders its exposition

Then it may expose the value as a Prometheus gauge.

---

### Requirement: Explicit External Sink Access

Observability Components SHALL access external systems only through explicitly
granted Capabilities.

External access MAY include:

- outbound HTTP
- filesystem write
- secret read
- log output
- future message queue access

#### Scenario: OTLP HTTP export

Given an OpenTelemetry Component sends telemetry over HTTP

When it contacts an OTLP collector

Then the Component uses an explicitly granted outbound HTTP Capability.

---

### Requirement: No Ambient Network Access

Observability Components SHALL NOT receive ambient network access by default.

#### Scenario: Exporter without HTTP permission

Given an exporter does not import an authorized outbound HTTP Capability

When it attempts network access

Then the access is unavailable or denied.

---

### Requirement: No Ambient Filesystem Access

Observability Components SHALL NOT receive ambient filesystem access by default.

#### Scenario: JSONL exporter

Given a JSONL exporter needs to write a file

When it is instantiated

Then filesystem write access must be explicitly granted and scoped.

---

### Requirement: Secret Access

Exporter credentials SHALL be obtained through explicit secret access
Capabilities.

Raw secrets SHOULD NOT be embedded directly into generic exporter
configuration.

#### Scenario: OTLP authentication token

Given an OTLP collector requires authentication

When the exporter needs the token

Then it obtains the token through an explicitly authorized secret Capability.

---

### Requirement: Capability Import Scoping

The Runtime SHALL apply policy to Observability Component imports.

When possible, unauthorized imported Capabilities SHOULD be absent from the
Component linker.

#### Scenario: Network access not granted

Given a Prometheus Component does not require outbound network access

When it is instantiated

Then the Runtime does not link an outbound HTTP Capability for that Component.

---

### Requirement: Value-Based Sink Scoping

Capabilities requiring value-based authorization SHALL validate their scoped
values.

Examples include:

- HTTP destination URL
- filesystem path
- secret identifier
- metric namespace

#### Scenario: Restricted OTLP endpoint

Given an exporter is authorized only for:

`https://otel.internal.example/**`

When it attempts to send telemetry to another endpoint

Then the outbound HTTP Capability rejects the request.

---

### Requirement: Observation Access Scoping

The Runtime SHALL enforce observation access policy when policy scopes are configured.\r\n\r\nThe Runtime MAY restrict which observation categories an exporter can consume.

#### Scenario: Security exporter

Given an exporter is allowed only Runtime security diagnostics

When it subscribes to all observations

Then the Runtime restricts or rejects categories outside its authorized scope.

---

### Requirement: Observability Component Lifecycle

Observability Components SHALL expose Runtime-visible lifecycle state.

Lifecycle states SHALL include:

- discovered
- loaded
- initializing
- active
- degraded
- saturated
- failed
- disabled
- stopped

#### Scenario: Exporter activates

Given an exporter Component loads and initializes successfully

When its required imports and policy are valid

Then its state becomes `active`.

---

### Requirement: Exporter Degradation

Exporter degradation SHALL be represented separately from compute execution correctness.\r\n\r\nAn exporter MAY become degraded when delivery partially fails.

Exporter degradation SHALL NOT degrade Compute Operation correctness.

#### Scenario: OTLP endpoint intermittently fails

Given the OTLP endpoint is intermittently unavailable

When export attempts fail

Then the exporter may become degraded while compute execution remains
unaffected.

---

### Requirement: Exporter Saturation

Exporter saturation SHALL be represented as exporter lifecycle state.\r\n\r\nAn exporter MAY enter `saturated` state when it cannot consume or deliver
observations at the required rate.

#### Scenario: Export queue overloaded

Given an exporter cannot keep up with observation production

When configured limits are exceeded

Then the Runtime may mark the exporter saturated and apply backpressure policy.

---

### Requirement: Exporter Failure Isolation

Exporter failures SHALL be isolated from Runtime compute execution.

Exporter failure SHALL NOT alter:

- successful Compute Graph results
- Tensor Resource state
- Scheduled Operation terminal state
- Provider execution state
- Resource Affinity
- Memory Plan correctness

#### Scenario: Exporter traps during inference

Given Provider execution completes successfully

And an exporter Component traps

When compute completion is reported

Then the Scheduled Operation remains successful

And the exporter failure is reported independently.

---

### Requirement: Sink Failure Isolation

External sink failure SHALL NOT become a compute execution failure.

#### Scenario: Prometheus or OTLP unavailable

Given an observability sink is unavailable

When compute execution completes successfully

Then compute execution remains successful.

---

### Requirement: Observability Backpressure Policy

The Runtime SHALL define an observability backpressure policy.

Supported policy actions MAY include:

- buffer
- drop
- degrade exporter
- disable exporter
- shed observability work

Critical compute paths SHALL NOT block on observability delivery by default.

#### Scenario: Exporter cannot keep up

Given observations are produced faster than an exporter consumes them

When configured buffers reach capacity

Then the Runtime applies the configured observability backpressure policy.

---

### Requirement: Slow Producer Policy

A policy that slows compute execution because of telemetry pressure SHALL NOT be
the default.

If such a policy is supported, it SHALL require explicit configuration.

#### Scenario: Default configuration

Given Magnetar uses its default observability policy

When exporters become saturated

Then compute execution is not deliberately slowed to preserve telemetry.

---

### Requirement: Observability Load Shedding

Observability load shedding SHALL preserve compute and Runtime control priority.\r\n\r\nThe Runtime MAY shed observability work under resource pressure.

Compute execution and Runtime control SHALL have higher priority than exporter
processing.

#### Scenario: Runtime under heavy compute load

Given the Runtime is under severe compute pressure

When observability Components compete for execution resources

Then the Runtime may delay, throttle or suspend observability Components.

---

### Requirement: Snapshot Exposer Load Shedding

Snapshot exposer load shedding SHALL be controlled by Runtime policy.\r\n\r\nSnapshot exposer requests MAY be rejected or delayed under severe Runtime
pressure.

#### Scenario: Prometheus scrape under severe load

Given the Runtime is protecting compute capacity

When a Prometheus scrape arrives

Then the Runtime may apply observability load-shedding policy without affecting
running compute operations.

---

### Requirement: Observability Policy

The Runtime SHALL define an Observability Policy.

Observability Policy MAY define:

- enabled categories
- severity levels
- sampling rates
- internal buffer limits
- exporter buffer limits
- batch sizes
- drop behavior
- exporter enabled state
- sink configuration
- access scopes

#### Scenario: Sampling policy

Given trace sampling is configured to a lower rate

When new operations execute

Then the Runtime applies the updated sampling policy.

---

### Requirement: Hot-Reloadable Observability Configuration

Observability configuration SHALL define which fields are hot-reloadable.\r\n\r\nObservability configuration SHOULD be updateable without restarting the Runtime.

#### Scenario: Enable debugging dynamically

Given one Provider requires additional diagnostics

When policy enables debug observations for that Provider

Then subsequent observations use the updated policy without Runtime restart.

---

### Requirement: Hot-Reloadable Exporter State

Exporter state changes SHALL be represented in observability policy.\r\n\r\nExporter Components MAY be enabled or disabled dynamically.

#### Scenario: Disable Jaeger exporter

Given the Jaeger exporter is active

When policy disables it

Then new observations stop being delivered to that exporter without restarting
the Runtime.

---

### Requirement: Stable Portable Observability Values

Observability contracts SHALL use stable portable values.

Observations SHALL NOT contain:

- Rust trait objects
- callbacks
- raw native handles
- raw pointers
- GPU pointers
- backend storage
- native queues
- native streams
- kernel objects
- allocator internals

#### Scenario: Provider failure observation

Given a CUDA Provider reports a native execution failure

When an observation is delivered to an exporter

Then the record contains stable Magnetar error categories and redacted
diagnostics rather than native CUDA objects.

---

### Requirement: Diagnostic Redaction

Diagnostics exposed to Observability Components SHALL be redacted according to
Runtime policy.

Sensitive values MAY include:

- credentials
- secret values
- private filesystem paths
- authentication tokens
- raw backend diagnostics
- user-provided sensitive data

#### Scenario: Backend error contains path

Given a Provider diagnostic contains a sensitive filesystem path

When the Runtime produces an observation

Then the path is omitted or redacted according to policy.

---

### Requirement: Observation Delivery Semantics

Observation streams SHALL explicitly define their delivery semantics.

Initial stream delivery MAY be best-effort.

The Runtime SHALL NOT imply durable telemetry delivery unless a future contract
explicitly provides durability.

#### Scenario: Runtime crashes before export

Given an observation was not durably persisted

When the Runtime terminates before an exporter consumes it

Then the observation may be lost under best-effort semantics.

---

### Requirement: Observation Ordering

Observation ordering guarantees SHALL be explicit.

Ordering MAY be preserved within:

- one TraceId
- one ScheduledOperationId
- one Observation Stream

Global total ordering SHALL NOT be assumed.

#### Scenario: Same Scheduled Operation

Given multiple observations belong to the same Scheduled Operation

When they are delivered through one ordered stream

Then their execution ordering is preserved according to the stream contract.

---

### Requirement: Structured Observability Errors

Observability contracts SHALL return stable structured errors.

Error categories SHALL include:

- invalid observation
- unsupported observation
- access denied
- invalid filter
- stream closed
- stream interrupted
- exporter unavailable
- exporter saturated
- exporter failed
- sink unavailable
- sink unauthorized
- sink timeout
- serialization failed
- observability policy rejected
- observation dropped

Backend-specific diagnostic strings MAY be attached.

They SHALL NOT define stable error semantics.

#### Scenario: Unauthorized observation stream

Given a Component is not authorized to consume Provider diagnostics

When it requests that observation category

Then the Runtime returns a stable access-denied error.

---

### Requirement: OpenTelemetry as Optional WASM Integration

OpenTelemetry integration SHALL remain optional.\r\n\r\nMagnetar MAY provide an OpenTelemetry integration as a WASM Component.

The Runtime SHALL NOT require an OpenTelemetry SDK or exporter dependency to
execute compute workloads.

#### Scenario: Magnetar without OpenTelemetry

Given no OpenTelemetry Component is installed

When Magnetar executes workloads

Then Runtime compute functionality remains fully operational.

---

### Requirement: OpenTelemetry Stream Mapping

OpenTelemetry stream mapping SHALL preserve Magnetar correlation identifiers when mapped.\r\n\r\nAn OpenTelemetry exporter SHOULD consume typed Runtime observations through
`magnetar:observability/stream`.

The exporter MAY map:

- Runtime traces to spans
- Runtime metrics to OTel metrics
- Runtime logs to OTel logs
- Runtime identifiers to OTel attributes

#### Scenario: Map execution trace

Given one execution trace contains planning, scheduling and Provider execution
observations

When the OpenTelemetry Component processes them

Then it may create corresponding OpenTelemetry spans while preserving Magnetar
correlation identifiers.

---

### Requirement: Prometheus as Snapshot Exposer

Prometheus integration SHALL use the snapshot model for aggregated Runtime metrics.\r\n\r\nA Prometheus integration SHOULD be implemented as a WASM Component consuming
`magnetar:observability/reader`.

#### Scenario: Prometheus scrape

Given the Prometheus Component receives a scrape request

When it generates its response

Then it reads one Runtime Metrics Snapshot and renders Prometheus-compatible
text.

---

### Requirement: Prometheus Does Not Require Raw Event Stream

A Prometheus Component SHALL NOT require the full Runtime observation stream for
metrics that are already available through aggregated snapshots.

#### Scenario: Render operation counters

Given completed operation count exists in the Runtime Metrics Snapshot

When Prometheus renders the metric

Then it uses the snapshot instead of replaying historical Runtime events.

---

### Requirement: Custom Observability Components

Magnetar SHALL support custom Observability Components when allowed by policy.

Custom Components SHALL use the same stable observability contracts as built-in
integrations.

#### Scenario: Organization-specific exporter

Given an organization implements a WASM Component consuming
`magnetar:observability/stream`

When the Runtime validates its imports and scopes

Then it may operate as a custom exporter without changes to the Runtime core.

---

### Requirement: No Mandatory Observability Backend

Magnetar SHALL NOT mandate OpenTelemetry, Prometheus, Jaeger or another external
observability backend.

#### Scenario: Local-only deployment

Given Magnetar runs without external telemetry infrastructure

When observability is enabled

Then local snapshots and diagnostics may operate without any external exporter.

---

### Requirement: Inference API Observability Is Redacted By Default

Observability produced by Runtime Inference API SHALL be redacted by default.

#### Scenario: Prompt submitted

Given caller submits prompt text

When inference observations are emitted

Then raw prompt text is not logged by default.

---

### Requirement: Inference API Observability Preserves Correlation

Runtime Inference API observations SHALL include stable correlation metadata for requests, sessions, generations, streams, cache events, and errors.

#### Scenario: Generation failure

Given generation fails

When observability emits error metadata

Then events can be correlated without exposing raw prompt or handles.

---

### Requirement: CLI And Runtime Observability Are Distinct

CLI-side observations and Runtime-side observations SHALL remain distinct.

#### Scenario: Command plus inference

Given CLI runs `magnetar run`

When observations are emitted

Then CLI may observe command parsing while Runtime observes inference execution.

---

### Requirement: CLI Observability Redacts Sensitive Context

CLI observability SHALL redact raw prompts, secrets, file contents, tokens,
model weights, handles, and memory pointers by default.

#### Scenario: File prompt

Given CLI reads file content for prompt

When CLI emits observations

Then raw file content is not logged by default.

---

### Requirement: Runtime Observability Does Not Log CLI Authority

Runtime observations SHALL not log CLI authority, workspace permissions, secret
providers, network credentials, or tool capabilities.

#### Scenario: CLI has tool access

Given CLI has tool access

When Runtime emits inference observations

Then Runtime does not log tool capability details unless explicitly included as
redacted request metadata.

---

### Requirement: E2E Observability Is Redacted

E2E conformance SHALL validate observability redaction for inference events.

#### Scenario: Prompt redaction

Given prompt text is submitted

When observability events are emitted

Then raw prompt text is absent by default.

---

### Requirement: E2E Observability Supports Correlation

E2E observations SHALL include correlation IDs that connect request, model
loading, session, generation, graph, kernel, and result events without exposing
raw data.

#### Scenario: Correlated failure

Given generation fails during Kernel dispatch

When observations are inspected

Then failure can be correlated across Runtime subsystems.

### Requirement: Source Cache Observability Is Redacted

Source/cache observations SHALL be redacted by default.

#### Scenario: Cache lookup

Given cache lookup occurs

When observation is emitted

Then raw cache paths, credentials, raw file contents, and raw weights are absent.

---

### Requirement: Source Cache Observability Preserves Correlation

Source/cache observations SHOULD include correlation IDs linking model resolution, cache lookup, normalization, validation, and loading, and correlation identifiers SHALL not themselves expose redacted metadata.

#### Scenario: Cache corrupt

Given cache entry is corrupt

When loading fails

Then observations can correlate source resolution and integrity failure.

### Requirement: Server Observability Is Redacted

Server observations SHALL be redacted by default.

#### Scenario: Request logged

Given generation request contains prompt text

When server emits observation

Then raw prompt text is absent by default.

---

### Requirement: Server Runtime Correlation

Server observations SHALL not expose raw data during correlation.

Server observations SHOULD correlate request, Runtime request, streaming,
cancellation, diagnostics, and errors.

#### Scenario: Stream interrupted

Given stream is interrupted

When observations are emitted

Then server and Runtime events can be correlated.

### Requirement: Release Observability Redaction

Release builds SHALL preserve default observability redaction.

#### Scenario: Release diagnostics

Given release binary emits diagnostics

When diagnostics are inspected

Then raw prompts, secrets, file contents, model weights, tensor values, KV cache
contents, handles, and memory pointers are absent by default.

---

### Requirement: Release Build Metadata Observability

Release observability MAY include build metadata, but included metadata SHALL
exclude secrets and local filesystem paths.

#### Scenario: Version observation

Given Runtime emits version observation

When metadata is inspected

Then version and feature flags may be included without secrets or local paths.

### Requirement: Release Observability Security Redaction

Release observability SHALL be redacted by default.

#### Scenario: Release observation

Given inference request includes prompt and model artifact

When observation is emitted

Then raw prompt, weights, tensors, cache contents, secrets, credentials, handles,
pointers, and raw file contents are absent.

---

### Requirement: Release Security Event Recording

Release process SHALL record security gate status, and recording SHOULD avoid
exposing sensitive content.

#### Scenario: Secret scan failed

Given secret scan fails

When release metadata is recorded

Then failure status is recorded without printing the secret.

### Requirement: Observability Release Redaction Gate

Observability redaction gate SHALL be required for `v0.1`.

#### Scenario: Secret logged

Given observation logs secret by default

When release validation runs

Then stable release is blocked.

---

### Requirement: Release Reports Redacted

Release reports SHALL be redacted by default.

#### Scenario: Report failure

Given failure includes prompt text

When release report is generated

Then raw prompt text is absent by default.

### Requirement: Cutover Observability Is Redacted

Cutover observations and reports SHALL be redacted by default.

#### Scenario: Release observation

Given cutover records failed secret scan

When report is emitted

Then secret value is not printed.

---

### Requirement: Cutover Events Are Correlatable

Cutover SHALL record correlation between gates, reports, artifacts, and release
metadata.

#### Scenario: Gate failure

Given Runtime gate fails

When cutover report is inspected

Then failure can be correlated to gate, target, feature set, and artifact.

---

### Requirement: Kernel Artifact Observations

Kernel Artifact lifecycle observations SHALL redact raw kernel source and
native handles by default. Runtime MAY emit observations for Kernel Artifact
lifecycle.

#### Scenario: Preparation completed

Given Provider successfully prepares compiled kernel

When observation is emitted

Then artifact identity and redacted preparation metadata may be included.

---

### Requirement: Raw Kernel Source Redacted

Raw Kernel Source Artifact contents SHALL be redacted by default.

#### Scenario: Compilation failure

Given source compilation fails

When diagnostic is emitted

Then complete kernel source is not logged by default.

---

### Requirement: Compiled Binary Redacted

Raw compiled binary bytes SHALL not be logged by default.

#### Scenario: Binary validation failure

Given compiled artifact is malformed

When error is observed

Then digest/format may be reported but raw binary bytes are absent.

---

### Requirement: Prepared Native Handle Redacted

Native Provider execution handles SHALL not appear in observability.

#### Scenario: Prepared kernel selected

Given Provider uses native function pointer internally

When selection is observed

Then only opaque stable metadata is reported.

---

### Requirement: Compilation Lifecycle Observability

Runtime SHALL make Provider compilation lifecycle observable through structured events.

#### Scenario: Compilation completes

Given Provider finishes compilation

When observation is emitted

Then request ID, artifact digest, compiler identity and duration may be
reported.

---

### Requirement: Source Contents Redacted

Raw kernel source SHALL be redacted by default.

#### Scenario: Compiler syntax error

Given compiler reports offending line

When diagnostic is exported

Then policy controls source excerpt and full source is not logged automatically.

---

### Requirement: Compiler Paths Redacted

Provider compiler temporary paths SHALL be redacted by default.

#### Scenario: NVCC reports temp file

Given compiler output contains `/tmp/...`

When diagnostic is exported

Then path is removed or normalized.

---

### Requirement: Compiler Environment Redacted

Environment variables and secrets SHALL not be emitted in compilation
observability.

#### Scenario: Compiler process inherits environment

Given Provider records compiler failure

When observation is emitted

Then environment contents are absent.

---

### Requirement: Native Compiler Handles Redacted

Native compiler/driver objects SHALL remain absent from diagnostics.

#### Scenario: Shader compiler object exists

Given Provider tracks native object internally

When diagnostic is emitted

Then only opaque Runtime metadata is exposed.

---

### Requirement: Qualification Lifecycle Observability

Observability SHALL NOT expose raw kernel source, compiled binaries, or test tensors by default.

Runtime SHOULD emit redacted qualification lifecycle observations.

#### Scenario: Differential mismatch

Given candidate output exceeds tolerance

When qualification fails

Then observation identifies Kernel/artifact/profile and mismatch category
without dumping raw tensor data by default.

---

### Requirement: Benchmark Observability

A benchmark summary without workload/context metadata SHALL NOT be treated as authoritative ranking evidence.

Runtime MAY emit benchmark summaries without exposing sensitive inputs.

#### Scenario: Candidate benchmark

Given benchmark completes

When observation is emitted

Then latency/throughput/profile metadata may be included.

---

### Requirement: Promotion Observability

Kernel promotion SHALL be observable.

#### Scenario: Generation 42 promoted

Given candidate becomes active

When promotion completes

Then event records logical Kernel, old generation, new generation and policy
decision.

---

### Requirement: Rollback Observability

Rollback SHALL be observable.

#### Scenario: Regression rollback

Given active kernel is rolled back

When event is emitted

Then reason and resulting active generation are recorded.

---

### Requirement: Revocation Observability

Revocation SHALL be observable without exposing executable internals.

#### Scenario: Kernel revoked

Given qualification is revoked

When event is emitted

Then artifact digest and reason may be reported without native handles.

---

### Requirement: Generated Kernel Observability Redaction

Observability SHALL redact raw source, compiled binaries, raw qualification
tensors and native handles by default.

#### Scenario: Qualification crash

Given Provider fails during generated kernel test

When diagnostic is emitted

Then sensitive artifact/tensor contents are absent by default.

---

### Requirement: Kernel Selection Observability

Recorded Kernel selection lifecycle events SHALL exclude native handles and raw tensor data; Runtime SHOULD record this redacted lifecycle.

#### Scenario: Candidate selected

Given five candidates are evaluated

When selection completes

Then selected Kernel and policy profile may be observed.

---

### Requirement: Candidate Exclusion Is Observable

Exposed exclusion reasons SHALL identify the specific constraint that excluded the candidate; Runtime SHOULD expose such structured reasons.

#### Scenario: Candidate excluded by affinity

Given candidate fails Resource Affinity

When diagnostics are requested

Then affinity incompatibility is reported.

---

### Requirement: Ranking Metadata Is Redacted

Selection observations SHALL not expose native handles or model data.

#### Scenario: Ranking report

Given candidates are ranked

When report is exported

Then it may contain score, KernelId and artifact digest but no tensor values or
native addresses.

---

### Requirement: Hysteresis Is Observable

The record of a hysteresis-retained active Kernel SHALL include the comparison that was suppressed; Runtime SHOULD record such retention events.

#### Scenario: Marginal improvement

Given new candidate is slightly faster but below promotion threshold

When active Kernel remains

Then decision reason is observable.

---

### Requirement: Fallback Is Observable

Explicit fallback SHALL record original failure class and selected fallback.

#### Scenario: GPU unavailable

Given Runtime falls back to Reference CPU

When observation is emitted

Then fallback policy and reason are included.

---

### Requirement: Optimization Observability Is Distinguishable

Optimization Campaign observations SHALL be distinguishable from inference
execution observations.

#### Scenario: Candidate benchmark starts

Given benchmark begins

When observation is emitted

Then event is classified as optimization-plane activity.

---

### Requirement: Campaign Correlation

Optimization events SHALL support correlation by campaign/candidate/artifact.

#### Scenario: Candidate promoted later

Given campaign created candidate

When Runtime promotes it

Then promotion may retain campaign/evidence correlation metadata.

---

### Requirement: Optimization Data Is Redacted

Optimization observations SHALL redact sensitive source/inference information
according to policy.

#### Scenario: Workload profile emitted

Given profile was derived from production workloads

When observation is exported

Then raw prompts and raw user documents are absent.

---

### Requirement: Generator Credentials Are Redacted

Observability SHALL never expose generator or artifact-registry credentials.

#### Scenario: External service authentication fails

Given API key is present internally

When error is logged

Then key value is absent.

---

### Requirement: Manifest Import Observability

Runtime MAY observe Kernel Manifest parsing and validation lifecycle, and when observability is enabled, emitted records SHALL comply with default redaction rules.

#### Scenario: Bundle imported

Given valid bundle

When imported

Then schema version, manifest digest and artifact counts may be observed.

---

### Requirement: Artifact Payloads Redacted

Observability SHALL not emit raw Kernel source or compiled blobs by default.

#### Scenario: Digest mismatch

Given binary integrity fails

When error is logged

Then expected/actual digest may appear while binary bytes do not.

---

### Requirement: Sensitive Locators Redacted

External artifact location diagnostics SHALL redact credentials and sensitive
path components.

#### Scenario: Credential-bearing URI rejected

Given malformed locator includes secret

When diagnostic is emitted

Then credential value is absent.

---

### Requirement: Trust Claims And Decisions Distinguished

Observability SHOULD distinguish asserted provenance from evaluated trust, and diagnostics SHALL NOT label an unauthenticated claim as an authenticated trust decision.

#### Scenario: Publisher claim

Given manifest says publisher A

When trust is denied

Then diagnostics do not represent publisher claim as authenticated fact.
