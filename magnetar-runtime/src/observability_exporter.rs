use crate::{
    Capability, CapabilityDescriptor, CapabilityId, CapabilityVersion, ComponentMetadata,
    DeviceBinding, ExecutionPlanId, HealthState, ProviderBinding, RuntimeDiagnostic, RuntimeEvent,
    RuntimeMetric, ScheduledOperationId, TraceId, WitInterface,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
};

pub const OBSERVABILITY_CAPABILITY_VERSION: CapabilityVersion = CapabilityVersion::new(1, 0, 0);
pub const OBSERVABILITY_EMIT_INTERFACE: &str = "magnetar:observability/emit";
pub const OBSERVABILITY_READER_INTERFACE: &str = "magnetar:observability/reader";
pub const OBSERVABILITY_STREAM_INTERFACE: &str = "magnetar:observability/stream";

pub fn observability_emit_wit() -> WitInterface {
    WitInterface::new(
        OBSERVABILITY_EMIT_INTERFACE,
        OBSERVABILITY_CAPABILITY_VERSION.to_string(),
    )
}

pub fn observability_reader_wit() -> WitInterface {
    WitInterface::new(
        OBSERVABILITY_READER_INTERFACE,
        OBSERVABILITY_CAPABILITY_VERSION.to_string(),
    )
}

pub fn observability_stream_wit() -> WitInterface {
    WitInterface::new(
        OBSERVABILITY_STREAM_INTERFACE,
        OBSERVABILITY_CAPABILITY_VERSION.to_string(),
    )
}

pub fn observability_emit_capability() -> Capability {
    Capability::new(
        CapabilityId::new(OBSERVABILITY_EMIT_INTERFACE),
        OBSERVABILITY_CAPABILITY_VERSION,
        CapabilityDescriptor::new("custom Component observation emission")
            .with_contract(observability_emit_wit()),
    )
}

pub fn observability_reader_capability() -> Capability {
    Capability::new(
        CapabilityId::new(OBSERVABILITY_READER_INTERFACE),
        OBSERVABILITY_CAPABILITY_VERSION,
        CapabilityDescriptor::new("aggregated Runtime observability snapshots")
            .with_contract(observability_reader_wit()),
    )
}

pub fn observability_stream_capability() -> Capability {
    Capability::new(
        CapabilityId::new(OBSERVABILITY_STREAM_INTERFACE),
        OBSERVABILITY_CAPABILITY_VERSION,
        CapabilityDescriptor::new("typed Runtime observation stream")
            .with_contract(observability_stream_wit()),
    )
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservabilityComponentRole {
    ObservabilityComponent,
    StreamExporter,
    SnapshotExposer,
    CustomObserver,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservabilityComponentState {
    Discovered,
    Loaded,
    Initializing,
    Active,
    Degraded,
    Saturated,
    Failed,
    Disabled,
    Stopped,
}

impl ObservabilityComponentState {
    pub const fn is_failure_or_pressure(self) -> bool {
        matches!(self, Self::Degraded | Self::Saturated | Self::Failed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityComponentDescriptor {
    pub metadata: ComponentMetadata,
    pub role: ObservabilityComponentRole,
    pub state: ObservabilityComponentState,
    pub imports: BTreeSet<WitInterface>,
    pub sinks: BTreeSet<ObservabilitySinkDependency>,
    pub filter: ObservationFilter,
}

impl ObservabilityComponentDescriptor {
    pub fn new(metadata: ComponentMetadata, role: ObservabilityComponentRole) -> Self {
        Self {
            metadata,
            role,
            state: ObservabilityComponentState::Discovered,
            imports: BTreeSet::new(),
            sinks: BTreeSet::new(),
            filter: ObservationFilter::default(),
        }
    }

    pub fn stream_exporter(metadata: ComponentMetadata) -> Self {
        let mut descriptor = Self::new(metadata, ObservabilityComponentRole::StreamExporter);
        descriptor.imports.insert(observability_stream_wit());
        descriptor
    }

    pub fn snapshot_exposer(metadata: ComponentMetadata) -> Self {
        let mut descriptor = Self::new(metadata, ObservabilityComponentRole::SnapshotExposer);
        descriptor.imports.insert(observability_reader_wit());
        descriptor
    }

    pub fn custom_observer(metadata: ComponentMetadata) -> Self {
        let mut descriptor = Self::new(metadata, ObservabilityComponentRole::CustomObserver);
        descriptor.imports.insert(observability_emit_wit());
        descriptor
    }

    pub const fn participates_in_compute_resolution(&self) -> bool {
        false
    }

    pub const fn is_provider(&self) -> bool {
        false
    }

    pub fn transition(&mut self, state: ObservabilityComponentState) {
        self.state = state;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CustomMetricKind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MetricTags(BTreeMap<String, String>);

impl MetricTags {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), ObservabilityError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(ObservabilityError::invalid_observation(
                "metric tag key must not be empty",
            ));
        }
        self.0.insert(key, value.into());
        Ok(())
    }

    pub fn as_map(&self) -> &BTreeMap<String, String> {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomMetricRecord {
    pub namespace: String,
    pub name: String,
    pub kind: CustomMetricKind,
    pub value: i128,
    pub tags: MetricTags,
    pub component: String,
}

impl CustomMetricRecord {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        kind: CustomMetricKind,
        value: i128,
        component: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            kind,
            value,
            tags: MetricTags::new(),
            component: component.into(),
        }
    }

    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.namespace, self.name)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LogSeverity {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomLogRecord {
    pub severity: LogSeverity,
    pub message: String,
    pub fields: BTreeMap<String, String>,
    pub component: String,
}

impl CustomLogRecord {
    pub fn new(
        severity: LogSeverity,
        message: impl Into<String>,
        component: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            message: message.into(),
            fields: BTreeMap::new(),
            component: component.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomEventRecord {
    pub namespace: String,
    pub name: String,
    pub attributes: BTreeMap<String, String>,
    pub component: String,
}

impl CustomEventRecord {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        component: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            attributes: BTreeMap::new(),
            component: component.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservationCategory {
    RuntimeEvent,
    RuntimeMetric,
    RuntimeTrace,
    RuntimeDiagnostic,
    ProviderHealth,
    DeviceHealth,
    Scheduler,
    ExecutionLifecycle,
    DataMovement,
    MemoryPlanning,
    CustomMetric,
    CustomLog,
    CustomEvent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationRecord {
    RuntimeEvent(RuntimeEvent),
    RuntimeMetric(RuntimeMetric),
    RuntimeTrace {
        trace_id: TraceId,
        events: Vec<RuntimeEvent>,
    },
    RuntimeDiagnostic(RuntimeDiagnostic),
    ProviderHealth {
        provider: ProviderBinding,
        state: HealthState,
    },
    DeviceHealth {
        device: DeviceBinding,
        state: HealthState,
    },
    Scheduler {
        operation: Option<ScheduledOperationId>,
        state: String,
    },
    ExecutionLifecycle {
        operation: ScheduledOperationId,
        plan: ExecutionPlanId,
        state: String,
    },
    DataMovement {
        plan: ExecutionPlanId,
        bytes: u64,
        description: String,
    },
    MemoryPlanning {
        plan: ExecutionPlanId,
        estimated_bytes: u64,
    },
    CustomMetric(CustomMetricRecord),
    CustomLog(CustomLogRecord),
    CustomEvent(CustomEventRecord),
}

impl ObservationRecord {
    pub const fn category(&self) -> ObservationCategory {
        match self {
            Self::RuntimeEvent(_) => ObservationCategory::RuntimeEvent,
            Self::RuntimeMetric(_) => ObservationCategory::RuntimeMetric,
            Self::RuntimeTrace { .. } => ObservationCategory::RuntimeTrace,
            Self::RuntimeDiagnostic(_) => ObservationCategory::RuntimeDiagnostic,
            Self::ProviderHealth { .. } => ObservationCategory::ProviderHealth,
            Self::DeviceHealth { .. } => ObservationCategory::DeviceHealth,
            Self::Scheduler { .. } => ObservationCategory::Scheduler,
            Self::ExecutionLifecycle { .. } => ObservationCategory::ExecutionLifecycle,
            Self::DataMovement { .. } => ObservationCategory::DataMovement,
            Self::MemoryPlanning { .. } => ObservationCategory::MemoryPlanning,
            Self::CustomMetric(_) => ObservationCategory::CustomMetric,
            Self::CustomLog(_) => ObservationCategory::CustomLog,
            Self::CustomEvent(_) => ObservationCategory::CustomEvent,
        }
    }

    pub fn provider(&self) -> Option<&ProviderBinding> {
        match self {
            Self::RuntimeEvent(event) => event.provider.as_ref(),
            Self::RuntimeMetric(metric) => metric.provider.as_ref(),
            Self::RuntimeDiagnostic(diagnostic) => diagnostic.provider.as_ref(),
            Self::ProviderHealth { provider, .. } => Some(provider),
            _ => None,
        }
    }

    pub fn device(&self) -> Option<&DeviceBinding> {
        match self {
            Self::RuntimeEvent(event) => event.device.as_ref(),
            Self::RuntimeMetric(metric) => metric.device.as_ref(),
            Self::RuntimeDiagnostic(diagnostic) => diagnostic.device.as_ref(),
            Self::DeviceHealth { device, .. } => Some(device),
            _ => None,
        }
    }

    pub fn component(&self) -> Option<&str> {
        match self {
            Self::CustomMetric(record) => Some(&record.component),
            Self::CustomLog(record) => Some(&record.component),
            Self::CustomEvent(record) => Some(&record.component),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObservationFilter {
    pub categories: BTreeSet<ObservationCategory>,
    pub min_severity: Option<LogSeverity>,
    pub providers: BTreeSet<ProviderBinding>,
    pub devices: BTreeSet<DeviceBinding>,
    pub components: BTreeSet<String>,
    pub subsystems: BTreeSet<String>,
}

impl ObservationFilter {
    pub fn all() -> Self {
        Self::default()
    }

    pub fn with_category(mut self, category: ObservationCategory) -> Self {
        self.categories.insert(category);
        self
    }

    pub fn permits(&self, record: &ObservationRecord) -> bool {
        if !self.categories.is_empty() && !self.categories.contains(&record.category()) {
            return false;
        }
        if !self.providers.is_empty()
            && !record
                .provider()
                .is_some_and(|p| self.providers.contains(p))
        {
            return false;
        }
        if !self.devices.is_empty() && !record.device().is_some_and(|d| self.devices.contains(d)) {
            return false;
        }
        if !self.components.is_empty()
            && !record
                .component()
                .is_some_and(|component| self.components.contains(component))
        {
            return false;
        }
        if let Some(min_severity) = self.min_severity
            && let ObservationRecord::CustomLog(log) = record
        {
            return log.severity >= min_severity;
        }
        true
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObservationBatch {
    pub records: Vec<ObservationRecord>,
    pub end_of_stream: bool,
    pub interrupted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationStream {
    filter: ObservationFilter,
    records: VecDeque<ObservationRecord>,
    closed: bool,
    interrupted: bool,
    max_batch: usize,
}

impl ObservationStream {
    pub fn new(
        records: impl IntoIterator<Item = ObservationRecord>,
        filter: ObservationFilter,
        max_batch: usize,
    ) -> Self {
        let records = records
            .into_iter()
            .filter(|record| filter.permits(record))
            .collect();
        Self {
            filter,
            records,
            closed: false,
            interrupted: false,
            max_batch: max_batch.max(1),
        }
    }

    pub fn pull(&mut self, requested: usize) -> Result<ObservationBatch, ObservabilityError> {
        if self.closed {
            return Err(ObservabilityError::stream_closed());
        }
        if self.interrupted {
            return Err(ObservabilityError::stream_interrupted());
        }
        let limit = requested.min(self.max_batch).max(1);
        let mut records = Vec::new();
        while records.len() < limit {
            match self.records.pop_front() {
                Some(record) => records.push(record),
                None => break,
            }
        }
        Ok(ObservationBatch {
            end_of_stream: self.records.is_empty(),
            interrupted: false,
            records,
        })
    }

    pub fn push(&mut self, record: ObservationRecord) {
        if !self.closed && self.filter.permits(&record) {
            self.records.push_back(record);
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn interrupt(&mut self) {
        self.interrupted = true;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservationOverflowPolicy {
    DropNewest,
    DropOldest,
    DegradeExporter,
    DisableExporter,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservabilityPriority {
    LowerThanCompute,
    ComputeBlocking,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityPolicy {
    pub enabled_categories: BTreeSet<ObservationCategory>,
    pub min_log_severity: LogSeverity,
    pub sampling_per_million: u32,
    pub internal_buffer_capacity: usize,
    pub exporter_buffer_capacity: usize,
    pub max_batch_size: usize,
    pub overflow: ObservationOverflowPolicy,
    pub priority: ObservabilityPriority,
    pub hot_reloadable: BTreeSet<ObservabilityPolicyField>,
    pub allowed_observation_categories: BTreeSet<ObservationCategory>,
    pub allowed_metric_namespaces: BTreeSet<String>,
    pub allowed_endpoints: BTreeSet<String>,
    pub allowed_filesystem_paths: BTreeSet<String>,
    pub allowed_secret_namespaces: BTreeSet<String>,
}

impl Default for ObservabilityPolicy {
    fn default() -> Self {
        Self {
            enabled_categories: BTreeSet::new(),
            min_log_severity: LogSeverity::Info,
            sampling_per_million: 1_000_000,
            internal_buffer_capacity: 1024,
            exporter_buffer_capacity: 1024,
            max_batch_size: 128,
            overflow: ObservationOverflowPolicy::DropNewest,
            priority: ObservabilityPriority::LowerThanCompute,
            hot_reloadable: BTreeSet::from([
                ObservabilityPolicyField::SamplingRate,
                ObservabilityPolicyField::LogSeverity,
                ObservabilityPolicyField::ObservationFilters,
                ObservabilityPolicyField::ExporterState,
                ObservabilityPolicyField::BatchingLimits,
                ObservabilityPolicyField::Endpoint,
            ]),
            allowed_observation_categories: BTreeSet::new(),
            allowed_metric_namespaces: BTreeSet::new(),
            allowed_endpoints: BTreeSet::new(),
            allowed_filesystem_paths: BTreeSet::new(),
            allowed_secret_namespaces: BTreeSet::new(),
        }
    }
}

impl ObservabilityPolicy {
    pub fn validate_custom_metric(
        &self,
        metric: &CustomMetricRecord,
    ) -> Result<(), ObservabilityError> {
        if metric.namespace.trim().is_empty() || metric.name.trim().is_empty() {
            return Err(ObservabilityError::invalid_observation(
                "metric namespace and name must not be empty",
            ));
        }
        if self.allowed_metric_namespaces.is_empty()
            || self.allowed_metric_namespaces.iter().any(|namespace| {
                metric.namespace == *namespace
                    || metric.namespace.starts_with(&format!("{namespace}."))
            })
        {
            Ok(())
        } else {
            Err(ObservabilityError::access_denied(format!(
                "metric namespace '{}' is not authorized",
                metric.namespace
            )))
        }
    }

    pub fn validate_filter(&self, filter: &ObservationFilter) -> Result<(), ObservabilityError> {
        if self.allowed_observation_categories.is_empty()
            || filter
                .categories
                .iter()
                .all(|category| self.allowed_observation_categories.contains(category))
        {
            Ok(())
        } else {
            Err(ObservabilityError::new(
                ObservabilityErrorCode::InvalidFilter,
                "observation filter requests unauthorized categories",
            ))
        }
    }

    pub fn can_hot_reload(&self, field: ObservabilityPolicyField) -> bool {
        self.hot_reloadable.contains(&field)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservabilityPolicyField {
    SamplingRate,
    LogSeverity,
    ObservationFilters,
    ExporterState,
    BatchingLimits,
    Endpoint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationBus {
    capacity: usize,
    queue: VecDeque<ObservationRecord>,
    dropped: u64,
    overflow: ObservationOverflowPolicy,
}

impl ObservationBus {
    pub fn new(capacity: usize, overflow: ObservationOverflowPolicy) -> Self {
        Self {
            capacity,
            queue: VecDeque::new(),
            dropped: 0,
            overflow,
        }
    }

    pub fn try_emit(&mut self, record: ObservationRecord) -> Result<(), ObservabilityError> {
        if self.capacity == 0 {
            self.dropped += 1;
            return Err(ObservabilityError::observation_dropped());
        }
        if self.queue.len() < self.capacity {
            self.queue.push_back(record);
            return Ok(());
        }
        match self.overflow {
            ObservationOverflowPolicy::DropNewest => {
                self.dropped += 1;
                Err(ObservabilityError::observation_dropped())
            }
            ObservationOverflowPolicy::DropOldest => {
                self.queue.pop_front();
                self.queue.push_back(record);
                self.dropped += 1;
                Ok(())
            }
            ObservationOverflowPolicy::DegradeExporter => {
                self.dropped += 1;
                Err(ObservabilityError::exporter_saturated())
            }
            ObservationOverflowPolicy::DisableExporter => {
                self.dropped += 1;
                Err(ObservabilityError::exporter_unavailable())
            }
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    pub fn stream(&self, filter: ObservationFilter, max_batch: usize) -> ObservationStream {
        ObservationStream::new(self.queue.iter().cloned(), filter, max_batch)
    }

    pub fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            observation_queue_depth: self.queue_depth() as u64,
            dropped_observation_count: self.dropped,
            ..RuntimeMetricsSnapshot::default()
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeMetricsSnapshot {
    pub submitted_operations: u64,
    pub running_operations: u64,
    pub completed_operations: u64,
    pub failed_operations: u64,
    pub cancelled_operations: u64,
    pub interrupted_operations: u64,
    pub queue_depth: u64,
    pub provider_count: u64,
    pub device_count: u64,
    pub available_provider_count: u64,
    pub available_device_count: u64,
    pub estimated_memory_pressure: u64,
    pub observation_queue_depth: u64,
    pub dropped_observation_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulerMetricsSnapshot {
    pub queue_depth: u64,
    pub submitted: u64,
    pub running: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderMetricsSnapshot {
    pub states: BTreeMap<ProviderBinding, HealthState>,
    pub submitted_executions: u64,
    pub failed_executions: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeviceMetricsSnapshot {
    pub states: BTreeMap<DeviceBinding, HealthState>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObservabilityMetricsSnapshot {
    pub exporters: BTreeMap<String, ObservabilityComponentState>,
    pub dropped_observations: u64,
    pub stream_count: u64,
    pub snapshot_requests: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObservabilitySnapshot {
    pub runtime: RuntimeMetricsSnapshot,
    pub scheduler: SchedulerMetricsSnapshot,
    pub providers: ProviderMetricsSnapshot,
    pub devices: DeviceMetricsSnapshot,
    pub observability: ObservabilityMetricsSnapshot,
}

impl ObservabilitySnapshot {
    pub fn from_bus(bus: &ObservationBus) -> Self {
        Self {
            runtime: bus.snapshot(),
            observability: ObservabilityMetricsSnapshot {
                dropped_observations: bus.dropped_count(),
                ..ObservabilityMetricsSnapshot::default()
            },
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ObservabilitySinkDependency {
    OutboundHttp { endpoint_scope: String },
    FilesystemWrite { path_scope: String },
    SecretRead { namespace: String },
    StdoutLog,
}

impl ObservabilitySinkDependency {
    pub fn is_authorized_by(&self, policy: &ObservabilityPolicy) -> bool {
        match self {
            Self::OutboundHttp { endpoint_scope } => {
                policy.allowed_endpoints.contains(endpoint_scope)
            }
            Self::FilesystemWrite { path_scope } => {
                policy.allowed_filesystem_paths.contains(path_scope)
            }
            Self::SecretRead { namespace } => policy.allowed_secret_namespaces.contains(namespace),
            Self::StdoutLog => true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExporterRuntimeStatus {
    pub component: String,
    pub state: ObservabilityComponentState,
    pub failure: Option<ObservabilityError>,
}

impl ExporterRuntimeStatus {
    pub fn active(component: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            state: ObservabilityComponentState::Active,
            failure: None,
        }
    }

    pub fn failed(component: impl Into<String>, failure: ObservabilityError) -> Self {
        Self {
            component: component.into(),
            state: ObservabilityComponentState::Failed,
            failure: Some(failure),
        }
    }
}

pub fn opentelemetry_exporter_component() -> ObservabilityComponentDescriptor {
    let mut component = ComponentMetadata::new(
        "magnetar-opentelemetry-exporter",
        "1.0.0",
        "translates Magnetar observations to OpenTelemetry",
    );
    component.imports.insert(observability_stream_wit());
    let mut descriptor = ObservabilityComponentDescriptor::stream_exporter(component);
    descriptor
        .sinks
        .insert(ObservabilitySinkDependency::OutboundHttp {
            endpoint_scope: "otlp".into(),
        });
    descriptor.filter.categories.extend([
        ObservationCategory::RuntimeTrace,
        ObservationCategory::RuntimeMetric,
        ObservationCategory::RuntimeDiagnostic,
        ObservationCategory::RuntimeEvent,
    ]);
    descriptor
}

pub fn jaeger_exporter_component() -> ObservabilityComponentDescriptor {
    let mut component = ComponentMetadata::new(
        "magnetar-jaeger-exporter",
        "1.0.0",
        "translates Magnetar traces to Jaeger-compatible data",
    );
    component.imports.insert(observability_stream_wit());
    ObservationFilter::default();
    let mut descriptor = ObservabilityComponentDescriptor::stream_exporter(component);
    descriptor
        .filter
        .categories
        .insert(ObservationCategory::RuntimeTrace);
    descriptor
}

pub fn jsonl_exporter_component(path_scope: impl Into<String>) -> ObservabilityComponentDescriptor {
    let mut component = ComponentMetadata::new(
        "magnetar-jsonl-exporter",
        "1.0.0",
        "writes authorized observations as JSONL-compatible records",
    );
    component.imports.insert(observability_stream_wit());
    let mut descriptor = ObservabilityComponentDescriptor::stream_exporter(component);
    descriptor
        .sinks
        .insert(ObservabilitySinkDependency::FilesystemWrite {
            path_scope: path_scope.into(),
        });
    descriptor
}

pub fn prometheus_exposer_component() -> ObservabilityComponentDescriptor {
    let mut component = ComponentMetadata::new(
        "magnetar-prometheus-exposer",
        "1.0.0",
        "renders Runtime metrics snapshots for Prometheus scraping",
    );
    component.imports.insert(observability_reader_wit());
    ObservabilityComponentDescriptor::snapshot_exposer(component)
}

pub fn prometheus_snapshot_lines(snapshot: &RuntimeMetricsSnapshot) -> Vec<(String, u64)> {
    vec![
        (
            "magnetar_runtime_submitted_operations_total".into(),
            snapshot.submitted_operations,
        ),
        (
            "magnetar_runtime_running_operations".into(),
            snapshot.running_operations,
        ),
        (
            "magnetar_runtime_completed_operations_total".into(),
            snapshot.completed_operations,
        ),
        (
            "magnetar_runtime_failed_operations_total".into(),
            snapshot.failed_operations,
        ),
        (
            "magnetar_observation_queue_depth".into(),
            snapshot.observation_queue_depth,
        ),
        (
            "magnetar_observations_dropped_total".into(),
            snapshot.dropped_observation_count,
        ),
    ]
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservabilityErrorCode {
    InvalidObservation,
    UnsupportedObservation,
    AccessDenied,
    InvalidFilter,
    StreamClosed,
    StreamInterrupted,
    ExporterUnavailable,
    ExporterSaturated,
    ExporterFailed,
    SinkUnavailable,
    SinkUnauthorized,
    SinkTimeout,
    SerializationFailed,
    ObservabilityPolicyRejected,
    ObservationDropped,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityError {
    pub code: ObservabilityErrorCode,
    pub message: String,
    pub diagnostic: Option<String>,
}

impl ObservabilityError {
    pub fn new(code: ObservabilityErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            diagnostic: None,
        }
    }

    pub fn invalid_observation(message: impl Into<String>) -> Self {
        Self::new(ObservabilityErrorCode::InvalidObservation, message)
    }

    pub fn access_denied(message: impl Into<String>) -> Self {
        Self::new(ObservabilityErrorCode::AccessDenied, message)
    }

    pub fn stream_closed() -> Self {
        Self::new(
            ObservabilityErrorCode::StreamClosed,
            "observation stream is closed",
        )
    }

    pub fn stream_interrupted() -> Self {
        Self::new(
            ObservabilityErrorCode::StreamInterrupted,
            "observation stream was interrupted",
        )
    }

    pub fn exporter_unavailable() -> Self {
        Self::new(
            ObservabilityErrorCode::ExporterUnavailable,
            "observability exporter is unavailable",
        )
    }

    pub fn exporter_saturated() -> Self {
        Self::new(
            ObservabilityErrorCode::ExporterSaturated,
            "observability exporter is saturated",
        )
    }

    pub fn observation_dropped() -> Self {
        Self::new(
            ObservabilityErrorCode::ObservationDropped,
            "observation was dropped",
        )
    }

    pub fn with_diagnostic(mut self, diagnostic: impl Into<String>) -> Self {
        self.diagnostic = Some(diagnostic.into());
        self
    }
}

impl fmt::Display for ObservabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl Error for ObservabilityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuntimeEventKind, RuntimeObservationPhase, SpanId};

    fn event(message: &str) -> ObservationRecord {
        ObservationRecord::RuntimeEvent(RuntimeEvent::new(
            TraceId::new("trace:test"),
            SpanId::new("span:test"),
            RuntimeObservationPhase::Scheduling,
            RuntimeEventKind::Scheduled,
            message,
        ))
    }

    #[test]
    fn observability_components_are_not_compute_providers() {
        let descriptor = ObservabilityComponentDescriptor::stream_exporter(ComponentMetadata::new(
            "otel",
            "1",
            "OpenTelemetry exporter",
        ));

        assert!(!descriptor.is_provider());
        assert!(!descriptor.participates_in_compute_resolution());
        assert!(descriptor.imports.contains(&observability_stream_wit()));
    }

    #[test]
    fn custom_metric_namespaces_are_policy_scoped() {
        let mut policy = ObservabilityPolicy::default();
        policy
            .allowed_metric_namespaces
            .insert("component.foo".into());
        let allowed = CustomMetricRecord::new(
            "component.foo.latency",
            "p50",
            CustomMetricKind::Gauge,
            42,
            "foo",
        );
        let denied = CustomMetricRecord::new(
            "system.scheduler",
            "queue_depth",
            CustomMetricKind::Gauge,
            3,
            "foo",
        );

        policy.validate_custom_metric(&allowed).unwrap();
        assert_eq!(
            policy.validate_custom_metric(&denied).unwrap_err().code,
            ObservabilityErrorCode::AccessDenied
        );
    }

    #[test]
    fn observation_bus_drops_without_blocking_when_full() {
        let mut bus = ObservationBus::new(1, ObservationOverflowPolicy::DropNewest);

        bus.try_emit(event("first")).unwrap();
        let error = bus.try_emit(event("second")).unwrap_err();

        assert_eq!(error.code, ObservabilityErrorCode::ObservationDropped);
        assert_eq!(bus.queue_depth(), 1);
        assert_eq!(bus.dropped_count(), 1);
        assert_eq!(bus.snapshot().dropped_observation_count, 1);
    }

    #[test]
    fn observation_stream_filters_and_bounds_batches() {
        let mut stream = ObservationStream::new(
            [event("one"), event("two")],
            ObservationFilter::all().with_category(ObservationCategory::RuntimeEvent),
            1,
        );

        let first = stream.pull(128).unwrap();
        let second = stream.pull(128).unwrap();

        assert_eq!(first.records.len(), 1);
        assert!(!first.end_of_stream);
        assert_eq!(second.records.len(), 1);
        assert!(second.end_of_stream);
        stream.close();
        assert_eq!(
            stream.pull(1).unwrap_err().code,
            ObservabilityErrorCode::StreamClosed
        );
    }

    #[test]
    fn snapshot_and_prometheus_mapping_are_runtime_core_neutral() {
        let mut snapshot = RuntimeMetricsSnapshot {
            completed_operations: 7,
            observation_queue_depth: 2,
            dropped_observation_count: 1,
            ..RuntimeMetricsSnapshot::default()
        };
        snapshot.running_operations = 3;

        let prometheus = prometheus_snapshot_lines(&snapshot);

        assert!(prometheus.contains(&("magnetar_runtime_completed_operations_total".into(), 7)));
        assert!(prometheus.contains(&("magnetar_runtime_running_operations".into(), 3)));
    }

    #[test]
    fn exporter_examples_declare_explicit_sinks_and_contracts() {
        let otel = opentelemetry_exporter_component();
        let prometheus = prometheus_exposer_component();
        let jsonl = jsonl_exporter_component("/var/log/magnetar");

        assert!(otel.imports.contains(&observability_stream_wit()));
        assert!(
            otel.sinks
                .contains(&ObservabilitySinkDependency::OutboundHttp {
                    endpoint_scope: "otlp".into()
                })
        );
        assert!(prometheus.imports.contains(&observability_reader_wit()));
        assert!(
            jsonl
                .sinks
                .contains(&ObservabilitySinkDependency::FilesystemWrite {
                    path_scope: "/var/log/magnetar".into()
                })
        );
    }

    #[test]
    fn sink_dependencies_require_explicit_authorization() {
        let mut policy = ObservabilityPolicy::default();
        policy.allowed_endpoints.insert("otlp".into());
        let http = ObservabilitySinkDependency::OutboundHttp {
            endpoint_scope: "otlp".into(),
        };
        let secret = ObservabilitySinkDependency::SecretRead {
            namespace: "prod".into(),
        };

        assert!(http.is_authorized_by(&policy));
        assert!(!secret.is_authorized_by(&policy));
    }

    #[test]
    fn exporter_failures_are_reported_separately() {
        let status = ExporterRuntimeStatus::failed(
            "otel",
            ObservabilityError::new(ObservabilityErrorCode::SinkUnavailable, "collector down"),
        );

        assert_eq!(status.state, ObservabilityComponentState::Failed);
        assert_eq!(
            status.failure.as_ref().unwrap().code,
            ObservabilityErrorCode::SinkUnavailable
        );
        assert!(ObservabilityComponentState::Saturated.is_failure_or_pressure());
    }
}
