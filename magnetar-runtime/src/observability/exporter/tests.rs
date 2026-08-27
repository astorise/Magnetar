//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

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
fn observation_bus_stream_delivers_every_admitted_record_across_pulls() {
    let mut bus = ObservationBus::new(8, ObservationOverflowPolicy::DropOldest);
    for index in 0..5 {
        bus.try_emit(event(&format!("event-{index}"))).unwrap();
    }

    // max_batch bounds each pull, not the stream: draining in batches of
    // two must still yield all five records.
    let mut stream = bus.stream(ObservationFilter::all(), 2);
    let mut delivered = 0;
    loop {
        let batch = stream.pull(128).unwrap();
        delivered += batch.records.len();
        if batch.end_of_stream {
            break;
        }
    }

    assert_eq!(delivered, 5);
}

#[test]
fn observation_bus_stream_admits_only_filtered_records() {
    let mut bus = ObservationBus::new(8, ObservationOverflowPolicy::DropOldest);
    bus.try_emit(event("kept")).unwrap();

    let mut stream = bus.stream(
        ObservationFilter::all().with_category(ObservationCategory::RuntimeEvent),
        8,
    );
    assert_eq!(stream.pull(128).unwrap().records.len(), 1);

    let mut excluded = bus.stream(
        ObservationFilter::all().with_category(ObservationCategory::RuntimeMetric),
        8,
    );
    assert!(excluded.pull(128).unwrap().records.is_empty());
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
