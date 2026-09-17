//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::ProviderBinding;
use crate::component::{ComponentMetadata, WitInterface};

#[test]
fn runtime_diagnostics_redact_native_details_and_exporters_are_components() {
    let diagnostic = RuntimeDiagnostic::new(
        RuntimeDiagnosticCode::ExecutionFailed,
        "backend handle=0xdeadbeef at C:\\native\\queue",
    )
    .with_trace(TraceId::new("trace:failure"))
    .with_provider(ProviderBinding::new("provider-a"));
    assert_eq!(diagnostic.message, "[redacted backend diagnostic]");

    let component = ComponentMetadata::new("otel-exporter", "1", "exports observations");
    let mut exporter =
        ObservabilityExporterDescriptor::new(component, ObservabilitySink::OpenTelemetry);
    exporter
        .accepted_events
        .insert(RuntimeEventKind::ExecutionCompleted);

    assert_eq!(
        exporter.input_contract,
        WitInterface::new("magnetar:runtime/observability", "1.0.0")
    );
    assert!(
        exporter
            .accepted_events
            .contains(&RuntimeEventKind::ExecutionCompleted)
    );
}
