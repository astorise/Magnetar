//! Assembles a real [`ReleaseSecurityGateInputs`] from other CI jobs'
//! actual outcomes (tasks.md section 1, item 1.6). This crate does not
//! reimplement secret scanning, redaction checks, or raw-handle-exposure
//! checks -- those already exist as separate CI jobs/tests
//! (`.github/workflows/quality.yml`); duplicating their logic here would
//! let the two drift apart. What this module DOES compute for itself is
//! parsing `cargo deny`'s own real JSON diagnostic stream: re-deriving
//! two of the ten fields directly from it is more honest than asking
//! every caller to hand-translate cargo-deny's exit code into a boolean.

use magnetar_roadmap_contracts::ReleaseSecurityGateInputs;
use serde_json::Value;

use crate::ReleasePublishError;

/// The two [`ReleaseSecurityGateInputs`] fields `cargo deny -f json check`'s
/// own newline-delimited JSON output can answer: whether its `advisories`
/// or `licenses` category recorded a real `errors` count above zero.
/// `cargo deny`'s own severity configuration (`deny.toml`) decides what
/// counts as an error versus a warning; this function trusts that
/// decision rather than re-implementing it. Returns
/// `(critical_advisory_unmitigated, incompatible_license_unapproved)`.
pub fn cargo_deny_gate_inputs_from_json(
    cargo_deny_jsonl: &str,
) -> Result<(bool, bool), ReleasePublishError> {
    for line in cargo_deny_jsonl.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        if value.get("type").and_then(Value::as_str) == Some("summary") {
            let fields = value.get("fields").ok_or_else(|| {
                ReleasePublishError::Metadata("cargo-deny summary line missing 'fields'".into())
            })?;
            let advisory_errors = fields["advisories"]["errors"].as_u64().unwrap_or(0);
            let license_errors = fields["licenses"]["errors"].as_u64().unwrap_or(0);
            return Ok((advisory_errors > 0, license_errors > 0));
        }
    }
    Err(ReleasePublishError::Metadata(
        "cargo-deny output did not contain a summary line".into(),
    ))
}

/// Composes a full [`ReleaseSecurityGateInputs`] from the cargo-deny-derived
/// fields plus every other field, which the caller supplies directly from
/// its own CI job's outcome -- this crate cannot observe a secret scan, a
/// redaction test suite, or an E2E conformance run itself.
#[allow(clippy::too_many_arguments)]
pub fn collect_security_gate_inputs(
    cargo_deny_jsonl: &str,
    secrets_detected: bool,
    redaction_gate_failed: bool,
    raw_handle_exposed: bool,
    trust_integrity_failed_in_fixtures: bool,
    e2e_conformance_bypassed: bool,
    openspec_validation_failed: bool,
    checksum_mismatch: bool,
    undocumented_security_exception: bool,
) -> Result<ReleaseSecurityGateInputs, ReleasePublishError> {
    let (critical_advisory_unmitigated, incompatible_license_unapproved) =
        cargo_deny_gate_inputs_from_json(cargo_deny_jsonl)?;
    Ok(ReleaseSecurityGateInputs {
        secrets_detected,
        critical_advisory_unmitigated,
        incompatible_license_unapproved,
        redaction_gate_failed,
        raw_handle_exposed,
        trust_integrity_failed_in_fixtures,
        e2e_conformance_bypassed,
        openspec_validation_failed,
        checksum_mismatch,
        undocumented_security_exception,
    })
}

/// Serializes a [`ReleaseSecurityGateInputs`] to the JSON this crate's CLI
/// writes as a release artifact, recording exactly what was evaluated
/// (not just pass/fail) for later audit. `ReleaseSecurityGateInputs`
/// carries no `Serialize` impl (it lives in `magnetar-roadmap-contracts`),
/// so this is the one place that shape is defined.
pub fn gate_inputs_to_json(inputs: &ReleaseSecurityGateInputs) -> serde_json::Value {
    serde_json::json!({
        "secrets_detected": inputs.secrets_detected,
        "critical_advisory_unmitigated": inputs.critical_advisory_unmitigated,
        "incompatible_license_unapproved": inputs.incompatible_license_unapproved,
        "redaction_gate_failed": inputs.redaction_gate_failed,
        "raw_handle_exposed": inputs.raw_handle_exposed,
        "trust_integrity_failed_in_fixtures": inputs.trust_integrity_failed_in_fixtures,
        "e2e_conformance_bypassed": inputs.e2e_conformance_bypassed,
        "openspec_validation_failed": inputs.openspec_validation_failed,
        "checksum_mismatch": inputs.checksum_mismatch,
        "undocumented_security_exception": inputs.undocumented_security_exception,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN_FIXTURE: &str = r#"
        {"fields":{"code":"duplicate","message":"found 2 duplicate entries"},"type":"diagnostic"}
        {"fields":{"advisories":{"errors":0,"helps":0,"notes":0,"warnings":1},"bans":{"errors":0,"helps":0,"notes":0,"warnings":8},"licenses":{"errors":0,"helps":130,"notes":0,"warnings":1},"sources":{"errors":0,"helps":0,"notes":0,"warnings":0}},"type":"summary"}
    "#;

    const BLOCKED_FIXTURE: &str = r#"
        {"fields":{"advisories":{"errors":1,"helps":0,"notes":0,"warnings":0},"bans":{"errors":0,"helps":0,"notes":0,"warnings":0},"licenses":{"errors":2,"helps":0,"notes":0,"warnings":0},"sources":{"errors":0,"helps":0,"notes":0,"warnings":0}},"type":"summary"}
    "#;

    #[test]
    fn clean_run_reports_no_advisory_or_license_blocking() {
        let (advisory, license) = cargo_deny_gate_inputs_from_json(CLEAN_FIXTURE).unwrap();
        assert!(!advisory);
        assert!(!license);
    }

    #[test]
    fn errors_above_zero_are_reported_as_blocking() {
        let (advisory, license) = cargo_deny_gate_inputs_from_json(BLOCKED_FIXTURE).unwrap();
        assert!(advisory);
        assert!(license);
    }

    #[test]
    fn missing_summary_line_is_an_error() {
        let result = cargo_deny_gate_inputs_from_json(
            r#"{"fields":{"message":"only a diagnostic"},"type":"diagnostic"}"#,
        );
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }

    #[test]
    fn collect_security_gate_inputs_composes_all_ten_fields() {
        let inputs = collect_security_gate_inputs(
            CLEAN_FIXTURE,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(inputs, ReleaseSecurityGateInputs::default());
    }

    #[test]
    fn gate_inputs_to_json_carries_every_field() {
        let inputs = collect_security_gate_inputs(
            BLOCKED_FIXTURE,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        let json = gate_inputs_to_json(&inputs);
        assert_eq!(json["secrets_detected"], true);
        assert_eq!(json["critical_advisory_unmitigated"], true);
        assert_eq!(json["incompatible_license_unapproved"], true);
        assert_eq!(json["checksum_mismatch"], false);
    }
}
