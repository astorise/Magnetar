//! Assembles [`ReleaseGateResult`]s from named CI check outcomes and
//! evaluates them against `release_packaging::release_may_publish_stable`
//! (tasks.md section 2, item 2.1). The gate names this module recognizes
//! are exactly `ReleaseGate`'s own `Debug` output (`"Formatting"`,
//! `"CargoCheck"`, ...), so a results file staying in sync with the real
//! enum is a compile-time-adjacent guarantee, not a convention someone has
//! to remember.

use magnetar_roadmap_contracts::{ReleaseGate, ReleaseGateResult, ReleasePackagingError};

use crate::ReleasePublishError;

/// Maps a gate's `Debug` name (e.g. `"CargoCheck"`) back to its
/// [`ReleaseGate`] variant. Returns `None` for any other string, so a typo
/// in a results file is reported as an unknown gate rather than silently
/// ignored.
pub fn gate_from_name(name: &str) -> Option<ReleaseGate> {
    Some(match name {
        "Formatting" => ReleaseGate::Formatting,
        "CargoCheck" => ReleaseGate::CargoCheck,
        "Clippy" => ReleaseGate::Clippy,
        "UnitTests" => ReleaseGate::UnitTests,
        "ContractTests" => ReleaseGate::ContractTests,
        "OpenSpecValidation" => ReleaseGate::OpenSpecValidation,
        "WitValidation" => ReleaseGate::WitValidation,
        "ReferenceCpuConformance" => ReleaseGate::ReferenceCpuConformance,
        "OperatorFirstScopeConformance" => ReleaseGate::OperatorFirstScopeConformance,
        "RuntimeInferenceApiTests" => ReleaseGate::RuntimeInferenceApiTests,
        "CliBoundaryTests" => ReleaseGate::CliBoundaryTests,
        "E2eLocalConformance" => ReleaseGate::E2eLocalConformance,
        "CoverageGate" => ReleaseGate::CoverageGate,
        "RedactionChecks" => ReleaseGate::RedactionChecks,
        "NoRawHandleExposureChecks" => ReleaseGate::NoRawHandleExposureChecks,
        _ => return None,
    })
}

/// Parses a `[{"gate": "Formatting", "passed": true}, ...]` JSON array
/// (the shape the release-publication workflow writes from real check-run
/// conclusions) into [`ReleaseGateResult`]s.
pub fn gate_results_from_json(json: &str) -> Result<Vec<ReleaseGateResult>, ReleasePublishError> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let entries = value.as_array().ok_or_else(|| {
        ReleasePublishError::Metadata("release gate results JSON must be an array".into())
    })?;
    entries
        .iter()
        .map(|entry| {
            let name = entry["gate"].as_str().ok_or_else(|| {
                ReleasePublishError::Metadata("gate result entry missing 'gate'".into())
            })?;
            let gate = gate_from_name(name).ok_or_else(|| {
                ReleasePublishError::Metadata(format!("unrecognized release gate '{name}'"))
            })?;
            let passed = entry["passed"].as_bool().ok_or_else(|| {
                ReleasePublishError::Metadata(format!(
                    "gate result entry for '{name}' missing boolean 'passed'"
                ))
            })?;
            Ok(ReleaseGateResult { gate, passed })
        })
        .collect()
}

/// Evaluates `results` against `release_may_publish_stable` and renders a
/// human-readable reason on failure (that function's own error carries a
/// `Debug`-formatted gate name and no further detail, which is enough to
/// act on but not friendly print output on its own).
pub fn evaluate_release_gates(results: &[ReleaseGateResult]) -> Result<(), String> {
    magnetar_roadmap_contracts::release_may_publish_stable(results).map_err(|error| match error {
        ReleasePackagingError::ReleaseGateMissing { gate } => {
            format!(
                "required release gate '{gate}' has no result -- it was not run or not reported"
            )
        }
        ReleasePackagingError::ReleaseGateFailed { gate } => {
            format!("required release gate '{gate}' failed")
        }
        other => format!("{other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_from_name_round_trips_every_required_gate() {
        for gate in magnetar_roadmap_contracts::REQUIRED_RELEASE_GATES {
            let name = format!("{gate:?}");
            assert_eq!(gate_from_name(&name), Some(*gate), "gate name '{name}'");
        }
    }

    #[test]
    fn gate_from_name_rejects_unknown_names() {
        assert_eq!(gate_from_name("NotARealGate"), None);
    }

    #[test]
    fn gate_results_from_json_parses_real_shape() {
        let json = r#"[{"gate":"Formatting","passed":true},{"gate":"Clippy","passed":false}]"#;
        let results = gate_results_from_json(json).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].gate, ReleaseGate::Formatting);
        assert!(results[0].passed);
        assert_eq!(results[1].gate, ReleaseGate::Clippy);
        assert!(!results[1].passed);
    }

    #[test]
    fn gate_results_from_json_rejects_unrecognized_gate_name() {
        let json = r#"[{"gate":"TotallyMadeUp","passed":true}]"#;
        let result = gate_results_from_json(json);
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }

    #[test]
    fn evaluate_release_gates_reports_missing_gate_by_name() {
        let results = vec![ReleaseGateResult {
            gate: ReleaseGate::Formatting,
            passed: true,
        }];
        let error = evaluate_release_gates(&results).unwrap_err();
        assert!(error.contains("CliBoundaryTests") || error.contains("required release gate"));
    }

    #[test]
    fn evaluate_release_gates_passes_when_every_required_gate_passed() {
        let results: Vec<ReleaseGateResult> = magnetar_roadmap_contracts::REQUIRED_RELEASE_GATES
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        assert!(evaluate_release_gates(&results).is_ok());
    }
}
