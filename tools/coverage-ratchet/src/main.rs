//! Enforces the coverage ratchet: the current run's line (and, when both
//! sides report it, function) coverage must be at or above
//! `quality/coverage-baseline.json`'s recorded figures.
//!
//! #60: previously parsed both JSON documents by raw `str::find` substring
//! search over the whole file text. That reads the *first occurrence*
//! of a key's text anywhere in the document, including inside an unrelated
//! string value -- the baseline's own `notes` field is exactly that kind of
//! prose, and a rewording that happened to quote a JSON-shaped fragment
//! (e.g. `"line_coverage_percent": 99.99`) inside it would have been read
//! as the real figure with no error. It also only ever enforced
//! `line_coverage_percent`, silently ignoring `function_coverage_percent`
//! despite the baseline recording it and CI's own step being named "Check
//! coverage ratchet".
//!
//! Parses with `serde_json` instead: [`find_totals_percentages`] walks the
//! actual JSON *structure* (never a byte-offset text scan) looking for the
//! first object carrying a `"totals"` key with `lines`/`functions` percent
//! figures underneath it -- the shape `cargo llvm-cov --json --summary-only`
//! exports (nested under `data[0].totals` today, though this does not
//! hardcode that exact path, matching the flexibility the previous
//! substring search had without its false-match risk). A string value can
//! never be mistaken for a real field this way: `Value::String` is a leaf
//! the tree walk does not search into. [`parse_coverage_report`] falls back
//! to this project's own flat baseline-summary shape
//! (`line_coverage_percent`/`function_coverage_percent`) via real object-key
//! lookups, then to a `covered_lines`/`count` ratio, matching the shapes the
//! original implementation supported.
//!
//! [`evaluate`] is the one place the pass/fail decision is made, used by
//! both the gate and `--summary` (previously computed twice, once in each
//! branch, from two separate literal expressions with nothing keeping them
//! in agreement).

use std::{env, fs, process};

use serde_json::Value;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let summary = args.first().is_some_and(|arg| arg == "--summary");
    let offset = usize::from(summary);

    if args.len() != offset + 2 {
        eprintln!("usage: coverage-ratchet [--summary] <coverage.json> <baseline.json>");
        process::exit(2);
    }

    let current_path = &args[offset];
    let baseline_path = &args[offset + 1];
    let current = parse_coverage_report(&read(current_path, "coverage report"), "coverage report")
        .unwrap_or_else(|message| {
            eprintln!("{message}");
            process::exit(2);
        });
    let baseline = parse_coverage_report(
        &read(baseline_path, "coverage baseline"),
        "coverage baseline",
    )
    .unwrap_or_else(|message| {
        eprintln!("{message}");
        process::exit(2);
    });

    let outcome = evaluate(current, baseline);

    if summary {
        print_summary(&outcome);
        return;
    }

    if !outcome.passed() {
        for line in outcome.failure_lines() {
            eprintln!("{line}");
        }
        process::exit(1);
    }
}

fn read(path: &str, label: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        eprintln!("could not read {label} '{path}': {error}");
        process::exit(2);
    })
}

/// Line coverage (always present) and function coverage (present when the
/// source report includes it -- both `cargo-llvm-cov`'s export and this
/// project's own baseline file do today, but a report that omits it is not
/// treated as a parse failure).
#[derive(Clone, Copy, Debug, PartialEq)]
struct CoverageFigures {
    line_percent: f64,
    function_percent: Option<f64>,
}

fn parse_coverage_report(json_text: &str, label: &str) -> Result<CoverageFigures, String> {
    let value: Value = serde_json::from_str(json_text)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;

    if let Some(figures) = find_totals_percentages(&value) {
        return Ok(figures);
    }

    // This project's own `quality/coverage-baseline.json` shape: flat
    // top-level keys, real object-key lookups (never a text scan that
    // could match the same text sitting inside an unrelated string value,
    // e.g. this file's own `notes` field).
    for key in [
        "line_coverage_percent",
        "line_percent",
        "lines_percent",
        "lineCoverage",
    ] {
        if let Some(line_percent) = value.get(key).and_then(Value::as_f64) {
            let function_percent = value
                .get("function_coverage_percent")
                .and_then(Value::as_f64);
            return Ok(CoverageFigures {
                line_percent,
                function_percent,
            });
        }
    }

    if let (Some(covered), Some(count)) = (
        value.get("covered_lines").and_then(Value::as_f64),
        value.get("count").and_then(Value::as_f64),
    ) && count > 0.0
    {
        return Ok(CoverageFigures {
            line_percent: covered * 100.0 / count,
            function_percent: None,
        });
    }

    Err(format!(
        "{label} does not contain a supported line coverage field"
    ))
}

/// Depth-first search for the first JSON object carrying a `"totals"` key
/// whose value has a `lines.percent` figure underneath it (the shape
/// `cargo llvm-cov --json --summary-only` exports). Real structural
/// traversal: only ever descends into `Value::Object`/`Value::Array`
/// nodes, so text that happens to read `"totals"` or `"percent"` inside an
/// unrelated `Value::String` leaf is never visited, let alone matched.
fn find_totals_percentages(value: &Value) -> Option<CoverageFigures> {
    match value {
        Value::Object(map) => {
            if let Some(totals) = map.get("totals")
                && let Some(line_percent) = totals.pointer("/lines/percent").and_then(Value::as_f64)
            {
                let function_percent = totals.pointer("/functions/percent").and_then(Value::as_f64);
                return Some(CoverageFigures {
                    line_percent,
                    function_percent,
                });
            }
            map.values().find_map(find_totals_percentages)
        }
        Value::Array(items) => items.iter().find_map(find_totals_percentages),
        _ => None,
    }
}

struct RatchetOutcome {
    current: CoverageFigures,
    baseline: CoverageFigures,
    line_pass: bool,
    function_pass: bool,
}

impl RatchetOutcome {
    fn passed(&self) -> bool {
        self.line_pass && self.function_pass
    }

    fn failure_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if !self.line_pass {
            lines.push(format!(
                "line coverage {:.2}% is below accepted baseline {:.2}%",
                self.current.line_percent, self.baseline.line_percent
            ));
        }
        if !self.function_pass
            && let (Some(current), Some(baseline)) = (
                self.current.function_percent,
                self.baseline.function_percent,
            )
        {
            lines.push(format!(
                "function coverage {current:.2}% is below accepted baseline {baseline:.2}%"
            ));
        }
        lines
    }
}

/// The one place the ratchet decision is made, on both axes it has data
/// for. A side missing `function_percent` (a report shape that does not
/// carry it) passes that axis rather than blocking on data that is not
/// there -- there is nothing to ratchet against.
fn evaluate(current: CoverageFigures, baseline: CoverageFigures) -> RatchetOutcome {
    let line_pass = current.line_percent + f64::EPSILON >= baseline.line_percent;
    let function_pass = match (current.function_percent, baseline.function_percent) {
        (Some(current), Some(baseline)) => current + f64::EPSILON >= baseline,
        _ => true,
    };
    RatchetOutcome {
        current,
        baseline,
        line_pass,
        function_pass,
    }
}

fn print_summary(outcome: &RatchetOutcome) {
    println!("## Coverage");
    println!();
    println!(
        "- Current line coverage: {:.2}%",
        outcome.current.line_percent
    );
    println!("- Accepted baseline: {:.2}%", outcome.baseline.line_percent);
    if let (Some(current), Some(baseline)) = (
        outcome.current.function_percent,
        outcome.baseline.function_percent,
    ) {
        println!("- Current function coverage: {current:.2}%");
        println!("- Accepted function baseline: {baseline:.2}%");
    }
    println!(
        "- Ratchet: {}",
        if outcome.passed() { "pass" } else { "fail" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn figures(line: f64, function: Option<f64>) -> CoverageFigures {
        CoverageFigures {
            line_percent: line,
            function_percent: function,
        }
    }

    #[test]
    fn parses_llvm_cov_totals_shape_nested_under_data() {
        let json = r#"{
            "data": [{"totals": {"lines": {"percent": 81.5}, "functions": {"percent": 79.25}}}],
            "type": "llvm.coverage.json.export",
            "version": "2.0.1"
        }"#;
        let parsed = parse_coverage_report(json, "coverage report").unwrap();
        assert_eq!(parsed, figures(81.5, Some(79.25)));
    }

    #[test]
    fn parses_llvm_cov_totals_shape_at_top_level() {
        let json = r#"{"totals": {"lines": {"percent": 60.0}, "functions": {"percent": 55.0}}}"#;
        let parsed = parse_coverage_report(json, "coverage report").unwrap();
        assert_eq!(parsed, figures(60.0, Some(55.0)));
    }

    #[test]
    fn parses_this_projects_own_baseline_summary_shape() {
        let json = r#"{
            "line_coverage_percent": 78.37312271884163,
            "function_coverage_percent": 77.9957953749124,
            "branch_coverage_percent": null,
            "notes": "irrelevant prose"
        }"#;
        let parsed = parse_coverage_report(json, "coverage baseline").unwrap();
        assert_eq!(parsed, figures(78.37312271884163, Some(77.9957953749124)));
    }

    #[test]
    fn parses_covered_lines_count_fallback_shape() {
        let json = r#"{"covered_lines": 50, "count": 200}"#;
        let parsed = parse_coverage_report(json, "coverage report").unwrap();
        assert_eq!(parsed, figures(25.0, None));
    }

    /// #60: a real danger of the previous byte-offset substring search --
    /// the exact key text sitting inside an unrelated string value (here,
    /// prose quoting what looks like a JSON field) must never be read as
    /// the real figure. The document below has no actual
    /// `line_coverage_percent` *key*, only that text inside a `description`
    /// string, so parsing must fail rather than silently return `99.99`.
    #[test]
    fn a_key_appearing_only_inside_a_string_value_is_never_treated_as_the_real_field() {
        let json = r#"{"description": "field \"line_coverage_percent\": 99.99 was renamed"}"#;
        let result = parse_coverage_report(json, "coverage report");
        assert!(result.is_err());
    }

    #[test]
    fn malformed_json_is_rejected_with_a_clear_error() {
        let result = parse_coverage_report("not json at all", "coverage report");
        let error = result.unwrap_err();
        assert!(error.contains("coverage report"));
        assert!(error.contains("not valid JSON"));
    }

    #[test]
    fn well_formed_json_with_no_recognized_coverage_field_is_rejected() {
        let result = parse_coverage_report(r#"{"unrelated": "value"}"#, "coverage report");
        assert!(result.is_err());
    }

    #[test]
    fn report_at_or_above_baseline_passes_on_both_axes() {
        let outcome = evaluate(figures(80.0, Some(80.0)), figures(78.0, Some(77.0)));
        assert!(outcome.passed());
        assert!(outcome.failure_lines().is_empty());
    }

    #[test]
    fn report_exactly_at_the_baseline_passes() {
        let outcome = evaluate(figures(78.0, Some(77.0)), figures(78.0, Some(77.0)));
        assert!(outcome.passed());
    }

    #[test]
    fn line_coverage_regression_fails_the_gate() {
        let outcome = evaluate(figures(70.0, Some(80.0)), figures(78.0, Some(77.0)));
        assert!(!outcome.passed());
        assert!(!outcome.line_pass);
        assert!(
            outcome
                .failure_lines()
                .iter()
                .any(|line| line.contains("line coverage"))
        );
    }

    /// #60: function coverage is now actually enforced -- previously a
    /// function-coverage regression passed silently no matter how large,
    /// despite the baseline recording the figure and CI's step being named
    /// "Check coverage ratchet".
    #[test]
    fn function_coverage_regression_fails_the_gate_even_when_line_coverage_improves() {
        let outcome = evaluate(figures(85.0, Some(50.0)), figures(78.0, Some(77.0)));
        assert!(!outcome.passed());
        assert!(outcome.line_pass);
        assert!(!outcome.function_pass);
        assert!(
            outcome
                .failure_lines()
                .iter()
                .any(|line| line.contains("function coverage"))
        );
    }

    #[test]
    fn missing_function_coverage_on_either_side_does_not_block_the_gate() {
        let outcome = evaluate(figures(80.0, None), figures(78.0, Some(77.0)));
        assert!(outcome.function_pass);
        assert!(outcome.passed());
    }
}
