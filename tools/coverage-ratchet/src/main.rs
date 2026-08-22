use std::{env, fs, process};

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
    let current = coverage_percent(&read(current_path, "coverage report"));
    let baseline = coverage_percent(&read(baseline_path, "coverage baseline"));

    if summary {
        println!("## Coverage");
        println!();
        println!("- Current line coverage: {:.2}%", current);
        println!("- Accepted baseline: {:.2}%", baseline);
        println!(
            "- Ratchet: {}",
            if current + f64::EPSILON >= baseline {
                "pass"
            } else {
                "fail"
            }
        );
        return;
    }

    if current + f64::EPSILON < baseline {
        eprintln!(
            "line coverage {:.2}% is below accepted baseline {:.2}%",
            current, baseline
        );
        process::exit(1);
    }
}

fn read(path: &str, label: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        eprintln!("could not read {label} '{path}': {error}");
        process::exit(2);
    })
}

fn coverage_percent(json: &str) -> f64 {
    if let Some(totals) = json.find("\"totals\"") {
        if let Some(lines) = json[totals..].find("\"lines\"").map(|index| totals + index) {
            if let Some(value) = number_after_key(&json[lines..], "\"percent\"") {
                return value;
            }
        }
    }

    for key in [
        "\"line_coverage_percent\"",
        "\"line_percent\"",
        "\"lines_percent\"",
        "\"lineCoverage\"",
    ] {
        if let Some(value) = number_after_key(json, key) {
            return value;
        }
    }

    if let (Some(covered), Some(count)) = (
        number_after_key(json, "\"covered_lines\""),
        number_after_key(json, "\"count\""),
    ) {
        if count > 0.0 {
            return covered * 100.0 / count;
        }
    }

    eprintln!("coverage report does not contain a supported line coverage field");
    process::exit(2);
}

fn number_after_key(input: &str, key: &str) -> Option<f64> {
    let start = input.find(key)? + key.len();
    let after_colon = input[start..].find(':')? + start + 1;
    let number = input[after_colon..]
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || matches!(ch, '.' | '-'))
        .collect::<String>();
    number.parse().ok()
}
