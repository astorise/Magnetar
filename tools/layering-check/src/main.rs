//! `layering-check`: the Runtime layering gate
//! (`layer-and-split-runtime-crate`, `quality` spec "Runtime Layering Gate").
//!
//! Reads a version-controlled layer map (`quality/layers.toml`) and the
//! source of one crate, and fails when a top-level module depends on a
//! module of a higher layer, or of a different layer of the same rank, or
//! when a module has no layer.
//!
//! Usage:
//!
//! ```text
//! layering-check <crate-src-dir> <layers.toml>            # gate
//! layering-check --report <crate-src-dir> <layers.toml>   # every edge, with the items behind it
//! layering-check --cycles <crate-src-dir>                 # dependency cycles, no map needed
//! layering-check --debt <crate-src-dir> <layers.toml>     # upward edges, as `debt.edges` entries
//! ```

mod analysis;
mod layers;

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, rest) = match args.first().map(String::as_str) {
        Some("--report") => ("report", &args[1..]),
        Some("--cycles") => ("cycles", &args[1..]),
        Some("--debt") => ("debt", &args[1..]),
        _ => ("check", &args[..]),
    };
    let needs_map = mode != "cycles";
    if rest.len() != if needs_map { 2 } else { 1 } {
        eprintln!(
            "usage: layering-check [--report|--debt] <crate-src-dir> <layers.toml>\n       layering-check --cycles <crate-src-dir>"
        );
        return ExitCode::from(2);
    }
    let analysis = match analysis::analyze(Path::new(&rest[0])) {
        Ok(analysis) => analysis,
        Err(error) => {
            eprintln!("layering-check: {error}");
            return ExitCode::from(2);
        }
    };
    if mode == "cycles" {
        let cycles = analysis.cycles();
        for cycle in &cycles {
            println!("cycle ({} modules): {}", cycle.len(), cycle.join(" "));
        }
        println!("{} dependency cycle(s)", cycles.len());
        return ExitCode::SUCCESS;
    }
    let map = match std::fs::read_to_string(&rest[1])
        .map_err(|error| format!("cannot read {}: {error}", rest[1]))
        .and_then(|text| layers::LayerMap::parse(&text))
    {
        Ok(map) => map,
        Err(error) => {
            eprintln!("layering-check: {error}");
            return ExitCode::from(2);
        }
    };
    if mode == "debt" {
        for edge in layers::debt_edges(&analysis, &map) {
            println!("    \"{edge}\",");
        }
        return ExitCode::SUCCESS;
    }
    if mode == "report" {
        print!("{}", layers::report(&analysis, &map));
        return ExitCode::SUCCESS;
    }
    let violations = layers::check(&analysis, &map);
    if violations.is_empty() {
        println!(
            "layering-check: {} modules, {} dependencies, no violation",
            analysis.modules.len(),
            analysis.deps.values().map(|deps| deps.len()).sum::<usize>()
        );
        ExitCode::SUCCESS
    } else {
        for violation in &violations {
            eprintln!("{violation}");
        }
        eprintln!("layering-check: {} violation(s)", violations.len());
        ExitCode::FAILURE
    }
}
