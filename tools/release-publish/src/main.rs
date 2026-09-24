//! CLI entry point for `tools/release-publish`, invoked from
//! `.github/workflows/release-publication.yml` and
//! `.github/workflows/component-artifact-distribution.yml` the same way
//! `tools/coverage-ratchet` is invoked from `quality.yml`:
//! `cargo run --manifest-path tools/release-publish/Cargo.toml -- <args>`.
//! Every subcommand is a thin wrapper over this crate's own library
//! functions (see `lib.rs` and its modules) -- no logic lives here beyond
//! argument parsing and wiring.

use std::{env, fs, path::PathBuf, process};

use release_publish::{
    checksum, oci::layout, oci::manifest::build_artifact_manifest, oci::media_types::ArtifactKind,
    provenance, release_gate, sbom, security_gate,
};

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage_and_exit();
    }
    let command = args.remove(0);
    let result = match command.as_str() {
        "checksum-bundle" => cmd_checksum_bundle(args),
        "verify-checksums" => cmd_verify_checksums(args),
        "sbom" => cmd_sbom(args),
        "provenance" => cmd_provenance(args),
        "release-gate" => cmd_release_gate(args),
        "security-gate" => cmd_security_gate(args),
        "oci-manifest" => cmd_oci_manifest(args),
        _ => usage_and_exit(),
    };
    if let Err(message) = result {
        eprintln!("error: {message}");
        process::exit(1);
    }
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: release-publish <command> [args]\n\n\
         commands:\n  \
         checksum-bundle <dir> --out <file>\n  \
         verify-checksums <bundle.json> <dir>\n  \
         sbom --cargo-metadata <file> --out <file> [--build-target T] [--feature F]...\n  \
         provenance --repo <dir> --out <file> [--openspec-dir D] [--wit-dir D] \
         [--release-tag T] [--ci-run-id I] [--build-target T] [--build-profile P] \
         [--conformance-digest D]\n  \
         release-gate --results <file>\n  \
         security-gate --cargo-deny-json <file> --out <file> [--secrets-detected] \
         [--redaction-gate-failed] [--raw-handle-exposed] [--trust-integrity-failed] \
         [--e2e-bypassed] [--openspec-failed] [--checksum-mismatch] \
         [--undocumented-exception]\n  \
         oci-manifest --kind component|kernel-bundle|conformance --input <file> \
         --title <name> --tag <tag> --out-dir <dir>"
    );
    process::exit(2);
}

fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    if index + 1 >= args.len() {
        return None;
    }
    args.remove(index);
    Some(args.remove(index))
}

fn take_all_values(args: &mut Vec<String>, flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    while let Some(value) = take_value(args, flag) {
        values.push(value);
    }
    values
}

fn take_switch(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn cmd_checksum_bundle(mut args: Vec<String>) -> Result<(), String> {
    let out = take_value(&mut args, "--out").ok_or("checksum-bundle requires --out <file>")?;
    let dir = args
        .into_iter()
        .next()
        .ok_or("checksum-bundle requires a directory argument")?;
    let bundle =
        checksum::checksum_bundle_for_dir(&PathBuf::from(&dir)).map_err(|e| e.to_string())?;
    let json = checksum::bundle_to_json(&bundle);
    fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote {} checksum(s) to {out}", bundle.len());
    Ok(())
}

fn cmd_verify_checksums(args: Vec<String>) -> Result<(), String> {
    if args.len() != 2 {
        return Err("verify-checksums requires <bundle.json> <dir>".to_string());
    }
    let bundle_json = fs::read_to_string(&args[0]).map_err(|e| e.to_string())?;
    let bundle = checksum::bundle_from_json(&bundle_json).map_err(|e| e.to_string())?;
    checksum::verify_bundle_against_dir(&bundle, &PathBuf::from(&args[1]))
        .map_err(|e| e.to_string())?;
    println!("verified {} checksum(s) against {}", bundle.len(), args[1]);
    Ok(())
}

fn cmd_sbom(mut args: Vec<String>) -> Result<(), String> {
    let metadata_path =
        take_value(&mut args, "--cargo-metadata").ok_or("sbom requires --cargo-metadata <file>")?;
    let out = take_value(&mut args, "--out").ok_or("sbom requires --out <file>")?;
    let build_target = take_value(&mut args, "--build-target");
    let feature_flags = take_all_values(&mut args, "--feature");

    let metadata_json = fs::read_to_string(&metadata_path).map_err(|e| e.to_string())?;
    let manifest = sbom::generate_sbom_manifest(&metadata_json, build_target, feature_flags)
        .map_err(|e| e.to_string())?;
    manifest.validate().map_err(|e| e.to_string())?;
    let json = sbom::manifest_to_json(&manifest);
    fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string())?;
    println!(
        "wrote SBOM with {} entries to {out}",
        manifest.entries.len()
    );
    Ok(())
}

fn cmd_provenance(mut args: Vec<String>) -> Result<(), String> {
    let repo = take_value(&mut args, "--repo").ok_or("provenance requires --repo <dir>")?;
    let out = take_value(&mut args, "--out").ok_or("provenance requires --out <file>")?;
    let openspec_dir = take_value(&mut args, "--openspec-dir").map(PathBuf::from);
    let wit_dir = take_value(&mut args, "--wit-dir").map(PathBuf::from);
    let release_tag = take_value(&mut args, "--release-tag");
    let ci_run_id = take_value(&mut args, "--ci-run-id");
    let build_target = take_value(&mut args, "--build-target");
    let build_profile = take_value(&mut args, "--build-profile");
    let conformance_digest = take_value(&mut args, "--conformance-digest");

    let provenance = provenance::collect_local_provenance(
        &PathBuf::from(&repo),
        openspec_dir.as_deref(),
        wit_dir.as_deref(),
        release_tag,
        ci_run_id,
        build_target,
        build_profile,
        conformance_digest,
    )
    .map_err(|e| e.to_string())?;
    let json = provenance::provenance_to_json(&provenance);
    fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote release provenance to {out}");
    Ok(())
}

fn cmd_release_gate(mut args: Vec<String>) -> Result<(), String> {
    let results_path =
        take_value(&mut args, "--results").ok_or("release-gate requires --results <file>")?;
    let results_json = fs::read_to_string(&results_path).map_err(|e| e.to_string())?;
    let results = release_gate::gate_results_from_json(&results_json).map_err(|e| e.to_string())?;
    release_gate::evaluate_release_gates(&results)?;
    println!("all required release gates passed");
    Ok(())
}

fn cmd_security_gate(mut args: Vec<String>) -> Result<(), String> {
    let deny_json_path = take_value(&mut args, "--cargo-deny-json")
        .ok_or("security-gate requires --cargo-deny-json <file>")?;
    let out = take_value(&mut args, "--out").ok_or("security-gate requires --out <file>")?;
    let secrets_detected = take_switch(&mut args, "--secrets-detected");
    let redaction_gate_failed = take_switch(&mut args, "--redaction-gate-failed");
    let raw_handle_exposed = take_switch(&mut args, "--raw-handle-exposed");
    let trust_integrity_failed = take_switch(&mut args, "--trust-integrity-failed");
    let e2e_bypassed = take_switch(&mut args, "--e2e-bypassed");
    let openspec_failed = take_switch(&mut args, "--openspec-failed");
    let checksum_mismatch = take_switch(&mut args, "--checksum-mismatch");
    let undocumented_exception = take_switch(&mut args, "--undocumented-exception");

    let deny_json = fs::read_to_string(&deny_json_path).map_err(|e| e.to_string())?;
    let inputs = security_gate::collect_security_gate_inputs(
        &deny_json,
        secrets_detected,
        redaction_gate_failed,
        raw_handle_exposed,
        trust_integrity_failed,
        e2e_bypassed,
        openspec_failed,
        checksum_mismatch,
        undocumented_exception,
    )
    .map_err(|e| e.to_string())?;

    let json = security_gate::gate_inputs_to_json(&inputs);
    fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string())?;

    magnetar_roadmap_contracts::evaluate_release_security_blocking(&inputs)
        .map_err(|e| e.to_string())?;
    println!("no release-blocking security condition detected");
    Ok(())
}

fn cmd_oci_manifest(mut args: Vec<String>) -> Result<(), String> {
    let kind_name = take_value(&mut args, "--kind").ok_or("oci-manifest requires --kind <kind>")?;
    let kind = match kind_name.as_str() {
        "component" => ArtifactKind::ComponentArtifact,
        "kernel-bundle" => ArtifactKind::KernelExchangeBundle,
        "conformance" => ArtifactKind::ConformanceFixture,
        other => {
            return Err(format!(
                "unknown --kind '{other}' (expected component, kernel-bundle, or conformance)"
            ));
        }
    };
    let input = take_value(&mut args, "--input").ok_or("oci-manifest requires --input <file>")?;
    let title = take_value(&mut args, "--title").ok_or("oci-manifest requires --title <name>")?;
    let tag = take_value(&mut args, "--tag").ok_or("oci-manifest requires --tag <tag>")?;
    let out_dir =
        take_value(&mut args, "--out-dir").ok_or("oci-manifest requires --out-dir <dir>")?;

    let layer_bytes = fs::read(&input).map_err(|e| e.to_string())?;
    let built = build_artifact_manifest(kind, &layer_bytes, &title).map_err(|e| e.to_string())?;
    layout::write_oci_layout(&PathBuf::from(&out_dir), &built, &layer_bytes, &tag)
        .map_err(|e| e.to_string())?;
    layout::verify_oci_layout(&PathBuf::from(&out_dir)).map_err(|e| e.to_string())?;

    println!("manifest digest: {}", built.manifest_digest);
    println!("layer digest: {}", built.layer_digest);
    println!("OCI layout written and verified at {out_dir}");
    Ok(())
}
