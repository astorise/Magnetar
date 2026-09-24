//! Exercises the real compiled `release-publish` binary end to end (not
//! just the library functions it wraps): every subcommand's argument
//! parsing, file I/O, and process exit code, via the actual `main.rs`
//! dispatch. `main.rs` itself has no unit tests of its own -- this is
//! deliberate, matching `tools/coverage-ratchet`'s convention of testing
//! its own thin CLI wrapper through its real binary rather than
//! unit-testing argument-parsing helpers in isolation.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("release-publish-cli-it-{prefix}-{nanos}-{n}"));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tools/")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_release-publish"))
}

const CARGO_METADATA_FIXTURE: &str = r#"{
    "packages": [
        {
            "name": "demo-crate",
            "version": "1.0.0",
            "license": "MIT",
            "license_file": null,
            "repository": "https://example.invalid/demo"
        }
    ]
}"#;

const CLEAN_CARGO_DENY_FIXTURE: &str = r#"
{"fields":{"advisories":{"errors":0,"helps":0,"notes":0,"warnings":0},"bans":{"errors":0,"helps":0,"notes":0,"warnings":0},"licenses":{"errors":0,"helps":0,"notes":0,"warnings":0},"sources":{"errors":0,"helps":0,"notes":0,"warnings":0}},"type":"summary"}
"#;

const BLOCKED_CARGO_DENY_FIXTURE: &str = r#"
{"fields":{"advisories":{"errors":1,"helps":0,"notes":0,"warnings":0},"bans":{"errors":0,"helps":0,"notes":0,"warnings":0},"licenses":{"errors":0,"helps":0,"notes":0,"warnings":0},"sources":{"errors":0,"helps":0,"notes":0,"warnings":0}},"type":"summary"}
"#;

#[test]
fn no_arguments_prints_usage_and_exits_2() {
    let output = bin().output().expect("run binary");
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: release-publish"));
}

#[test]
fn unknown_command_prints_usage_and_exits_2() {
    let output = bin()
        .arg("not-a-real-command")
        .output()
        .expect("run binary");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn checksum_bundle_and_verify_checksums_round_trip() {
    let dir = temp_dir("checksum");
    fs::write(dir.join("a.bin"), b"some release artifact bytes").unwrap();
    let out = dir.join("checksums.json");

    let status = bin()
        .args(["checksum-bundle"])
        .arg(&dir)
        .args(["--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let json = fs::read_to_string(&out).unwrap();
    assert!(json.contains("\"a.bin\""));
    assert!(json.contains("\"sha256\""));

    let status = bin()
        .arg("verify-checksums")
        .arg(&out)
        .arg(&dir)
        .status()
        .unwrap();
    assert!(status.success());

    // Tamper and confirm verify-checksums now fails.
    fs::write(dir.join("a.bin"), b"tampered bytes").unwrap();
    let output = bin()
        .arg("verify-checksums")
        .arg(&out)
        .arg(&dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
}

#[test]
fn checksum_bundle_requires_out_flag() {
    let dir = temp_dir("checksum-missing-out");
    let output = bin().arg("checksum-bundle").arg(&dir).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--out"));
}

#[test]
fn sbom_generates_a_valid_manifest_from_real_cargo_metadata_json() {
    let dir = temp_dir("sbom");
    let metadata_path = dir.join("cargo-metadata.json");
    fs::write(&metadata_path, CARGO_METADATA_FIXTURE).unwrap();
    let out = dir.join("sbom.json");

    let status = bin()
        .arg("sbom")
        .args(["--cargo-metadata"])
        .arg(&metadata_path)
        .args(["--out"])
        .arg(&out)
        .args(["--build-target", "x86_64-unknown-linux-gnu"])
        .status()
        .unwrap();
    assert!(status.success());

    let json = fs::read_to_string(&out).unwrap();
    assert!(json.contains("demo-crate"));
    assert!(json.contains("\"generated\""));
}

#[test]
fn provenance_collects_this_checkouts_real_state() {
    let dir = temp_dir("provenance");
    let out = dir.join("provenance.json");

    let status = bin()
        .arg("provenance")
        .args(["--repo"])
        .arg(repo_root())
        .args(["--out"])
        .arg(&out)
        .args(["--release-tag", "v0.0.0-cli-test"])
        .status()
        .unwrap();
    assert!(status.success());

    let json = fs::read_to_string(&out).unwrap();
    assert!(json.contains("v0.0.0-cli-test"));
    assert!(json.contains("source_commit"));
    assert!(json.contains("rustc_version"));
}

#[test]
fn release_gate_fails_closed_when_a_required_gate_is_missing() {
    let dir = temp_dir("release-gate-missing");
    let results = dir.join("results.json");
    fs::write(&results, r#"[{"gate":"Formatting","passed":true}]"#).unwrap();

    let output = bin()
        .arg("release-gate")
        .args(["--results"])
        .arg(&results)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("has no result"));
}

#[test]
fn release_gate_passes_when_every_required_gate_passed() {
    let dir = temp_dir("release-gate-pass");
    let results = dir.join("results.json");
    let gates = magnetar_roadmap_contracts::REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| format!(r#"{{"gate":"{gate:?}","passed":true}}"#))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&results, format!("[{gates}]")).unwrap();

    let output = bin()
        .arg("release-gate")
        .args(["--results"])
        .arg(&results)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("passed"));
}

#[test]
fn security_gate_passes_on_a_clean_cargo_deny_report() {
    let dir = temp_dir("security-gate-clean");
    let deny_json = dir.join("deny.jsonl");
    fs::write(&deny_json, CLEAN_CARGO_DENY_FIXTURE).unwrap();
    let out = dir.join("security-gate.json");

    let status = bin()
        .arg("security-gate")
        .args(["--cargo-deny-json"])
        .arg(&deny_json)
        .args(["--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let json = fs::read_to_string(&out).unwrap();
    assert!(json.contains("\"secrets_detected\": false"));
}

#[test]
fn security_gate_fails_closed_on_a_blocked_cargo_deny_report() {
    let dir = temp_dir("security-gate-blocked");
    let deny_json = dir.join("deny.jsonl");
    fs::write(&deny_json, BLOCKED_CARGO_DENY_FIXTURE).unwrap();
    let out = dir.join("security-gate.json");

    let output = bin()
        .arg("security-gate")
        .args(["--cargo-deny-json"])
        .arg(&deny_json)
        .args(["--out"])
        .arg(&out)
        .args(["--secrets-detected"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    // The report is still written before the blocking check runs, so a
    // human debugging a blocked release can see exactly what was evaluated.
    assert!(out.exists());
}

#[test]
fn oci_manifest_builds_and_verifies_a_local_layout() {
    let dir = temp_dir("oci-manifest");
    let input = dir.join("component.wasm");
    fs::write(&input, b"pretend wasm component bytes").unwrap();
    let out_dir = dir.join("layout");

    let output = bin()
        .arg("oci-manifest")
        .args(["--kind", "component"])
        .args(["--input"])
        .arg(&input)
        .args(["--title", "demo"])
        .args(["--tag", "v1"])
        .args(["--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("manifest digest: sha256:"));
    assert!(stdout.contains("layer digest: sha256:"));
    assert!(out_dir.join("oci-layout").is_file());
    assert!(out_dir.join("index.json").is_file());
}

#[test]
fn oci_manifest_rejects_an_unknown_kind() {
    let dir = temp_dir("oci-manifest-bad-kind");
    let input = dir.join("component.wasm");
    fs::write(&input, b"bytes").unwrap();

    let output = bin()
        .arg("oci-manifest")
        .args(["--kind", "not-a-real-kind"])
        .args(["--input"])
        .arg(&input)
        .args(["--title", "demo"])
        .args(["--tag", "v1"])
        .args(["--out-dir"])
        .arg(dir.join("layout"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown --kind"));
}
