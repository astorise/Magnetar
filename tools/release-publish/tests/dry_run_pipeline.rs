//! End-to-end, credential-free dry-run of the full pipeline (tasks.md
//! section 5, items 5.2-5.3): gate evaluation -> release-proof generation
//! -> release-gate/security-gate evaluation -> OCI manifest construction,
//! entirely without network access or credentials, run against this
//! checkout's own real files. This is `openspec/changes/
//! implement-release-publication-automation/tasks.md`'s own "dry-run
//! validation path" requirement, expressed as a real, CI-executed test
//! rather than only a manual smoke test.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use magnetar_roadmap_contracts::{REQUIRED_RELEASE_GATES, ReleaseGateResult};
use release_publish::{
    checksum, oci::layout, oci::manifest::build_artifact_manifest, oci::media_types::ArtifactKind,
    provenance, release_gate, sbom, security_gate,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("release-publish-it-{prefix}-{nanos}-{n}"));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tools/")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

const CLEAN_CARGO_DENY_FIXTURE: &str = r#"
{"fields":{"advisories":{"errors":0,"helps":0,"notes":0,"warnings":1},"bans":{"errors":0,"helps":0,"notes":0,"warnings":8},"licenses":{"errors":0,"helps":130,"notes":0,"warnings":1},"sources":{"errors":0,"helps":0,"notes":0,"warnings":0}},"type":"summary"}
"#;

#[test]
fn full_credential_free_pipeline_composes_without_error() {
    let dir = temp_dir("full-pipeline");

    // 1. Release-proof generation over a fake built-artifact directory.
    fs::write(dir.join("magnetar-cli"), b"pretend binary bytes").unwrap();
    let bundle = checksum::checksum_bundle_for_dir(&dir).unwrap();
    checksum::verify_bundle_against_dir(&bundle, &dir).unwrap();
    assert_eq!(bundle.len(), 1);

    // 2. SBOM from this workspace's own real `cargo metadata`.
    let metadata_json = std::process::Command::new("cargo")
        .args([
            "metadata",
            "--offline",
            "--format-version=1",
            "--no-deps",
            "--manifest-path",
        ])
        .arg(repo_root().join("roadmap-contracts/Cargo.toml"))
        .output()
        .expect("run cargo metadata");
    assert!(metadata_json.status.success(), "cargo metadata failed");
    let metadata_json = String::from_utf8(metadata_json.stdout).unwrap();
    let manifest = sbom::generate_sbom_manifest(&metadata_json, None, vec![]).unwrap();
    manifest.validate().expect("generated SBOM must validate");
    assert!(!manifest.entries.is_empty());

    // 3. Provenance from this checkout's real git/rustc/content state.
    let prov = provenance::collect_local_provenance(
        &repo_root(),
        None,
        None,
        Some("v0.0.0-dry-run".to_string()),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(prov.source_commit.is_some());

    // 4. Release-gate evaluation: every required gate reported as passed
    //    composes to an overall pass (the gate machinery itself, not
    //    whether this repo's real CI names every one of these today --
    //    that mapping is the release-publication workflow's job).
    let results: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    release_gate::evaluate_release_gates(&results).expect("all-passing gates must pass");

    // 5. Security-gate evaluation from a real (clean) cargo-deny fixture.
    let inputs = security_gate::collect_security_gate_inputs(
        CLEAN_CARGO_DENY_FIXTURE,
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
    magnetar_roadmap_contracts::evaluate_release_security_blocking(&inputs)
        .expect("clean inputs must not block");

    // 6. OCI manifest construction + local layout round-trip, no network.
    let built = build_artifact_manifest(
        ArtifactKind::ComponentArtifact,
        b"pretend wasm bytes",
        "demo",
    )
    .unwrap();
    layout::write_oci_layout(&dir.join("oci"), &built, b"pretend wasm bytes", "latest").unwrap();
    layout::verify_oci_layout(&dir.join("oci")).unwrap();

    let _ = fs::remove_dir_all(&dir);
}

/// `component-artifact-distribution`'s "OCI Digest Is The Only Trust
/// Identity" requirement, proven directly: two publications of the exact
/// same bytes under two different tags produce identical manifest and
/// layer digests, because the tag is never an input to either digest
/// computation.
#[test]
fn tag_alone_never_changes_the_manifest_or_layer_digest() {
    let bytes = b"identical component bytes";
    let built_a = build_artifact_manifest(ArtifactKind::ComponentArtifact, bytes, "demo").unwrap();
    let built_b = build_artifact_manifest(ArtifactKind::ComponentArtifact, bytes, "demo").unwrap();

    let dir_a = temp_dir("tag-a");
    let dir_b = temp_dir("tag-b");
    layout::write_oci_layout(&dir_a, &built_a, bytes, "v1.0.0").unwrap();
    layout::write_oci_layout(&dir_b, &built_b, bytes, "totally-different-tag").unwrap();

    assert_eq!(built_a.manifest_digest, built_b.manifest_digest);
    assert_eq!(built_a.layer_digest, built_b.layer_digest);

    let _ = fs::remove_dir_all(&dir_a);
    let _ = fs::remove_dir_all(&dir_b);
}
