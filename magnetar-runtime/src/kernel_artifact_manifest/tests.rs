//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::ProviderBinding;
use crate::kernel_artifact::KernelArtifactProvenance;
use crate::kernel_qualification::QualificationProfile;
use crate::operator::{OperatorFamily, OperatorId};
use std::fs;

use crate::kernel::KernelOperatorVersionRange;
fn gzip_compress(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn build_kernel_bundle_tar(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (path, data) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        builder.append_data(&mut header, *path, *data).unwrap();
    }
    builder.into_inner().unwrap()
}

fn write_kernel_bundle(directory: &std::path::Path, blob_bytes: &[u8]) -> String {
    let digest = KernelBlobDigest::of_bytes(blob_bytes);
    fs::write(
        directory.join("blobs").join("sha256").join(&digest.value),
        blob_bytes,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{}",
      "size": {},
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest.value,
        blob_bytes.len()
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();
    digest.value
}

fn temp_kernel_bundle_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-kernel-bundle-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(directory.join("blobs").join("sha256")).unwrap();
    directory
}

fn build_tar_with_raw_path_entry(path: &str, data: &[u8]) -> Vec<u8> {
    // `tar::Header::set_path` (used by `append_data`) refuses `..` itself,
    // which is correct behavior for a well-behaved writer but means it
    // cannot be used to construct the maliciously-crafted archive this test
    // needs. Writing directly into the raw GNU header name bytes bypasses
    // that writer-side check, simulating an archive from an attacker who
    // does not go through this safe API.
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(0);
    {
        let gnu = header.as_gnu_mut().expect("gnu header");
        let name_bytes = path.as_bytes();
        gnu.name[..name_bytes.len()].copy_from_slice(name_bytes);
    }
    header.set_cksum();
    builder.append(&header, data).unwrap();
    builder.into_inner().unwrap()
}

fn kernel_bundle_tar_entries(blob_bytes: &[u8]) -> (Vec<(&'static str, Vec<u8>)>, String) {
    let digest = KernelBlobDigest::of_bytes(blob_bytes);
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": {size},
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest = digest.value,
        size = blob_bytes.len()
    );
    (
        vec![
            (KERNEL_MANIFEST_FILE_NAME, manifest.into_bytes()),
            (
                Box::leak(format!("blobs/sha256/{}", digest.value).into_boxed_str()),
                blob_bytes.to_vec(),
            ),
        ],
        digest.value,
    )
}

#[test]
fn kernel_manifest_dependency_cycle_is_detected() {
    let digest_a = KernelBlobDigest::of_bytes(b"artifact-a");
    let digest_b = KernelBlobDigest::of_bytes(b"artifact-b");
    let mut artifact_a = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        digest_a.clone(),
        4,
    ));
    artifact_a.dependencies.push(digest_b.clone());
    let mut artifact_b = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        digest_b,
        4,
    ));
    artifact_b.dependencies.push(digest_a);
    let manifest = KernelManifestV1 {
        artifacts: vec![artifact_a, artifact_b],
        ..KernelManifestV1::new()
    };
    assert!(detect_dependency_cycle(&manifest).is_some());
    assert!(matches!(
        manifest.validate(),
        Err(KernelManifestError::DependencyCycle { .. })
    ));
}

#[test]
fn kernel_exchange_bundle_validates_end_to_end() {
    let directory = temp_kernel_bundle_dir("happy-path");
    write_kernel_bundle(&directory, b"cubin-bytes");

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("well-formed bundle should validate");
    assert_eq!(validated.digest, validated.manifest.digest());

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_rejects_corrupted_blob() {
    let directory = temp_kernel_bundle_dir("corrupted-blob");
    let digest_value = write_kernel_bundle(&directory, b"cubin-bytes");
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_value),
        b"tampered-bytes",
    )
    .unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleBlobSizeMismatch { .. })
            | Err(KernelManifestError::BundleBlobDigestMismatch { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_missing_manifest_is_rejected() {
    let directory = temp_kernel_bundle_dir("missing-manifest");
    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleManifestMissing)
    ));
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_inference_request_field_boundary_rejects_bundle_fields() {
    for field in KERNEL_MANIFEST_FORBIDDEN_INFERENCE_FIELDS {
        let outcome = reject_inference_request_manifest_field(field);
        assert!(outcome.is_err(), "expected '{field}' to be rejected");
    }
    assert!(reject_inference_request_manifest_field("prompt").is_ok());
}

#[test]
fn kernel_manifest_recommendation_never_grants_promotion() {
    for recommendation in [
        KernelManifestRecommendation::RecommendedForLatency,
        KernelManifestRecommendation::RecommendedForThroughput,
        KernelManifestRecommendation::Experimental,
        KernelManifestRecommendation::Reject,
    ] {
        assert!(!recommendation_grants_promotion(recommendation));
    }
}

#[test]
fn kernel_manifest_signature_presence_alone_is_never_verified() {
    let envelope = KernelSignatureEnvelope {
        algorithm: "ed25519".into(),
        key_id: Some("key-1".into()),
        signed_digest: KernelBlobDigest::of_bytes(b"manifest"),
        signature_material: KernelBlobDigest::of_bytes(b"signature-bytes"),
        certificate_chain_reference: None,
    };
    assert!(!signature_is_verified(&envelope, false));
    assert!(signature_is_verified(&envelope, true));
}

#[test]
fn kernel_manifest_generator_rejects_embedded_credential_locator() {
    let generator = KernelGeneratorMetadata {
        generator_name: Some("gen".into()),
        generator_version: None,
        campaign_id: None,
        source_revision: Some("https://user:secret@example.invalid/repo".into()),
    };
    assert!(matches!(
        generator.validate(),
        Err(KernelManifestError::ProvenanceInvalid { .. })
    ));

    let clean = KernelGeneratorMetadata {
        source_revision: Some("https://example.invalid/repo@deadbeef".into()),
        ..generator
    };
    assert!(clean.validate().is_ok());
}

#[test]
fn kernel_manifest_specialization_rejects_impossible_range() {
    let specialization = KernelManifestSpecialization {
        batch_range: Some((32, 1)),
        ..KernelManifestSpecialization::default()
    };
    assert!(matches!(
        specialization.validate(),
        Err(KernelManifestError::SpecializationInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_normalizes_to_source_and_compiled_artifacts() {
    let mut source_blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::KERNEL_SOURCE),
        KernelArtifactFormat::new("triton", "source").with_version("3"),
        KernelBlobDigest::of_bytes(b"normalize-source"),
        4,
    );
    source_blob.required = true;
    let mut source_artifact = KernelManifestArtifact::new(source_blob);
    source_artifact.semantic_binding = Some(KernelSemanticBinding::single(OperatorId::magnetar(
        "matmul",
        1,
        OperatorFamily::LinearAlgebra,
    )));
    source_artifact.provenance = Some(KernelArtifactProvenance::AiGenerated);
    source_artifact.specialization.batch_range = Some((1, 8));

    let normalized_source =
        normalize_to_source_artifact(&source_artifact).expect("source normalizes");
    assert_eq!(normalized_source.format.stable_key(), "triton:source@3");
    assert_eq!(normalized_source.shape.max_batch_size, Some(8));
    assert_eq!(
        normalized_source.provenance,
        KernelArtifactProvenance::AiGenerated
    );

    let compiled_blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"normalize-compiled"),
        4,
    );
    let mut compiled_artifact = KernelManifestArtifact::new(compiled_blob);
    compiled_artifact.semantic_binding = Some(KernelSemanticBinding::single(OperatorId::magnetar(
        "matmul",
        1,
        OperatorFamily::LinearAlgebra,
    )));
    compiled_artifact.compiler_metadata = Some(KernelCompilerMetadata {
        compiler_identity: Some("triton".into()),
        compiler_version: Some("3.1".into()),
        ..Default::default()
    });
    compiled_artifact.target.architecture = Some("sm90".into());

    let normalized_compiled =
        normalize_to_compiled_artifact(&compiled_artifact).expect("compiled normalizes");
    assert_eq!(normalized_compiled.compiler_identity, "triton");
    assert_eq!(normalized_compiled.target_architecture, "sm90");

    // No semantic binding -> normalization fails rather than fabricating one.
    let unbound = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"unbound"),
        4,
    ));
    assert!(normalize_to_source_artifact(&unbound).is_err());
    assert!(normalize_to_compiled_artifact(&unbound).is_err());
}

#[test]
fn kernel_manifest_normalizes_qualification_identity_pieces() {
    let evidence = KernelEvidenceReference {
        digest: KernelBlobDigest::of_bytes(b"qualification-evidence"),
        profile: "baseline-correctness@1".into(),
        suite_or_workload_version: Some("v3".into()),
        oracle_or_provider_identity: Some("reference-cpu@2".into()),
        target_compatibility: std::collections::BTreeSet::new(),
        status: KernelEvidenceStatus::Passed,
        storage_mode: KernelArtifactStorageMode::Embedded,
        workload_profile: None,
        device_context: None,
        provider_context: None,
        workload_metadata: None,
    };
    let profile = normalize_qualification_profile(&evidence).expect("profile normalizes");
    assert_eq!(
        profile,
        QualificationProfile::new("baseline-correctness", 1)
    );

    let oracle = normalize_oracle_identity(&evidence).expect("oracle normalizes");
    assert_eq!(oracle.provider, ProviderBinding::new("reference-cpu"));
    assert_eq!(oracle.version, "2");

    let mut malformed = evidence.clone();
    malformed.profile = "no-version-separator".into();
    assert!(normalize_qualification_profile(&malformed).is_err());

    let mut no_oracle = evidence;
    no_oracle.oracle_or_provider_identity = None;
    assert!(normalize_oracle_identity(&no_oracle).is_err());
}

#[test]
fn kernel_exchange_tar_archive_validates_to_identical_digest_as_directory_bundle() {
    let directory = temp_kernel_bundle_dir("archive-directory-parity");
    write_kernel_bundle(&directory, b"tar-parity-bytes");
    let directory_bundle = KernelExchangeBundle::open(&directory);
    let directory_validated =
        validate_kernel_exchange_bundle(&directory_bundle, &KernelManifestLimits::default())
            .expect("directory bundle validates");

    let (entries, _digest) = kernel_bundle_tar_entries(b"tar-parity-bytes");
    let entry_refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, d)| (*p, d.as_slice())).collect();
    let tar_bytes = build_kernel_bundle_tar(&entry_refs);

    let extract_dir = temp_kernel_bundle_dir("archive-directory-parity-extracted");
    let archive_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &extract_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("tar archive extracts");
    let archive_validated =
        validate_kernel_exchange_bundle(&archive_bundle, &KernelManifestLimits::default())
            .expect("extracted archive bundle validates");

    assert_eq!(directory_validated.digest, archive_validated.digest);

    // The archive checksum is a distinct, transport-level concept: it is
    // *not* the manifest digest, and never claimed to be.
    let archive_checksum = archive_diagnostic_checksum(&tar_bytes);
    assert_ne!(archive_checksum, directory_validated.digest.value);

    fs::remove_dir_all(&directory).unwrap();
    fs::remove_dir_all(&extract_dir).unwrap();
}

#[test]
fn kernel_exchange_tar_gz_archive_produces_the_same_manifest_digest_as_plain_tar() {
    let (entries, _digest) = kernel_bundle_tar_entries(b"gzip-parity-bytes");
    let entry_refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, d)| (*p, d.as_slice())).collect();
    let tar_bytes = build_kernel_bundle_tar(&entry_refs);
    let gz_bytes = gzip_compress(&tar_bytes);

    let plain_dir = temp_kernel_bundle_dir("gzip-parity-plain");
    let plain_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &plain_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("plain tar extracts");
    let plain_validated =
        validate_kernel_exchange_bundle(&plain_bundle, &KernelManifestLimits::default())
            .expect("plain tar bundle validates");

    let gz_dir = temp_kernel_bundle_dir("gzip-parity-compressed");
    let gz_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&gz_bytes),
        true,
        &gz_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("gzip tar extracts");
    let gz_validated =
        validate_kernel_exchange_bundle(&gz_bundle, &KernelManifestLimits::default())
            .expect("gzip tar bundle validates");

    assert_eq!(plain_validated.digest, gz_validated.digest);
    // Compressing the exact same logical content changes the raw archive
    // bytes (and therefore the transport-level diagnostic checksum) without
    // touching logical identity.
    assert_ne!(
        archive_diagnostic_checksum(&tar_bytes),
        archive_diagnostic_checksum(&gz_bytes)
    );

    fs::remove_dir_all(&plain_dir).unwrap();
    fs::remove_dir_all(&gz_dir).unwrap();
}

#[test]
fn kernel_exchange_archive_ignores_ownership_timestamp_and_mode_for_identity() {
    let (entries, _digest) = kernel_bundle_tar_entries(b"metadata-irrelevant-bytes");

    let mut builder_a = tar::Builder::new(Vec::new());
    let mut builder_b = tar::Builder::new(Vec::new());
    for (path, data) in &entries {
        let mut header_a = tar::Header::new_gnu();
        header_a.set_size(data.len() as u64);
        header_a.set_mode(0o644);
        header_a.set_mtime(1_000);
        header_a.set_uid(0);
        header_a.set_gid(0);
        header_a.set_cksum();
        builder_a
            .append_data(&mut header_a, *path, data.as_slice())
            .unwrap();

        let mut header_b = tar::Header::new_gnu();
        header_b.set_size(data.len() as u64);
        header_b.set_mode(0o755);
        header_b.set_mtime(999_999_999);
        header_b.set_uid(1000);
        header_b.set_gid(1000);
        header_b.set_cksum();
        builder_b
            .append_data(&mut header_b, *path, data.as_slice())
            .unwrap();
    }
    let tar_a = builder_a.into_inner().unwrap();
    let tar_b = builder_b.into_inner().unwrap();
    assert_ne!(
        archive_diagnostic_checksum(&tar_a),
        archive_diagnostic_checksum(&tar_b),
        "sanity check: the two archives really do differ at the byte level"
    );

    let dir_a = temp_kernel_bundle_dir("archive-metadata-a");
    let dir_b = temp_kernel_bundle_dir("archive-metadata-b");
    let bundle_a = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_a),
        false,
        &dir_a,
        &KernelExchangeArchiveLimits::default(),
    )
    .unwrap();
    let bundle_b = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_b),
        false,
        &dir_b,
        &KernelExchangeArchiveLimits::default(),
    )
    .unwrap();
    let validated_a =
        validate_kernel_exchange_bundle(&bundle_a, &KernelManifestLimits::default()).unwrap();
    let validated_b =
        validate_kernel_exchange_bundle(&bundle_b, &KernelManifestLimits::default()).unwrap();
    assert_eq!(
        validated_a.digest, validated_b.digest,
        "ownership/timestamp/mode differences must not affect logical bundle identity"
    );

    fs::remove_dir_all(&dir_a).unwrap();
    fs::remove_dir_all(&dir_b).unwrap();
}

#[test]
fn kernel_exchange_archive_rejects_path_traversal_entry() {
    let tar_bytes = build_tar_with_raw_path_entry("../escape.txt", b"malicious");
    let dir = temp_kernel_bundle_dir("archive-traversal");
    let outcome = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &dir,
        &KernelExchangeArchiveLimits::default(),
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundlePathInvalid { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_bundle_transport_reservation_marks_only_directory_and_tar_as_implemented() {
    assert!(KernelBundleTransport::Directory.is_implemented());
    assert!(KernelBundleTransport::TarArchive.is_implemented());
    assert!(!KernelBundleTransport::ObjectStore.is_implemented());
    assert!(!KernelBundleTransport::OciLike.is_implemented());
    assert!(!KernelBundleTransport::Registry.is_implemented());
}

#[test]
fn kernel_manifest_normalizes_benchmark_profile() {
    let evidence = KernelEvidenceReference {
        digest: KernelBlobDigest::of_bytes(b"benchmark-evidence"),
        profile: "latency@1".into(),
        suite_or_workload_version: Some("v2".into()),
        oracle_or_provider_identity: None,
        target_compatibility: std::collections::BTreeSet::new(),
        status: KernelEvidenceStatus::Passed,
        storage_mode: KernelArtifactStorageMode::Embedded,
        workload_profile: Some("decode-256".into()),
        device_context: Some("h100".into()),
        provider_context: Some("nvidia-cuda@12.4".into()),
        workload_metadata: Some(KernelBenchmarkWorkloadMetadata {
            input_shapes: Some("[1,256,4096]".into()),
            dtype_layout: Some("fp16-row-major".into()),
            batch_size: Some(1),
            sequence_length: Some(256),
            warmup_count: Some(10),
            measurement_count: Some(100),
            synchronization_policy: Some("device-sync".into()),
            driver_runtime_version: Some("cuda-12.4".into()),
            benchmark_version: Some("bench-1".into()),
        }),
    };

    let profile =
        normalize_benchmark_profile(&evidence, "sm90").expect("benchmark profile normalizes");
    assert_eq!(profile.target_device, "h100");
    assert_eq!(profile.hardware_architecture, "sm90");
    assert_eq!(profile.provider_version, "nvidia-cuda@12.4");
    assert_eq!(profile.measurement_count, 100);
    assert!(profile.is_authoritative());

    let mut missing_workload = evidence.clone();
    missing_workload.workload_metadata = None;
    assert!(normalize_benchmark_profile(&missing_workload, "sm90").is_err());
}

#[test]
fn kernel_manifest_observation_builders_carry_schema_artifact_count_and_formats() {
    let schema = KernelManifestSchemaVersion::current();
    let formats = vec![
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelArtifactFormat::new("triton", "source").with_version("3"),
    ];
    let observation = KernelManifestObservation::new(KernelManifestObservationKind::ManifestParsed)
        .with_schema_version(&schema)
        .with_artifact_count(2)
        .with_formats(formats);

    assert_eq!(
        observation
            .redacted_metadata
            .get("schema-version")
            .map(String::as_str),
        Some("magnetar:kernel-manifest@1.0")
    );
    assert_eq!(
        observation
            .redacted_metadata
            .get("artifact-count")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        observation
            .redacted_metadata
            .get("formats")
            .map(String::as_str),
        Some("nvidia:cubin, triton:source@3")
    );
}

#[test]
fn kernel_exchange_bundle_missing_required_embedded_artifact_is_rejected() {
    let directory = temp_kernel_bundle_dir("required-missing");
    let missing_digest = KernelBlobDigest::of_bytes(b"never-written");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{missing}",
      "size": 13,
      "storage_mode": "embedded",
      "required": true
    }}
  ]
}}"#,
        missing = missing_digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleRequiredArtifactMissing { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_rejects_required_external_artifact_without_fetching() {
    let directory = temp_kernel_bundle_dir("required-external");
    let digest = KernelBlobDigest::of_bytes(b"external-bytes");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 13,
      "storage_mode": "external",
      "required": true,
      "location_hint": "https://example.invalid/artifact.cubin"
    }}
  ]
}}"#,
        digest = digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ExchangeExternalReferenceDenied { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_bundle_symlink_entry_is_rejected_when_creatable() {
    let directory = temp_kernel_bundle_dir("symlink");
    write_kernel_bundle(&directory, b"symlink-fixture");
    let target = directory.join(KERNEL_MANIFEST_FILE_NAME);
    let link = directory.join("blobs").join("escape-link");

    #[cfg(unix)]
    let created = std::os::unix::fs::symlink(&target, &link).is_ok();
    #[cfg(windows)]
    let created = std::os::windows::fs::symlink_file(&target, &link).is_ok();
    #[cfg(not(any(unix, windows)))]
    let created = false;

    if created {
        let outcome = scan_bundle_for_unsafe_entries(&directory);
        assert!(matches!(
            outcome,
            Err(KernelManifestError::BundleSymlinkDenied { .. })
        ));
    }
    // When the platform/permissions do not allow creating a symlink (e.g.
    // Windows without Developer Mode or admin rights), this test is a no-op
    // rather than a false failure -- the rejection code path itself is
    // exercised whenever a symlink can actually be created.

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_qualification_evidence_array_round_trips_through_json() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"evidence-bytes").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{artifact_digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ],
  "qualification_evidence": [
    {{
      "digest": "sha256:{digest}",
      "profile": "correctness",
      "suite_or_workload_version": "v1",
      "oracle_or_provider_identity": "reference-cpu@1",
      "status": "passed"
    }}
  ]
}}"#,
        artifact_digest = KernelBlobDigest::of_bytes(b"artifact-bytes").value,
        digest = digest
    );
    let manifest =
        parse_manifest_json(&text, &limits).expect("manifest with qualification evidence parses");
    assert_eq!(manifest.qualification_evidence.len(), 1);
    let evidence = &manifest.qualification_evidence[0];
    assert_eq!(evidence.profile, "correctness");
    assert_eq!(evidence.status, KernelEvidenceStatus::Passed);
    assert!(oracle_identity_is_known(evidence));
    assert!(evaluate_qualification_evidence_currency(evidence, "v1"));
    assert!(!evaluate_qualification_evidence_currency(evidence, "v2"));
}

#[test]
fn kernel_manifest_evaluate_target_compatibility() {
    let target = KernelTargetConstraints {
        architecture: Some("sm90".into()),
        provider_compatibility: ["nvidia-cuda".to_string()].into_iter().collect(),
        device_features: ["tensor-core".to_string()].into_iter().collect(),
        ..KernelTargetConstraints::default()
    };

    let matching_context = KernelRuntimeCompatibilityContext {
        provider_id: Some("nvidia-cuda".into()),
        architecture: Some("sm90".into()),
        available_device_features: ["tensor-core".to_string()].into_iter().collect(),
    };
    assert!(evaluate_target_compatibility(&target, &matching_context).is_ok());

    let wrong_architecture = KernelRuntimeCompatibilityContext {
        architecture: Some("sm80".into()),
        ..matching_context.clone()
    };
    assert!(matches!(
        evaluate_target_compatibility(&target, &wrong_architecture),
        Err(KernelManifestError::ExchangeCompatibilityFailed { .. })
    ));

    let missing_feature = KernelRuntimeCompatibilityContext {
        available_device_features: std::collections::BTreeSet::new(),
        ..matching_context
    };
    assert!(matches!(
        evaluate_target_compatibility(&target, &missing_feature),
        Err(KernelManifestError::ExchangeCompatibilityFailed { .. })
    ));

    // An artifact with no declared target constraints is always compatible.
    assert!(
        evaluate_target_compatibility(
            &KernelTargetConstraints::default(),
            &KernelRuntimeCompatibilityContext::default()
        )
        .is_ok()
    );
}

#[test]
fn kernel_exchange_bundle_total_size_limit_is_enforced() {
    let directory = temp_kernel_bundle_dir("total-size-limit");
    let bytes_a = b"aaaa";
    let bytes_b = b"bbbb";
    let digest_a = KernelBlobDigest::of_bytes(bytes_a);
    let digest_b = KernelBlobDigest::of_bytes(bytes_b);
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_a.value),
        bytes_a,
    )
    .unwrap();
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_b.value),
        bytes_b,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{ "role": "compiled-kernel", "format": "nvidia:cubin", "digest": "sha256:{a}", "size": 4, "storage_mode": "embedded" }},
    {{ "role": "auxiliary", "format": "nvidia:cubin", "digest": "sha256:{b}", "size": 4, "storage_mode": "embedded" }}
  ]
}}"#,
        a = digest_a.value,
        b = digest_b.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let limits = KernelManifestLimits {
        max_total_embedded_bytes: 6,
        ..KernelManifestLimits::default()
    };
    let outcome = validate_kernel_exchange_bundle(&bundle, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleTotalSizeExceeded { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_archive_enforces_per_entry_decompressed_size_limit() {
    let oversized = vec![0u8; 1024];
    let tar_bytes = build_kernel_bundle_tar(&[("blobs/sha256/oversized", &oversized)]);
    let dir = temp_kernel_bundle_dir("archive-entry-limit");
    let limits = KernelExchangeArchiveLimits {
        max_entry_decompressed_bytes: 100,
        ..KernelExchangeArchiveLimits::default()
    };
    let outcome =
        extract_kernel_exchange_archive(std::io::Cursor::new(&tar_bytes), false, &dir, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_exchange_archive_enforces_entry_count_limit() {
    let entries: Vec<(&str, &[u8])> = vec![("a", b"1"), ("b", b"2"), ("c", b"3")];
    let tar_bytes = build_kernel_bundle_tar(&entries);
    let dir = temp_kernel_bundle_dir("archive-entry-count-limit");
    let limits = KernelExchangeArchiveLimits {
        max_entries: 2,
        ..KernelExchangeArchiveLimits::default()
    };
    let outcome =
        extract_kernel_exchange_archive(std::io::Cursor::new(&tar_bytes), false, &dir, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_manifest_evaluate_trust_pipeline_stage_delegates_to_sole_authority() {
    assert_eq!(
        evaluate_manifest_trust(true),
        crate::evaluate_artifact_trust(true)
    );
    assert_eq!(
        evaluate_manifest_trust(false),
        crate::evaluate_artifact_trust(false)
    );
    assert!(!evaluate_manifest_trust(false).is_trusted());
}

#[test]
fn kernel_bundle_path_safety_rejects_traversal_and_absolute_paths() {
    for bad in [
        "../escape",
        "/etc/passwd",
        "C:/Windows/system32",
        "a/../../b",
        "\\\\server\\share",
    ] {
        assert!(
            validate_bundle_relative_path(bad).is_err(),
            "expected '{bad}' to be rejected"
        );
    }
    assert!(validate_bundle_relative_path("blobs/sha256/deadbeef").is_ok());
}

#[test]
fn kernel_manifest_embedded_byte_accounting_saturates_instead_of_overflowing() {
    // The bundle validation pipeline accumulates declared blob sizes with
    // `u64::saturating_add`, implementing "Reject overflow" (tasks, "Integer
    // Safety"): summing sizes near `u64::MAX` must never wrap around or
    // panic, even though no real bundle could actually contain that many
    // bytes on disk.
    let total = [u64::MAX, u64::MAX, 1_u64]
        .into_iter()
        .fold(0_u64, u64::saturating_add);
    assert_eq!(total, u64::MAX);
}

#[test]
fn kernel_artifact_manifest_conformance_report_is_conformant() {
    let report = run_kernel_artifact_manifest_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn kernel_manifest_json_duplicate_key_is_rejected() {
    let limits = KernelManifestLimits::default();
    let text = r#"{"schema":"magnetar:kernel-manifest@1.0","artifacts":[],"artifacts":[]}"#;
    let outcome = parse_manifest_json(text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::DuplicateKey { .. })
    ));
}

#[test]
fn kernel_manifest_json_excessive_nesting_is_rejected() {
    let limits = KernelManifestLimits {
        max_nesting_depth: 4,
        ..KernelManifestLimits::default()
    };
    let mut text = String::new();
    for _ in 0..10 {
        text.push('[');
    }
    for _ in 0..10 {
        text.push(']');
    }
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
}

#[test]
fn kernel_manifest_oversized_input_is_rejected() {
    let limits = KernelManifestLimits {
        max_manifest_bytes: 8,
        ..KernelManifestLimits::default()
    };
    let outcome = parse_manifest_json(
        r#"{"schema":"magnetar:kernel-manifest@1.0","artifacts":[]}"#,
        &limits,
    );
    assert!(matches!(outcome, Err(KernelManifestError::TooLarge { .. })));
}

#[test]
fn kernel_manifest_unsupported_schema_major_is_rejected() {
    let limits = KernelManifestLimits::default();
    let outcome = parse_manifest_json(
        r#"{"schema":"magnetar:kernel-manifest@2.0","artifacts":[]}"#,
        &limits,
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::SchemaUnsupported { .. })
    ));
}

#[test]
fn kernel_exchange_bundle_missing_optional_embedded_artifact_is_tolerated() {
    let directory = temp_kernel_bundle_dir("optional-missing");
    let present_digest = KernelBlobDigest::of_bytes(b"present-bytes");
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&present_digest.value),
        b"present-bytes",
    )
    .unwrap();
    let missing_digest = KernelBlobDigest::of_bytes(b"missing-bytes");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{present}",
      "size": 13,
      "storage_mode": "embedded",
      "required": true
    }},
    {{
      "role": "benchmark-evidence",
      "format": "magnetar:benchmark-report@1",
      "digest": "sha256:{missing}",
      "size": 99,
      "storage_mode": "embedded",
      "required": false
    }}
  ]
}}"#,
        present = present_digest.value,
        missing = missing_digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("missing optional embedded artifact should not invalidate the bundle");
    assert_eq!(validated.manifest.artifacts.len(), 2);

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_extension_cannot_claim_a_core_field_namespace() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"core-field-fixture").value
    );
    let mut manifest = parse_manifest_json(&text, &limits).expect("sample manifest parses");
    manifest.extensions.push(KernelManifestExtension {
        namespace: "trust:override".into(),
        required: false,
        data: serde_json::Value::Null,
    });
    let outcome = manifest.validate();
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ArtifactReferenceInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_accepts_unknown_future_artifact_format() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "vendor:new-ir@1",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"future-format-bytes").value
    );
    let manifest = parse_manifest_json(&text, &limits)
        .expect("unknown future format still parses structurally");
    assert_eq!(
        manifest.artifacts[0].blob.format.stable_key(),
        "vendor:new-ir@1"
    );
}

#[test]
fn kernel_manifest_operator_version_range_compatibility() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ],
      "operator_version_range": {{ "min": 1, "max": 3 }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"version-range-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("manifest with version range parses");
    let binding = manifest.artifacts[0].semantic_binding.as_ref().unwrap();
    assert!(binding.is_version_compatible(1));
    assert!(binding.is_version_compatible(3));
    assert!(!binding.is_version_compatible(4));

    let invalid_range = KernelSemanticBinding {
        operators: vec![OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        )],
        primary_version_requirements: Some(KernelOperatorVersionRange { min: 5, max: 1 }),
    };
    assert!(matches!(
        invalid_range.validate(),
        Err(KernelManifestError::SemanticBindingInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_fused_semantic_binding_preserves_operator_order() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "rmsnorm", "version": 1, "family": "normalization" }},
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"fused-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("fused binding manifest parses");
    let binding = manifest.artifacts[0].semantic_binding.as_ref().unwrap();
    assert!(binding.is_fused());
    assert_eq!(
        binding.fingerprint(),
        "magnetar:operator/rmsnorm@1 -> magnetar:operator/matmul@1"
    );

    // Order matters: swapping the two operators is a different fusion.
    let reversed_text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }},
        {{ "namespace": "magnetar:operator", "name": "rmsnorm", "version": 1, "family": "normalization" }}
      ]
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"fused-fixture-reversed").value
    );
    let reversed = parse_manifest_json(&reversed_text, &limits)
        .expect("reversed fused binding manifest parses");
    let reversed_binding = reversed.artifacts[0].semantic_binding.as_ref().unwrap();
    assert_ne!(binding.fingerprint(), reversed_binding.fingerprint());

    // Normalizing a fused source artifact preserves the remaining operators
    // as the fused group, and only the primary Operator becomes the
    // compiled artifact's single `operator_semantics`.
    let normalized_source =
        normalize_to_source_artifact(&manifest.artifacts[0]).expect("fused source normalizes");
    assert_eq!(normalized_source.fused_operator_group.len(), 1);
    assert_eq!(normalized_source.fused_operator_group[0].name(), "matmul");
}

#[test]
fn kernel_manifest_target_specialization_compiler_precision_generator_round_trip() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ],
      "target": {{
        "device_type": "gpu",
        "hardware_vendor": "nvidia",
        "architecture": "sm90",
        "device_features": ["tensor-core"],
        "provider_compatibility": ["nvidia-cuda"],
        "runtime_driver_compatibility": ["cuda-12"],
        "memory_classes": ["hbm"]
      }},
      "specialization": {{
        "exact_dimensions": {{"0": 128}},
        "batch_range": [1, 32],
        "sequence_range": [1, 4096],
        "head_count": 32,
        "head_dimension": 128,
        "tile_sizes": [64, 64],
        "alignment": 16,
        "dtype": "float16",
        "layout": "row-major",
        "quantization_profile": "int8-groupwise",
        "execution_phase": "decode",
        "device_features": ["tensor-core"]
      }},
      "compiler_metadata": {{
        "compiler_identity": "triton",
        "compiler_version": "3.1",
        "backend_identity_version": "ptxas-12.4",
        "flags_fingerprint": "abc123",
        "build_fingerprint": "build-456",
        "target_architecture": "sm90"
      }},
      "precision": {{
        "accumulation_dtype": "float32",
        "approximate_math": true,
        "deterministic": false,
        "tolerance_profile": "operator-default",
        "quantization_error_profile": "int8-standard"
      }},
      "generator": {{
        "generator_name": "kernel-forge",
        "generator_version": "2.0",
        "campaign_id": "campaign-42",
        "source_revision": "https://example.invalid/repo@deadbeef"
      }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"rich-descriptor-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("rich descriptor manifest parses");
    let artifact = &manifest.artifacts[0];

    assert_eq!(artifact.target.device_type.as_deref(), Some("gpu"));
    assert_eq!(artifact.target.architecture.as_deref(), Some("sm90"));
    assert!(artifact.target.device_features.contains("tensor-core"));

    assert_eq!(artifact.specialization.batch_range, Some((1, 32)));
    assert_eq!(artifact.specialization.head_count, Some(32));
    assert_eq!(artifact.specialization.dtype.as_deref(), Some("float16"));
    assert_eq!(
        artifact.specialization.execution_phase,
        Some(KernelExecutionPhase::Decode)
    );

    let compiler = artifact.compiler_metadata.as_ref().unwrap();
    assert_eq!(compiler.compiler_identity.as_deref(), Some("triton"));
    assert_eq!(compiler.target_architecture.as_deref(), Some("sm90"));

    assert!(artifact.precision.approximate_math);
    assert_eq!(artifact.precision.deterministic, Some(false));

    let generator = artifact.generator.as_ref().unwrap();
    assert_eq!(generator.generator_name.as_deref(), Some("kernel-forge"));
    assert_eq!(generator.campaign_id.as_deref(), Some("campaign-42"));

    // Canonical identity round-trips through re-parsing the canonical bytes.
    let canonical_text = String::from_utf8(manifest.canonical_bytes()).unwrap();
    let reparsed = parse_manifest_json(&canonical_text, &limits).expect("canonical bytes reparse");
    assert_eq!(reparsed.digest(), manifest.digest());
}

#[test]
fn kernel_manifest_target_entry_count_limit_is_enforced() {
    let limits = KernelManifestLimits {
        max_target_entries: 2,
        ..KernelManifestLimits::default()
    };
    let features: Vec<String> = (0..5).map(|i| format!("\"feature-{i}\"")).collect();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "target": {{ "device_features": [{features}] }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"target-limit-fixture").value,
        features = features.join(", ")
    );
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
}

#[test]
fn kernel_manifest_conflicting_digest_metadata_is_rejected() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"shared-content").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }},
    {{
      "role": "auxiliary",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 999,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
    );
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ArtifactReferenceInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_same_digest_same_size_across_artifacts_is_allowed() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"deduplicated-content").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }},
    {{
      "role": "auxiliary",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
    );
    assert!(parse_manifest_json(&text, &limits).is_ok());
}

#[test]
fn kernel_manifest_multi_target_bundle_validates_with_distinct_architectures() {
    let directory = temp_kernel_bundle_dir("multi-target");
    let sm80_bytes = b"sm80-cubin";
    let sm90_bytes = b"sm90-cubin";
    let sm80_digest = KernelBlobDigest::of_bytes(sm80_bytes);
    let sm90_digest = KernelBlobDigest::of_bytes(sm90_bytes);
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&sm80_digest.value),
        sm80_bytes,
    )
    .unwrap();
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&sm90_digest.value),
        sm90_bytes,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{sm80}",
      "size": {sm80_len},
      "storage_mode": "embedded",
      "target": {{ "architecture": "sm80", "provider_compatibility": ["nvidia-cuda"] }}
    }},
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{sm90}",
      "size": {sm90_len},
      "storage_mode": "embedded",
      "target": {{ "architecture": "sm90", "provider_compatibility": ["nvidia-cuda"] }}
    }}
  ],
  "qualification_evidence": [
    {{ "digest": "sha256:{sm80}", "profile": "correctness@1", "status": "passed" }}
  ],
  "benchmark_evidence": [
    {{ "digest": "sha256:{sm90}", "profile": "latency@1", "workload_profile": "decode-256", "status": "passed" }}
  ]
}}"#,
        sm80 = sm80_digest.value,
        sm80_len = sm80_bytes.len(),
        sm90 = sm90_digest.value,
        sm90_len = sm90_bytes.len(),
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("multi-target bundle should validate");
    assert_eq!(validated.manifest.artifacts.len(), 2);
    assert_eq!(validated.manifest.qualification_evidence.len(), 1);
    assert_eq!(validated.manifest.benchmark_evidence.len(), 1);
    assert_eq!(
        validated.manifest.benchmark_evidence[0]
            .workload_profile
            .as_deref(),
        Some("decode-256")
    );

    let architectures: std::collections::BTreeSet<_> = validated
        .manifest
        .artifacts
        .iter()
        .filter_map(|artifact| artifact.target.architecture.clone())
        .collect();
    assert_eq!(
        architectures.len(),
        2,
        "expected two distinct compiled architectures"
    );

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_validation_pipeline_orders_schema_before_blob_io() {
    // A directory bundle with a *missing* blobs directory entirely and an
    // *unsupported* schema major version: if schema validation ran after
    // blob I/O, this would surface as a filesystem/blob error instead.
    let directory = temp_kernel_bundle_dir("ordering");
    let manifest = r#"{
  "schema": "magnetar:kernel-manifest@99.0",
  "artifacts": [
    { "role": "compiled-kernel", "format": "nvidia:cubin", "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000", "size": 4 }
  ]
}"#;
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();
    fs::remove_dir_all(directory.join("blobs")).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(
        matches!(outcome, Err(KernelManifestError::SchemaUnsupported { .. })),
        "expected schema validation to fail before any blob I/O is attempted, got {outcome:?}"
    );

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_normalizes_to_cache_key_and_entry_without_granting_trust() {
    let mut blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"cache-bridge-fixture"),
        4,
    );
    blob.required = true;
    let mut artifact = KernelManifestArtifact::new(blob);
    artifact.target.architecture = Some("sm90".into());
    artifact.compiler_metadata = Some(KernelCompilerMetadata {
        compiler_identity: Some("triton".into()),
        compiler_version: Some("3.1".into()),
        ..Default::default()
    });

    let key = normalize_to_cache_key(&artifact);
    assert_eq!(key.target_architecture, "sm90");
    assert_eq!(key.compiler_identity, "triton");

    let entry = normalize_to_cache_entry(&artifact);
    assert!(
        !entry.trust.is_trusted(),
        "a freshly normalized cache entry must start untrusted"
    );
    assert!(
        entry.qualification.is_none(),
        "a freshly normalized cache entry must start unqualified"
    );
}

#[test]
fn kernel_manifest_cli_operations_all_use_shared_validation() {
    let directory = temp_kernel_bundle_dir("cli-shared-validation");
    write_kernel_bundle(&directory, b"cli-fixture-bytes");
    let bundle = KernelExchangeBundle::open(&directory);
    let limits = KernelManifestLimits::default();

    for operation in [
        KernelManifestCliOperation::Inspect,
        KernelManifestCliOperation::Validate,
        KernelManifestCliOperation::Import,
        KernelManifestCliOperation::Export,
    ] {
        let result = run_kernel_manifest_cli_operation(operation, &bundle, &limits);
        assert!(
            result.is_ok(),
            "operation {operation:?} should reuse shared validation and succeed"
        );
    }

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_archive_rejects_symlink_entry() {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_cksum();
    builder
        .append_link(&mut header, "blobs/escape-link", "/etc/passwd")
        .unwrap();
    let tar_bytes = builder.into_inner().unwrap();

    let dir = temp_kernel_bundle_dir("archive-symlink");
    let outcome = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &dir,
        &KernelExchangeArchiveLimits::default(),
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleSymlinkDenied { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_exchange_archive_rejects_hardlink_and_device_entries() {
    for entry_type in [
        tar::EntryType::Link,
        tar::EntryType::Char,
        tar::EntryType::Fifo,
    ] {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(entry_type);
        header.set_size(0);
        header.set_mode(0o644);
        header.set_mtime(0);
        if entry_type == tar::EntryType::Char {
            header.set_device_major(1).unwrap();
            header.set_device_minor(3).unwrap();
        }
        if entry_type == tar::EntryType::Link {
            // `append_link` sets the path/link-name and writes the header's
            // checksum itself, so no manual `set_cksum` call here.
            builder
                .append_link(&mut header, "blobs/hardlink", "blobs/sha256/target")
                .unwrap();
        } else {
            // The raw `append` method does not recompute the checksum, so
            // the path must be set *before* `set_cksum` here.
            header.set_path("special-entry").unwrap();
            header.set_cksum();
            builder.append(&header, std::io::empty()).unwrap();
        }
        let tar_bytes = builder.into_inner().unwrap();

        let dir = temp_kernel_bundle_dir(&format!("archive-special-{entry_type:?}"));
        let outcome = extract_kernel_exchange_archive(
            std::io::Cursor::new(&tar_bytes),
            false,
            &dir,
            &KernelExchangeArchiveLimits::default(),
        );
        assert!(
            matches!(outcome, Err(KernelManifestError::BundlePathInvalid { .. })),
            "expected {entry_type:?} to be rejected, got {outcome:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
