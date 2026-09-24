//! Generates a real [`SbomManifest`] from `cargo metadata --format-version=1`
//! output (tasks.md section 1, item 1.4): every [`SbomEntry`] comes from an
//! actual workspace/dependency package Cargo already resolved, never a
//! hand-maintained list that can drift from the real dependency graph.

use magnetar_roadmap_contracts::{SbomAvailability, SbomEntry, SbomManifest};
use serde::Deserialize;

use crate::ReleasePublishError;

#[derive(Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
}

#[derive(Deserialize)]
struct CargoPackage {
    name: String,
    version: String,
    license: Option<String>,
    license_file: Option<String>,
    repository: Option<String>,
}

/// Parses `cargo metadata --format-version=1` JSON (as produced by, e.g.,
/// `cargo metadata --format-version=1 --no-deps`) into one [`SbomEntry`]
/// per package. A package with no declared `license` but a `license_file`
/// records `"see LICENSE file"` rather than a fabricated SPDX identifier;
/// a package with neither records `"UNSPECIFIED"` so the gap is visible in
/// the SBOM itself instead of silently omitted.
pub fn sbom_entries_from_cargo_metadata_json(
    json: &str,
) -> Result<Vec<SbomEntry>, ReleasePublishError> {
    let metadata: CargoMetadata = serde_json::from_str(json)?;
    Ok(metadata
        .packages
        .into_iter()
        .map(|package| SbomEntry {
            package_name: package.name,
            package_version: package.version,
            licenses: vec![package.license.unwrap_or_else(|| {
                package
                    .license_file
                    .map(|_| "see LICENSE file".to_string())
                    .unwrap_or_else(|| "UNSPECIFIED".to_string())
            })],
            source_repository: package.repository,
        })
        .collect())
}

/// Builds a [`SbomManifest`] with [`SbomAvailability::Generated`] from real
/// `cargo metadata` output. Callers with no successfully resolved metadata
/// (e.g. an offline or broken build) should build
/// `SbomManifest { availability: SbomAvailability::PlaceholderDocumented, limitation_note: Some(..), .. }`
/// directly instead of calling this function with empty input, per
/// [`SbomManifest::validate`]'s own requirement that `Generated` carry at
/// least one real entry.
pub fn generate_sbom_manifest(
    cargo_metadata_json: &str,
    build_target: Option<String>,
    feature_flags: Vec<String>,
) -> Result<SbomManifest, ReleasePublishError> {
    let entries = sbom_entries_from_cargo_metadata_json(cargo_metadata_json)?;
    Ok(SbomManifest {
        availability: SbomAvailability::Generated,
        limitation_note: None,
        entries,
        build_target,
        feature_flags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "packages": [
            {
                "name": "magnetar-runtime",
                "version": "0.1.0",
                "license": "MIT",
                "license_file": null,
                "repository": "https://github.com/astorise/Magnetar"
            },
            {
                "name": "some-dep",
                "version": "2.3.4",
                "license": null,
                "license_file": "LICENSE-CUSTOM",
                "repository": null
            },
            {
                "name": "unlicensed-dep",
                "version": "0.0.1",
                "license": null,
                "license_file": null,
                "repository": null
            }
        ]
    }"#;

    #[test]
    fn extracts_one_entry_per_package() {
        let entries = sbom_entries_from_cargo_metadata_json(FIXTURE).unwrap();
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].package_name, "magnetar-runtime");
        assert_eq!(entries[0].licenses, vec!["MIT".to_string()]);
        assert_eq!(
            entries[0].source_repository.as_deref(),
            Some("https://github.com/astorise/Magnetar")
        );

        assert_eq!(entries[1].licenses, vec!["see LICENSE file".to_string()]);

        assert_eq!(entries[2].licenses, vec!["UNSPECIFIED".to_string()]);
        assert_eq!(entries[2].source_repository, None);
    }

    #[test]
    fn generated_manifest_passes_its_own_validation() {
        let manifest = generate_sbom_manifest(
            FIXTURE,
            Some("x86_64-unknown-linux-gnu".to_string()),
            vec![],
        )
        .unwrap();
        assert_eq!(manifest.availability, SbomAvailability::Generated);
        manifest.validate().expect("generated SBOM must validate");
    }

    #[test]
    fn invalid_json_is_reported_as_an_error() {
        let result = sbom_entries_from_cargo_metadata_json("not json");
        assert!(matches!(result, Err(ReleasePublishError::Json(_))));
    }
}
