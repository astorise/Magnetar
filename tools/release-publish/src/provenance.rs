//! Populates a real [`ReleaseProvenance`] from this checkout's actual git,
//! toolchain, and content state (tasks.md section 1, item 1.5) -- never a
//! placeholder record. `openspec_baseline_digest` and `wit_package_digest`
//! are computed over the real `openspec/` and WIT directories so a change
//! to either changes the recorded digest; `conformance_report_digest` is
//! supplied by the caller because generating that report is a separate,
//! already-existing CI job (`quality / e2e conformance` et al in
//! `.github/workflows/quality.yml`), not this crate's job to reproduce.

use std::{path::Path, process::Command};

use magnetar_roadmap_contracts::ReleaseProvenance;

use crate::{ReleasePublishError, checksum::sha256_hex};

fn run(command: &mut Command) -> Result<String, ReleasePublishError> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(ReleasePublishError::Metadata(format!(
            "command {command:?} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The real commit `repo_dir` is checked out at (`git rev-parse HEAD`).
pub fn git_commit_sha(repo_dir: &Path) -> Result<String, ReleasePublishError> {
    run(Command::new("git")
        .arg("-C")
        .arg(repo_dir)
        .arg("rev-parse")
        .arg("HEAD"))
}

/// The real toolchain building this release (`rustc --version`).
pub fn rustc_version() -> Result<String, ReleasePublishError> {
    run(Command::new("rustc").arg("--version"))
}

/// SHA-256 digest over every file's relative path and bytes under `dir`,
/// in deterministic (sorted relative-path) order -- a change to any
/// file's content, or a file being added or removed under `dir`, changes
/// this digest. Used for both `openspec_baseline_digest` (over
/// `openspec/`) and `wit_package_digest` (over a WIT package directory).
pub fn directory_digest(dir: &Path) -> Result<String, ReleasePublishError> {
    let mut paths = Vec::new();
    collect_relative_paths(dir, dir, &mut paths)?;
    paths.sort();

    let mut hasher_input = Vec::new();
    for relative in &paths {
        let bytes = std::fs::read(dir.join(relative))?;
        hasher_input.extend_from_slice(relative.as_bytes());
        hasher_input.push(0);
        hasher_input.extend_from_slice(&bytes);
        hasher_input.push(0);
    }
    Ok(sha256_hex(&hasher_input))
}

fn collect_relative_paths(
    root: &Path,
    dir: &Path,
    out: &mut Vec<String>,
) -> Result<(), ReleasePublishError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_relative_paths(root, &path, out)?;
        } else {
            out.push(
                path.strip_prefix(root)
                    .expect("walked path is under its own root")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

/// Assembles a real [`ReleaseProvenance`] from `repo_dir`'s actual git
/// commit, the toolchain actually invoking this function, and content
/// digests of `openspec_dir`/`wit_dir` when given. `release_tag`,
/// `ci_run_id`, `build_target`, `build_profile`, and
/// `conformance_report_digest` come from the caller because this crate has
/// no way to observe them itself: a CI run's own identity, the build
/// invocation's own target/profile flags, and a conformance report this
/// crate did not generate.
#[allow(clippy::too_many_arguments)]
pub fn collect_local_provenance(
    repo_dir: &Path,
    openspec_dir: Option<&Path>,
    wit_dir: Option<&Path>,
    release_tag: Option<String>,
    ci_run_id: Option<String>,
    build_target: Option<String>,
    build_profile: Option<String>,
    conformance_report_digest: Option<String>,
) -> Result<ReleaseProvenance, ReleasePublishError> {
    let lockfile_digest = sha256_hex(&std::fs::read(repo_dir.join("Cargo.lock"))?);

    Ok(ReleaseProvenance {
        source_commit: Some(git_commit_sha(repo_dir)?),
        release_tag,
        ci_run_id,
        build_target,
        build_profile,
        rustc_version: Some(rustc_version()?),
        lockfile_digest: Some(lockfile_digest),
        openspec_baseline_digest: openspec_dir.map(directory_digest).transpose()?,
        wit_package_digest: wit_dir.map(directory_digest).transpose()?,
        conformance_report_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("tools/")
            .parent()
            .expect("repo root")
            .to_path_buf()
    }

    #[test]
    fn git_commit_sha_returns_a_real_40_char_hex_sha() {
        let sha = git_commit_sha(&repo_root()).unwrap();
        assert_eq!(sha.len(), 40);
        assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn rustc_version_mentions_rustc() {
        let version = rustc_version().unwrap();
        assert!(version.contains("rustc"), "unexpected output: {version}");
    }

    #[test]
    fn directory_digest_is_deterministic_and_content_sensitive() {
        let dir = TempDir::new("dir-digest");
        std::fs::write(dir.join("a.txt"), b"one").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/b.txt"), b"two").unwrap();

        let first = directory_digest(&dir).unwrap();
        let second = directory_digest(&dir).unwrap();
        assert_eq!(first, second);

        std::fs::write(dir.join("a.txt"), b"changed").unwrap();
        let third = directory_digest(&dir).unwrap();
        assert_ne!(first, third);
    }

    #[test]
    fn collect_local_provenance_populates_real_fields() {
        let root = repo_root();
        let provenance = collect_local_provenance(
            &root,
            None,
            None,
            Some("v0.1.0".to_string()),
            Some("ci-run-123".to_string()),
            Some("x86_64-unknown-linux-gnu".to_string()),
            Some("release".to_string()),
            None,
        )
        .unwrap();

        assert!(provenance.source_commit.is_some());
        assert_eq!(provenance.release_tag.as_deref(), Some("v0.1.0"));
        assert_eq!(provenance.ci_run_id.as_deref(), Some("ci-run-123"));
        assert!(provenance.rustc_version.is_some());
        assert!(provenance.lockfile_digest.is_some());
        assert_eq!(provenance.openspec_baseline_digest, None);
    }
}
