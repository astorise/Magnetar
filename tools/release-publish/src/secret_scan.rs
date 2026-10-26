//! Real secret-shaped-content scanner feeding the previously-unmeasured
//! `secrets_detected` release security gate input (MAG-07 follow-up to
//! `openspec/changes/implement-release-publication-automation`). Before
//! this module existed, `security-gate`'s `--secrets-detected` switch was
//! never passed by the workflow, so it silently defaulted to `false` with
//! no scan ever having run -- the gate recorded "no secret found" when the
//! honest state was "never looked."
//!
//! Deliberately pattern-based, not regex-based: this crate has no `regex`
//! dependency today, and a handful of substring/character-class checks
//! covers the common secret shapes (cloud provider keys, VCS/chat platform
//! tokens, PEM private key blocks, generic `key = "..."`-shaped
//! assignments) without adding one. This is a release-time scan over
//! source/doc text, not a cryptographic guarantee -- it complements, and
//! does not replace, a human reviewing anything flagged, per
//! `release_security::SecretScanTarget`'s own non-exhaustive target list.

use std::{fs, path::Path};

use crate::ReleasePublishError;

/// One line of text that matched a known secret-shaped pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecretScanFinding {
    pub path: String,
    pub line_number: usize,
    pub pattern: String,
}

/// The result of scanning a set of paths: every finding, plus the single
/// boolean `security_gate::collect_security_gate_inputs` needs for its
/// `secrets_detected` field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SecretScanResult {
    pub findings: Vec<SecretScanFinding>,
}

impl SecretScanResult {
    pub fn secrets_detected(&self) -> bool {
        !self.findings.is_empty()
    }
}

/// Directory names never descended into: version control metadata and
/// build output, neither of which is release-shipped source/doc content
/// and both of which are large enough to make scanning them pure cost.
const SKIPPED_DIR_NAMES: &[&str] = &[".git", "target", "node_modules"];

/// File extensions treated as text and scanned; anything else (binaries,
/// images, compiled artifacts) is skipped without being read, matching
/// `SecretScanTarget::SourceFiles`/`GeneratedDocs`/`ReleaseNotes`'s own
/// scope -- this scan covers release-shipped text, not arbitrary binary
/// payloads (a binary secret leak is a different, checksum/provenance-
/// shaped problem, not a text-pattern one).
const SCANNED_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "yml", "yaml", "json", "txt", "sh", "py", "wit",
];

fn is_scanned_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| SCANNED_EXTENSIONS.contains(&ext))
}

fn walk_files(root: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), ReleasePublishError> {
    if root.is_file() {
        if is_scanned_file(root) {
            out.push(root.to_path_buf());
        }
        return Ok(());
    }
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        // A path named on the command line that does not exist by scan
        // time (e.g. an optional generated-docs directory) is not a scan
        // failure -- there is nothing there to leak a secret from.
        Err(_) => return Ok(()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let name = entry.file_name();
            if SKIPPED_DIR_NAMES
                .iter()
                .any(|skipped| name.to_str() == Some(*skipped))
            {
                continue;
            }
            walk_files(&path, out)?;
        } else if file_type.is_file() && is_scanned_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// Checks whether `bytes[start..]` begins with at least `min_len`
/// consecutive bytes satisfying `pred`. Requires at least `min_len` bytes
/// to actually remain -- `Iterator::all` is vacuously `true` on a short or
/// empty input, so without this check a token cut short by line end (e.g.
/// a bare `"AKIA"` or `"ghp_"` with nothing after it) would wrongly count
/// as a full-length match.
fn has_run(bytes: &[u8], start: usize, min_len: usize, pred: impl Fn(u8) -> bool) -> bool {
    bytes
        .get(start..)
        .is_some_and(|rest| rest.len() >= min_len && rest.iter().take(min_len).all(|b| pred(*b)))
}

fn is_upper_alnum(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit()
}

fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'+' || b == b'/' || b == b'='
}

/// Trailing-comment marker for a line that is known, deliberately
/// secret-shaped and not a leak -- a pattern's own definition naming the
/// shape it looks for (e.g. this module's PEM-header check), or a test
/// fixture built to exercise detection. Scanning this crate's own source
/// as release-shipped text would otherwise self-match both: a check for
/// "does this line contain both PEM-header substrings" cannot itself
/// avoid spelling out the two substrings it is looking for. Narrowly
/// scoped to the exact marked line, never a whole file or block, so it
/// cannot hide an unrelated real secret added later on a nearby line.
const SUPPRESSION_MARKER: &str = "secret-scan:allow";

/// Scans one line for every known secret-shaped pattern, returning the
/// first pattern name matched (a line is flagged once, not once per
/// pattern -- the finding's `path`/`line_number` is enough to go look).
fn match_line(line: &str) -> Option<&'static str> {
    if line.contains(SUPPRESSION_MARKER) {
        return None;
    }
    let bytes = line.as_bytes();

    // AWS access key id: literal "AKIA" followed by 16 more uppercase
    // letters/digits (the fixed AWS key-id shape).
    if let Some(index) = line.find("AKIA")
        && has_run(bytes, index + 4, 16, is_upper_alnum)
    {
        return Some("aws-access-key-id");
    }

    // GitHub personal/app/OAuth tokens: fixed prefixes followed by a long
    // alphanumeric token body.
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
        if let Some(index) = line.find(prefix)
            && has_run(bytes, index + prefix.len(), 20, |b| {
                b.is_ascii_alphanumeric()
            })
        {
            return Some("github-token");
        }
    }

    // Slack tokens: "xoxb-"/"xoxp-"/"xoxa-"/"xoxr-"/"xoxs-" followed by
    // digits/hyphens.
    for prefix in ["xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"] {
        if let Some(index) = line.find(prefix)
            && has_run(bytes, index + prefix.len(), 8, |b| {
                b.is_ascii_digit() || b == b'-'
            })
        {
            return Some("slack-token");
        }
    }

    // PEM private key block header.
    let is_pem_private_key_header = line.contains("-----BEGIN") && line.contains("PRIVATE KEY"); // secret-scan:allow
    if is_pem_private_key_header {
        return Some("pem-private-key");
    }

    // Generic `api_key = "..."`-shaped assignment: one of a handful of
    // secret-suggesting identifiers, a `:`/`=` separator, and a
    // sufficiently long token-shaped value after it.
    let lower = line.to_ascii_lowercase();
    const SECRET_LIKE_IDENTIFIERS: &[&str] = &[
        "api_key",
        "apikey",
        "secret_key",
        "access_key",
        "private_key",
        "client_secret",
        "password",
        "passwd",
    ];
    for identifier in SECRET_LIKE_IDENTIFIERS {
        if let Some(identifier_index) = lower.find(identifier) {
            let after_identifier = identifier_index + identifier.len();
            if let Some(separator_offset) = bytes[after_identifier..]
                .iter()
                .take(8)
                .position(|b| *b == b'=' || *b == b':')
            {
                let mut value_start = after_identifier + separator_offset + 1;
                while bytes
                    .get(value_start)
                    .is_some_and(|b| *b == b' ' || *b == b'"' || *b == b'\'')
                {
                    value_start += 1;
                }
                if has_run(bytes, value_start, 16, is_token_char) {
                    return Some("generic-secret-assignment");
                }
            }
        }
    }

    None
}

/// Scans every scanned-extension file under `paths` (files or
/// directories) for secret-shaped content, implementing the real evidence
/// source `security-gate`'s `secrets_detected` field now derives from
/// (previously always `false`, never measured).
pub fn scan_paths_for_secrets(
    paths: &[std::path::PathBuf],
) -> Result<SecretScanResult, ReleasePublishError> {
    let mut files = Vec::new();
    for path in paths {
        walk_files(path, &mut files)?;
    }
    files.sort();

    let mut findings = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            // Not valid UTF-8 text -- not one of this scan's targets
            // (binary artifact integrity is the checksum/provenance
            // gate's job, not this one's).
            continue;
        };
        for (zero_based_line, line) in text.lines().enumerate() {
            if let Some(pattern) = match_line(line) {
                findings.push(SecretScanFinding {
                    path: path.display().to_string(),
                    line_number: zero_based_line + 1,
                    pattern: pattern.to_string(),
                });
            }
        }
    }

    Ok(SecretScanResult { findings })
}

/// Serializes a [`SecretScanResult`] to the JSON this crate's CLI writes
/// as a release artifact and `security-gate --secret-scan-result` reads
/// back.
pub fn result_to_json(result: &SecretScanResult) -> serde_json::Value {
    serde_json::json!({
        "secrets_detected": result.secrets_detected(),
        "findings": result.findings.iter().map(|finding| serde_json::json!({
            "path": finding.path,
            "line_number": finding.line_number,
            "pattern": finding.pattern,
        })).collect::<Vec<_>>(),
    })
}

/// Parses the JSON [`result_to_json`] writes, used by `security-gate
/// --secret-scan-result <file>` to read a prior scan's outcome without
/// re-scanning.
pub fn secrets_detected_from_json(json: &str) -> Result<bool, ReleasePublishError> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    value["secrets_detected"].as_bool().ok_or_else(|| {
        ReleasePublishError::Metadata(
            "secret scan result JSON missing boolean 'secrets_detected'".into(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use std::fs;

    fn write(dir: &Path, relative: &str, contents: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn clean_source_reports_no_secrets() {
        let dir = TempDir::new("secret-scan-clean");
        write(
            &dir,
            "src/lib.rs",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        );
        write(&dir, "README.md", "# A perfectly ordinary project\n");
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(!result.secrets_detected());
        assert!(result.findings.is_empty());
    }

    #[test]
    fn detects_aws_access_key_id() {
        let dir = TempDir::new("secret-scan-aws");
        write(&dir, "notes.md", "export KEY=AKIAABCDEFGHIJKLMNOP\n"); // secret-scan:allow
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(result.secrets_detected());
        assert_eq!(result.findings[0].pattern, "aws-access-key-id");
    }

    #[test]
    fn detects_github_token() {
        let dir = TempDir::new("secret-scan-gh");
        write(
            &dir,
            "config.yml",
            "token: ghp_1234567890abcdefghij1234567890abcdef\n", // secret-scan:allow
        );
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(result.secrets_detected());
        assert_eq!(result.findings[0].pattern, "github-token");
    }

    #[test]
    fn detects_pem_private_key_header() {
        let dir = TempDir::new("secret-scan-pem");
        write(&dir, "key.txt", "-----BEGIN RSA PRIVATE KEY-----\n"); // secret-scan:allow
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(result.secrets_detected());
        assert_eq!(result.findings[0].pattern, "pem-private-key");
    }

    #[test]
    fn detects_generic_secret_assignment() {
        let dir = TempDir::new("secret-scan-generic");
        write(
            &dir,
            "settings.toml",
            // Deliberately not shaped like any real vendor's secret prefix
            // (e.g. Stripe's `sk_live_`) -- this only needs to exercise the
            // generic `identifier = "long-token-shaped-value"` check, and a
            // vendor-recognizable shape here would itself be flagged by
            // GitHub's own push protection on this very commit.
            "api_key = \"not-a-real-secret-just-a-long-placeholder-value\"\n",
        );
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(result.secrets_detected());
        assert_eq!(result.findings[0].pattern, "generic-secret-assignment");
    }

    #[test]
    fn does_not_flag_the_word_password_alone_without_a_value() {
        let dir = TempDir::new("secret-scan-word-only");
        write(
            &dir,
            "docs.md",
            "The password field accepts any UTF-8 string.\n",
        );
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(!result.secrets_detected());
    }

    #[test]
    fn does_not_flag_a_token_prefix_cut_short_by_end_of_line() {
        // Regression test: `has_run` used to accept a short or empty
        // remaining slice as a vacuous match (`Iterator::all` on fewer
        // than `min_len` items is trivially true), so ordinary text
        // ending in a bare prefix like "AKIA" was wrongly flagged.
        let dir = TempDir::new("secret-scan-short-suffix");
        write(&dir, "notes.md", "the string ends in AKIA\n");
        write(&dir, "other.md", "the identifier is ghp_abc\n");
        write(&dir, "empty.md", "password = \n");
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(!result.secrets_detected(), "{:?}", result.findings);
    }

    #[test]
    fn skips_git_and_target_directories() {
        let dir = TempDir::new("secret-scan-skip-dirs");
        write(&dir, ".git/config.rs", "AKIAABCDEFGHIJKLMNOP\n"); // secret-scan:allow
        write(&dir, "target/debug/build.rs", "AKIAABCDEFGHIJKLMNOP\n"); // secret-scan:allow
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(!result.secrets_detected());
    }

    #[test]
    fn skips_non_text_extensions() {
        let dir = TempDir::new("secret-scan-skip-ext");
        write(&dir, "fixture.wasm", "AKIAABCDEFGHIJKLMNOP\n"); // secret-scan:allow
        let result = scan_paths_for_secrets(std::slice::from_ref(&dir.0)).unwrap();
        assert!(!result.secrets_detected());
    }

    #[test]
    fn missing_path_is_not_an_error() {
        let result = scan_paths_for_secrets(&[std::path::PathBuf::from(
            "/this/path/does/not/exist/anywhere",
        )])
        .unwrap();
        assert!(!result.secrets_detected());
    }

    #[test]
    fn json_round_trips_secrets_detected() {
        let detected = SecretScanResult {
            findings: vec![SecretScanFinding {
                path: "foo.rs".into(),
                line_number: 3,
                pattern: "aws-access-key-id".into(),
            }],
        };
        let json = result_to_json(&detected);
        assert!(secrets_detected_from_json(&json.to_string()).unwrap());

        let clean = SecretScanResult::default();
        let json = result_to_json(&clean);
        assert!(!secrets_detected_from_json(&json.to_string()).unwrap());
    }
}
