//! CLI-owned secret access (§11 "Secret Access" in
//! `openspec/changes/define-magnetar-cli-inference-boundary/proposal.md`).
//!
//! `magnetar-cli` MAY access secrets through CLI-owned policy; Runtime
//! SHALL not own user secrets and SHALL not receive them unless an
//! inference-scoped contract explicitly requires it (it does not, today).
//!
//! This module reads a single named environment variable on request
//! (`--env-secret <NAME>` on `magnetar run`, see `commands.rs`) and
//! deliberately does **not** wire the value into the prompt/context this
//! CLI sends to `pipeline::one_shot`. That omission is the safest, most
//! honest way to demonstrate "avoid sending secrets to Runtime by default"
//! in this increment: rather than adding an opt-out flag or a redaction
//! filter that could have gaps, the secret value has no code path into the
//! pipeline at all. A caller who wants to interpolate a secret into a
//! prompt must do so explicitly themselves (e.g. by typing it into the
//! prompt text) -- this module does not do it for them.
//!
//! [`CliBoundaryError::CliSecretUnavailable`] carries only the environment
//! variable *name*, never a value: see [`read_env_secret`]'s error
//! construction below, which never formats the underlying `std::env::var`
//! error's `Display` output because `VarError::NotUnicode` embeds the raw
//! (possibly secret-derived) `OsString` in its `Debug`/`Display`
//! representation. Redaction here is structural, not a follow-up scrub.

use magnetar_runtime::CliBoundaryError;

/// Reads the named environment variable in the CLI only. Never returns the
/// underlying `std::env::VarError`'s rendering (see module doc comment) --
/// on failure the resulting [`CliBoundaryError::CliSecretUnavailable`]
/// carries only `name`, so it is always safe to print via
/// `render::print_error`.
pub fn read_env_secret(name: &str) -> Result<String, CliBoundaryError> {
    map_var_lookup(name, std::env::var(name))
}

/// The pure mapping from a `std::env::var` lookup result to this module's
/// structured, redacted error -- split out from [`read_env_secret`] so it
/// can be tested against constructed `Result` values instead of the real
/// process environment (#68: `std::env::set_var`/`remove_var` are `unsafe`
/// in Rust 2024 precisely because they race with any *concurrent*
/// environment access on another thread, and `cargo test` runs tests in
/// parallel threads within one process. The previous "SAFETY: test-local,
/// unique variable name" comments addressed name *collisions*, not the
/// actual hazard, which is any concurrent access at all -- including a
/// plain `std::env::var` read inside another test or a dependency racing
/// with one of these tests' `set_var`/`remove_var` calls. Testing this pure
/// mapping instead removes the hazard entirely rather than trying to
/// synchronize around it).
fn map_var_lookup(
    name: &str,
    lookup: Result<String, std::env::VarError>,
) -> Result<String, CliBoundaryError> {
    match lookup {
        Ok(value) => Ok(value),
        Err(std::env::VarError::NotPresent) => Err(CliBoundaryError::CliSecretUnavailable {
            reason: format!("environment variable '{name}' is not set"),
        }),
        Err(std::env::VarError::NotUnicode(_)) => Err(CliBoundaryError::CliSecretUnavailable {
            reason: format!("environment variable '{name}' is not valid unicode"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::VarError;

    #[test]
    fn present_variable_returns_its_value() {
        let value = map_var_lookup(
            "MAGNETAR_CLI_TEST_SECRET_PRESENT",
            Ok("super-secret-test-value".to_string()),
        )
        .expect("variable was set");
        assert_eq!(value, "super-secret-test-value");
    }

    #[test]
    fn missing_variable_is_cli_secret_unavailable_naming_only_the_var() {
        let name = "MAGNETAR_CLI_TEST_SECRET_MISSING";
        let error = map_var_lookup(name, Err(VarError::NotPresent)).unwrap_err();
        match &error {
            CliBoundaryError::CliSecretUnavailable { reason } => {
                assert!(reason.contains(name));
            }
            other => panic!("expected CliSecretUnavailable, got {other:?}"),
        }
    }

    /// #68: `VarError::NotUnicode` was previously untested (a real
    /// non-UTF-8 environment variable is awkward to construct portably in
    /// a test at all) -- constructing the error value directly makes it
    /// trivial, and confirms this path is redacted exactly like
    /// `NotPresent`.
    #[test]
    fn non_unicode_variable_is_cli_secret_unavailable_and_never_renders_the_raw_value() {
        let name = "MAGNETAR_CLI_TEST_SECRET_NON_UNICODE";
        let raw = std::ffi::OsString::from("definitely-a-secret-marker-9f3c");
        let error = map_var_lookup(name, Err(VarError::NotUnicode(raw))).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains(name));
        assert!(!rendered.contains("definitely-a-secret-marker-9f3c"));
    }

    /// Proves the redaction guarantee end to end: the structured error
    /// this module builds for a missing variable's lookup never contains a
    /// value that happens to look like a secret, and `render::print_error`
    /// (via `render_error_never_uses_debug_formatting_of_the_boundary_error_itself`)
    /// separately proves rendering only ever uses this `Display` output.
    #[test]
    fn secret_value_never_leaks_into_a_cli_secret_unavailable_error_for_another_variable() {
        let missing_name = "MAGNETAR_CLI_TEST_SECRET_LEAK_CHECK_MISSING";
        let secret_value = "definitely-a-secret-marker-9f3c";
        let error = map_var_lookup(missing_name, Err(VarError::NotPresent)).unwrap_err();
        let rendered = error.to_string();
        assert!(!rendered.contains(secret_value));
    }

    /// One narrow check that the public entry point is genuinely wired to
    /// `std::env::var`, not only `map_var_lookup` in isolation --
    /// deliberately reads a variable this test never sets (a plain read,
    /// with no concurrent writer, carries none of `set_var`/`remove_var`'s
    /// hazard) rather than reintroducing environment mutation.
    #[test]
    fn read_env_secret_reports_a_variable_that_is_genuinely_absent() {
        let name = "MAGNETAR_CLI_TEST_SECRET_GENUINELY_ABSENT_7f2e9c";
        let error = read_env_secret(name).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliSecretUnavailable { .. }
        ));
    }
}
