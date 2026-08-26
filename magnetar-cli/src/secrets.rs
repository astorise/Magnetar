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
    match std::env::var(name) {
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

    #[test]
    fn read_env_secret_returns_the_value_of_a_present_variable() {
        let name = "MAGNETAR_CLI_TEST_SECRET_PRESENT";
        // SAFETY: test-local, unique variable name not read/written by any
        // other test in this crate.
        unsafe {
            std::env::set_var(name, "super-secret-test-value");
        }
        let value = read_env_secret(name).expect("variable was set");
        assert_eq!(value, "super-secret-test-value");
        unsafe {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn read_env_secret_missing_variable_is_cli_secret_unavailable_naming_only_the_var() {
        let name = "MAGNETAR_CLI_TEST_SECRET_MISSING";
        unsafe {
            std::env::remove_var(name);
        }
        let error = read_env_secret(name).unwrap_err();
        match &error {
            CliBoundaryError::CliSecretUnavailable { reason } => {
                assert!(reason.contains(name));
            }
            other => panic!("expected CliSecretUnavailable, got {other:?}"),
        }
    }

    /// Proves the redaction guarantee end to end: a real secret value is
    /// set in the environment, but the structured error this module builds
    /// when a *different* (missing) variable is requested never contains
    /// that value, and `render::print_error`'s rendering of the error is
    /// built only from the error's `Display` impl, which in turn only ever
    /// sees the variable name for this variant.
    #[test]
    fn secret_value_never_leaks_into_a_cli_secret_unavailable_error_for_another_variable() {
        let present_name = "MAGNETAR_CLI_TEST_SECRET_LEAK_CHECK_PRESENT";
        let missing_name = "MAGNETAR_CLI_TEST_SECRET_LEAK_CHECK_MISSING";
        let secret_value = "definitely-a-secret-marker-9f3c";
        unsafe {
            std::env::set_var(present_name, secret_value);
            std::env::remove_var(missing_name);
        }
        let error = read_env_secret(missing_name).unwrap_err();
        let rendered = error.to_string();
        assert!(!rendered.contains(secret_value));
        unsafe {
            std::env::remove_var(present_name);
        }
    }
}
