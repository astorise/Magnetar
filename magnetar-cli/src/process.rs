//! CLI-owned shell/process execution (§13 "Shell And Process Execution" in
//! the change proposal).
//!
//! `magnetar-cli` MAY execute shell/processes according to CLI policy;
//! Runtime SHALL not execute shell commands or processes -- already
//! asserted structurally by `magnetar_runtime::cli_boundary::reject_cli_owned_authority`,
//! which rejects the `"process"`/`"process-execution"`/`"shell"` capability
//! names. This module runs a single named program with no arguments and no
//! shell interpretation (`std::process::Command::new(program).output()`,
//! never a shell string) -- the same injection-safe pattern
//! `commands::collect_git_diff_context` already uses for `git diff`: a
//! program name never passes through a shell that could reinterpret
//! metacharacters in it.

use magnetar_runtime::CliBoundaryError;

/// CLI-owned process execution policy (§13/§21 "Keep tool policy in CLI").
/// A process only ever runs when both this policy allows it and the caller
/// explicitly requests it (see `commands::cmd_run`'s `--tool` flag and
/// `config::CliConfig::tool_policy`) -- a flag alone is never sufficient.
///
/// This type's own [`Default`] is [`Self::Deny`], but that is not what
/// ships: [`crate::config::CliConfig::default`] sets `tool_policy` to
/// [`Self::AllowExplicit`], so `--tool` works out of the box today (see
/// that type's doc comment for why, and #55 for the discrepancy between
/// this comment's previous "deny by default" framing and the actually
/// shipped, permissive default). `Self::Deny` is real and reachable --
/// once a persistent configuration mechanism exists, setting it there
/// disables `--tool` regardless of the flag -- it is simply not selected
/// by anything today.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProcessPolicy {
    #[default]
    Deny,
    AllowExplicit,
}

/// Runs `program` with no arguments when `policy` allows it, returning its
/// captured stdout as a plain `String`. Never interprets `program` through a
/// shell -- `Command::new` execs it directly, so shell metacharacters in
/// `program` are not special and cannot chain a second command.
pub fn run_process(program: &str, policy: ProcessPolicy) -> Result<String, CliBoundaryError> {
    if !matches!(policy, ProcessPolicy::AllowExplicit) {
        return Err(CliBoundaryError::CliShellDenied {
            reason: format!("process execution denied by CLI policy for '{program}'"),
        });
    }
    let output = std::process::Command::new(program)
        .output()
        .map_err(|error| CliBoundaryError::CliShellDenied {
            reason: format!("failed to run '{program}': {error}"),
        })?;
    if !output.status.success() {
        return Err(CliBoundaryError::CliShellDenied {
            reason: format!("'{program}' exited with status {}", output.status),
        });
    }
    String::from_utf8(output.stdout).map_err(|error| CliBoundaryError::CliShellDenied {
        reason: format!("'{program}' output was not valid utf-8: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §13/§29 "Test shell stays in CLI": deny is the default and never
    /// spawns anything, regardless of what `program` names.
    #[test]
    fn deny_policy_never_spawns_a_process() {
        let error =
            run_process("definitely-not-a-real-program-xyz", ProcessPolicy::Deny).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliShellDenied { .. }));
    }

    /// A real, portable program runs and its stdout is captured -- or, if
    /// this environment does not have it on `PATH`, the failure is still a
    /// structured `CliShellDenied`, never a panic.
    #[test]
    fn allow_explicit_runs_a_real_program_and_captures_stdout() {
        match run_process("whoami", ProcessPolicy::AllowExplicit) {
            Ok(output) => assert!(!output.trim().is_empty()),
            Err(error) => assert!(matches!(error, CliBoundaryError::CliShellDenied { .. })),
        }
    }
}
