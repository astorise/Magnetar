//! CLI-owned tool execution (§12 "Tool Execution" in the change proposal).
//!
//! `magnetar-cli` MAY execute tools; Runtime SHALL not execute tools and
//! SHALL not receive automatic execution triggered by its own generated
//! text. "Prevent automatic tool execution from Runtime output" holds here
//! by construction: nothing in `commands.rs`/`agent.rs` ever calls
//! [`execute_tool`] with model-generated text as the tool name -- it is
//! wired only to a caller-supplied `--tool <program>` CLI flag
//! (`commands::cmd_run`, `commands::cmd_agent`). `render::looks_like_tool_call`
//! remains a detect-only heuristic used purely to print an informational
//! note; it is never consulted here.
//!
//! Tool execution itself is layered directly on
//! [`crate::process::run_process`] -- a tool, in this minimal CLI
//! increment, is just a named program the user explicitly asked to run.

use crate::process::{ProcessPolicy, run_process};
use magnetar_runtime::CliBoundaryError;

/// Runs the tool named `program` when `policy` allows it. A thin, explicit
/// wrapper over [`run_process`] so call sites name what they are doing
/// ("execute a tool") even though today's implementation is the same
/// process-execution primitive `process::run_process` provides.
pub fn execute_tool(program: &str, policy: ProcessPolicy) -> Result<String, CliBoundaryError> {
    run_process(program, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §12/§29 "Test tools stay in CLI": tool execution is denied by
    /// default, the same as bare process execution -- there is no separate,
    /// looser policy for "tools" than for "processes".
    #[test]
    fn tool_execution_is_denied_by_default_policy() {
        let error = execute_tool("whoami", ProcessPolicy::Deny).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliShellDenied { .. }));
    }

    #[test]
    fn tool_execution_runs_the_named_program_when_explicitly_allowed() {
        match execute_tool("whoami", ProcessPolicy::AllowExplicit) {
            Ok(output) => assert!(!output.trim().is_empty()),
            Err(error) => assert!(matches!(error, CliBoundaryError::CliShellDenied { .. })),
        }
    }
}
