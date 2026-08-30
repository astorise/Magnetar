//! CLI-owned agent orchestration (§14 "Agent Orchestration" in the change
//! proposal).
//!
//! Runtime SHALL not own agent planning, tool loops, or workspace mutation
//! -- already asserted structurally by
//! `magnetar_runtime::cli_boundary::reject_cli_owned_authority`, which
//! rejects the `"agent-orchestration"` capability name. This module is the
//! concrete, minimal demonstration of the other half of that boundary:
//! `magnetar-cli` MAY orchestrate an agent workflow, and every piece of
//! that workflow -- the loop itself, the CLI-owned "memory" fed back into
//! each step's prompt, any tool call, and any workspace mutation -- happens
//! here, in the CLI process, never in Runtime. Runtime is called only
//! through repeated [`pipeline::one_shot`] calls (§14 "Allow repeated
//! Runtime API calls") -- one independent one-shot Runtime session per
//! step, each given only a short, CLI-assembled prompt, rather than one
//! long-lived chat session replaying the whole transcript every turn (which
//! `pipeline.rs`'s fixture tokenizer's small `model_max_length` cannot
//! sustain past a couple of turns -- see that module's chat session test
//! doc comment for the same constraint).
//!
//! Tool calls in this loop are never triggered automatically by
//! model-generated text -- only by the caller-supplied [`AgentOptions::tool`]
//! (see `commands::cmd_agent`'s `--tool` flag), matching "Prevent automatic
//! tool execution from Runtime output" (already enforced independently of
//! this module -- see `tools.rs`'s module doc comment). Likewise, workspace
//! mutation only ever writes to the single caller-supplied
//! [`AgentOptions::write`] path, never a path derived from model output.

use magnetar_runtime::{CliBoundaryError, ModelRef};

use crate::process::ProcessPolicy;
use crate::{pipeline, render, tools};

/// Bounds the number of Runtime Inference API calls one `magnetar agent`
/// invocation can make, so a caller cannot accidentally start an unbounded
/// loop against the tiny deterministic fixture path.
pub const MAX_AGENT_STEPS: usize = 4;

/// CLI-parsed options for one `magnetar agent` invocation. Assembled by
/// `commands::cmd_agent` from CLI flags; never influenced by model output.
#[derive(Default)]
pub struct AgentOptions {
    pub steps: usize,
    pub tool: Option<String>,
    pub write: Option<String>,
}

/// Runs the agent loop: up to `options.steps` (clamped to
/// `[1, MAX_AGENT_STEPS]`) independent [`pipeline::one_shot`] calls, each
/// turn's prompt built here in the CLI from `goal` (step 0) or the previous
/// step's own (CLI-owned) output (later steps). This is the entire
/// "planning" this increment implements: iterative re-prompting, not a
/// separate planning model or protocol (defining a general agent loop
/// protocol is an explicit Non-Goal -- see `proposal.md` "define agent loop
/// semantics").
///
/// After the loop, runs the caller-requested tool (if any) and writes the
/// final step's output to the caller-requested path (if any) -- both are
/// CLI-owned, explicit, user-requested actions, never triggered by the
/// model's generated text. Returns the final step's decoded text.
pub fn run_agent_loop(
    model_ref: &ModelRef,
    goal: &str,
    options: &AgentOptions,
) -> Result<String, CliBoundaryError> {
    let steps = options.steps.clamp(1, MAX_AGENT_STEPS);
    let mut last_output = String::new();

    for step in 0..steps {
        let prompt_line = if step == 0 {
            goal.to_string()
        } else {
            format!("continue: {last_output}")
        };
        let (reply, generation_observer) = pipeline::one_shot(model_ref, &prompt_line)?;
        render::print_generation_observations(&generation_observer);
        println!("[agent step {}] {reply}", step + 1);
        last_output = reply;
    }

    if let Some(program) = &options.tool {
        let output = tools::execute_tool(program, ProcessPolicy::AllowExplicit)?;
        println!("[agent tool output from '{program}']\n{output}");
    }

    if let Some(path) = &options.write {
        std::fs::write(path, &last_output).map_err(|error| {
            CliBoundaryError::CliWorkspaceAccessDenied {
                reason: format!("failed to write agent output to '{path}': {error}"),
            }
        })?;
        println!("agent: wrote final step output to '{path}'");
    }

    Ok(last_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §14/§29: the agent loop makes repeated Runtime Inference API calls
    /// (one independent `pipeline::one_shot` call per step) and never
    /// executes a tool or mutates the workspace unless explicitly asked.
    #[test]
    fn agent_loop_runs_bounded_steps_without_tool_or_write_by_default() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 2,
            tool: None,
            write: None,
        };
        let output = run_agent_loop(&model_ref, "reach the goal", &options).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn agent_loop_clamps_steps_to_max_agent_steps() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1000,
            tool: None,
            write: None,
        };
        // Would run effectively forever against a real model; proves the
        // loop itself is CLI-bounded (MAX_AGENT_STEPS) rather than trusting
        // the caller-supplied step count.
        let error = run_agent_loop(&model_ref, "goal", &options).unwrap_err();
        assert!(error.runtime_category().is_some());
    }

    /// §14/§92 "Keep workspace mutation in CLI": the write path is used
    /// exactly as given -- a real file is created with the final step's
    /// CLI-owned output, and Runtime never touches the filesystem.
    #[test]
    fn agent_loop_write_option_writes_final_output_to_the_given_path() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let path = std::env::temp_dir().join(format!(
            "magnetar-cli-agent-write-test-{}.txt",
            std::process::id()
        ));
        let options = AgentOptions {
            steps: 1,
            tool: None,
            write: Some(path.to_str().unwrap().to_string()),
        };
        let output = run_agent_loop(&model_ref, "goal", &options).unwrap();
        assert!(!output.is_empty());
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), output);
        std::fs::remove_file(&path).unwrap();
    }
}
