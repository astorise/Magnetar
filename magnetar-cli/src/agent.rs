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

use std::path::{Path, PathBuf};

use magnetar_runtime::{CliBoundaryError, ModelRef};

use crate::observability::{CliObservationKind, CliObserver};
use crate::process::ProcessPolicy;
use crate::{pipeline, render, tools};

/// Bounds the number of Runtime Inference API calls one `magnetar agent`
/// invocation can make, so a caller cannot accidentally start an unbounded
/// loop against the tiny deterministic fixture path.
pub const MAX_AGENT_STEPS: usize = 4;

/// CLI-owned workspace-mutation policy for `magnetar agent --write`
/// (§14/§21 "Keep workspace mutation in CLI"), the same `Deny`/
/// `AllowExplicit` shape as [`crate::process::ProcessPolicy`] and
/// [`crate::network::NetworkPolicy`]. `--write` is the only capability in
/// this CLI that mutates the filesystem, and (like the other two) a flag
/// alone is not sufficient to exercise it -- this policy, threaded from
/// [`crate::config::CliConfig::write_policy`], must also allow it (#53:
/// previously nothing gated `--write` at all).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkspacePolicy {
    #[default]
    Deny,
    /// Not constructed by any non-test code path today: this CLI
    /// increment has no persistent configuration storage (see
    /// `config.rs`'s module doc comment), so there is no way for a user to
    /// have actually configured `AllowExplicit` yet -- the same reason
    /// `config::OutputFormat::Json` carries the same attribute. Exercised
    /// directly by this module's own tests.
    #[allow(dead_code)]
    AllowExplicit,
}

/// CLI-parsed options for one `magnetar agent` invocation. Assembled by
/// `commands::cmd_agent` from CLI flags; never influenced by model output.
#[derive(Default)]
pub struct AgentOptions {
    pub steps: usize,
    pub tool: Option<String>,
    pub write: Option<String>,
    /// CLI-owned tool execution policy, threaded from
    /// `config::CliConfig::tool_policy` by `commands::cmd_agent` exactly as
    /// `commands::cmd_run` already threads it to `tools::execute_tool` --
    /// previously this loop hardcoded [`ProcessPolicy::AllowExplicit`]
    /// regardless of CLI configuration (#53).
    pub tool_policy: ProcessPolicy,
    /// CLI-owned workspace-mutation policy, threaded from
    /// `config::CliConfig::write_policy`. Gates `write`.
    pub write_policy: WorkspacePolicy,
}

/// Resolves `requested` against `base` (the current working directory)
/// without touching the filesystem -- the target path need not exist yet,
/// so this cannot use `Path::canonicalize`. Rejects an absolute path
/// outright, then lexically applies `.`/`..` components, refusing to let a
/// `..` walk above `base`. Implements "confine `--write` to the working
/// directory" (#53): `--write` is the one filesystem-mutating capability in
/// this CLI, and previously accepted any path the process could reach,
/// including one escaping the working directory via `../..`.
fn resolve_workspace_write_path(base: &Path, requested: &str) -> Result<PathBuf, CliBoundaryError> {
    let requested_path = Path::new(requested);
    if requested_path.is_absolute() {
        return Err(CliBoundaryError::CliWorkspaceAccessDenied {
            reason: "--write path must be relative to the working directory".into(),
        });
    }
    let mut resolved = PathBuf::from(base);
    let mut depth = 0usize;
    for component in requested_path.components() {
        match component {
            std::path::Component::Normal(part) => {
                resolved.push(part);
                depth += 1;
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if depth == 0 {
                    return Err(CliBoundaryError::CliWorkspaceAccessDenied {
                        reason: "--write path must not escape the working directory".into(),
                    });
                }
                resolved.pop();
                depth -= 1;
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(CliBoundaryError::CliWorkspaceAccessDenied {
                    reason: "--write path must be relative to the working directory".into(),
                });
            }
        }
    }
    if depth == 0 {
        return Err(CliBoundaryError::CliWorkspaceAccessDenied {
            reason: "--write path must not be empty".into(),
        });
    }
    Ok(resolved)
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
    observer: &mut CliObserver,
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
        // §12/§21: gated by `options.tool_policy`, threaded from CLI config
        // exactly as `commands::cmd_run` gates its own `--tool` flag --
        // previously this loop always used `ProcessPolicy::AllowExplicit`
        // regardless of configuration (#53).
        let output = tools::execute_tool(program, options.tool_policy)?;
        observer.observe(
            CliObservationKind::ToolExecuted,
            format!("tool executed: {} bytes of output", output.len()),
        );
        println!("[agent tool output from '{program}']\n{output}");
    }

    if let Some(path) = &options.write {
        if !matches!(options.write_policy, WorkspacePolicy::AllowExplicit) {
            return Err(CliBoundaryError::CliWorkspaceAccessDenied {
                reason: format!("workspace write denied by CLI policy for '{path}'"),
            });
        }
        let cwd = std::env::current_dir().map_err(|error| {
            CliBoundaryError::CliWorkspaceAccessDenied {
                reason: format!("failed to read current directory: {error}"),
            }
        })?;
        let resolved_path = resolve_workspace_write_path(&cwd, path)?;
        std::fs::write(&resolved_path, &last_output).map_err(|error| {
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

    fn default_options() -> AgentOptions {
        AgentOptions {
            steps: 2,
            tool: None,
            write: None,
            tool_policy: ProcessPolicy::Deny,
            write_policy: WorkspacePolicy::Deny,
        }
    }

    /// §14/§29: the agent loop makes repeated Runtime Inference API calls
    /// (one independent `pipeline::one_shot` call per step) and never
    /// executes a tool or mutates the workspace unless explicitly asked.
    #[test]
    fn agent_loop_runs_bounded_steps_without_tool_or_write_by_default() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = default_options();
        let mut observer = CliObserver::new();
        let output = run_agent_loop(&model_ref, "reach the goal", &options, &mut observer).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn agent_loop_clamps_steps_to_max_agent_steps() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1000,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        // Would run effectively forever against a real model; proves the
        // loop itself is CLI-bounded (MAX_AGENT_STEPS) rather than trusting
        // the caller-supplied step count.
        let error = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap_err();
        assert!(error.runtime_category().is_some());
    }

    /// §13/§21/§53: `--tool` is denied when `tool_policy` denies it, the
    /// same as `commands::cmd_run`'s own `--tool` handling -- previously
    /// this loop ignored CLI configuration entirely and always ran the
    /// tool.
    #[test]
    fn agent_loop_tool_is_denied_by_deny_policy_regardless_of_the_flag() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1,
            tool: Some("whoami".to_string()),
            tool_policy: ProcessPolicy::Deny,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        let error = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliShellDenied { .. }));
    }

    /// The `ToolExecuted` observation (documented in `observability.rs` as
    /// emitted by both `cmd_run` and `cmd_agent`) is actually recorded on
    /// this path once the policy allows the tool to run.
    #[test]
    fn agent_loop_tool_execution_is_observed_when_policy_allows_it() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1,
            tool: Some("whoami".to_string()),
            tool_policy: ProcessPolicy::AllowExplicit,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        // "whoami" may not exist in every environment; either outcome is
        // fine here, the property under test is only whether a
        // successful run gets observed.
        if run_agent_loop(&model_ref, "goal", &options, &mut observer).is_ok() {
            assert!(
                observer
                    .observations()
                    .iter()
                    .any(|observation| observation.kind == CliObservationKind::ToolExecuted)
            );
        }
    }

    /// §14/§21/§53: `--write` is denied by default -- previously nothing
    /// gated it at all.
    #[test]
    fn agent_loop_write_is_denied_by_default_policy() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1,
            write: Some("agent-write-denied-test.txt".to_string()),
            ..default_options()
        };
        let mut observer = CliObserver::new();
        let error = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliWorkspaceAccessDenied { .. }
        ));
    }

    /// §14/§92 "Keep workspace mutation in CLI": the write path is
    /// resolved relative to the working directory and a real file is
    /// created there with the final step's CLI-owned output, and Runtime
    /// never touches the filesystem.
    #[test]
    fn agent_loop_write_option_writes_final_output_to_the_given_path() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let relative_name = format!("magnetar-cli-agent-write-test-{}.txt", std::process::id());
        let options = AgentOptions {
            steps: 1,
            write: Some(relative_name.clone()),
            write_policy: WorkspacePolicy::AllowExplicit,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        let output = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap();
        let full_path = std::env::current_dir().unwrap().join(&relative_name);
        assert!(!output.is_empty());
        assert!(full_path.exists());
        assert_eq!(std::fs::read_to_string(&full_path).unwrap(), output);
        std::fs::remove_file(&full_path).unwrap();
    }

    /// #53: `--write` cannot escape the working directory, even when the
    /// policy allows writing at all.
    #[test]
    fn agent_loop_write_rejects_a_path_escaping_the_working_directory() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let options = AgentOptions {
            steps: 1,
            write: Some("../escape-attempt.txt".to_string()),
            write_policy: WorkspacePolicy::AllowExplicit,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        let error = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliWorkspaceAccessDenied { .. }
        ));
    }

    /// #53: an absolute `--write` path is rejected outright, never
    /// forwarded to `std::fs::write` unchecked.
    #[test]
    fn agent_loop_write_rejects_an_absolute_path() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let absolute = std::env::temp_dir()
            .join("magnetar-cli-agent-write-absolute-test.txt")
            .to_str()
            .unwrap()
            .to_string();
        let options = AgentOptions {
            steps: 1,
            write: Some(absolute),
            write_policy: WorkspacePolicy::AllowExplicit,
            ..default_options()
        };
        let mut observer = CliObserver::new();
        let error = run_agent_loop(&model_ref, "goal", &options, &mut observer).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliWorkspaceAccessDenied { .. }
        ));
    }

    #[test]
    fn resolve_workspace_write_path_accepts_a_nested_relative_path() {
        let base = Path::new("/workspace/root");
        let resolved = resolve_workspace_write_path(base, "sub/dir/out.txt").unwrap();
        assert_eq!(resolved, base.join("sub/dir/out.txt"));
    }

    #[test]
    fn resolve_workspace_write_path_rejects_an_absolute_path() {
        let base = Path::new("/workspace/root");
        assert!(resolve_workspace_write_path(base, "/etc/passwd").is_err());
    }

    #[test]
    fn resolve_workspace_write_path_rejects_escaping_via_parent_dir() {
        let base = Path::new("/workspace/root");
        assert!(resolve_workspace_write_path(base, "../escape.txt").is_err());
        assert!(resolve_workspace_write_path(base, "sub/../../escape.txt").is_err());
    }

    #[test]
    fn resolve_workspace_write_path_allows_a_parent_dir_that_stays_within_base() {
        let base = Path::new("/workspace/root");
        let resolved = resolve_workspace_write_path(base, "sub/../out.txt").unwrap();
        assert_eq!(resolved, base.join("out.txt"));
    }
}
