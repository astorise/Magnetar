//! Subcommand dispatch and implementations.
//!
//! Every command here either calls into [`crate::pipeline`] (which itself
//! only calls the Runtime Inference API) or, for `providers`/`devices`,
//! builds a [`Runtime`] directly to read redacted diagnostics through
//! `magnetar_runtime::runtime::Runtime::providers`/`devices`. Only
//! [`crate::pipeline`] and this module are allowed to call into
//! `magnetar_runtime`, and neither ever grants the Runtime workspace, Git,
//! network, secret, tool, shell, or agent authority.
//!
//! `magnetar run` now supports opt-in `--file`/`--git-diff`/`--env-secret`
//! flags (§4/§8/§9/§11): file reads (`read_file_context`) and Git access
//! (`collect_git_diff_context`) happen only in this file, and only when
//! explicitly requested; their results are folded into a plain `String`
//! prompt (`assemble_prompt`) before ever reaching `pipeline::one_shot`,
//! which accepts `prompt: &str` -- never a path, a `Command`, or any other
//! CLI-owned authority. `--env-secret` reads a named environment variable
//! (`secrets::read_env_secret`) but its value is deliberately never wired
//! into the prompt in this increment (see `secrets.rs`).

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use magnetar_runtime::{
    CliBoundaryError, InferenceApiError, MemoryManager, MemoryManagerConfig, ModelArtifactSource,
    ModelInstanceId, ModelInstanceUnloadPolicy, ModelLoadingApiRequest, ModelLoadingCoordinator,
    ModelLoadingRequest, ModelLoadingRequestId, ModelRef, ModelRegistry, ModelResolutionRequest,
    ModelTrustStore, ReferenceCpuProvider, ReleaseVersion, Runtime,
    build_release_binary_version_report, load_model, unload_model_instance,
};

use crate::observability::{CliObservationKind, CliObserver};
use crate::{agent, aliases, config, network, pipeline, render, secrets, serve, tools};

pub fn dispatch(args: &[String]) -> Result<(), CliBoundaryError> {
    // `-v`/`--verbose` is a global flag: recognized anywhere in `args`,
    // stripped before subcommand parsing, and controls only whether a
    // one-line CLI observability summary (counts only, see
    // `observability.rs`'s redaction guarantee) is printed at the end.
    // Default output is unchanged when it is absent.
    let verbose = args.iter().any(|arg| arg == "--verbose" || arg == "-v");
    let filtered: Vec<String> = args
        .iter()
        .filter(|arg| arg.as_str() != "--verbose" && arg.as_str() != "-v")
        .cloned()
        .collect();

    let mut observer = CliObserver::new();
    observer.observe(
        CliObservationKind::CommandReceived,
        format!("received {} argument(s)", filtered.len()),
    );

    let Some(subcommand) = filtered.first() else {
        render::print_usage();
        return Ok(());
    };
    observer.observe(
        CliObservationKind::CommandParsed,
        format!("parsed subcommand '{subcommand}'"),
    );

    let result = match subcommand.as_str() {
        "--help" | "-h" | "help" => {
            render::print_usage();
            Ok(())
        }
        "run" => cmd_run(&filtered[1..], &mut observer),
        "chat" => cmd_chat(&filtered[1..], &mut observer),
        "agent" => cmd_agent(&filtered[1..], &mut observer),
        "model" => cmd_model(&filtered[1..]),
        "providers" => cmd_providers(),
        "devices" => cmd_devices(),
        "sessions" => cmd_sessions(),
        "serve" => cmd_serve(&filtered[1..]),
        "version" | "--version" | "-V" => cmd_version(),
        other => {
            render::print_usage();
            Err(CliBoundaryError::CliCommandInvalid {
                reason: format!("unknown subcommand '{other}'"),
            })
        }
    };

    // Coarse, kind-only labels -- never the error's `Display` text, which
    // for some variants (e.g. `CliFileReadFailed`) carries a filesystem
    // path. Keeping observability messages to static labels here is a
    // stronger redaction guarantee than trusting every error variant's
    // text to stay content-free.
    match &result {
        Ok(()) => observer.observe(CliObservationKind::CommandCompleted, "command completed"),
        Err(_) => observer.observe(CliObservationKind::CommandFailed, "command failed"),
    }

    if verbose {
        eprintln!(
            "[cli] {} observation(s) recorded for this invocation",
            observer.observations().len()
        );
    }

    result
}

/// Opt-in flags recognized by `magnetar run` (§4/§8/§9/§10/§11/§12/§15). All
/// are deliberately off by default: file reads, Git access, workspace
/// inspection, network retrieval, tool execution, and secret access only
/// happen when the user explicitly asks for them via these flags, per the
/// change proposal's "Workspace And File Access" / "Git Access" / "Network
/// Access" / "Tool Execution" / "Secret Access" sections ("MAY access ...
/// where explicitly requested").
#[derive(Default)]
struct RunFlags {
    file: Option<String>,
    git_diff: bool,
    workspace: bool,
    url: Option<String>,
    tool: Option<String>,
    env_secret: Option<String>,
}

/// Splits `args` into recognized [`RunFlags`] and the remaining positional
/// arguments (model reference + prompt words), preserving relative order.
/// Pure CLI-side argument parsing; never touches Runtime.
fn parse_run_flags(args: &[String]) -> (RunFlags, Vec<String>) {
    let mut flags = RunFlags::default();
    let mut rest = Vec::new();
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--file" => flags.file = iter.next(),
            "--git-diff" => flags.git_diff = true,
            "--workspace" => flags.workspace = true,
            "--url" => flags.url = iter.next(),
            "--tool" => flags.tool = iter.next(),
            "--env-secret" => flags.env_secret = iter.next(),
            _ => rest.push(arg),
        }
    }
    (flags, rest)
}

/// Collects a shallow CLI-owned "workspace context" snapshot: the current
/// working directory's path and immediate entry names, not recursive and
/// never file contents (§15 "Assemble workspace context in CLI"). Distinct
/// from [`read_file_context`] (a single file's content) -- this is
/// workspace *inspection*. Runtime never receives a directory handle or
/// scanning authority, only this already-collected `String`.
fn collect_workspace_context() -> Result<String, CliBoundaryError> {
    let cwd =
        std::env::current_dir().map_err(|error| CliBoundaryError::CliWorkspaceAccessDenied {
            reason: format!("failed to read current directory: {error}"),
        })?;
    let mut entries: Vec<String> = std::fs::read_dir(&cwd)
        .map_err(|error| CliBoundaryError::CliWorkspaceAccessDenied {
            reason: format!("failed to list '{}': {error}", cwd.display()),
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    Ok(format!(
        "cwd: {}\nentries: {}",
        cwd.display(),
        entries.join(", ")
    ))
}

/// Reads `path`'s UTF-8 content in the CLI only (`std::fs::read_to_string`).
/// The CLI never hands a filesystem authority to Runtime -- only the
/// resulting `String` ever crosses into `pipeline::one_shot`'s `prompt:
/// &str` parameter (a `&str`, not a path, by construction). Implements
/// "Read requested files in CLI only" / "Keep arbitrary file reads in CLI"
/// (§8).
fn read_file_context(path: &str) -> Result<String, CliBoundaryError> {
    std::fs::read_to_string(path).map_err(|error| CliBoundaryError::CliFileReadFailed {
        reason: format!("{path}: {error}"),
    })
}

/// Runs a fixed, hardcoded `git diff` in the current working directory
/// (§9). This never interprets or executes any model-generated or
/// user-supplied string as a shell command -- the `Command` invocation
/// below is a constant, not built from untrusted input -- so it does not
/// reopen the general "execute arbitrary shell text" risk this whole
/// change fences off (§13).
fn collect_git_diff_context() -> Result<String, CliBoundaryError> {
    let output = std::process::Command::new("git")
        .args(["diff"])
        .output()
        .map_err(|error| CliBoundaryError::CliGitFailed {
            reason: format!("failed to run 'git diff': {error}"),
        })?;
    if !output.status.success() {
        return Err(CliBoundaryError::CliGitFailed {
            reason: format!("'git diff' exited with status {}", output.status),
        });
    }
    String::from_utf8(output.stdout).map_err(|error| CliBoundaryError::CliGitFailed {
        reason: format!("'git diff' output was not valid utf-8: {error}"),
    })
}

/// Assembles the final prompt string sent to the pipeline from the base
/// prompt plus any labeled context sections, entirely in the CLI (§15
/// "Assemble file/Git/workspace/tool/network context in CLI"). The
/// pipeline only ever receives this owned `String` -- never a path, a
/// `Command`, a socket, or any other filesystem/process/network authority.
fn assemble_prompt(base: &str, sections: &[(&str, &str)]) -> String {
    let mut parts: Vec<String> = sections
        .iter()
        .map(|(label, content)| format!("[{label}]\n{content}"))
        .collect();
    parts.push(base.to_string());
    parts.join("\n\n")
}

const RUN_USAGE: &str = "usage: magnetar run <model-ref> [--file <path>] [--git-diff] [--workspace] [--url <http-url>] [--tool <program>] [--env-secret <NAME>] <prompt text...>";

/// `magnetar run <model-ref> [--file <path>] [--git-diff] [--workspace]
/// [--url <http-url>] [--tool <program>] [--env-secret <NAME>] <prompt
/// text...>`. See `pipeline::one_shot` for the full flow: parse args and
/// assemble prompt context here in the CLI, then hand off to the Runtime
/// Inference API for everything inference-scoped.
fn cmd_run(args: &[String], observer: &mut CliObserver) -> Result<(), CliBoundaryError> {
    let (flags, rest) = parse_run_flags(args);
    let config = config::CliConfig::default();

    // Model reference resolution: positional arg 0 if present, else CLI
    // config's default alias (§21 "Keep default model alias in CLI" --
    // inert today since `CliConfig::default()` always has `None` here, see
    // `config.rs`'s doc comment and tests).
    let (model_ref_arg, prompt_words) = match rest.split_first() {
        Some((first, remainder)) => (first.clone(), remainder.to_vec()),
        None => match config::resolve_model_ref_arg(&rest, &config) {
            Some(alias) => (alias.to_string(), Vec::new()),
            None => {
                return Err(CliBoundaryError::CliCommandInvalid {
                    reason: RUN_USAGE.into(),
                });
            }
        },
    };
    if prompt_words.is_empty() {
        return Err(CliBoundaryError::CliPromptInputInvalid {
            reason: "prompt text is required".into(),
        });
    }
    let base_prompt = prompt_words.join(" ");

    // Model Aliases (§6/§22): resolve a friendly alias to a literal
    // reference before it goes through the exact same `ModelRef::new`
    // validation as any literal input -- alias resolution cannot bypass
    // that validation by construction.
    let model_ref = ModelRef::new(aliases::resolve_alias(&model_ref_arg))?;

    let file_context = match &flags.file {
        Some(path) => {
            let content = read_file_context(path)?;
            observer.observe(
                CliObservationKind::FileContextCollected,
                format!("file context collected: {} bytes", content.len()),
            );
            Some(content)
        }
        None => None,
    };
    let git_context = if flags.git_diff {
        let diff = collect_git_diff_context()?;
        observer.observe(
            CliObservationKind::GitContextCollected,
            format!("git diff context collected: {} bytes", diff.len()),
        );
        Some(diff)
    } else {
        None
    };
    // Workspace Access (§8/§15): opt-in, CLI-owned, shallow (non-recursive)
    // directory listing -- see `collect_workspace_context`'s doc comment.
    let workspace_context = if flags.workspace {
        Some(collect_workspace_context()?)
    } else {
        None
    };
    // Network Access (§10/§15/§21): opt-in, and gated by both the `--url`
    // flag and CLI config's `network_policy` -- a flag alone is not
    // sufficient (see `config.rs`'s doc comment on `network_policy`).
    let network_context = match &flags.url {
        Some(url) => Some(network::fetch_url_context(url, config.network_policy)?),
        None => None,
    };
    // Tool Execution (§12/§15/§21/§26): opt-in, gated by both the `--tool`
    // flag and CLI config's `tool_policy`, and never triggered by
    // model-generated text (see `tools.rs`'s module doc comment).
    let tool_context = match &flags.tool {
        Some(program) => {
            let output = tools::execute_tool(program, config.tool_policy)?;
            observer.observe(
                CliObservationKind::ToolExecuted,
                format!("tool executed: {} bytes of output", output.len()),
            );
            Some(output)
        }
        None => None,
    };
    // Secret Access (§11): read-only, opt-in, and deliberately not wired
    // into the prompt in this increment -- see `secrets.rs`'s module doc
    // comment for why that omission (rather than a redaction filter) is
    // the safest way to keep the secret value off the Runtime-bound path.
    if let Some(name) = &flags.env_secret {
        match secrets::read_env_secret(name) {
            Ok(_value) => {
                println!(
                    "cli: read secret from env var '{name}' (value not sent to Runtime in this increment)"
                );
            }
            Err(error) => render::print_error(&error),
        }
    }

    let mut sections: Vec<(&str, &str)> = Vec::new();
    if let Some(content) = file_context.as_deref() {
        sections.push(("file context", content));
    }
    if let Some(diff) = git_context.as_deref() {
        sections.push(("git diff", diff));
    }
    if let Some(workspace) = workspace_context.as_deref() {
        sections.push(("workspace context", workspace));
    }
    if let Some(retrieved) = network_context.as_deref() {
        sections.push(("network context", retrieved));
    }
    if let Some(tool_output) = tool_context.as_deref() {
        sections.push(("tool output", tool_output));
    }
    let prompt = assemble_prompt(&base_prompt, &sections);

    observer.observe(
        CliObservationKind::RuntimeRequestSubmitted,
        "run: submitting one-shot pipeline request",
    );
    let (text, generation_observer) = pipeline::one_shot(&model_ref, &prompt)?;
    render::print_generation_observations(&generation_observer);
    observer.observe(
        CliObservationKind::StreamRendered,
        format!(
            "stream rendered: {} observation(s)",
            generation_observer.observations().len()
        ),
    );
    if render::looks_like_tool_call(&text) {
        println!(
            "note: this looks like it might be a tool call; magnetar-cli does not execute tool calls in this increment"
        );
    }
    println!("{text}");
    Ok(())
}

/// `magnetar chat <model-ref>`. Manages the interactive loop, terminal
/// prompt, CLI-owned transcript, and CLI-owned command history entirely in
/// this function; only each turn's prompt text/chat messages cross into
/// `pipeline::ChatSession::turn` (and from there into the Runtime
/// Inference API).
fn cmd_chat(args: &[String], observer: &mut CliObserver) -> Result<(), CliBoundaryError> {
    let Some(model_ref_arg) = args.first() else {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: "usage: magnetar chat <model-ref>".into(),
        });
    };
    let model_ref = ModelRef::new(aliases::resolve_alias(model_ref_arg))?;
    let mut chat = pipeline::ChatSession::open(&model_ref)?;
    // Records that this chat's turns are bound to one persistent Runtime
    // session (task 8.3) -- the id itself, not any prompt or transcript
    // content, so it stays within the redaction rules this module's
    // `CliObservation.message` doc comment requires.
    observer.observe(
        CliObservationKind::RuntimeRequestSubmitted,
        format!("chat session opened: session={}", chat.session_id()),
    );
    println!("magnetar chat -- native fixture inference; type 'exit', 'cancel', or Ctrl-D to quit");

    let stdin = io::stdin();
    let mut input = stdin.lock();
    // Command History (§5): distinct from `ChatSession::transcript` (the
    // CLI-owned conversational transcript, some of which may be rendered
    // into Runtime-bound chat messages -- see `pipeline::ChatSession::turn`).
    // This records every non-empty line the user typed at the `>` prompt,
    // including `exit`, purely as REPL-style command history; it is never
    // sent to Runtime in any form.
    let mut command_history: Vec<String> = Vec::new();
    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| CliBoundaryError::InternalCliError {
                reason: error.to_string(),
            })?;

        let mut line = String::new();
        let bytes_read =
            input
                .read_line(&mut line)
                .map_err(|error| CliBoundaryError::InternalCliError {
                    reason: error.to_string(),
                })?;
        if bytes_read == 0 {
            break; // EOF (Ctrl-D)
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.is_empty() {
            continue;
        }
        command_history.push(line.to_string());
        if line == "exit" {
            break;
        }
        // Cancellation (§19 "CLI Cancellation Calls Runtime Cancellation"):
        // calls the real `pipeline::ChatSession::cancel`, which calls
        // Runtime's own session cancellation. This is the CLI's user-facing
        // cancellation entry point; CLI-owned file/Git/network/tool work
        // has nothing to cancel here (see `ChatSession::cancel`'s doc
        // comment) since this synchronous CLI never has any of that work
        // in flight when `cancel` is typed.
        if line == "cancel" {
            match chat.cancel() {
                Ok(()) => {
                    println!(
                        "magnetar chat -- cancellation requested: Runtime inference session cancelled"
                    );
                    println!(
                        "magnetar chat -- this CLI is synchronous, so there is no separate CLI-owned file/Git/network/tool work in flight to cancel here (see openspec proposal 'Cancellation' section)"
                    );
                }
                Err(error) => render::print_error(&error),
            }
            println!(
                "magnetar chat -- {} transcript line(s) kept CLI-side (never sent to Runtime as a whole)",
                chat.transcript().len()
            );
            println!(
                "magnetar chat -- {} command(s) recorded in CLI-owned command history (never sent to Runtime)",
                command_history.len()
            );
            return Ok(());
        }
        observer.observe(
            CliObservationKind::RuntimeRequestSubmitted,
            "chat: submitting turn",
        );
        match chat.turn(line) {
            Ok((reply, generation_observer)) => {
                render::print_generation_observations(&generation_observer);
                observer.observe(
                    CliObservationKind::StreamRendered,
                    format!(
                        "stream rendered: {} observation(s)",
                        generation_observer.observations().len()
                    ),
                );
                if render::looks_like_tool_call(&reply) {
                    println!(
                        "note: this looks like it might be a tool call; magnetar-cli does not execute tool calls in this increment"
                    );
                }
                println!("{reply}");
            }
            Err(error) => render::print_error(&error),
        }
    }

    println!(
        "magnetar chat -- {} transcript line(s) kept CLI-side (never sent to Runtime as a whole)",
        chat.transcript().len()
    );
    println!(
        "magnetar chat -- {} command(s) recorded in CLI-owned command history (never sent to Runtime)",
        command_history.len()
    );
    chat.close()
}

const AGENT_USAGE: &str = "usage: magnetar agent <model-ref> [--steps N] [--tool <program>] [--write <path>] <goal text...>";

/// `magnetar agent <model-ref> [--steps N] [--tool <program>] [--write
/// <path>] <goal text...>` (§14 "Agent Orchestration"). Parses agent-loop
/// flags and the goal text entirely in the CLI, then hands off to
/// `agent::run_agent_loop` -- see that module's doc comment for what "agent
/// planning"/"tool loop"/"workspace mutation" mean in this minimal,
/// bounded increment.
fn cmd_agent(args: &[String], observer: &mut CliObserver) -> Result<(), CliBoundaryError> {
    let mut steps = 2usize;
    let mut tool: Option<String> = None;
    let mut write: Option<String> = None;
    let mut rest = Vec::new();
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--steps" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliBoundaryError::CliCommandInvalid {
                        reason: AGENT_USAGE.into(),
                    })?;
                steps = value
                    .parse()
                    .map_err(|_| CliBoundaryError::CliCommandInvalid {
                        reason: format!("invalid --steps value '{value}'"),
                    })?;
            }
            "--tool" => tool = iter.next(),
            "--write" => write = iter.next(),
            _ => rest.push(arg),
        }
    }
    let Some((model_ref_arg, goal_words)) = rest
        .split_first()
        .map(|(first, remainder)| (first.clone(), remainder.to_vec()))
    else {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: AGENT_USAGE.into(),
        });
    };
    if goal_words.is_empty() {
        return Err(CliBoundaryError::CliPromptInputInvalid {
            reason: "agent goal text is required".into(),
        });
    }
    let model_ref = ModelRef::new(aliases::resolve_alias(&model_ref_arg))?;
    let goal = goal_words.join(" ");
    observer.observe(
        CliObservationKind::RuntimeRequestSubmitted,
        "agent: starting agent loop",
    );
    let options = agent::AgentOptions { steps, tool, write };
    agent::run_agent_loop(&model_ref, &goal, &options)?;
    Ok(())
}

fn cmd_model(args: &[String]) -> Result<(), CliBoundaryError> {
    let Some(action) = args.first() else {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: "usage: magnetar model <list|inspect|load|unload> [model-ref]".into(),
        });
    };
    match action.as_str() {
        "list" => cmd_model_list(),
        "inspect" => cmd_model_inspect(&args[1..]),
        "load" => cmd_model_load(&args[1..]),
        "unload" => cmd_model_unload(&args[1..]),
        other => Err(CliBoundaryError::CliCommandInvalid {
            reason: format!("unknown model subcommand '{other}'"),
        }),
    }
}

fn require_model_ref(args: &[String], usage: &str) -> Result<ModelRef, CliBoundaryError> {
    let Some(raw) = args.first() else {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: format!("usage: magnetar {usage} <model-ref>"),
        });
    };
    Ok(ModelRef::new(raw.as_str())?)
}

/// `magnetar model list`. This minimal CLI does not yet persist model
/// aliases across process invocations (a stated non-goal of this change --
/// see `proposal.md`'s "define model download UX" / CLI configuration
/// ownership non-goals); the registry constructed here is always empty.
/// This is intentionally honest about current scope, not a fake feature.
fn cmd_model_list() -> Result<(), CliBoundaryError> {
    let registry = ModelRegistry::new();
    println!("registered model aliases (this process only):");
    let mut any = false;
    for (reference, artifact) in registry.entries() {
        any = true;
        println!("  {reference} -> {artifact:?}");
    }
    if !any {
        println!("  (none -- magnetar-cli does not yet persist model aliases across invocations)");
    }
    Ok(())
}

/// `magnetar model inspect <model-ref>`. Calls the real
/// `ModelRegistry::resolve`. Since no alias is registered, this legitimately
/// surfaces a structured `InferenceApiError::ModelResolutionFailed` -- the
/// CLI does not fabricate a resolved model to make this command look more
/// complete than it is.
fn cmd_model_inspect(args: &[String]) -> Result<(), CliBoundaryError> {
    let model_ref = require_model_ref(args, "model inspect")?;
    let registry = ModelRegistry::new();
    let request = ModelResolutionRequest::new(model_ref.clone());
    let result = registry.resolve(&request)?;
    println!(
        "resolved model reference '{model_ref}' -> {:?}",
        result.artifact
    );
    Ok(())
}

/// Resolves whether `input` names a local model file path rather than a
/// friendly alias/literal `ModelRef` string (§23 "Resolve local paths in
/// CLI"). A conservative heuristic: `input` is treated as a local path only
/// if it contains a path separator or starts with `.` -- a bare
/// alphanumeric name like `"qwen-small"` is never misread as a path. This
/// mirrors (and is only needed because of) `ModelRef::new`'s own
/// validation, which rejects anything that resembles a filesystem path by
/// construction (see `magnetar_runtime::inference_api::validate_model_reference`)
/// -- a local model file can never be smuggled through as an opaque
/// `ModelRef`. Entirely CLI-side; never touches Runtime.
fn resolve_local_model_path(input: &str) -> Option<std::path::PathBuf> {
    let looks_like_path = input.contains('/') || input.contains('\\') || input.starts_with('.');
    if !looks_like_path {
        return None;
    }
    Some(std::path::PathBuf::from(input))
}

/// `magnetar model load <model-ref>` / `magnetar model load --file <path>`.
/// Per "CLI Model Resolution Does Not Bypass Loading"
/// (`specs/model-loading/spec.md`), this first attempts the real
/// `ModelRegistry` resolution and prints its outcome either way -- since
/// this CLI increment has no persistent alias/artifact storage, resolution
/// legitimately fails for every reference today, which is printed rather
/// than propagated so the Runtime model loading call below (this command's
/// actual point, §6/§42 "Call Runtime model loading") still runs.
///
/// That call goes through the real `magnetar_runtime::load_model` against a
/// CLI-constructed fixture manifest for `model_ref`
/// (`pipeline::fixture_model_manifest`) and a real (not fabricated) trust
/// evaluation from an empty `ModelTrustStore` (no CLI-side trust policy is
/// configured yet). Since no digest/publisher is pre-trusted, this
/// legitimately and honestly fails with a structured "model artifact trust
/// status is unknown" Runtime error rather than fabricating a successful
/// load -- proving the CLI calls real Runtime model loading without
/// bypassing its trust check (§45/§46).
fn cmd_model_load(args: &[String]) -> Result<(), CliBoundaryError> {
    if args.first().map(String::as_str) == Some("--file") {
        return cmd_model_load_local_file(&args[1..]);
    }
    // A friendlier, CLI-side-only error when the caller typed a local path
    // as a bare model reference: `ModelRef::new` would already reject this
    // (it rejects anything resembling a filesystem path by construction),
    // but pointing the caller at `--file` directly is clearer than
    // surfacing that generic rejection.
    if let Some(first) = args.first()
        && resolve_local_model_path(first).is_some()
    {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: format!(
                "'{first}' looks like a local path; use 'magnetar model load --file {first}' instead"
            ),
        });
    }

    let model_ref = require_model_ref(args, "model load")?;
    let registry = ModelRegistry::new();
    match registry.resolve(&ModelResolutionRequest::new(model_ref.clone())) {
        Ok(resolved) => println!(
            "model reference '{model_ref}' resolved to artifact {:?} via CLI alias registry",
            resolved.artifact
        ),
        Err(error) => println!(
            "model reference '{model_ref}' did not resolve via CLI alias registry: {error}"
        ),
    }

    let manifest = pipeline::fixture_model_manifest(model_ref.as_str());
    run_load_model(&manifest)?;
    println!("model instance for '{model_ref}' loaded");
    Ok(())
}

/// `magnetar model load --file <path>` (§23 "Local Model Files"). Path
/// resolution -- a single `std::fs::canonicalize` existence check, never a
/// directory scan -- happens here in the CLI (§150). Runtime only ever
/// receives the resulting `ModelArtifactSource::LocalPath` as inert
/// manifest metadata (§151/§152), and goes through the exact same
/// trust/artifact validation as [`cmd_model_load`]'s `<model-ref>` path
/// (§153/§154) -- `run_load_model` is shared, not duplicated.
fn cmd_model_load_local_file(args: &[String]) -> Result<(), CliBoundaryError> {
    let Some(raw_path) = args.first() else {
        return Err(CliBoundaryError::CliCommandInvalid {
            reason: "usage: magnetar model load --file <path>".into(),
        });
    };
    let canonical =
        std::fs::canonicalize(raw_path).map_err(|error| CliBoundaryError::CliFileReadFailed {
            reason: format!("local model path '{raw_path}' is not accessible: {error}"),
        })?;
    if !canonical.is_file() {
        return Err(CliBoundaryError::CliFileReadFailed {
            reason: format!("local model path '{}' is not a file", canonical.display()),
        });
    }

    let mut manifest = pipeline::fixture_model_manifest(&canonical.to_string_lossy());
    manifest.source = Some(ModelArtifactSource::LocalPath(canonical.clone()));
    run_load_model(&manifest)?;
    println!(
        "model instance for local file '{}' loaded",
        canonical.display()
    );
    Ok(())
}

/// Shared Runtime model loading call used by both [`cmd_model_load`] and
/// [`cmd_model_load_local_file`]: builds a fresh `ModelLoadingCoordinator`
/// and `MemoryManager`, evaluates real (not fabricated) trust from an empty
/// `ModelTrustStore`, and calls the real `magnetar_runtime::load_model`.
fn run_load_model(manifest: &magnetar_runtime::ModelManifest) -> Result<(), CliBoundaryError> {
    let trust = ModelTrustStore::default().evaluate(manifest);
    let mut coordinator = ModelLoadingCoordinator::new();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let core = ModelLoadingRequest::new(
        ModelLoadingRequestId::new(format!("cli-load-{}", manifest.id.digest.value)),
        manifest.id.clone(),
    );
    load_model(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(core),
        manifest,
        &trust,
    )?;
    Ok(())
}

/// `magnetar model unload <model-ref>`. Calls the real
/// `unload_model_instance` Runtime function against a fresh Runtime with no
/// instances. Since no instance was ever created, this legitimately
/// surfaces a structured Runtime error -- the CLI does not fabricate a
/// successful unload.
fn cmd_model_unload(args: &[String]) -> Result<(), CliBoundaryError> {
    let model_ref = require_model_ref(args, "model unload")?;
    let mut runtime =
        Runtime::builder()
            .build()
            .map_err(|error| CliBoundaryError::CliRuntimeUnavailable {
                reason: error.to_string(),
            })?;
    let instance = ModelInstanceId::new(model_ref.as_str()).map_err(InferenceApiError::from)?;
    unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::RejectActiveUse,
    )?;
    println!("model instance for '{model_ref}' unloaded");
    Ok(())
}

/// Prints the CLI's current working directory to stderr as CLI-only
/// diagnostic enrichment (§20 "Keep workspace path context CLI-side" /
/// "Optionally enrich with CLI command metadata"). `cmd_providers` and
/// `cmd_devices` never pass this path to any Runtime call -- `Runtime::providers`/
/// `Runtime::devices` take no path argument at all, so there is no code
/// path by which this could reach Runtime even by accident.
fn print_cli_workspace_diagnostic_context() {
    match std::env::current_dir() {
        Ok(cwd) => eprintln!(
            "[cli] workspace path: {} (not sent to Runtime)",
            cwd.display()
        ),
        Err(error) => eprintln!("[cli] workspace path unavailable: {error}"),
    }
}

/// `magnetar providers`. Builds a Runtime with the built-in Reference CPU
/// Provider registered and prints only redacted `ProviderMetadata` fields.
fn cmd_providers() -> Result<(), CliBoundaryError> {
    print_cli_workspace_diagnostic_context();
    let runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .build()
        .map_err(|error| CliBoundaryError::CliRuntimeUnavailable {
            reason: error.to_string(),
        })?;
    for name in runtime.providers().provider_names() {
        if let Some(provider) = runtime.providers().provider(name) {
            render::print_provider(&provider.metadata());
        }
    }
    Ok(())
}

/// `magnetar devices`. Builds a Runtime with the built-in Reference CPU
/// Provider registered and prints only redacted `DeviceMetadata` fields.
fn cmd_devices() -> Result<(), CliBoundaryError> {
    print_cli_workspace_diagnostic_context();
    let runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .build()
        .map_err(|error| CliBoundaryError::CliRuntimeUnavailable {
            reason: error.to_string(),
        })?;
    let mut any = false;
    for device in runtime.devices() {
        any = true;
        render::print_device(device.metadata());
    }
    if !any {
        println!("no devices registered");
    }
    Ok(())
}

/// `magnetar version` / `magnetar --version` / `magnetar -V`. Builds and
/// prints the release binary version report defined by
/// `magnetar_runtime::release_packaging::build_release_binary_version_report`
/// (see `openspec/changes/define-release-packaging-and-versioning-policy`).
/// `magnetar-cli`'s own crate version (`env!("CARGO_PKG_VERSION")`) is the
/// binary version; the build profile is derived from `debug_assertions`
/// since this CLI has no separate release-automation step yet; no feature
/// flags are compiled into this binary today, so the enabled list is empty
/// rather than fabricated; and the commit hash is only included when
/// provided at build time via `MAGNETAR_COMMIT_HASH` (unset by default --
/// honestly `None`, not a fabricated value).
fn cmd_version() -> Result<(), CliBoundaryError> {
    let binary_version = parse_release_version(env!("CARGO_PKG_VERSION"));
    let build_profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let commit_hash = option_env!("MAGNETAR_COMMIT_HASH").map(str::to_string);
    let report =
        build_release_binary_version_report(binary_version, Vec::new(), build_profile, commit_hash);
    render::print_version(&report);
    Ok(())
}

/// Parses `env!("CARGO_PKG_VERSION")` (always a valid `major.minor.patch`
/// semantic version, enforced by Cargo) into a [`ReleaseVersion`].
fn parse_release_version(raw: &str) -> ReleaseVersion {
    let mut parts = raw.split('.').map(|part| part.parse::<u64>().unwrap_or(0));
    ReleaseVersion::new(
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// `magnetar sessions`. This minimal CLI does not persist sessions across
/// process invocations -- sessions are created per `run`/`chat` invocation
/// and closed on exit. Stated honestly rather than faking a session list.
fn cmd_sessions() -> Result<(), CliBoundaryError> {
    println!(
        "magnetar-cli does not persist sessions across process invocations: sessions are \
         created per `run`/`chat` invocation and closed on exit."
    );
    Ok(())
}

/// `magnetar serve`. HTTP/server API is out of scope for this change (see
/// `proposal.md` Non-Goals: "define HTTP server API"). Rather than
/// pretending to start a server, this command fails structurally.
fn cmd_serve(args: &[String]) -> Result<(), CliBoundaryError> {
    // Binding a real HTTP listener is out of scope for this change (see
    // `proposal.md` Non-Goals: "define HTTP server API"), so bare `magnetar
    // serve` still fails structurally. `--demo-request` instead
    // demonstrates -- without a socket -- exactly what a served request
    // would do: call `serve::handle_serve_generation_request`, the real
    // Runtime Inference API entry point a future HTTP handler would also
    // call (§24 "Ensure serve mode calls Runtime Inference API").
    if args.first().map(String::as_str) == Some("--demo-request") {
        let Some((model_ref_arg, prompt_words)) = args[1..]
            .split_first()
            .map(|(first, remainder)| (first.clone(), remainder.to_vec()))
        else {
            return Err(CliBoundaryError::CliCommandInvalid {
                reason: "usage: magnetar serve --demo-request <model-ref> <prompt text...>".into(),
            });
        };
        if prompt_words.is_empty() {
            return Err(CliBoundaryError::CliPromptInputInvalid {
                reason: "prompt text is required".into(),
            });
        }
        let model_ref = ModelRef::new(aliases::resolve_alias(&model_ref_arg))?;
        let prompt = prompt_words.join(" ");
        let (text, generation_observer) =
            serve::handle_serve_generation_request(&model_ref, &prompt)?;
        render::print_generation_observations(&generation_observer);
        println!("{text}");
        return Ok(());
    }
    Err(CliBoundaryError::CliCommandInvalid {
        reason: "serve mode (HTTP server API) is out of scope for this change; see \
                 openspec/changes/define-magnetar-cli-inference-boundary/proposal.md Non-Goals"
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_run_flags_extracts_file_git_and_secret_flags_from_positional_args() {
        let args = vec![
            "model-ref".to_string(),
            "--file".to_string(),
            "notes.txt".to_string(),
            "--git-diff".to_string(),
            "--env-secret".to_string(),
            "MY_SECRET".to_string(),
            "prompt".to_string(),
            "words".to_string(),
        ];
        let (flags, rest) = parse_run_flags(&args);
        assert_eq!(flags.file.as_deref(), Some("notes.txt"));
        assert!(flags.git_diff);
        assert_eq!(flags.env_secret.as_deref(), Some("MY_SECRET"));
        assert_eq!(
            rest,
            vec![
                "model-ref".to_string(),
                "prompt".to_string(),
                "words".to_string()
            ]
        );
    }

    #[test]
    fn parse_run_flags_with_no_flags_returns_all_args_as_positional() {
        let args = vec!["model-ref".to_string(), "hello".to_string()];
        let (flags, rest) = parse_run_flags(&args);
        assert!(flags.file.is_none());
        assert!(!flags.git_diff);
        assert!(flags.env_secret.is_none());
        assert_eq!(rest, args);
    }

    /// §8/§29 "Test CLI file read stays in CLI": a missing/unreadable path
    /// produces a structured `CliFileReadFailed`, never a panic.
    #[test]
    fn read_file_context_nonexistent_path_is_cli_file_read_failed() {
        let error = read_file_context("this/path/definitely/does/not/exist/magnetar-cli-test.txt")
            .unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliFileReadFailed { .. }));
    }

    /// §8/§29 "Test CLI file read stays in CLI": the file's content reaches
    /// the prompt string via `assemble_prompt` -- a plain `String`, never a
    /// path or filesystem handle.
    #[test]
    fn file_content_reaches_the_assembled_prompt_string() {
        let marker = "MAGNETAR_CLI_FILE_CONTEXT_MARKER_9d21";
        let path = std::env::temp_dir().join(format!(
            "magnetar-cli-file-context-test-{}-{}.txt",
            std::process::id(),
            marker
        ));
        std::fs::write(&path, format!("some file content with {marker} inside")).unwrap();

        let content = read_file_context(path.to_str().unwrap()).unwrap();
        assert!(content.contains(marker));

        let prompt = assemble_prompt("base prompt text", &[("file context", &content)]);
        assert!(prompt.contains(marker));
        assert!(prompt.contains("base prompt text"));

        std::fs::remove_file(&path).unwrap();
    }

    /// Proves the assembled prompt (built from file content) flows into
    /// `pipeline::one_shot` as a plain `&str` -- the function signature
    /// itself only accepts `prompt: &str`, so there is no way to hand
    /// Runtime a path here even by accident.
    #[test]
    fn assembled_file_context_prompt_flows_through_pipeline_as_plain_text() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let prompt = assemble_prompt("hi", &[("file context", "x")]);
        let (text, observer) = pipeline::one_shot(&model_ref, &prompt).unwrap();
        assert!(!text.is_empty());
        assert!(!observer.observations().is_empty());
    }

    /// §9/§29 "Test Git stays in CLI": Git access happens through a fixed,
    /// hardcoded `Command` (no interpretation of arbitrary text as a shell
    /// command) and never panics -- either it succeeds with Git's stdout as
    /// a plain `String`, or it fails with a structured `CliGitFailed`
    /// (e.g. `git` not on PATH in some environments).
    #[test]
    fn collect_git_diff_context_never_panics_and_returns_a_structured_result() {
        match collect_git_diff_context() {
            Ok(diff) => {
                // Whatever Git returns is already a plain String -- proven
                // by the function's return type -- and can be folded into
                // the prompt the same way file context is.
                let prompt = assemble_prompt("base", &[("git diff", &diff)]);
                assert!(prompt.contains("base"));
            }
            Err(error) => assert!(matches!(error, CliBoundaryError::CliGitFailed { .. })),
        }
    }

    #[test]
    fn assemble_prompt_includes_file_and_git_sections_and_base_prompt() {
        let prompt = assemble_prompt(
            "do the thing",
            &[
                ("file context", "file-section-content"),
                ("git diff", "git-section-content"),
            ],
        );
        assert!(prompt.contains("file-section-content"));
        assert!(prompt.contains("git-section-content"));
        assert!(prompt.contains("do the thing"));
    }

    #[test]
    fn assemble_prompt_with_no_context_is_just_the_base_prompt() {
        let prompt = assemble_prompt("only base", &[]);
        assert_eq!(prompt, "only base");
    }

    /// §15/§29: workspace/network/tool context sections are assembled the
    /// same way file/Git context is -- labeled sections before the base
    /// prompt.
    #[test]
    fn assemble_prompt_includes_workspace_network_and_tool_sections() {
        let prompt = assemble_prompt(
            "goal",
            &[
                ("workspace context", "cwd: /tmp\nentries: a, b"),
                ("network context", "fetched body"),
                ("tool output", "tool stdout"),
            ],
        );
        assert!(prompt.contains("cwd: /tmp"));
        assert!(prompt.contains("fetched body"));
        assert!(prompt.contains("tool stdout"));
        assert!(prompt.contains("goal"));
    }

    /// §23/§29 "Add local model file tests": a bare alias-like string is
    /// never mistaken for a local path, but any string containing a path
    /// separator or a leading `.` is.
    #[test]
    fn resolve_local_model_path_only_matches_path_like_input() {
        assert!(resolve_local_model_path("qwen-small").is_none());
        assert!(resolve_local_model_path("cpu-default").is_none());
        assert!(resolve_local_model_path("./model.bin").is_some());
        assert!(resolve_local_model_path("models/model.bin").is_some());
        assert!(resolve_local_model_path("models\\model.bin").is_some());
    }

    /// §23/§29: local model loading resolves the path in the CLI (a single
    /// existence check) and fails with a structured `CliFileReadFailed` for
    /// a path that does not exist -- never a panic, never a directory scan.
    #[test]
    fn cmd_model_load_local_file_nonexistent_path_is_cli_file_read_failed() {
        let error = cmd_model_load_local_file(&[
            "this/path/definitely/does/not/exist/magnetar-cli-test-model.bin".to_string(),
        ])
        .unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliFileReadFailed { .. }));
    }

    /// §23/§29: a real local file resolves, is attached to the manifest as
    /// an authorized `ModelArtifactSource::LocalPath`, and the real
    /// `load_model` call still runs (and legitimately fails on trust, same
    /// as the `<model-ref>` path -- see `cmd_model_load`'s doc comment).
    #[test]
    fn cmd_model_load_local_file_existing_file_reaches_load_model_and_fails_on_trust() {
        let path = std::env::temp_dir().join(format!(
            "magnetar-cli-local-model-test-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"not a real model artifact").unwrap();

        let error = cmd_model_load_local_file(&[path.to_str().unwrap().to_string()]).unwrap_err();
        assert!(error.runtime_category().is_some());

        std::fs::remove_file(&path).unwrap();
    }

    /// §42/§29: `magnetar model load <model-ref>` calls the real
    /// `load_model` and legitimately fails with a structured Runtime error
    /// (unknown trust) rather than fabricating a successful load.
    #[test]
    fn cmd_model_load_calls_real_load_model_and_fails_on_trust() {
        let error = cmd_model_load(&["qwen-test".to_string()]).unwrap_err();
        assert!(error.runtime_category().is_some());
    }

    #[test]
    fn cmd_model_load_with_a_path_like_argument_points_at_the_file_flag() {
        let error = cmd_model_load(&["./local-model.bin".to_string()]).unwrap_err();
        match error {
            CliBoundaryError::CliCommandInvalid { reason } => {
                assert!(reason.contains("--file"));
            }
            other => panic!("expected CliCommandInvalid, got {other:?}"),
        }
    }

    /// §24/§29 "Add serve boundary tests": bare `magnetar serve` still
    /// fails structurally (HTTP server API is out of scope).
    #[test]
    fn cmd_serve_without_args_is_out_of_scope() {
        let error = cmd_serve(&[]).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliCommandInvalid { .. }));
    }

    /// §24/§29: `--demo-request` calls the real Runtime Inference API
    /// pipeline (the same one `magnetar run` uses), proving serve mode's
    /// request handling does not bypass Runtime validation.
    #[test]
    fn cmd_serve_demo_request_calls_the_runtime_inference_api() {
        cmd_serve(&[
            "--demo-request".to_string(),
            "qwen-test".to_string(),
            "hi".to_string(),
        ])
        .unwrap();
    }
}
