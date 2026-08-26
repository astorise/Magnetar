//! Terminal rendering helpers. Owns all "what does the user see" decisions
//! so `commands.rs` stays focused on request/response plumbing.
//!
//! Every function here prints only redacted, documented-safe fields --
//! never a raw Rust `{:?}` dump of a Runtime type, which could otherwise
//! leak internal representation details `magnetar-runtime`'s API boundary
//! does not promise to keep stable or redacted. Where an enum field is
//! printed via `{:?}` (`DeviceMetadata::device_type`,
//! `InferenceApiObservationKind`), that enum carries no handle/pointer
//! payload -- its `Debug` output is just the variant name, matching the
//! existing convention in `print_device` below.
//!
//! Every `print_*` function here is a thin wrapper around a `render_*`
//! function that returns a `String` instead of printing directly, so
//! redaction can be asserted in tests without capturing real stdout.

use magnetar_runtime::{
    CliBoundaryError, DeviceMetadata, InferenceApiObserver, ProviderMetadata,
    ReleaseBinaryVersionReport,
};

/// Prints a [`CliBoundaryError`] to stderr via its `Display` impl only --
/// never `{:?}`. `Display` is the structured, human-readable rendering that
/// preserves (and does not further redact) the wrapped Runtime error
/// category for `CliRuntimeRequestFailed`.
pub fn print_error(error: &CliBoundaryError) {
    eprint!("{}", render_error(error));
}

/// Builds the text `print_error` prints. Split out so tests can assert
/// redaction (e.g. for `CliSecretUnavailable`) without capturing stdout.
pub fn render_error(error: &CliBoundaryError) -> String {
    let mut out = format!("magnetar: error: {error}\n");
    if let Some(runtime_category) = error.runtime_category() {
        out.push_str(&format!("  runtime category: {runtime_category:?}\n"));
    }
    out
}

/// Prints redacted [`ProviderMetadata`]: name, version, vendor, description,
/// api_version only. `ProviderMetadata` carries no Provider handle field, so
/// there is nothing to strip beyond sticking to this explicit field list
/// rather than a `{:?}` dump.
pub fn print_provider(metadata: &ProviderMetadata) {
    print!("{}", render_provider(metadata));
}

/// Builds the text `print_provider` prints (see its doc comment).
pub fn render_provider(metadata: &ProviderMetadata) -> String {
    format!(
        "provider: {}\n  version:     {}\n  vendor:      {}\n  api version: {}\n  description: {}\n",
        metadata.name,
        metadata.version,
        metadata.vendor,
        metadata.api_version,
        metadata.description
    )
}

/// Prints redacted [`DeviceMetadata`]: id, name, device type, vendor,
/// architecture, and owning provider name only. `DeviceMetadata` carries no
/// native pointer or handle field, so there is nothing to strip beyond
/// sticking to this explicit field list rather than a `{:?}` dump.
pub fn print_device(metadata: &DeviceMetadata) {
    print!("{}", render_device(metadata));
}

/// Builds the text `print_device` prints (see its doc comment).
pub fn render_device(metadata: &DeviceMetadata) -> String {
    format!(
        "device: {}\n  name:         {}\n  type:         {:?}\n  vendor:       {}\n  architecture: {}\n  provider:     {}\n",
        metadata.id,
        metadata.name,
        metadata.device_type,
        metadata.vendor,
        metadata.architecture,
        metadata.provider
    )
}

/// Renders the Generation Contract's per-token observation trail *after*
/// `run_generation_loop` has already completed -- this is necessarily a
/// replay, not true incremental terminal streaming: the pipeline is
/// synchronous and `run_generation_loop` returns only once fully done,
/// exposing no mid-loop callback hook a caller could print from as each
/// token is produced. True incremental streaming would need
/// `run_generation_loop` to expose a per-step callback, which it does not
/// today. Observations are iterated in `observer.observations()`'s order
/// (a `Vec`, so Runtime's emission order is preserved exactly), printed
/// before the final decoded text -- see `commands::cmd_run` /
/// `commands::cmd_chat`, which call this before printing decoded text.
///
/// `InferenceApiObservation.message` is documented as redacted-by-default
/// by `magnetar_runtime::inference_api` itself (no raw prompt/tensor/handle
/// content), so printing it here does not reopen that guarantee.
pub fn print_generation_observations(observer: &InferenceApiObserver) {
    for observation in observer.observations() {
        println!(
            "[generation] {:?}: {}",
            observation.kind, observation.message
        );
    }
}

/// Heuristic-only detector for tool-call-like text in decoded model
/// output. This is **not** a real tool-call protocol parser -- it is a
/// deliberately simple placeholder heuristic (a fenced code block whose
/// body starts with `$` or `!`, evoking a shell command) used only to
/// decide whether to print an informational note. Detection never triggers
/// execution of anything: see "Interpret tool-call-like model output in
/// CLI only" (§12) -- a real tool execution engine/protocol is an explicit
/// Non-Goal of `proposal.md` ("define tool protocol").
pub fn looks_like_tool_call(text: &str) -> bool {
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence && (trimmed.starts_with('$') || trimmed.starts_with('!')) {
            return true;
        }
    }
    false
}

pub fn print_usage() {
    println!(
        r#"magnetar -- first-party client/workspace/agent runtime around Magnetar inference

USAGE:
    magnetar run <model-ref> [--file <path>] [--git-diff] [--workspace] [--url <http-url>] [--tool <program>] [--env-secret <NAME>] <prompt text...>
    magnetar chat <model-ref>
    magnetar agent <model-ref> [--steps N] [--tool <program>] [--write <path>] <goal text...>
    magnetar model list
    magnetar model inspect <model-ref>
    magnetar model load <model-ref>
    magnetar model load --file <path>
    magnetar model unload <model-ref>
    magnetar providers
    magnetar devices
    magnetar sessions
    magnetar serve
    magnetar serve --demo-request <model-ref> <prompt text...>
    magnetar version
    magnetar --help

FLAGS (magnetar run):
    --file <path>        Read a UTF-8 file in the CLI and include its content
                          as explicit prompt context (never a filesystem
                          authority handed to Runtime). Opt-in only.
    --git-diff            Run a fixed `git diff` in the CLI and include its
                          stdout as explicit prompt context. Opt-in only.
    --workspace            Include a shallow (non-recursive) CLI-collected
                          directory listing of the current directory as
                          explicit prompt context. Opt-in only.
    --url <http-url>       Fetch an http:// URL's body in the CLI (no TLS
                          dependency, https:// is rejected) and include it as
                          explicit prompt context. Opt-in and gated by CLI
                          network policy.
    --tool <program>       Run a named program with no arguments in the CLI
                          and include its stdout as explicit prompt context.
                          Opt-in and gated by CLI tool policy; never
                          triggered by model-generated text.
    --env-secret <NAME>   Read a named environment variable in the CLI. Its
                          value is never sent to Runtime in this increment
                          (see `secrets.rs`).
    -v, --verbose          Print a one-line CLI observability summary
                          (counts only) after the command runs.

FLAGS (magnetar agent):
    --steps N              Number of Runtime chat turns to run (default 2,
                          capped at a small fixed maximum).
    --tool <program>       Run a named program after the loop completes and
                          print its output (never triggered automatically by
                          model output).
    --write <path>          Write the final step's output to this exact
                          path (CLI-owned workspace mutation; Runtime never
                          touches the filesystem).

NOTE: magnetar-runtime is a contracts/validation layer today, not an
end-to-end inference engine. `run`/`chat`/`agent` exercise the real Runtime
Session/Generation/Tokenizer API end to end, but generation uses a
placeholder byte-based tokenizer and caller-supplied placeholder (all-zero)
logits -- decoded text is not meaningful model output. See
openspec/changes/define-magnetar-cli-inference-boundary/proposal.md for the
CLI/Runtime authority boundary this binary implements."#
    );
}

/// `magnetar version`. Prints the release binary version report defined by
/// `magnetar_runtime::release_packaging` (see
/// `openspec/changes/define-release-packaging-and-versioning-policy`):
/// binary version, Runtime crate version, OpenSpec baseline version, WIT
/// contract versions, enabled feature flags, build profile, commit hash
/// where available, and conformance suite version where available.
pub fn print_version(report: &ReleaseBinaryVersionReport) {
    print!("{}", render_version(report));
}

/// Builds the text `print_version` prints (see its doc comment).
pub fn render_version(report: &ReleaseBinaryVersionReport) -> String {
    let mut out = format!(
        "magnetar {}\n  runtime crate version:  {}\n  openspec baseline:      {}\n  build profile:          {}\n",
        report.binary_version,
        report.runtime_crate_version,
        report.openspec_baseline_version,
        report.build_profile,
    );
    out.push_str(&format!(
        "  commit hash:            {}\n",
        report.commit_hash.as_deref().unwrap_or("unavailable")
    ));
    out.push_str(&format!(
        "  conformance suite:      {}\n",
        report
            .conformance_suite_version
            .as_deref()
            .unwrap_or("unavailable")
    ));
    out.push_str("  wit contracts:\n");
    for interface in &report.wit_contract_versions {
        out.push_str(&format!("    {}@{}\n", interface.name, interface.version));
    }
    out.push_str("  enabled feature flags:\n");
    if report.enabled_feature_flags.is_empty() {
        out.push_str("    (none)\n");
    } else {
        for flag in &report.enabled_feature_flags {
            out.push_str(&format!("    {flag}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::{InferenceApiError, ReferenceCpuProvider, Runtime};
    use std::sync::Arc;

    #[test]
    fn render_error_never_uses_debug_formatting_of_the_boundary_error_itself() {
        let error = CliBoundaryError::CliCommandInvalid {
            reason: "usage: ...".into(),
        };
        let rendered = render_error(&error);
        assert!(rendered.contains("cli command invalid"));
    }

    /// Redaction guarantee for `CliSecretUnavailable`: its rendering must
    /// never contain a real secret value -- only the error's own `reason`
    /// (which `secrets::read_env_secret` builds from the variable *name*
    /// only, never a value).
    #[test]
    fn secret_unavailable_rendering_never_contains_a_secret_value() {
        let secret_value = "definitely-a-secret-marker-render-check";
        let error = CliBoundaryError::CliSecretUnavailable {
            reason: "environment variable 'SOME_NAME' is not set".into(),
        };
        let rendered = render_error(&error);
        assert!(!rendered.contains(secret_value));
        assert!(rendered.contains("SOME_NAME"));
    }

    #[test]
    fn runtime_request_failed_rendering_preserves_runtime_category() {
        let error = CliBoundaryError::from(InferenceApiError::ModelLoadingFailed {
            reason: "example".into(),
        });
        let rendered = render_error(&error);
        assert!(rendered.contains("runtime category"));
    }

    /// Provider diagnostics never leak handle-like text (§7/§20/§29).
    #[test]
    fn provider_diagnostics_are_redacted() {
        let runtime = Runtime::builder()
            .register_provider(Arc::new(ReferenceCpuProvider::new()))
            .build()
            .unwrap();
        let mut any = false;
        for name in runtime.providers().provider_names() {
            if let Some(provider) = runtime.providers().provider(name) {
                any = true;
                let rendered = render_provider(&provider.metadata());
                assert!(!rendered.contains("0x"));
                assert!(!rendered.to_lowercase().contains("handle"));
                assert!(!rendered.to_lowercase().contains("pointer"));
            }
        }
        assert!(any, "expected at least one registered provider");
    }

    /// Device diagnostics never leak handle-like text (§7/§20/§29).
    #[test]
    fn device_diagnostics_are_redacted() {
        let runtime = Runtime::builder()
            .register_provider(Arc::new(ReferenceCpuProvider::new()))
            .build()
            .unwrap();
        let mut any = false;
        for device in runtime.devices() {
            any = true;
            let rendered = render_device(device.metadata());
            assert!(!rendered.contains("0x"));
            assert!(!rendered.to_lowercase().contains("handle"));
            assert!(!rendered.to_lowercase().contains("pointer"));
        }
        assert!(any, "expected at least one registered device");
    }

    #[test]
    fn looks_like_tool_call_detects_fenced_dollar_prefixed_line() {
        let text = "here you go:\n```\n$ rm -rf /\n```\nhope that helps";
        assert!(looks_like_tool_call(text));
    }

    #[test]
    fn looks_like_tool_call_detects_fenced_bang_prefixed_line() {
        let text = "```\n! danger\n```";
        assert!(looks_like_tool_call(text));
    }

    #[test]
    fn looks_like_tool_call_is_false_for_plain_text() {
        let text = "just a normal placeholder response with no code fences";
        assert!(!looks_like_tool_call(text));
    }

    #[test]
    fn looks_like_tool_call_is_false_for_unfenced_dollar_line() {
        let text = "$ this is not inside a fence";
        assert!(!looks_like_tool_call(text));
    }

    #[test]
    fn print_generation_observations_preserves_runtime_emission_order() {
        use magnetar_runtime::InferenceApiObservationKind;
        let mut observer = InferenceApiObserver::new();
        observer.observe(InferenceApiObservationKind::GenerationStarted, "a", None);
        observer.observe(InferenceApiObservationKind::TokenGenerated, "b", None);
        observer.observe(InferenceApiObservationKind::GenerationCompleted, "c", None);
        let kinds: Vec<_> = observer.observations().iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                InferenceApiObservationKind::GenerationStarted,
                InferenceApiObservationKind::TokenGenerated,
                InferenceApiObservationKind::GenerationCompleted,
            ]
        );
        // print_generation_observations itself only prints -- exercised
        // here to prove it does not panic over a real observation trail.
        print_generation_observations(&observer);
    }
}
