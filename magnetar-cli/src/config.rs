//! CLI-owned configuration (§21 "Configuration" in the change proposal).
//!
//! [`CliConfig`] is a real Rust type demonstrating the structural boundary
//! between CLI-side defaults/formatting choices and Runtime-owned inference
//! *policy* (`magnetar_runtime::SessionPolicy`, `GenerationParameters`,
//! etc.). Runtime policy is never set from this struct directly -- it is
//! only ever consulted by CLI code to decide what to pass to the normal
//! Runtime API request types (e.g. `pipeline::session_creation_request`
//! still builds `SessionPolicy::default()` and
//! `GenerationParameters::greedy()` independently; this type does not reach
//! into that construction).
//!
//! This is intentionally not a persistent config file or a new
//! argument-parsing system (both out of scope -- see `proposal.md`
//! Non-Goals "define model download UX" / this change's stated minimal,
//! zero-new-dependency increment): [`CliConfig::default`] always reflects
//! today's actual in-process defaults, and the only wiring is
//! [`resolve_model_ref_arg`], called from `commands::cmd_run` /
//! `commands::cmd_chat`.

use crate::{network, pipeline, process};

/// CLI-owned output formatting choice. Deliberately just an enum with no
/// formatting engine behind it in this increment -- it exists to give
/// "Keep output formatting in CLI" a real type to point at, not to
/// implement a second output pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputFormat {
    #[default]
    Text,
    /// Reserved for future wiring (see module doc comment: this increment
    /// deliberately does not build a second output pipeline). Kept to
    /// demonstrate the type is a real, extensible ownership point rather
    /// than a single hardcoded choice.
    #[allow(dead_code)]
    Json,
}

/// CLI-side defaults/formatting configuration.
///
/// This type only owns CLI-side defaults/formatting choices. Runtime
/// inference *policy* (`SessionPolicy`, `GenerationParameters`, and
/// friends) remains Runtime-owned and is never set from this struct
/// directly -- any Runtime policy value the CLI needs still goes through
/// the normal Runtime API request types (`SessionCreationRequest`,
/// `GenerationRequest`, ...), never through a field on `CliConfig`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliConfig {
    /// Falls back to a positional model-ref argument when the user gave
    /// none. `None` today because this CLI increment has no persistent
    /// configuration storage (see module doc comment) -- constructing a
    /// `CliConfig` with `Some(..)` is exercised in tests only.
    pub default_model_alias: Option<String>,
    /// Mirrors `pipeline::DEFAULT_MAX_NEW_TOKENS`, the CLI-side generation
    /// parameter profile default used by `pipeline::one_shot` /
    /// `pipeline::ChatSession::turn`.
    pub default_max_new_tokens: usize,
    pub output_format: OutputFormat,
    /// CLI-owned tool/process execution policy (§13/§21 "Keep tool policy
    /// in CLI"). Consulted by `commands::cmd_run`/`commands::cmd_agent`
    /// alongside their `--tool` flag: a flag alone is not sufficient, this
    /// policy must also allow it. Defaults to `AllowExplicit` so today's
    /// flag-gated behavior is unchanged out of the box (this CLI increment
    /// has no persistent configuration storage -- see module doc comment --
    /// so there is no way for a user to have actually configured a
    /// stricter policy yet); a future config loader can set this to `Deny`
    /// to disable `--tool` regardless of the flag.
    pub tool_policy: process::ProcessPolicy,
    /// CLI-owned network access policy (§10/§21 "Keep network policy in
    /// CLI"), the network sibling of `tool_policy` above -- same default
    /// and same reasoning.
    pub network_policy: network::NetworkPolicy,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_model_alias: None,
            default_max_new_tokens: pipeline::DEFAULT_MAX_NEW_TOKENS,
            output_format: OutputFormat::default(),
            tool_policy: process::ProcessPolicy::AllowExplicit,
            network_policy: network::NetworkPolicy::AllowExplicit,
        }
    }
}

/// Resolves the model-reference-or-alias argument for `run`/`chat`: the
/// first positional argument if present, else `config.default_model_alias`.
/// Pure CLI-side argument resolution -- never calls into Runtime. This is
/// the concrete wiring behind "Keep default model alias in CLI" (§21).
pub fn resolve_model_ref_arg<'a>(
    positional_args: &'a [String],
    config: &'a CliConfig,
) -> Option<&'a str> {
    positional_args
        .first()
        .map(String::as_str)
        .or(config.default_model_alias.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_todays_actual_pipeline_default() {
        let config = CliConfig::default();
        assert_eq!(
            config.default_max_new_tokens,
            pipeline::DEFAULT_MAX_NEW_TOKENS
        );
        assert_eq!(config.default_model_alias, None);
        assert_eq!(config.output_format, OutputFormat::Text);
        assert_eq!(config.tool_policy, process::ProcessPolicy::AllowExplicit);
        assert_eq!(config.network_policy, network::NetworkPolicy::AllowExplicit);
    }

    #[test]
    fn resolve_model_ref_arg_prefers_positional_argument_over_config_default() {
        let config = CliConfig {
            default_model_alias: Some("configured-default".to_string()),
            ..CliConfig::default()
        };
        let args = vec!["explicit-ref".to_string()];
        assert_eq!(resolve_model_ref_arg(&args, &config), Some("explicit-ref"));
    }

    #[test]
    fn resolve_model_ref_arg_falls_back_to_config_default_when_no_positional_argument() {
        let config = CliConfig {
            default_model_alias: Some("configured-default".to_string()),
            ..CliConfig::default()
        };
        let args: Vec<String> = Vec::new();
        assert_eq!(
            resolve_model_ref_arg(&args, &config),
            Some("configured-default")
        );
    }

    #[test]
    fn resolve_model_ref_arg_is_none_when_neither_is_available() {
        let config = CliConfig::default();
        let args: Vec<String> = Vec::new();
        assert_eq!(resolve_model_ref_arg(&args, &config), None);
    }
}
