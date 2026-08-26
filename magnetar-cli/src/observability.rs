//! CLI-side observability (§26 "Observability" in the change proposal).
//!
//! Deliberately mirrors `magnetar_runtime::InferenceApiObserver`'s shape --
//! an append-only, caller-owned `Vec` of redacted-by-default observations,
//! rather than a global sink or a logging framework dependency. Runtime
//! observations remain inference-only (`magnetar_runtime::inference_api`);
//! this module never reads or wraps them, it only records CLI-side command
//! lifecycle events.
//!
//! `CliObservation.message` MUST NOT contain the raw prompt, secret values,
//! file contents, Git diff contents, or token IDs -- only counts and kind
//! labels (e.g. `"file context collected: {byte_count} bytes"`, never the
//! file's actual content). See the `redaction` test module below for the
//! executable guarantee.

/// Kinds of CLI-side command lifecycle observations. Mirrors the
/// "CLI-side observations MAY include" list in `proposal.md`'s
/// "Observability" section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliObservationKind {
    CommandReceived,
    CommandParsed,
    FileContextCollected,
    GitContextCollected,
    /// Emitted when `commands::cmd_run`/`commands::cmd_agent` executes a
    /// tool via `tools::execute_tool` (§12/§26). Only a byte count of the
    /// tool's output, never the output itself -- see the redaction test
    /// below.
    ToolExecuted,
    RuntimeRequestSubmitted,
    StreamRendered,
    CommandCompleted,
    CommandFailed,
}

/// A single redacted-by-default CLI-side observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliObservation {
    pub kind: CliObservationKind,
    pub message: String,
}

/// Collects [`CliObservation`]s for one CLI invocation. Caller-owned and
/// explicitly threaded, never a global/static sink.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliObserver {
    observations: Vec<CliObservation>,
}

impl CliObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, kind: CliObservationKind, message: impl Into<String>) {
        self.observations.push(CliObservation {
            kind,
            message: message.into(),
        });
    }

    pub fn observations(&self) -> &[CliObservation] {
        &self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_records_observations_in_call_order() {
        let mut observer = CliObserver::new();
        observer.observe(
            CliObservationKind::CommandReceived,
            "received 3 argument(s)",
        );
        observer.observe(CliObservationKind::CommandParsed, "parsed subcommand 'run'");
        observer.observe(CliObservationKind::CommandCompleted, "command completed");
        let kinds: Vec<_> = observer.observations().iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                CliObservationKind::CommandReceived,
                CliObservationKind::CommandParsed,
                CliObservationKind::CommandCompleted,
            ]
        );
    }

    /// Redaction guarantee: recording a `FileContextCollected` observation
    /// for file content containing a known marker string never lets that
    /// marker reach `.message` -- only a byte count may, matching how
    /// `commands::cmd_run` actually calls `observe` (it never passes file
    /// content itself).
    #[test]
    fn file_context_observation_never_contains_the_files_raw_content() {
        let marker = "SECRET_FILE_CONTENT_MARKER_7f2a";
        let file_content = format!("line one\nline two with {marker}\nline three");
        let mut observer = CliObserver::new();
        observer.observe(
            CliObservationKind::FileContextCollected,
            format!("file context collected: {} bytes", file_content.len()),
        );
        for observation in observer.observations() {
            assert!(!observation.message.contains(marker));
            assert!(!observation.message.contains(&file_content));
        }
    }

    /// Same redaction guarantee as above, for `ToolExecuted`: only a byte
    /// count may reach `.message`, never the tool's actual output.
    #[test]
    fn tool_executed_observation_never_contains_the_tools_raw_output() {
        let marker = "SECRET_TOOL_OUTPUT_MARKER_c418";
        let tool_output = format!("line one\nline two with {marker}\nline three");
        let mut observer = CliObserver::new();
        observer.observe(
            CliObservationKind::ToolExecuted,
            format!("tool executed: {} bytes of output", tool_output.len()),
        );
        for observation in observer.observations() {
            assert!(!observation.message.contains(marker));
            assert!(!observation.message.contains(&tool_output));
        }
    }
}
