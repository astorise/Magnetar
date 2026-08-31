//! First native implementation cut tracking.
//!
//! This module intentionally does not add an execution abstraction. It records
//! the implementation freeze and the known migration bypasses so Phase 0 can
//! be tested instead of living only in OpenSpec prose.

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FreezeReopenReason {
    CorrectnessBlocker,
    SecurityBlocker,
    ImpossibleImplementationContract,
    UnavoidableAbiBreak,
    AcceptedSpecContradiction,
    FeatureExpansion,
    PerformanceOptimization,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MigrationBypassKind {
    CallerProvidedLogits,
    CallerForwardCallback,
    CliPlaceholderLogits,
    DirectReferenceCpuExecution,
    FullSequenceDecodeShortcut,
    CandleModelExecution,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MigrationBypassDisposition {
    Deprecated,
    IsolatedTestOnly,
    NonConformantMigrationPath,
    TrackedForRemovalBeforeFinalCut,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureFreeze {
    pub name: &'static str,
    pub scope: &'static str,
}

impl ArchitectureFreeze {
    pub const fn reopens_for(self, reason: FreezeReopenReason) -> bool {
        matches!(
            reason,
            FreezeReopenReason::CorrectnessBlocker
                | FreezeReopenReason::SecurityBlocker
                | FreezeReopenReason::ImpossibleImplementationContract
                | FreezeReopenReason::UnavoidableAbiBreak
                | FreezeReopenReason::AcceptedSpecContradiction
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationBypassInventoryEntry {
    pub kind: MigrationBypassKind,
    pub path: &'static str,
    pub symbol: &'static str,
    pub disposition: MigrationBypassDisposition,
    pub final_cut_removal_required: bool,
}

impl MigrationBypassInventoryEntry {
    pub const fn is_final_conformance_allowed(self) -> bool {
        !self.final_cut_removal_required
            && matches!(
                self.disposition,
                MigrationBypassDisposition::IsolatedTestOnly
            )
    }
}

pub const fn architecture_freeze_1() -> ArchitectureFreeze {
    ArchitectureFreeze {
        name: "ARCHITECTURE FREEZE #1",
        scope: "first native local single-Device model-execution path",
    }
}

pub const fn phase_0_migration_inventory() -> &'static [MigrationBypassInventoryEntry] {
    &[
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::CliPlaceholderLogits,
            path: "magnetar-cli/src/pipeline.rs",
            symbol: "removed; normal path calls run_first_native_generation",
            disposition: MigrationBypassDisposition::Deprecated,
            final_cut_removal_required: false,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::CallerForwardCallback,
            path: "magnetar-runtime/src/inference_api.rs",
            symbol: "removed; normal path uses Runtime-owned model execution engine",
            disposition: MigrationBypassDisposition::Deprecated,
            final_cut_removal_required: false,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::CallerProvidedLogits,
            path: "magnetar-runtime/src/inference_api.rs",
            symbol: "removed from production caller API",
            disposition: MigrationBypassDisposition::Deprecated,
            final_cut_removal_required: false,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::DirectReferenceCpuExecution,
            path: "magnetar-runtime/src/first_native_runtime.rs",
            symbol: "e2e_forward_hidden_states / dispatch_matmul",
            disposition: MigrationBypassDisposition::IsolatedTestOnly,
            final_cut_removal_required: false,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::FullSequenceDecodeShortcut,
            path: "magnetar-runtime/src/first_native_runtime.rs",
            symbol: "removed; decode uses execute_qwen_decode_hidden_states_through_dispatch",
            disposition: MigrationBypassDisposition::Deprecated,
            final_cut_removal_required: false,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::CandleModelExecution,
            path: "Cargo.toml / magnetar-runtime/Cargo.toml",
            symbol: "no candle dependency in workspace manifests",
            disposition: MigrationBypassDisposition::Deprecated,
            final_cut_removal_required: false,
        },
    ]
}

pub fn validate_phase_0_migration_inventory() -> Result<(), &'static str> {
    let inventory = phase_0_migration_inventory();
    for required in [
        MigrationBypassKind::CallerProvidedLogits,
        MigrationBypassKind::CallerForwardCallback,
        MigrationBypassKind::CliPlaceholderLogits,
        MigrationBypassKind::DirectReferenceCpuExecution,
        MigrationBypassKind::FullSequenceDecodeShortcut,
        MigrationBypassKind::CandleModelExecution,
    ] {
        if !inventory.iter().any(|entry| entry.kind == required) {
            return Err("migration inventory is missing a required bypass kind");
        }
    }
    if inventory
        .iter()
        .any(|entry| entry.final_cut_removal_required && entry.is_final_conformance_allowed())
    {
        return Err("removal-required bypass cannot be final-conformance allowed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime crate has workspace parent")
            .to_path_buf()
    }

    fn read_workspace_file(path: &str) -> String {
        std::fs::read_to_string(workspace_root().join(path))
            .unwrap_or_else(|error| panic!("read {path}: {error}"))
    }

    #[test]
    fn architecture_freeze_reopens_only_for_blockers() {
        let freeze = architecture_freeze_1();

        assert!(freeze.reopens_for(FreezeReopenReason::CorrectnessBlocker));
        assert!(freeze.reopens_for(FreezeReopenReason::SecurityBlocker));
        assert!(freeze.reopens_for(FreezeReopenReason::ImpossibleImplementationContract));
        assert!(freeze.reopens_for(FreezeReopenReason::UnavoidableAbiBreak));
        assert!(freeze.reopens_for(FreezeReopenReason::AcceptedSpecContradiction));
        assert!(!freeze.reopens_for(FreezeReopenReason::FeatureExpansion));
        assert!(!freeze.reopens_for(FreezeReopenReason::PerformanceOptimization));
    }

    #[test]
    fn phase_0_inventory_covers_required_bypass_classes() {
        validate_phase_0_migration_inventory().unwrap();
    }

    #[test]
    fn placeholder_cli_logits_are_removed_from_the_normal_path() {
        let entry = phase_0_migration_inventory()
            .iter()
            .find(|entry| entry.kind == MigrationBypassKind::CliPlaceholderLogits)
            .expect("CLI placeholder logits entry present");

        assert_eq!(entry.disposition, MigrationBypassDisposition::Deprecated);
        assert!(!entry.final_cut_removal_required);
        assert!(!entry.is_final_conformance_allowed());
    }

    #[test]
    fn cli_and_public_runtime_surface_do_not_export_e2e_harness_generation() {
        let cli_pipeline = read_workspace_file("magnetar-cli/src/pipeline.rs");
        assert!(!cli_pipeline.contains("e2e_conformance"));
        assert!(!cli_pipeline.contains("run_first_native_fixture_generation"));
        assert!(cli_pipeline.contains("run_first_native_generation"));

        let runtime_lib = read_workspace_file("magnetar-runtime/src/lib.rs");
        assert!(!runtime_lib.contains("pub mod e2e_conformance"));
        assert!(!runtime_lib.contains("pub use e2e_conformance::*"));

        let inference_api = read_workspace_file("magnetar-runtime/src/inference_api.rs");
        assert!(!inference_api.contains("pub struct RuntimeModelExecutionStep"));
        assert!(!inference_api.contains("pub trait RuntimeModelExecutionEngine"));
        assert!(!inference_api.contains("pub struct SharedRuntimeModelExecutionEngine"));

        let runtime = read_workspace_file("magnetar-runtime/src/runtime.rs");
        assert!(!runtime.contains("pub fn model_execution_engine"));
    }

    #[test]
    fn public_logits_bypass_inventory_is_cleared_after_engine_cutover() {
        let inventory = phase_0_migration_inventory();
        for kind in [
            MigrationBypassKind::CallerForwardCallback,
            MigrationBypassKind::CallerProvidedLogits,
        ] {
            let entry = inventory
                .iter()
                .find(|entry| entry.kind == kind)
                .expect("public logits bypass entry present");
            assert_eq!(entry.disposition, MigrationBypassDisposition::Deprecated);
            assert!(!entry.final_cut_removal_required);
            assert!(!entry.is_final_conformance_allowed());
        }

        let inference_api = read_workspace_file("magnetar-runtime/src/inference_api.rs");
        for needle in [
            "pub struct RuntimeModelExecutionStep",
            "pub trait RuntimeModelExecutionEngine",
            "pub struct SharedRuntimeModelExecutionEngine",
            "pub fn new(logits: Vec<f32>",
            "pub fn execute_generation_step",
        ] {
            assert!(
                !inference_api.contains(needle),
                "{needle} must stay removed from the public inference API"
            );
        }

        let runtime = read_workspace_file("magnetar-runtime/src/runtime.rs");
        for needle in [
            "pub fn model_execution_engine",
            "pub fn model_execution_engine(",
        ] {
            assert!(
                !runtime.contains(needle),
                "{needle} must stay removed from the public Runtime surface"
            );
        }
    }

    #[test]
    fn production_callers_cannot_reach_model_execution_engine_or_logits_step() {
        let forbidden = [
            "magnetar-cli/src/agent.rs",
            "magnetar-cli/src/commands.rs",
            "magnetar-cli/src/main.rs",
            "magnetar-cli/src/pipeline.rs",
            "magnetar-cli/src/serve.rs",
            "magnetar-runtime/src/lib.rs",
        ];

        for path in forbidden {
            let source = read_workspace_file(path);
            for needle in [
                "RuntimeModelExecutionEngine",
                "RuntimeModelExecutionStep::new",
                ".model_execution_engine(",
            ] {
                assert!(
                    !source.contains(needle),
                    "{needle} must not appear in {path}"
                );
            }
        }
    }

    #[test]
    fn final_cut_inventory_has_no_removal_required_p0_bypass() {
        let blocking: Vec<_> = phase_0_migration_inventory()
            .iter()
            .filter(|entry| entry.final_cut_removal_required)
            .collect();
        assert_eq!(blocking, Vec::<&MigrationBypassInventoryEntry>::new());
    }

    #[test]
    fn first_native_decode_does_not_materialize_full_history() {
        let source = read_workspace_file("magnetar-runtime/src/first_native_runtime.rs");
        for forbidden in [
            "extend_from_slice(generated_tokens)",
            "request.input_token_ids.clone()",
            "prompt + generated_tokens",
        ] {
            assert!(
                !source.contains(forbidden),
                "{forbidden} must not appear in the normal first-native decode path"
            );
        }
        assert!(source.contains("execute_qwen_decode_hidden_states_through_dispatch"));
        assert!(source.contains("model_input_tokens={model_input_token_count}"));
        assert!(source.contains("fn commit_generation_step"));
        assert!(source.contains("runtime.append_decode_kv_cache(&state.cache, 1)"));
    }
}
