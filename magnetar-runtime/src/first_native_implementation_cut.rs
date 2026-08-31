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
            symbol: "crate-internal RuntimeGenerationExecutor",
            disposition: MigrationBypassDisposition::TrackedForRemovalBeforeFinalCut,
            final_cut_removal_required: true,
        },
        MigrationBypassInventoryEntry {
            kind: MigrationBypassKind::CallerProvidedLogits,
            path: "magnetar-runtime/src/inference_api.rs",
            symbol: "crate-internal RuntimeGenerationStep::new(logits, evidence)",
            disposition: MigrationBypassDisposition::TrackedForRemovalBeforeFinalCut,
            final_cut_removal_required: true,
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
            symbol: "E2eRuntimeGenerationExecutor::execute_generation_step",
            disposition: MigrationBypassDisposition::TrackedForRemovalBeforeFinalCut,
            final_cut_removal_required: true,
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
        assert!(!inference_api.contains("pub struct RuntimeGenerationStep"));
        assert!(!inference_api.contains("pub trait RuntimeGenerationExecutor"));
        assert!(!inference_api.contains("pub struct SharedRuntimeGenerationExecutor"));

        let runtime = read_workspace_file("magnetar-runtime/src/runtime.rs");
        assert!(!runtime.contains("pub fn generation_executor"));
    }

    #[test]
    fn public_logits_bypass_inventory_is_downgraded_after_visibility_removal() {
        let inventory = phase_0_migration_inventory();
        for kind in [
            MigrationBypassKind::CallerForwardCallback,
            MigrationBypassKind::CallerProvidedLogits,
        ] {
            let entry = inventory
                .iter()
                .find(|entry| entry.kind == kind)
                .expect("public logits bypass entry present");
            assert_eq!(
                entry.disposition,
                MigrationBypassDisposition::TrackedForRemovalBeforeFinalCut
            );
            assert!(entry.final_cut_removal_required);
            assert!(!entry.is_final_conformance_allowed());
        }

        let inference_api = read_workspace_file("magnetar-runtime/src/inference_api.rs");
        for needle in [
            "pub struct RuntimeGenerationStep",
            "pub trait RuntimeGenerationExecutor",
            "pub struct SharedRuntimeGenerationExecutor",
            "pub fn new(logits: Vec<f32>",
            "pub fn execute_generation_step",
        ] {
            assert!(
                !inference_api.contains(needle),
                "{needle} must stay removed from the public inference API"
            );
        }

        let runtime = read_workspace_file("magnetar-runtime/src/runtime.rs");
        for needle in ["pub fn generation_executor", "pub fn generation_executor("] {
            assert!(
                !runtime.contains(needle),
                "{needle} must stay removed from the public Runtime surface"
            );
        }
    }

    #[test]
    fn legacy_logits_seam_is_confined_to_tracked_migration_locations() {
        let allowed = [
            "magnetar-runtime/src/first_native_runtime.rs",
            "magnetar-runtime/src/inference_api.rs",
            "magnetar-runtime/src/runtime.rs",
            "magnetar-runtime/src/tests.rs",
            "magnetar-runtime/src/first_native_implementation_cut.rs",
        ];
        let forbidden = [
            "magnetar-cli/src/agent.rs",
            "magnetar-cli/src/commands.rs",
            "magnetar-cli/src/main.rs",
            "magnetar-cli/src/pipeline.rs",
            "magnetar-cli/src/serve.rs",
        ];

        for path in forbidden {
            let source = read_workspace_file(path);
            for needle in [
                "RuntimeGenerationExecutor",
                "RuntimeGenerationStep::new",
                ".generation_executor(",
            ] {
                assert!(
                    !source.contains(needle),
                    "{needle} must not appear in {path}"
                );
            }
        }

        for path in allowed {
            let source = read_workspace_file(path);
            let contains_legacy = source.contains("RuntimeGenerationExecutor")
                || source.contains("RuntimeGenerationStep::new")
                || source.contains(".generation_executor(");
            assert!(
                contains_legacy,
                "allowed migration location {path} should stay explicit while bypass inventory tracks it"
            );
        }
    }
}
