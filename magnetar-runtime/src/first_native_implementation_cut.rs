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
mod tests;
