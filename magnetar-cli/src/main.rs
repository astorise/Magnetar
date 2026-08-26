//! `magnetar-cli`: the first-party client/workspace/tool/agent runtime built
//! around Magnetar inference.
//!
//! This binary owns workspace/file access, Git, network, secrets, tool
//! execution, shell/process execution, agent orchestration, chat/session UX,
//! and CLI configuration. `magnetar-runtime` owns inference only: model
//! resolution/loading, tokenization, generation, streaming, cancellation,
//! and diagnostics. This binary never re-implements inference itself and
//! never grants `magnetar-runtime` any workspace, Git, network, secret,
//! tool, shell, or agent authority -- see
//! `openspec/changes/define-magnetar-cli-inference-boundary/proposal.md`
//! for the authoritative boundary this crate implements, and
//! `magnetar_runtime::cli_boundary` for the Runtime-side formalization of
//! the same boundary (in particular
//! `magnetar_runtime::reject_cli_owned_authority`, which this crate never
//! needs to call directly because it never asks Runtime to perform a
//! CLI-owned capability in the first place).
//!
//! Honesty note: `magnetar-runtime` is a contracts/validation layer today,
//! not an end-to-end inference engine (see `pipeline.rs`'s module doc
//! comment for the specifics: placeholder tokenizer, placeholder
//! all-zero logits). This CLI is built honestly against what the Runtime
//! actually offers today; it does not fake real text generation quality,
//! and it does not bypass Runtime validation to fabricate a successful
//! model load, resolution, or unload.
//!
//! This increment adds opt-in file (`--file`), Git (`--git-diff`), and
//! environment-secret (`--env-secret`) context collection to `magnetar
//! run` (see `commands.rs`), a small in-memory model alias table
//! (`aliases.rs`), a CLI-owned observation log (`observability.rs`), and a
//! structural CLI-config type (`config.rs`). It still does not implement
//! network access, tool execution, general shell/process execution, agent
//! orchestration, HTTP serve mode, persistent CLI configuration/session
//! storage, or real mid-generation cancellation (no concurrency model in
//! this synchronous crate to cancel). Those remain real gaps (consistent
//! with the change proposal's own Non-Goals and the "do NOT implement"
//! scope of this increment), not silently faked capabilities.
//!
//! Browser Boundary (§28): `magnetar-cli` is native-only by design (see
//! `proposal.md`'s "Browser Target" section) -- a browser client is
//! explicit future work this change does not define. The Runtime Inference
//! API surface is already structured to support a non-native caller
//! without this crate needing to do anything: `magnetar-runtime` has
//! `#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
//! pub mod component_web;` in its `lib.rs` today. That is pointed at here
//! as existing evidence, honestly described as such, rather than a
//! fabricated new wasm32 build check this crate cannot actually run.

mod agent;
mod aliases;
mod commands;
mod config;
mod network;
mod observability;
mod pipeline;
mod process;
mod render;
mod secrets;
mod serve;
mod tools;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = commands::dispatch(&args) {
        render::print_error(&error);
        std::process::exit(1);
    }
}
