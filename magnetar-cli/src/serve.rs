//! CLI/companion `magnetar serve` boundary (§24 "Serve Mode" in the change
//! proposal).
//!
//! HTTP/server API itself is out of scope for this change (see
//! `proposal.md` Non-Goals: "define HTTP server API") -- `commands::cmd_serve`
//! still fails structurally rather than pretending to bind a real HTTP
//! listener. This module proves the narrower claim §24 actually requires:
//! whatever function eventually handles one served generation request calls
//! the real Runtime Inference API and cannot bypass Runtime validation,
//! because it is not a new code path -- it is the exact same
//! [`pipeline::one_shot`] function `magnetar run` uses.

use magnetar_runtime::{CliBoundaryError, InferenceApiObserver, ModelRef};

use crate::pipeline;

/// Handles one served generation request the same way `magnetar run` does:
/// by calling [`pipeline::one_shot`], the Runtime Inference API entry
/// point. A future HTTP server implementation would call this (or
/// `pipeline::one_shot` directly) per request; nothing about serving over
/// HTTP would need a different Runtime-facing code path or a way to bypass
/// Runtime validation.
pub fn handle_serve_generation_request(
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<(String, InferenceApiObserver), CliBoundaryError> {
    pipeline::one_shot(model_ref, prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §24/§29: serve mode's request handling calls the real Runtime
    /// Inference API pipeline, not a bypass.
    #[test]
    fn handle_serve_generation_request_calls_the_real_runtime_inference_api() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let (text, observer) = handle_serve_generation_request(&model_ref, "hello").unwrap();
        assert!(!text.is_empty());
        assert!(!observer.observations().is_empty());
    }
}
