# close-tachyon-scope-audit-gaps

Close the real Magnetar gaps found reconciling the codebase against the Tachyon-authored Magnetar scope charter: real chat-template rendering from a bundle's own tokenizer_config.json instead of a generic placeholder, an explicit fail-closed Unsupported signal for CUDA multi-token decode instead of a residency error leaking from deep inside KV-cache resolution, and real tokens-per-second measurement.
