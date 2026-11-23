# Qwen3.5 gated-delta mixer fixture

This synthetic fixture is copied from the Qwen Component's independently
generated PyTorch reference. Its generator executes the pinned Transformers
model and cache classes after verifying source hashes; see
`components/qwen/tests/fixtures/README.md` for provenance and reproduction.
It includes no checkpoint weights. Graph tests use explicit state inputs and
outputs; they do not establish a Runtime-managed recurrent session.

The same source also documents how to generate an optional real-weight
reference and checksum-verified checkpoint segments. Set
MAGNETAR_QWEN35_CHECKPOINT_DIR to that output directory, then run:

    cargo test -p magnetar-runtime --lib gated_delta_graph_matches_real_checkpoint_mixer -- --ignored

This runs one real base-checkpoint mixer as an explicit-state f32 CPU graph on
synthetic hidden states. It does not load a complete model or execute a
Runtime-managed recurrent session.
