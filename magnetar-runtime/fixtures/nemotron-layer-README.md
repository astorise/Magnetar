# Nemotron-H mixer fixtures

These synthetic inputs and outputs were produced by the pinned NVIDIA
Nemotron-H mixer, with the upstream mamba-ssm PyTorch gated-normalization
reference. The files are copies of the fixtures maintained in the Nemotron
component repository. Their generator, exact source revisions, SHA-256
digests, dependencies, and regeneration commands are documented in
`components/nemotron/tests/fixtures/README.md`. No checkpoint weights are
included. The second fixture additionally exercises linear biases.

The runtime test compares full prefill, chunked prefill and one-token
continuation with explicit convolution and SSM tensors. It does not imply
that a runtime-managed recurrent session or full Nemotron model is supported.
