---
name: Hardware / Provider report
about: A Provider (CUDA, ROCm, Metal, WGPU, NPU, TPU, ...) misbehaves on real hardware, or you have hardware this project's own tooling lacks
title: "[P?] "
labels: []
---

<!--
Several Providers are real-device-discovery-only or fully unverified in
this repository's own CI because no matching hardware exists there (see
README.md's "Deferred or unsupported for v0.1" section) -- Metal, ROCm
compute, NPU compute, TPU compute. Reports and contributions from real
hardware are genuinely useful here.
-->

## Provider and hardware

- Provider: <!-- providers/cuda, providers/rocm, providers/metal, providers/wgpu, providers/npu, providers/tpu, providers/cpu -->
- Device / GPU model:
- Driver version:
- OS and version:
- `nvidia-smi` / `rocm-smi` / equivalent output (if applicable):

## What happened

<!-- Device discovery failure, a kernel producing a wrong result, a crash,
a hang, or a capability this Provider doesn't yet support. -->

## Expected behaviour

## Reproduction

<!-- The `cargo test` invocation or command that reproduces it, e.g.
`cargo test --locked --manifest-path providers/cuda/Cargo.toml -- --include-ignored`. -->

## Logs / output

<!-- Full output, not a summary -- a Provider failure is often diagnosed
from the exact error text. -->
