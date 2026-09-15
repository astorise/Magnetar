---
name: Bug report
about: Something in Magnetar is wrong -- a defect, a doc/reality mismatch, a gap in a quality gate
title: "[P?] "
labels: []
---

<!--
Title convention this project uses: prefix with a rough priority,
[P1] (correctness bug, security-relevant, or blocks a quality gate) through
[P3] (hygiene, documentation, minor gap). Not sure? Leave it off and a
maintainer will triage.
-->

## Summary

<!-- One or two sentences: what's wrong. -->

## Where

<!-- File and line (file_path:line_number), or the command/CI job that
surfaces it. -->

## Reproduction

<!-- The smallest input/command that shows the problem. For a Rust bug, a
short snippet or a failing `cargo test` invocation is ideal. -->

## Impact

<!-- Who or what is affected, and how badly -- a wrong answer, a panic, a
silently ignored flag, a CI gate that doesn't actually gate anything, a
doc claim that isn't true. -->

## Suggested fix

<!-- Optional: a concrete direction, even a rough one. -->

## Environment (if relevant)

- OS / target:
- Rust toolchain (`rustc --version`, should match `rust-toolchain.toml` unless testing MSRV):
- Relevant feature flags (`wasmtime-component-engine`, `web-component-engine`):
