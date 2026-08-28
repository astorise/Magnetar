# Tasks

## 1. Compilation Capability

- [x] Define Provider Kernel Compilation Capability.
- [x] Give capability an independent version.
- [x] Make capability optional.
- [x] Add capability discovery.
- [x] Ensure Providers without compilation remain valid.
- [x] Add discovery tests.

## 2. Capability Descriptor

- [x] Declare accepted source formats.
- [x] Declare produced compiled formats.
- [x] Declare supported Device targets.
- [x] Declare supported architectures.
- [x] Declare compilation modes.
- [x] Declare async support.
- [x] Declare cancellation support.
- [x] Declare deadline support.
- [x] Declare maximum source size.
- [x] Declare maximum output size.
- [x] Declare compilation concurrency limits.
- [x] Declare isolation model.
- [x] Declare compiler identity support.
- [x] Add descriptor validation.

## 3. Compilation Modes

- [x] Add SourceCompilation.
- [x] Add IntermediateTranslation.
- [x] Add BinarySpecialization.
- [x] Add ShaderCompilation.
- [x] Add PipelineCompilation.
- [x] Add OfflineAot.
- [x] Add LoadTimeJit.
- [x] Add ProviderManaged.
- [x] Document that LoadTimeJit is cold-path only.

## 4. Format Negotiation

- [x] Validate source format against Provider capability.
- [x] Validate output format.
- [x] Do not infer format from filename.
- [x] Do not infer format from Provider name.
- [x] Add unsupported format tests.

## 5. Compilation Request

- [x] Define KernelCompilationRequest.
- [x] Add request ID.
- [x] Add Kernel Source Artifact reference.
- [x] Add compilation target.
- [x] Add specialization requirements.
- [x] Add policy.
- [x] Use bytes rather than assuming UTF-8 String.
- [x] Reject arbitrary host paths.

## 6. Compilation Target

- [x] Add Provider binding.
- [x] Add Device binding.
- [x] Add target architecture.
- [x] Add hardware feature set.
- [x] Add target ABI.
- [x] Add execution environment.
- [x] Add dtype/layout specialization.
- [x] Add shape specialization.
- [x] Add precision requirements.
- [x] Add determinism requirements.
- [x] Ensure no native Device handle is present.

## 7. Runtime Authority

- [x] Preserve Runtime Provider selection.
- [x] Preserve Runtime Device selection.
- [x] Prevent compiler from changing selected Provider.
- [x] Prevent compiler from changing selected Device silently.
- [x] Add authority tests.

## 8. Compiler Identity

- [x] Add compiler name metadata.
- [x] Add compiler version metadata.
- [x] Add backend version metadata.
- [x] Add toolchain fingerprint.
- [x] Add compiler flags fingerprint.
- [x] Add compiler identity tests.

## 9. Specialization

- [x] Support shape specialization metadata.
- [x] Support dtype specialization metadata.
- [x] Support layout specialization metadata.
- [x] Support hardware feature specialization.
- [x] Ensure specialization is explicit.
- [x] Add specialization compatibility tests.

## 10. Compilation Jobs

- [x] Define CompilationJobId.
- [x] Define queued state.
- [x] Define compiling state.
- [x] Define succeeded state.
- [x] Define failed state.
- [x] Define cancelled state.
- [x] Define timed-out state.
- [x] Define valid state transitions.
- [x] Add lifecycle tests.

## 11. Asynchronous Compilation

- [x] Support submit.
- [x] Support poll.
- [x] Support await at Runtime layer if appropriate.
- [x] Support completion result.
- [x] Add asynchronous compilation tests.

## 12. Cancellation

- [x] Declare cancellation semantics.
- [x] Implement cancellation request contract.
- [x] Ensure partial output is never ready.
- [x] Add cancellation tests.

## 13. Deadlines

- [x] Add compilation deadlines.
- [x] Declare Provider deadline enforcement support.
- [x] Fail closed when required deadline cannot be enforced.
- [x] Add timeout tests.

## 14. Resource Limits

- [x] Enforce maximum source size.
- [x] Enforce maximum output size.
- [x] Enforce concurrent job limits.
- [x] Define workspace limits.
- [x] Define host memory limit metadata.
- [x] Add limit tests.

## 15. Isolation Model

- [x] Add InProcessTrustedCompiler.
- [x] Add RestrictedSubprocess.
- [x] Add SandboxedSubprocess.
- [x] Add ExternalCompilationService.
- [x] Add PlatformManagedCompiler.
- [x] Add BrowserManagedCompiler.
- [x] Add Unavailable.
- [x] Add Runtime policy evaluation.
- [x] Add insufficient-isolation tests.

## 16. Source Trust

- [x] Treat source as potentially untrusted.
- [x] Ensure compilation success does not grant trust.
- [x] Ensure compiler trust does not grant artifact trust.
- [x] Add untrusted-source tests.

## 17. Network Boundary

- [x] Prevent implicit compiler dependency downloads.
- [x] Require explicit policy for compilation network access.
- [x] Add network boundary tests.

## 18. Filesystem Boundary

- [x] Pass source bytes rather than arbitrary source paths.
- [x] Keep temporary workspaces Provider-private.
- [x] Redact temporary paths.
- [x] Add filesystem boundary tests.

## 19. Process Boundary

- [x] Allow declared compiler subprocesses.
- [x] Avoid shell string construction from untrusted metadata.
- [x] Normalize compiler process failures.
- [x] Add process boundary tests.

## 20. Environment Boundary

- [x] Deny/sanitize/allowlist relevant environment variables.
- [x] Fingerprint environment inputs that affect compilation where necessary.
- [x] Prevent secret inheritance by default.
- [x] Add environment tests.

## 21. Compilation Result

- [x] Produce CompiledKernelArtifact.
- [x] Add output format.
- [x] Add output digest.
- [x] Add source artifact digest.
- [x] Add target metadata.
- [x] Add compiler metadata.
- [x] Add specialization metadata.
- [x] Add compatibility metadata.
- [x] Add integrity state.
- [x] Add result tests.

## 22. Output Integrity

- [x] Compute compiled artifact digest.
- [x] Verify immutable identity.
- [x] Reject digest mismatch.
- [x] Add integrity tests.

## 23. Failure Atomicity

- [x] Keep partial artifacts non-ready.
- [x] Do not mutate known-good artifact on failed compile.
- [x] Do not mutate existing PreparedKernel on failed compile.
- [x] Add atomicity tests.

## 24. Preparation Boundary

- [x] Keep compilation and preparation logically distinct.
- [x] Allow preparation-only Providers.
- [x] Return opaque PreparedKernelId.
- [x] Keep native preparation handles private.
- [x] Add preparation tests.

## 25. AOT-Only Providers

- [x] Support Provider with no source compilation capability.
- [x] Allow Provider to prepare external AOT artifact.
- [x] Add AOT-only conformance fixture.

## 26. Platform-Managed Compilation

- [x] Support logical compile/prepare distinction where platform combines them.
- [x] Preserve cold/hot path boundary.
- [x] Add platform-managed compilation tests.

## 27. Hot Path

- [x] Prevent Provider execute from compiling source.
- [x] Detect unexpected compile requirement at execute time.
- [x] Return structured hot-path compilation error.
- [x] Add hot-path tests.

## 28. Model Loading Integration

- [x] Allow compilation jobs during Model Loading.
- [x] Allow compilation jobs to execute concurrently.
- [x] Gate Model Instance readiness on required preparation.
- [x] Add Model Loading tests.

## 29. Concurrency

- [x] Track Provider compilation job capacity.
- [x] Respect Provider concurrency limit.
- [x] Respect Runtime global resource policy.
- [x] Add concurrent compilation tests.

## 30. Crash Containment

- [x] Normalize compiler crash.
- [x] Prevent crash from producing ready artifact.
- [x] Isolate compiler crash where isolation model allows.
- [x] Add compiler-crash tests.

## 31. Provider ABI Extension

- [x] Define versioned compilation capability descriptor.
- [x] Keep it optional.
- [x] Avoid Rust trait object ABI.
- [x] Define submit ABI.
- [x] Define poll ABI.
- [x] Define cancel ABI.
- [x] Define release-job ABI.
- [x] Add ABI version checks.

## 32. ABI Buffer Ownership

- [x] Define request buffer ownership.
- [x] Define result buffer ownership.
- [x] Define output artifact ownership.
- [x] Define diagnostic buffer ownership.
- [x] Define release functions.
- [x] Add ownership violation tests.

## 33. ABI Failure Safety

- [x] Prevent Rust unwind across ABI.
- [x] Normalize panic/exception.
- [x] Normalize compiler crash.
- [x] Add failure-boundary tests.

## 34. Prepared Kernel ABI

- [x] Allow opaque numeric PreparedKernelId.
- [x] Prevent pointer semantics.
- [x] Keep native mapping inside Provider.
- [x] Add handle opacity tests.

## 35. Observability

- [x] Observe compilation submitted.
- [x] Observe compilation queued.
- [x] Observe compiler started.
- [x] Observe compilation completed.
- [x] Observe compilation failed.
- [x] Observe cancellation.
- [x] Observe timeout.
- [x] Observe artifact created.
- [x] Observe preparation started.
- [x] Observe preparation completed.
- [x] Record source digest.
- [x] Record source/compiled formats.
- [x] Record compiler identity.
- [x] Record duration.
- [x] Record output size.
- [x] Redact source and binaries.
- [x] Redact native handles.
- [x] Redact temp paths/environment/secrets.

## 36. Diagnostics

- [x] Define compiler-stage diagnostic.
- [x] Allow source-location metadata.
- [x] Redact unrestricted compiler stdout.
- [x] Redact unrestricted compiler stderr.
- [x] Add diagnostic tests.

## 37. Error Model

- [x] Add kernel-compilation-unavailable.
- [x] Add kernel-compilation-capability-version-unsupported.
- [x] Add kernel-compilation-source-format-unsupported.
- [x] Add kernel-compilation-output-format-unsupported.
- [x] Add kernel-compilation-target-unsupported.
- [x] Add kernel-compilation-specialization-unsupported.
- [x] Add kernel-compilation-policy-denied.
- [x] Add kernel-compilation-isolation-insufficient.
- [x] Add kernel-compilation-source-too-large.
- [x] Add kernel-compilation-output-too-large.
- [x] Add kernel-compilation-concurrency-limit.
- [x] Add kernel-compilation-deadline-unsupported.
- [x] Add kernel-compilation-timeout.
- [x] Add kernel-compilation-cancellation-unsupported.
- [x] Add kernel-compilation-cancelled.
- [x] Add kernel-compilation-compiler-unavailable.
- [x] Add kernel-compilation-compiler-crashed.
- [x] Add kernel-compilation-failed.
- [x] Add kernel-compilation-output-invalid.
- [x] Add kernel-compilation-output-integrity-failed.
- [x] Add kernel-compilation-job-not-found.
- [x] Add kernel-compilation-job-state-invalid.
- [x] Add kernel-compilation-abi-incompatible.
- [x] Add kernel-compilation-buffer-ownership-violation.
- [x] Add kernel-compilation-hot-path-denied.
- [x] Add internal-kernel-compilation-error.

## 38. Conformance

- [x] Test optional capability discovery.
- [x] Test source format negotiation.
- [x] Test output format declaration.
- [x] Test unsupported source rejection.
- [x] Test Runtime Device authority.
- [x] Test async job lifecycle.
- [x] Test cancellation.
- [x] Test deadlines.
- [x] Test limits.
- [x] Test isolation policy.
- [x] Test trust separation.
- [x] Test hot-path prohibition.
- [x] Test ABI buffer ownership.
- [x] Test no native pointer exposure.
- [x] Test failed compile atomicity.
- [x] Test opaque PreparedKernelId.

## 39. Documentation

- [x] Document Provider compilation capability.
- [x] Document compile versus prepare.
- [x] Document LoadTimeJit semantics.
- [x] Document AOT-only Providers.
- [x] Document isolation model.
- [x] Document ABI extension.
- [x] Document cold/hot path.
- [x] Document non-goals.

## 40. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify Device remains compilation-free.
- [x] Verify Scheduler remains compilation-free.
- [x] Verify execution hot path remains compilation-free.
- [x] Verify Provider handles remain private.
- [x] Verify compilation capability remains optional.