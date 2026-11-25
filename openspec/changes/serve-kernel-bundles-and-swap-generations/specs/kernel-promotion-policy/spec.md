## ADDED Requirements

### Requirement: Promotion Evidence Per Hardware Target

The Runtime SHALL decide a kernel generation's promotion per hardware target, from an A/B comparison against the active generation measured on that target. A hardware target is the Provider identity, the Device architecture or CPU model with its feature set, and the driver or runtime major version. The comparison SHALL run on a qualified workload with the same model, weights, prompts, precision, batch and memory policy, interleaving the two generations on the same host. Evidence from another target, including published evidence, SHALL NOT promote a generation.

#### Scenario: Evidence from another architecture

- **WHEN** a generation shows a 15% gain on `sm_90` and has no comparison on the local `sm_86`
- **THEN** it is not promoted on `sm_86`

#### Scenario: Promotion is scoped to its target

- **WHEN** a generation is promoted on one target
- **THEN** other targets keep their active generation until their own comparison decides

### Requirement: Gain And Significance

The Runtime SHALL measure the gain as the relative reduction of the workload's primary latency metric (end-to-end request latency unless the workload profile names TTFT or TPOT). It SHALL estimate the gain from interleaved runs, at least the policy's minimum per generation (10 by default), with a 95% confidence interval. A gain SHALL count only when that interval excludes zero.

#### Scenario: A large but noisy gain

- **WHEN** the median gain is 12% but its 95% interval includes zero
- **THEN** the gain does not count, and the generation is not promoted

### Requirement: No Significant Regression

The Runtime SHALL refuse any promotion when a guarded metric regresses significantly, beyond the policy tolerance (2% by default). The guarded metrics are TTFT, TPOT, p95 request latency, peak memory and error rate. It SHALL also refuse it on any correctness failure: generated tokens or outputs outside the qualification policy.

#### Scenario: Faster decode, more memory

- **WHEN** a generation cuts TPOT by 12% and raises peak device memory by 8%, significantly
- **THEN** it is not promoted, and the decision names the memory regression

### Requirement: Promotion Bands

With a significant gain and no significant regression, the Runtime SHALL promote a generation automatically when the gain is at least 10%. It SHALL offer it for manual promotion by an operator when the gain is at least 5% and below 10%, and SHALL NOT promote it below 5%. Every decision SHALL be recorded with its target, evidence, gain, interval, guarded metrics and band.

#### Scenario: Automatic promotion

- **WHEN** a generation shows a significant 11% gain on the local target with no significant regression
- **THEN** it is promoted automatically on that target, at the next plan boundary

#### Scenario: Manual band

- **WHEN** the significant gain is 7%
- **THEN** the generation is ready and offered for manual promotion, and stays inactive until an operator promotes it

#### Scenario: Below the bands

- **WHEN** the significant gain is 3%
- **THEN** the generation is not promoted, automatically or manually

### Requirement: Automatic Rollback

After any promotion, automatic or manual, the Runtime SHALL watch the guarded metrics on that target for the policy's observation window. It SHALL roll back to the previous generation automatically, within the rollback window, when a guarded metric regresses significantly against the pre-promotion baseline or a correctness check or error occurs. It SHALL record the reason, and SHALL bar that generation from automatic re-promotion on that target until new evidence is measured.

#### Scenario: A regression after promotion

- **WHEN** p95 request latency regresses by 6%, significantly, during the observation window after an automatic promotion
- **THEN** the previous generation is promoted back, the regressed one drains, and it is not promoted again automatically on that target without a new comparison
