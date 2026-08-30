## ADDED Requirements
### Requirement: Minimal First Profile CI Exists

CI SHALL include a configuration that runs the native first-profile vertical
slice without relying on deferred accelerator/optimization features.

#### Scenario: CPU-only CI runner

Given no GPU exists

When minimal profile job runs

Then Qwen fixture native E2E can execute.

### Requirement: Native E2E Is Required Gate

The native profile E2E SHALL be a required CI status for baseline stabilization
once implementation reaches the final integration phase.

#### Scenario: E2E fails

Given all unit tests pass

But native Qwen E2E fails

When merge/cut gate evaluates

Then baseline is not complete.

### Requirement: OpenSpec And WIT Remain Required

Implementation SHALL continue to validate OpenSpec and WIT while code is added.

#### Scenario: Code passes but contract validation fails

Given Rust tests succeed

When OpenSpec/WIT validation fails

Then CI remains failing.

### Requirement: Minimal Profile Must Not Accidentally Use Candle

CI SHALL execute a configuration where Candle model execution cannot silently
satisfy native profile.

#### Scenario: Candle feature disabled

Given minimal profile build

When E2E succeeds

Then native Magnetar execution is proven independently.

### Requirement: Coverage Gate Continues

Existing workspace coverage policy SHALL remain applicable to production code
introduced by the first-native implementation.

#### Scenario: New Kernel code untested

Given coverage falls below accepted project policy

When CI evaluates

Then existing coverage gate behavior applies.
