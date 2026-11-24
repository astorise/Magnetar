## ADDED Requirements

### Requirement: Compact bfloat16 weights on the optimized CPU Provider

The optimized CPU Provider SHALL store a Model Instance weight written through `write_weight_tensor` as its bfloat16 bits when it is a matrix of at least 65,536 values and every value is exactly a bfloat16 value, unless `MAGNETAR_CPU_COMPACT_WEIGHTS=0`. Every read of the weight SHALL return its values unchanged. Any later write of the resource SHALL replace it.

#### Scenario: A bfloat16 checkpoint's weight

- **WHEN** a 256 x 256 weight of exact bfloat16 values is written as a weight
- **THEN** it is held compact, and reading it returns the same values and shape

#### Scenario: A weight that is not exact

- **WHEN** a weight has a value with nonzero low 16 bits, or fewer than 65,536 values
- **THEN** it is stored as f32

### Requirement: Pooled vector matmuls against dense weights

The optimized CPU Provider SHALL compute `matmul`, `matmul-bias` and `matmul-residual` against a compact or f32 weight across its worker pool, with vector kernels where the host supports them. Their results SHALL agree with the Reference CPU kernel's within 2e-5 of the largest value. Forms the kernels do not take SHALL run on the Reference CPU kernel.

#### Scenario: Decode and prefill shapes

- **WHEN** 1, 3, 5 or 8 rows multiply a compact or f32 weight, used transposed or not, with inner sizes and output counts off a multiple of the vector width and of the kernels' row and output groups, on 1 or 3 threads, with and without vector kernels, at both vector widths
- **THEN** every result agrees with the Reference CPU matmul within tolerance

#### Scenario: No expansion

- **WHEN** a matmul-family Kernel or an embedding lookup reads a compact weight
- **THEN** the weight is not expanded to f32, and a lookup's rows equal the Reference CPU lookup's

### Requirement: Stored orientation of compact weights

The optimized CPU Provider SHALL hold a compact weight with more rows than columns transposed, so that its stored rows are its longer dimension, and SHALL use it the other way round in the matmul family. Reads, the resident shape and embedding lookups of the weight SHALL return its logical values and shape.

#### Scenario: A tall weight

- **WHEN** a 640 x 128 weight of exact bfloat16 values is written as a weight, then read, multiplied by 1 or 3 rows (used transposed and not) and looked up as an embedding table
- **THEN** the read returns the same values and shape, the products agree with the Reference CPU matmul within tolerance, and the lookup's rows equal the Reference CPU lookup's

#### Scenario: A wide weight

- **WHEN** a 128 x 640 weight goes through the same reads, products and lookups
- **THEN** it is held as written, with the same results

### Requirement: Worker pool size of the optimized CPU Provider

The optimized CPU Provider SHALL size its worker pool to one thread per physical core the process may run on, and never more than the logical CPUs it may use, unless `MAGNETAR_CPU_THREADS` sets a positive count. Where the host does not report which logical CPUs share a core, it SHALL use every logical CPU it may use.

#### Scenario: Linux CPU lists

- **WHEN** Linux CPU lists such as `0-3`, `0,2` and `0-1,4,6-7` are read
- **THEN** they give the listed CPUs, and a malformed list (`3-1`, `a`) gives none

#### Scenario: Default size

- **WHEN** the pool takes its default size
- **THEN** it has at least one thread and no more than the process's available parallelism

### Requirement: The benchmark's CPU Provider

The real-checkpoint qualification campaign SHALL accept `cpu-native` in `MAGNETAR_QUALIFICATION_PROVIDERS`, and the benchmark's CPU cell SHALL select it.

#### Scenario: CPU benchmark cell

- **WHEN** the CPU benchmark runs Magnetar
- **THEN** its report names `magnetar:provider/cpu-native`
