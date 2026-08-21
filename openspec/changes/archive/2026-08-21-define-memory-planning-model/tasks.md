# Tasks

## Memory Planning Types

- [x] Define MemoryPlan
- [x] Define MemoryRequirement
- [x] Define MemoryRegionKind
- [x] Define TensorLifetime
- [x] Define BufferLifetime
- [x] Define MemoryPressureReport
- [x] Define MemoryPlanningDecision
- [x] Define MemoryPlanningDiagnostic

## Graph Analysis

- [x] Analyze Compute Graph inputs
- [x] Analyze Compute Graph outputs
- [x] Analyze intermediate tensor lifetimes
- [x] Analyze temporary buffer requirements
- [x] Analyze view dependencies
- [x] Analyze materialization requirements
- [x] Analyze data movement buffer requirements

## Resource Placement

- [x] Use Resource Affinity during memory planning
- [x] Use Provider Compute Advertisements during memory planning
- [x] Use Device memory constraints during memory planning
- [x] Preserve Provider-pinned resources
- [x] Prevent implicit cross-Provider memory usage

## Memory Reuse

- [x] Identify reusable intermediate buffers
- [x] Prevent reuse while a resource is still live
- [x] Prevent reuse across incompatible affinity groups
- [x] Track output resource ownership

## Materialization

- [x] Detect when views require materialization
- [x] Require explicit materialization when needed
- [x] Estimate materialization memory cost
- [x] Attach Resource Affinity to materialized resources

## Transfer Planning

- [x] Estimate upload memory requirements
- [x] Estimate download memory requirements
- [x] Estimate copy memory requirements
- [x] Estimate transfer memory requirements
- [x] Detect host-staged transfer requirements
- [x] Prevent hidden CPU staging

## Validation

- [x] Validate memory requirements before Provider execution
- [x] Validate Provider memory limits
- [x] Validate Device memory limits
- [x] Validate tensor byte-size calculations
- [x] Return structured out-of-memory errors
- [x] Return structured memory-planning errors

## Documentation

- [x] Document Memory Planning as a Runtime responsibility
- [x] Document relationship with Tensor Resources
- [x] Document relationship with Provider Compute Advertisements
- [x] Document relationship with Data Movement
- [x] Document exclusions from WIT
