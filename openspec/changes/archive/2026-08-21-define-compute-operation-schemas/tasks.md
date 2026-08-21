# Tasks

## Operation Schema Model

- [x] Define ComputeOperationSchema
- [x] Define ComputeOperationId
- [x] Define ComputeOperationFamily
- [x] Define ComputeOperationAttribute
- [x] Define ComputeOperationInputRule
- [x] Define ComputeOperationOutputRule
- [x] Define ComputeOperationValidationResult

## Descriptor and View Schemas

- [x] Define reshape schema
- [x] Define transpose schema
- [x] Define permute schema
- [x] Define slice schema
- [x] Define broadcast schema
- [x] Define squeeze schema
- [x] Define unsqueeze schema

## Elementwise Schemas

- [x] Define unary elementwise schema
- [x] Define binary elementwise schema
- [x] Define supported unary operator identifiers
- [x] Define supported binary operator identifiers
- [x] Define dtype compatibility validation
- [x] Define broadcasting validation

## Comparison and Selection Schemas

- [x] Define comparison schema
- [x] Define where/select schema
- [x] Define boolean mask validation
- [x] Define output dtype rules

## Reduction Schemas

- [x] Define reduction schema
- [x] Define supported reduction operator identifiers
- [x] Define axis validation
- [x] Define keep-dimension behavior
- [x] Define output dtype rules
- [x] Define empty-input behavior placeholder

## Linear Algebra Schemas

- [x] Define matmul schema
- [x] Define batched matmul schema
- [x] Define transpose flags
- [x] Define accumulation dtype placeholder
- [x] Define precision policy placeholder

## Indexing Schemas

- [x] Define gather schema
- [x] Define index-select schema
- [x] Define scatter schema
- [x] Define scatter-add schema
- [x] Define index dtype rules
- [x] Define duplicate-index behavior placeholder

## Concatenation Schema

- [x] Define concat schema
- [x] Define axis validation
- [x] Define input shape compatibility validation

## Random Generation Schema

- [x] Define random-uniform schema
- [x] Define random-normal schema
- [x] Define optional seed behavior
- [x] Define Provider determinism disclaimer

## Provider Advertisement

- [x] Allow Providers to advertise supported operation schemas
- [x] Allow Providers to advertise dtype support per operation schema
- [x] Allow Providers to advertise layout support per operation schema
- [x] Allow Providers to advertise precision support per operation schema

## Runtime Validation

- [x] Validate operation identifier
- [x] Validate operation attributes
- [x] Validate input arity
- [x] Validate input descriptors
- [x] Validate output descriptors
- [x] Validate dtype compatibility
- [x] Validate shape compatibility
- [x] Validate Provider support before execution

## Exclusions

- [x] Exclude convolution schemas from this change
- [x] Exclude pooling schemas from this change
- [x] Exclude attention-specific fused schemas from this change
- [x] Exclude quantized operation schemas from this change
- [x] Exclude custom kernel schemas from this change
- [x] Exclude autograd and training graphs from this change

## Documentation

- [x] Document operation schema format
- [x] Document how operation schemas fit inside Compute Graph
- [x] Document why operation schemas are not WIT functions
- [x] Document initial supported operation schemas
- [x] Document excluded schemas and future changes
