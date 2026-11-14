// What a Provider advertises lives in `compute_support`; it keeps its paths here.
use crate::affinity::*;
use crate::capability::*;
pub use crate::compute_support::{
    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION, ComputeAttributeKind,
    ComputeCapabilitySupport, ComputeDataMovementDescriptor, ComputeDataMovementSource,
    ComputeDataMovementSupport, ComputeDiagnostic, ComputeError, ComputeErrorCode,
    ComputeErrorPhase, ComputeErrorSeverity, ComputeOperationAttribute,
    ComputeOperationAttributeRule, ComputeOperationDescriptor, ComputeOperationInputRule,
    ComputeOperationOutputRule, ComputeOperationSchema, ComputeOperationSupport,
    ComputeOutputDTypeRule, ComputeOutputShapeRule, ComputePlacementIntent, DTypeSupport,
    DataMovementSupport, DeviceComputeSupport, HostBufferDescriptor, HostBufferEncoding,
    LayoutSupport, OperationFamilySupport, OperationSchemaSupport, PrecisionSupport,
    ProviderComputeAdvertisement, RecoveryHint, ShapeLimitSupport,
};
pub use crate::compute_types::*;
use crate::provider::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};
/// WIT package that defines the Compute capability contract.
pub const COMPUTE_WIT_PACKAGE: &str = "magnetar:compute";
/// WIT interface implemented by Compute providers.
pub const COMPUTE_WIT_INTERFACE: &str = COMPUTE_CAPABILITY_ID;

pub(crate) fn effective_compute_advertisement(
    metadata: &ProviderMetadata,
) -> ProviderComputeAdvertisement {
    let mut advertisement = metadata.compute_advertisement.clone();
    for capability in metadata
        .capabilities
        .iter()
        .filter(|capability| capability.id.as_str() == COMPUTE_CAPABILITY_ID)
    {
        advertisement.capability.versions.insert(capability.version);
    }
    for (family, support) in &metadata.compute_operation_support {
        advertisement
            .operation_families
            .entry(*family)
            .or_insert_with(|| {
                OperationFamilySupport::from_operation_support(*family, support.clone())
            });
    }
    for (operation, support) in &metadata.compute_operation_schema_support {
        let family = initial_compute_operation_schemas()
            .get(operation)
            .map(|schema| schema.family)
            .unwrap_or(ComputeOperationFamily::DescriptorAndView);
        advertisement
            .operation_schemas
            .entry(operation.clone())
            .or_insert_with(|| {
                OperationSchemaSupport::from_operation_support(
                    operation.clone(),
                    family,
                    support.clone(),
                )
            });
    }
    for (kind, support) in &metadata.compute_data_movement_support {
        advertisement
            .data_movement
            .entry(*kind)
            .or_insert_with(|| DataMovementSupport::from_compute_support(*kind, support.clone()));
    }
    advertisement
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationValidationResult {
    pub schema: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub input_count: usize,
    pub output_count: usize,
}

pub fn initial_compute_operation_schemas() -> BTreeMap<ComputeOperationId, ComputeOperationSchema> {
    let mut schemas = BTreeMap::new();
    let same_output = || {
        ComputeOperationOutputRule::new(
            1,
            ComputeOutputDTypeRule::SameAsInput(0),
            ComputeOutputShapeRule::SameAsInput(0),
        )
    };
    for id in [
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.squeeze",
        "tensor.unsqueeze",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::DescriptorAndView,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.reshape",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.broadcast",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1).with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    for op in [
        "abs", "neg", "exp", "log", "sqrt", "recip", "sin", "cos", "tanh", "relu", "silu", "gelu",
        "erf", "floor", "ceil", "round",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.unary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    for op in ["add", "sub", "mul", "div", "maximum", "minimum"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.binary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    for op in ["eq", "ne", "lt", "le", "gt", "ge"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("comparison.{op}"),
                ComputeOperationFamily::ComparisonAndSelection,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::Boolean,
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "selection.where",
            ComputeOperationFamily::ComparisonAndSelection,
            ComputeOperationInputRule::exactly(3)
                .with_boolean_input(0)
                .with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(1),
                ComputeOutputShapeRule::BroadcastInputs,
            ),
        ),
    );
    for op in ["sum", "mean", "min", "max", "argmin", "argmax"] {
        let dtype = if matches!(op, "argmin" | "argmax") {
            ComputeOutputDTypeRule::IntegerIndex
        } else {
            ComputeOutputDTypeRule::SameAsInput(0)
        };
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("reduction.{op}"),
                ComputeOperationFamily::Reduction,
                ComputeOperationInputRule::exactly(1),
                ComputeOperationOutputRule::new(
                    1,
                    dtype,
                    ComputeOutputShapeRule::Reduction {
                        axes_attribute: "axes".into(),
                        keep_dimensions_attribute: "keep-dimensions".into(),
                    },
                ),
            )
            .with_attribute(
                "axes",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Axes),
            )
            .with_attribute(
                "keep-dimensions",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            ),
        );
    }
    for (id, shape_rule) in [
        (
            "linalg.matmul",
            ComputeOutputShapeRule::MatrixMultiplication,
        ),
        (
            "linalg.batched-matmul",
            ComputeOutputShapeRule::BatchedMatrixMultiplication,
        ),
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::LinearAlgebra,
                ComputeOperationInputRule::exactly(2).with_same_dtype(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    shape_rule,
                ),
            )
            .with_attribute(
                "transpose-a",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "transpose-b",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "accumulation-dtype",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "precision",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::String),
            ),
        );
    }
    for id in ["tensor.gather", "tensor.index-select"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(2).with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::ProviderDefined,
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            ),
        );
    }
    for id in ["tensor.scatter", "tensor.scatter-add"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(3)
                    .with_same_dtype()
                    .with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::SameAsInput(0),
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            )
            .with_provider_specific_semantics(),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.concat",
            ComputeOperationFamily::IndexingAndUpdate,
            ComputeOperationInputRule::at_least(1).with_same_dtype(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::Concatenation {
                    axis_attribute: "axis".into(),
                },
            ),
        )
        .with_attribute(
            "axis",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
        ),
    );
    for id in ["random.uniform", "random.normal"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::RandomGeneration,
                ComputeOperationInputRule::exactly(0),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::ExplicitAttribute("dtype".into()),
                    ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
                ),
            )
            .with_attribute(
                "shape",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
            )
            .with_attribute(
                "dtype",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "seed",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Integer),
            ),
        );
    }
    schemas
}

fn insert_schema(
    schemas: &mut BTreeMap<ComputeOperationId, ComputeOperationSchema>,
    schema: ComputeOperationSchema,
) {
    schemas.insert(schema.id.clone(), schema);
}

pub(crate) fn validate_compute_operation_schema(
    operation: &ComputeOperationDescriptor,
) -> Result<Option<ComputeOperationValidationResult>, ComputeValidationError> {
    let Some(schema_id) = &operation.schema_id else {
        return Ok(None);
    };
    let schemas = initial_compute_operation_schemas();
    let schema = schemas
        .get(schema_id)
        .ok_or_else(|| ComputeValidationError::UnknownOperationSchema(schema_id.clone()))?;
    if schema.family != operation.family {
        return Err(ComputeValidationError::UnknownOperationFamily(
            operation.family.id().into(),
        ));
    }
    validate_operation_attributes(schema, operation)?;

    if operation.tensors.len() < schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "operation declares {} tensor descriptors but schema requires {} output(s)",
                operation.tensors.len(),
                schema.output_rule.output_count
            ),
        });
    }
    let input_count = operation.tensors.len() - schema.output_rule.output_count;
    validate_operation_arity(schema, input_count)?;
    let (inputs, outputs) = operation.tensors.split_at(input_count);
    validate_operation_input_rule(schema, inputs)?;
    validate_operation_output_rule(schema, operation, inputs, outputs)?;
    Ok(Some(ComputeOperationValidationResult {
        schema: schema.id.clone(),
        family: schema.family,
        input_count,
        output_count: outputs.len(),
    }))
}

fn validate_operation_attributes(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
) -> Result<(), ComputeValidationError> {
    for name in operation.attributes.keys() {
        if !schema.attributes.contains_key(name) {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: schema.id.clone(),
                attribute: name.clone(),
                reason: "attribute is not defined by the operation schema".into(),
            });
        }
    }
    for (name, rule) in &schema.attributes {
        match operation.attributes.get(name) {
            Some(value) if value.kind() == rule.kind => {}
            Some(value) => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: format!("expected {:?}, found {:?}", rule.kind, value.kind()),
                });
            }
            None if rule.required => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: "required attribute is missing".into(),
                });
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_operation_arity(
    schema: &ComputeOperationSchema,
    input_count: usize,
) -> Result<(), ComputeValidationError> {
    let too_few = input_count < schema.input_rule.min_inputs;
    let too_many = schema
        .input_rule
        .max_inputs
        .is_some_and(|max| input_count > max);
    if too_few || too_many {
        let expected = match schema.input_rule.max_inputs {
            Some(max) if max == schema.input_rule.min_inputs => max.to_string(),
            Some(max) => format!("{}..={max}", schema.input_rule.min_inputs),
            None => format!("at least {}", schema.input_rule.min_inputs),
        };
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: schema.id.clone(),
            expected,
            found: input_count,
        });
    }
    Ok(())
}

fn validate_operation_input_rule(
    schema: &ComputeOperationSchema,
    inputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    for index in &schema.input_rule.boolean_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(ComputeDType::Boolean)) => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "input must have boolean dtype".into(),
                });
            }
        }
    }
    for index in &schema.input_rule.integer_index_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(dtype)) if dtype.is_integer() => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "index input must have an integer dtype".into(),
                });
            }
        }
    }
    if schema.input_rule.require_same_dtype {
        let Some(first) = inputs.first().map(|tensor| &tensor.dtype) else {
            return Ok(());
        };
        if inputs.iter().any(|tensor| &tensor.dtype != first) {
            return Err(ComputeValidationError::UnsupportedDType {
                family: schema.family,
                dtype: portable_dtype(first).unwrap_or(ComputeDType::UInt8),
            });
        }
    }
    if schema.input_rule.allow_broadcast && inputs.len() > 1 {
        broadcast_shape(inputs.iter().map(|tensor| &tensor.shape)).map_err(|reason| {
            ComputeValidationError::InvalidShape {
                reason: format!("operation schema '{}': {reason}", schema.id),
            }
        })?;
    }
    Ok(())
}

fn validate_operation_output_rule(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    if outputs.len() != schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "expected {} output(s), found {}",
                schema.output_rule.output_count,
                outputs.len()
            ),
        });
    }
    for output in outputs {
        validate_output_dtype(schema, operation, inputs, output)?;
    }
    validate_output_shape(schema, operation, inputs, outputs)
}

fn validate_output_dtype(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), ComputeValidationError> {
    match &schema.output_rule.dtype {
        ComputeOutputDTypeRule::SameAsInput(index) => {
            let Some(input) = inputs.get(*index) else {
                return Ok(());
            };
            if output.dtype != input.dtype {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must match input dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::Boolean => {
            if output.dtype != DTypeDescriptor::Portable(ComputeDType::Boolean) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be boolean".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ExplicitAttribute(name) => {
            if let Some(ComputeOperationAttribute::DType(dtype)) = operation.attributes.get(name)
                && output.dtype != DTypeDescriptor::Portable(*dtype)
            {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: format!("output dtype must match '{name}' attribute"),
                });
            }
        }
        ComputeOutputDTypeRule::IntegerIndex => {
            if !matches!(&output.dtype, DTypeDescriptor::Portable(dtype) if dtype.is_integer()) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be an integer index dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ProviderDefined => {}
    }
    Ok(())
}

fn validate_output_shape(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    let Some(output) = outputs.first() else {
        return Ok(());
    };
    let expected = match &schema.output_rule.shape {
        ComputeOutputShapeRule::SameAsInput(index) => inputs.get(*index).map(|t| t.shape.clone()),
        ComputeOutputShapeRule::ExplicitAttribute(name) => match operation.attributes.get(name) {
            Some(ComputeOperationAttribute::Shape(shape)) => Some(shape.clone()),
            _ => None,
        },
        ComputeOutputShapeRule::BroadcastInputs => {
            Some(broadcast_shape(inputs.iter().map(|t| &t.shape))?)
        }
        ComputeOutputShapeRule::Reduction {
            axes_attribute,
            keep_dimensions_attribute,
        } => Some(reduction_shape(
            &schema.id,
            inputs.first(),
            operation.attributes.get(axes_attribute),
            operation.attributes.get(keep_dimensions_attribute),
        )?),
        ComputeOutputShapeRule::MatrixMultiplication => Some(matmul_shape(&schema.id, inputs)?),
        ComputeOutputShapeRule::BatchedMatrixMultiplication => {
            Some(batched_matmul_shape(&schema.id, inputs)?)
        }
        ComputeOutputShapeRule::Concatenation { axis_attribute } => Some(concat_shape(
            &schema.id,
            inputs,
            operation.attributes.get(axis_attribute),
        )?),
        ComputeOutputShapeRule::ProviderDefined => None,
    };
    if let Some(expected) = expected
        && output.shape != expected
    {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "output shape {:?} does not match expected {:?}",
                output.shape.dimensions, expected.dimensions
            ),
        });
    }
    Ok(())
}

fn portable_dtype(dtype: &DTypeDescriptor) -> Option<ComputeDType> {
    match dtype {
        DTypeDescriptor::Portable(dtype) => Some(*dtype),
        DTypeDescriptor::ProviderSpecific { .. } => None,
    }
}

fn broadcast_shape<'a>(
    shapes: impl IntoIterator<Item = &'a ShapeDescriptor>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let shapes = shapes.into_iter().collect::<Vec<_>>();
    let max_rank = shapes
        .iter()
        .map(|shape| shape.dimensions.len())
        .max()
        .unwrap_or(0);
    let mut result = vec![1_u64; max_rank];
    for shape in shapes {
        for (offset, dimension) in shape.dimensions.iter().rev().enumerate() {
            let index = max_rank - 1 - offset;
            let current = result[index];
            if current == 1 {
                result[index] = *dimension;
            } else if *dimension == 1 || current == *dimension {
                continue;
            } else {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "dimensions {current} and {dimension} are not broadcast-compatible"
                    ),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(result))
}

fn reduction_shape(
    operation: &ComputeOperationId,
    input: Option<&TensorDescriptor>,
    axes: Option<&ComputeOperationAttribute>,
    keep_dimensions: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let input = input.ok_or_else(|| ComputeValidationError::InvalidOperationArity {
        operation: operation.clone(),
        expected: "1".into(),
        found: 0,
    })?;
    let axes = match axes {
        Some(ComputeOperationAttribute::Axes(axes)) => axes,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axes".into(),
                reason: "axes attribute is required".into(),
            });
        }
    };
    let keep = matches!(
        keep_dimensions,
        Some(ComputeOperationAttribute::Boolean(true))
    );
    let rank = input.shape.dimensions.len() as u64;
    for axis in axes {
        if *axis >= rank {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!("reduction axis {axis} is outside rank {rank}"),
            });
        }
    }
    let axis_set = axes.iter().copied().collect::<BTreeSet<_>>();
    let dimensions = input
        .shape
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| {
            if axis_set.contains(&(index as u64)) {
                keep.then_some(1)
            } else {
                Some(*dimension)
            }
        })
        .collect::<Vec<_>>();
    Ok(ShapeDescriptor::new(dimensions))
}

fn matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() != 2 || rhs.shape.dimensions.len() != 2 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication requires rank-2 inputs".into(),
        });
    }
    if lhs.shape.dimensions[1] != rhs.shape.dimensions[0] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    Ok(ShapeDescriptor::new([
        lhs.shape.dimensions[0],
        rhs.shape.dimensions[1],
    ]))
}

fn batched_matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() < 3 || rhs.shape.dimensions.len() < 3 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication requires rank >= 3 inputs".into(),
        });
    }
    let lhs_rank = lhs.shape.dimensions.len();
    let rhs_rank = rhs.shape.dimensions.len();
    if lhs.shape.dimensions[lhs_rank - 1] != rhs.shape.dimensions[rhs_rank - 2] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    let mut batch = broadcast_shape(
        [
            ShapeDescriptor::new(lhs.shape.dimensions[..lhs_rank - 2].to_vec()),
            ShapeDescriptor::new(rhs.shape.dimensions[..rhs_rank - 2].to_vec()),
        ]
        .iter(),
    )?
    .dimensions;
    batch.push(lhs.shape.dimensions[lhs_rank - 2]);
    batch.push(rhs.shape.dimensions[rhs_rank - 1]);
    Ok(ShapeDescriptor::new(batch))
}

fn concat_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
    axis: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let Some(first) = inputs.first() else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "at least 1".into(),
            found: 0,
        });
    };
    let axis = match axis {
        Some(ComputeOperationAttribute::Integer(axis)) if *axis >= 0 => *axis as usize,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axis".into(),
                reason: "axis must be a non-negative integer".into(),
            });
        }
    };
    if axis >= first.shape.dimensions.len() {
        return Err(ComputeValidationError::InvalidShape {
            reason: "concatenation axis is outside input rank".into(),
        });
    }
    let mut dimensions = first.shape.dimensions.clone();
    for input in &inputs[1..] {
        if input.shape.dimensions.len() != dimensions.len() {
            return Err(ComputeValidationError::InvalidShape {
                reason: "all concatenation inputs must have the same rank".into(),
            });
        }
        for (index, dimension) in input.shape.dimensions.iter().enumerate() {
            if index == axis {
                dimensions[index] = dimensions[index].checked_add(*dimension).ok_or(
                    ComputeValidationError::SizeOverflow {
                        reason: "concatenated dimension overflows u64".into(),
                    },
                )?;
            } else if dimensions[index] != *dimension {
                return Err(ComputeValidationError::InvalidShape {
                    reason: "non-concatenated dimensions must match".into(),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(dimensions))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationRequest {
    pub family_id: String,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
    pub tensors: Vec<TensorDescriptor>,
}
impl ComputeOperationRequest {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            dtype: None,
            layout: None,
            precision: None,
            tensors: Vec::new(),
        }
    }
    pub fn with_dtype(mut self, dtype: ComputeDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn with_layout(mut self, layout: ComputeLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn with_precision(mut self, precision: ComputePrecision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_tensor(mut self, tensor: TensorDescriptor) -> Self {
        self.tensors.push(tensor);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeInputValue {
    TensorResource(TensorResourceDescriptor),
    TensorDescriptor(TensorDescriptor),
    Constant(TensorDescriptor),
}
impl ComputeInputValue {
    pub(crate) fn descriptor(&self) -> &TensorDescriptor {
        match self {
            Self::TensorResource(resource) => &resource.descriptor,
            Self::TensorDescriptor(descriptor) | Self::Constant(descriptor) => descriptor,
        }
    }
    pub(crate) fn affinity(&self) -> Option<&ResourceAffinity> {
        match self {
            Self::TensorResource(resource) => Some(&resource.affinity),
            Self::TensorDescriptor(_) | Self::Constant(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeInput {
    pub id: ComputeInputId,
    pub value: ComputeInputValue,
}
impl ComputeInput {
    pub fn new(id: ComputeInputId, value: ComputeInputValue) -> Self {
        Self { id, value }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeValueRef {
    Input(ComputeInputId),
    NodeOutput {
        node: ComputeNodeId,
        output: ComputeOutputId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNodeOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
}
impl ComputeNodeOutput {
    pub fn new(id: ComputeOutputId, descriptor: TensorDescriptor) -> Self {
        Self { id, descriptor }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNode {
    pub id: ComputeNodeId,
    pub operation: ComputeOperationDescriptor,
    pub inputs: Vec<ComputeValueRef>,
    pub outputs: Vec<ComputeNodeOutput>,
}
impl ComputeNode {
    pub fn new(id: ComputeNodeId, operation: ComputeOperationDescriptor) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeValueRef) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_output(mut self, output: ComputeNodeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOutput {
    pub id: ComputeOutputId,
    pub source: ComputeValueRef,
}
impl ComputeOutput {
    pub fn new(id: ComputeOutputId, source: ComputeValueRef) -> Self {
        Self { id, source }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeGraph {
    pub id: ComputeGraphId,
    pub inputs: Vec<ComputeInput>,
    pub nodes: Vec<ComputeNode>,
    pub outputs: Vec<ComputeOutput>,
}
impl ComputeGraph {
    pub fn new(id: ComputeGraphId) -> Self {
        Self {
            id,
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeInput) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_node(mut self, node: ComputeNode) -> Self {
        self.nodes.push(node);
        self
    }
    pub fn with_output(mut self, output: ComputeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSubmissionState {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}
impl ComputeSubmissionState {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionResult {
    pub state: ComputeSubmissionState,
    pub outputs: Vec<TensorResourceDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeSubmission {
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub affinity: ResourceAffinity,
    state: ComputeSubmissionState,
    result: Option<ComputeExecutionResult>,
}
impl ComputeSubmission {
    pub fn new(
        graph: ComputeGraphId,
        provider: ProviderBinding,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            graph,
            provider,
            affinity,
            state: ComputeSubmissionState::Pending,
            result: None,
        }
    }
    pub fn state(&self) -> ComputeSubmissionState {
        self.state
    }
    pub fn start(&mut self) -> Result<(), ComputeValidationError> {
        if self.state != ComputeSubmissionState::Pending {
            return Err(ComputeValidationError::InvalidState {
                reason: format!("cannot start submission from {:?} state", self.state),
            });
        }
        self.state = ComputeSubmissionState::Running;
        Ok(())
    }
    pub fn complete(
        &mut self,
        outputs: Vec<TensorResourceDescriptor>,
    ) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Completed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs,
        });
        Ok(())
    }
    pub fn cancel(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Cancelled;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn fail(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Failed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn result(&self) -> Option<&ComputeExecutionResult> {
        self.result.as_ref()
    }
}
pub struct ComputeGraphValidationReport {
    pub provider: ProviderBinding,
    pub graph: ComputeGraphId,
    pub node_count: usize,
    pub input_count: usize,
    pub output_count: usize,
}

pub(crate) fn ensure_non_empty_id(kind: &str, value: &str) -> Result<(), ComputeValidationError> {
    if value.trim().is_empty() {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("{kind} identifier must not be empty"),
        });
    }
    Ok(())
}

pub(crate) fn insert_unique<T: Ord + Clone + fmt::Display>(
    ids: &mut BTreeSet<T>,
    kind: &str,
    id: &T,
) -> Result<(), ComputeValidationError> {
    if !ids.insert(id.clone()) {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("duplicate {kind} identifier '{id}'"),
        });
    }
    Ok(())
}

pub(crate) fn resolve_compute_value_descriptor<'a>(
    current_node: Option<&ComputeNodeId>,
    value: &ComputeValueRef,
    input_descriptors: &'a BTreeMap<ComputeInputId, TensorDescriptor>,
    output_descriptors: &'a BTreeMap<(ComputeNodeId, ComputeOutputId), TensorDescriptor>,
    completed_nodes: &BTreeSet<ComputeNodeId>,
) -> Result<&'a TensorDescriptor, ComputeValidationError> {
    match value {
        ComputeValueRef::Input(input) => {
            input_descriptors
                .get(input)
                .ok_or_else(|| ComputeValidationError::MissingInput {
                    input: input.clone(),
                })
        }
        ComputeValueRef::NodeOutput { node, output } => {
            if !completed_nodes.contains(node) {
                return Err(ComputeValidationError::CyclicGraph {
                    node: current_node.cloned().unwrap_or_else(|| node.clone()),
                    depends_on: node.clone(),
                });
            }
            output_descriptors
                .get(&(node.clone(), output.clone()))
                .ok_or_else(|| ComputeValidationError::MissingOutput {
                    node: node.clone(),
                    output: output.clone(),
                })
        }
    }
}

/// Returns the canonical hardware-independent Compute capability declaration.
///
/// Providers add this value to their metadata to advertise support for the
/// `magnetar:compute/run@2.0.0` WIT contract.
pub fn compute_capability() -> Capability {
    Capability::new(
        CapabilityId::new(COMPUTE_CAPABILITY_ID),
        COMPUTE_CAPABILITY_VERSION,
        CapabilityDescriptor::new("coarse provider-owned graph execution").with_contract(
            WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ),
        ),
    )
}

#[cfg(test)]
mod tests;
