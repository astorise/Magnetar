//! Weight binding and transactional weight materialization for a Model Instance.

use super::*;

/// Rejects a weight payload whose bytes do not match the known-correct
/// digest (task 6.5: artifact byte changes / digest rejection) before this
/// fixture's weights are materialized -- a corrupted or substituted payload
/// must fail closed here, not silently bind whatever bytes happen to be in
/// memory. The E2E fixture's own concern (it has a fixed, hard-coded
/// expected digest; a real Model Artifact loader would check against the
/// digest declared in its manifest instead), kept separate from
/// `materialize_model_instance_weights`, which has no fixture dependency at
/// all (Correctif 9: no model-family- or fixture-specific logic in the
/// generic weight-materialization step itself).
pub(super) fn bind_qwen_fixture_weights(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let actual_digest = e2e_fixture_weight_digest(&fixture.weights);
    if actual_digest != E2E_FIXTURE_WEIGHT_DIGEST {
        return Err(E2eConformanceError::FixtureInvalid {
            reason: format!(
                "fixture weight payload digest {actual_digest} does not match expected {E2E_FIXTURE_WEIGHT_DIGEST}"
            ),
        });
    }
    // Task group 8 / Correctif 6: materialize from the real, checked-in
    // Safetensors artifact's actual bytes (`e2e_fixture_weights_from_real_artifact`),
    // not the in-memory `fixture.weights` the digest check above just
    // verified -- `materialize-weights-from-real-model-artifact`'s parity
    // test (`tests::e2e_fixture_real_artifact_weights_match_in_memory_weights`)
    // proves the two are bit-identical for an untampered fixture, so this
    // is a real change in *source*, not in observable behavior for every
    // existing caller of this function.
    let mut real_weights = e2e_fixture_weights_from_real_artifact(&fixture.config)?;
    qwen_weights_with_derived_lm_head(fixture, &mut real_weights)?;
    materialize_model_instance_weights(
        runtime,
        instance,
        fixture.manifest.id.name.as_str(),
        &real_weights,
    )
    .map_err(E2eConformanceError::from)
}

/// `make-first-native-cuda-hot-path-device-resident`'s Decision 5: a
/// tied-embeddings model's `lm_head` projection is the transpose of
/// `token_embedding`, computed once here (Model Load, called once per
/// instance) and staged as a genuine, independent weight under its own
/// name -- not recomputed by `resolve_first_native_weight_edge` on every
/// generation step's dispatch. Deliberately Qwen-fixture-specific code
/// (shared by every path that materializes this fixture's weights for a
/// tied-embeddings configuration), not the generic
/// `materialize_model_instance_weights`: the transpose is a
/// tied-embeddings *model* concept, not a Runtime concept. A no-op when
/// `weights` already has an explicit `lm_head` entry (a non-tied
/// configuration, or a caller that already staged one itself).
pub(super) fn qwen_weights_with_derived_lm_head(
    fixture: &E2eFixture,
    weights: &mut BTreeMap<String, HostTensor>,
) -> Result<(), E2eConformanceError> {
    if fixture.config.tied_embeddings && !weights.contains_key("lm_head") {
        let reason = "tied-embeddings fixture has no token_embedding weight to derive lm_head from";
        let token_embedding =
            weights
                .get("token_embedding")
                .ok_or(E2eConformanceError::FixtureInvalid {
                    reason: reason.into(),
                })?;
        let lm_head = transpose_rows_cols(token_embedding)?;
        weights.insert("lm_head".to_string(), lm_head);
    }
    Ok(())
}

/// Materializes a Model Instance's weight tensors as Runtime resources:
/// creates one `TensorResourceId` per declared weight, writes its bytes
/// into the registered Reference CPU Provider's storage (the same
/// Provider-owned storage compute intermediates live in -- not a private
/// Rust-side map), accounts the allocation through Runtime's
/// `MemoryManager`, and records the name -> resource binding on the Model
/// Instance so graph execution can look weights up by name through the
/// instance rather than a caller-held map. Implements
/// `model-loading-materializes-weight-resources`'s "Model Loading
/// Materializes Weight Resources" requirement.
///
/// Generic over any name -> tensor payload; contains no Qwen- or
/// fixture-specific logic (Correctif 9). It is still called as a step
/// *after* `load_model`/`create_model_instance` rather than from inside
/// `ModelLoadingCoordinator::load()` itself -- `load()` has no Provider
/// access and materializes nothing for any artifact type today (task group
/// 8's still-open, deeper gap; see design.md and this change's task list).
/// This function is the generic replacement for what used to be a
/// Qwen-fixture-only helper baked into this file; it belongs to the same
/// architectural layer as `create_model_instance`, not to any one model
/// family. Deliberately *not* relocated into `model_loading.rs`: that
/// module has no dependency on `Runtime` (`load()` itself takes only
/// `&mut MemoryManager`), and this function needs `&mut Runtime` for
/// Provider resolution and the failure-path lifecycle transition below --
/// moving it would make `model_loading.rs` depend on `runtime.rs`, which
/// already depends on `model_loading.rs`, for no behavioral benefit.
///
/// `pub`: this is the one legitimate way -- for production code or an
/// external embedder alike -- to turn named weight bytes into bound
/// resources for a Model Instance
/// (`bind-model-loading-evidence-to-validated-artifact`). Its success is
/// itself the proof `derive_effective_readiness_checks` trusts (via the
/// `MaterializationEvidence` this call's `WeightMaterializationTransaction::commit`
/// mints); there is no separate token for a caller to construct or forge.
///
/// `Ok(())` means the supplied `weights` staged and committed
/// successfully -- it does **not** by itself mean the instance reached
/// `Ready`. `commit` only marks the instance `Ready` if the full
/// Runtime-derived readiness gate (mandatory inventory complete, Provider
/// and Device ready, ...) is satisfied afterward; an empty or partial
/// `weights` map against an instance with an unmet mandatory inventory
/// commits and evidences what it was given, correctly leaving the
/// instance non-`Ready`. Check `runtime.model_instance(instance)?.
/// lifecycle()`/`.readiness()` (or call `warm_model_instance` again once
/// conditions change) to find out. A further audit of PR #36 found the
/// previous behavior -- unconditionally marking `Ready` once staging
/// succeeded -- let an incomplete materialization reach `Ready`, violating
/// `model-loading`'s pre-existing "Model Loading Does Not Bypass Instance
/// Readiness" and "Partial Loading Policy" requirements.
pub fn materialize_model_instance_weights(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    artifact_owner: &str,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<(), InferenceApiError> {
    let mut transaction = WeightMaterializationTransaction::begin(runtime, instance)?;
    for (name, tensor) in weights {
        if let Err(error) =
            transaction.stage_weight(runtime, instance, artifact_owner, name, tensor)
        {
            transaction.abort(runtime);
            // The instance never reached Ready for this attempt (`create()`
            // no longer auto-readies, and `commit` -- which alone calls
            // `mark_ready` -- is never reached on this path), so there is
            // nothing to demote; transition it to `Failed` so the failure
            // is durably visible on the instance itself, not only in this
            // `Result::Err`. Best-effort: if the instance cannot be found,
            // or is not presently in a state `Failed` legally transitions
            // from, the original materialization error below is still what
            // is returned either way.
            if let Ok(model_instance) = runtime.model_instances_mut().instance_mut(instance) {
                let _ = model_instance.transition_to(ModelInstanceLifecycleState::Failed);
            }
            return Err(error);
        }
    }
    transaction.commit(runtime, instance)
}

/// Streaming counterpart to [`materialize_model_instance_weights`]
/// (`implement-production-qwen-model-loading` task group 6 / "Production
/// Loading Supports Bounded Tensor Payload Access"): reads, validates,
/// converts, and stages each of `tensors` one at a time from
/// `payload_source`, never requiring the entire model to first exist as a
/// whole-model `BTreeMap<String, HostTensor>`. Each tensor's `HostTensor`
/// is staged into Provider-owned storage and dropped before the next one
/// is even read (`host_tensor_from_payload_source`'s only allocation does
/// not outlive this loop body's iteration), so peak transient host staging
/// stays bounded by one tensor's size, independent of total model size.
///
/// Reuses the same `WeightMaterializationTransaction` commit/rollback
/// semantics as the whole-map entry point: `commit` publishes bindings/
/// readiness only after every tensor in `tensors` stages successfully, and
/// any failure releases every Provider-side tensor, Memory Manager
/// allocation, and pending binding staged so far in this attempt.
pub fn stream_materialize_model_instance_weights(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    artifact_owner: &str,
    tensors: &[ModelTensorMetadata],
    payload_source: &dyn ProductionArtifactPayloadSource,
) -> Result<(), InferenceApiError> {
    let mut transaction = WeightMaterializationTransaction::begin(runtime, instance)?;
    for tensor in tensors {
        let stage_result = stream_stage_tensor(
            &mut transaction,
            runtime,
            instance,
            artifact_owner,
            tensor,
            payload_source,
        );
        if let Err(error) = stage_result {
            transaction.abort(runtime);
            if let Ok(model_instance) = runtime.model_instances_mut().instance_mut(instance) {
                let _ = model_instance.transition_to(ModelInstanceLifecycleState::Failed);
            }
            return Err(error);
        }
    }
    transaction.commit(runtime, instance)
}

/// Reads one tensor's storage bytes and stages it: a block-quantized
/// matrix is first offered packed to the Provider
/// (`execute-direct-quantized-weights`), and only dequantized to `f32` when
/// the Provider declines it; everything else is converted to `f32` as
/// before. The bytes are read once either way.
pub(super) fn stream_stage_tensor(
    transaction: &mut WeightMaterializationTransaction,
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    artifact_owner: &str,
    tensor: &ModelTensorMetadata,
    payload_source: &dyn ProductionArtifactPayloadSource,
) -> Result<(), InferenceApiError> {
    let loading_failed =
        |error: crate::model_loading::ModelLoadingError| InferenceApiError::ModelLoadingFailed {
            reason: error.to_string(),
        };
    let raw = crate::model_loading::read_tensor_payload(tensor, payload_source)
        .map_err(loading_failed)?;
    if let Some(packed) =
        crate::model_loading::packed_weight_from_raw_bytes(tensor, &raw).map_err(loading_failed)?
        && transaction.stage_packed_weight(
            runtime,
            instance,
            artifact_owner,
            &tensor.name,
            packed,
        )?
    {
        return Ok(());
    }
    let host_tensor =
        crate::model_loading::tensor_from_raw_bytes(tensor, &raw).map_err(loading_failed)?;
    transaction.stage_weight(
        runtime,
        instance,
        artifact_owner,
        &tensor.name,
        &host_tensor,
    )
}

/// Stages weight materialization one resource at a time -- Memory Manager
/// admission, then Provider write, then residency registration, each
/// step's error propagated rather than discarded -- and either rolls back
/// everything staged so far ([`Self::abort`]) or publishes it all at once
/// and marks the instance Ready ([`Self::commit`]), mirroring
/// [`KvUpdateTransaction`]'s already-correct pattern for the same class of
/// problem (Correctif 11 / task group 9). Implements
/// `transactional-weight-materialization`'s "Weight Materialization Is
/// Transactional" requirement.
pub(super) struct WeightMaterializationTransaction {
    pub(super) provider_binding: ProviderBinding,
    pub(super) executor: Arc<dyn ProviderExecutionApi>,
    pub(super) staged: Vec<StagedWeight>,
}

pub(super) struct StagedWeight {
    pub(super) name: String,
    pub(super) resource_id: TensorResourceId,
    pub(super) allocation: MemoryAllocationId,
}

impl WeightMaterializationTransaction {
    /// Resolves the Provider bound to `instance`'s placement
    /// (`Runtime::create_model_instance` already cross-validated this
    /// against the prepared plan when the instance was created), falling
    /// back to Reference CPU only when placement left no Provider bound --
    /// `generalize-first-native-provider-dispatch`'s "Runtime Treats
    /// Reference CPU As Normal Provider" fix: this used to hardcode
    /// Reference CPU unconditionally, making a non-CPU-bound Model Instance
    /// materialize its weights through the wrong Provider's storage.
    pub(super) fn begin(
        runtime: &Runtime,
        instance: &ModelInstanceId,
    ) -> Result<Self, InferenceApiError> {
        let provider_binding = runtime
            .model_instance(instance)?
            .definition()
            .placement
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
        Ok(Self {
            provider_binding,
            executor,
            staged: Vec::new(),
        })
    }

    /// Admits `tensor` through the Memory Manager, then writes it into
    /// Provider-owned storage, then registers its residency -- in that
    /// order, each step's error surfacing immediately rather than being
    /// discarded. Does not touch the Model Instance's `resource_bindings`;
    /// that is [`Self::commit`]'s responsibility once every weight in this
    /// attempt has staged successfully.
    pub(super) fn stage_weight(
        &mut self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
        artifact_owner: &str,
        name: &str,
        tensor: &HostTensor,
    ) -> Result<(), InferenceApiError> {
        // Shape/dtype agreement precedes even the content digest check:
        // a declared storage dtype this Runtime cannot materialize at all
        // (anything outside F32/F16/BF16/Q8_0/Q4_K/Q5_K) has no digest
        // check reachable here regardless of what bytes a caller supplies
        // under its name, so without this check a caller could fabricate
        // F32 content under such a tensor's name and have nothing reject
        // it. This applies to every tensor with a declared shape,
        // independent of whether that tensor also has a declared digest
        // -- the two checks guard different things
        // (`seal-runtime-model-trust-and-provenance-authority`).
        let declared_storage_dtype = runtime
            .model_instance(instance)
            .map_err(InferenceApiError::from)?
            .definition()
            .required_weight_shapes
            .get(name)
            .cloned();
        if let Some((expected_shape, expected_dtype)) = declared_storage_dtype.clone()
            && (tensor.shape != expected_shape
                || !matches!(
                    expected_dtype,
                    ModelDType::F32
                        | ModelDType::F16
                        | ModelDType::Bf16
                        | ModelDType::Q8
                        | ModelDType::Q2K
                        | ModelDType::Q3K
                        | ModelDType::Iq4Nl
                        | ModelDType::Iq4Xs
                        | ModelDType::Iq2Xxs
                        | ModelDType::Iq2Xs
                        | ModelDType::Iq2S
                        | ModelDType::Iq3Xxs
                        | ModelDType::Iq3S
                        | ModelDType::Q4K
                        | ModelDType::Q5K
                        | ModelDType::Q6K
                ))
        {
            return Err(InferenceApiError::WeightShapeOrDtypeMismatch {
                reason: format!(
                    "weight resource '{name}' declared shape {expected_shape:?} and dtype {expected_dtype:?}, which this Runtime can only materialize as F32 content matching that shape"
                ),
            });
        }
        // Content verification precedes Memory Manager admission: if the
        // loaded artifact declared a content digest for this tensor name,
        // the bytes supplied here must match it before anything is
        // admitted or written. A caller with legitimate access to this
        // transaction (the one authorized path to bind a weight) can still
        // supply the *wrong* bytes under the *right* name -- this check is
        // what stops that from being accepted as the declared tensor's
        // content (`bind-materialized-weight-content-to-model-artifact-
        // digests`). A tensor whose inventory entry declared no digest is
        // unaffected (`None` means unknown, not "no content required" --
        // the same precedent `required_weight_names` already established).
        //
        // Skipped when the declared storage dtype is not F32: for F16/BF16
        // storage, `tensor` here is already the *converted* F32 content
        // (`host_tensors_from_artifact_bytes` produced it), and the
        // artifact's declared digest describes the original 2-byte-per-
        // element storage bytes, not this 4-byte-per-element conversion --
        // a digest computed over the former can never match the latter, so
        // re-checking here would be structurally wrong, not merely
        // redundant. That original-bytes check already happened inside
        // `host_tensors_from_artifact_bytes` (Decision 8).
        let storage_dtype_is_f32 = matches!(
            declared_storage_dtype.map(|(_, dtype)| dtype),
            None | Some(ModelDType::F32)
        );
        if storage_dtype_is_f32
            && let Some(expected_digest) = runtime
                .model_instance(instance)
                .map_err(InferenceApiError::from)?
                .definition()
                .required_weight_digests
                .get(name)
                .cloned()
            && let Err(error) = expected_digest.verify_bytes(&tensor.content_bytes())
        {
            return Err(InferenceApiError::WeightContentDigestMismatch {
                reason: format!(
                    "weight resource '{name}' content does not match the artifact's declared digest: {error}"
                ),
            });
        }
        let byte_size = tensor.data.len() as u64 * std::mem::size_of::<f32>() as u64;
        self.admit_and_write(
            runtime,
            instance,
            artifact_owner,
            name,
            byte_size,
            |executor, resource_id| {
                executor
                    .write_weight_tensor(resource_id, tensor.clone())
                    .map(|()| true)
            },
        )
        .map(|_| ())
    }

    /// Offers `weight`, the packed form of the declared weight `name`, to
    /// the Provider (`execute-direct-quantized-weights`). `Ok(true)`: the
    /// Provider stored it and it is staged exactly like a dequantized
    /// weight, admitted at its packed size. `Ok(false)`: the Provider
    /// declined and nothing was staged; the caller stages the dequantized
    /// weight instead. The caller has verified the packed bytes against
    /// the tensor's declared content digest.
    pub(super) fn stage_packed_weight(
        &mut self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
        artifact_owner: &str,
        name: &str,
        weight: crate::packed_weight::PackedWeight,
    ) -> Result<bool, InferenceApiError> {
        let declared = runtime
            .model_instance(instance)
            .map_err(InferenceApiError::from)?
            .definition()
            .required_weight_shapes
            .get(name)
            .cloned();
        if let Some((expected_shape, _)) = &declared
            && expected_shape.as_slice() != weight.logical_shape().as_slice()
        {
            return Err(InferenceApiError::WeightShapeOrDtypeMismatch {
                reason: format!(
                    "packed weight '{name}' has logical shape {:?} but the artifact declares {expected_shape:?}",
                    weight.logical_shape()
                ),
            });
        }
        let byte_size = weight.bytes().len() as u64;
        self.admit_and_write(
            runtime,
            instance,
            artifact_owner,
            name,
            byte_size,
            |executor, resource_id| {
                executor
                    .write_packed_weight_tensor(resource_id, weight)
                    .map(|written| written == crate::packed_weight::PackedWeightWrite::Stored)
            },
        )
    }

    /// Admits `byte_size` bytes for `name` through the Memory Manager, runs
    /// `write` against the Provider, then registers the resource's
    /// residency -- in that order, each step's error surfacing immediately.
    /// `write` answers whether the Provider stored the resource; `false`
    /// (a declined packed weight) releases the admission and stages
    /// nothing.
    pub(super) fn admit_and_write(
        &mut self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
        artifact_owner: &str,
        name: &str,
        byte_size: u64,
        write: impl FnOnce(
            &dyn ProviderExecutionApi,
            TensorResourceId,
        ) -> Result<bool, crate::scheduler::ProviderExecutionError>,
    ) -> Result<bool, InferenceApiError> {
        let resource_id = TensorResourceId::new(format!("model.{instance}.weight.{name}"));
        // Admission SHALL precede Provider materialization: reserve the
        // resource's memory before writing its bytes into Provider-owned
        // storage, not after.
        let allocation = runtime
            .memory_mut()
            .allocate(MemoryAllocationRequest::new(
                MemoryAllocationClass::ModelArtifact,
                byte_size,
                MemoryPlacement::ProviderOwnedOpaque(self.provider_binding.clone()),
                MemoryAllocationOwner::InferenceArtifact(artifact_owner.into()),
            ))
            .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
                reason: format!("failed to account weight resource '{name}': {error}"),
            })?;
        match write(self.executor.as_ref(), resource_id.clone()) {
            Ok(true) => {}
            // Declined: nothing was stored, so nothing is admitted either.
            Ok(false) => {
                let _ = runtime.memory_mut().release(allocation.id);
                return Ok(false);
            }
            Err(error) => {
                // Admission above already succeeded: release it before
                // propagating, so a write failure leaves no trace in the
                // Memory Manager ledger.
                let _ = runtime.memory_mut().release(allocation.id);
                return Err(InferenceApiError::ProviderTensorWriteFailed {
                    reason: format!(
                        "failed to write weight resource '{name}' to Provider storage: {error}"
                    ),
                });
            }
        }
        if let Err(error) = runtime.memory_mut().record_tensor_residency(
            TensorResidency::new(
                resource_id.clone(),
                MemoryPlacement::ProviderOwnedOpaque(self.provider_binding.clone()),
                ResourceAffinity::new(FallbackClass::Transparent)
                    .with_provider(self.provider_binding.clone()),
            )
            .with_allocation(allocation.id),
        ) {
            // Residency registration failed after admission and write both
            // succeeded: release what was just staged for this one weight
            // before propagating, so the caller's subsequent `abort()`
            // over `self.staged` never sees this half-staged entry (it was
            // never pushed). Best-effort: this call is already on a
            // failure path over a different error, so a further failure
            // releasing the just-written tensor is not itself surfaced --
            // the original residency-registration error is what the caller
            // needs to see.
            let _ = self.executor.release_tensor(&resource_id);
            let _ = runtime.memory_mut().release(allocation.id);
            return Err(InferenceApiError::MemoryAdmissionFailed {
                reason: format!(
                    "failed to register residency for weight resource '{name}': {error}"
                ),
            });
        }
        self.staged.push(StagedWeight {
            name: name.to_string(),
            resource_id,
            allocation: allocation.id,
        });
        Ok(true)
    }

    /// Releases every resource staged so far this attempt -- Provider
    /// tensor then Memory Manager allocation, per weight -- leaving no
    /// trace of this attempt behind. Reached whenever any weight in the
    /// attempt fails to stage.
    pub(super) fn abort(self, runtime: &mut Runtime) {
        for staged in &self.staged {
            // Best-effort: this is already unwinding a failed attempt, with
            // nothing further to roll back to if the Provider release
            // itself fails.
            let _ = self.executor.release_tensor(&staged.resource_id);
            // Remove the residency record before releasing the allocation
            // it references: once the Provider tensor and allocation are
            // both gone, a lingering `TensorResidency` entry would
            // misreport this resource as still resident (Correctif:
            // `invalidate-tensor-residency-on-release`).
            runtime
                .memory_mut()
                .remove_tensor_residency(&staged.resource_id);
            let _ = runtime.memory_mut().release(staged.allocation);
        }
    }

    /// Publishes every staged weight's binding onto the Model Instance,
    /// mints/replaces its Runtime-issued `MaterializationEvidence` to match
    /// the resulting *full* current weight-binding set (not just this
    /// attempt's own staged subset -- a second materialization attempt on
    /// an already-partially-materialized instance must still produce
    /// evidence covering every previously-bound weight, or a legitimate
    /// instance would fail its own evidence-matching readiness check), and
    /// -- only if the full Runtime-derived readiness gate (the same one
    /// `warm_model_instance`/`resume_model_instance` use, not a parallel or
    /// weaker check) is actually satisfied -- marks the instance Ready.
    ///
    /// Does *not* unconditionally mark Ready once staging succeeds: a
    /// materialization attempt supplying an empty or partial `weights` map
    /// (legitimate for incremental/progressive materialization, per this
    /// method's own evidence-recomputation above) must still leave the
    /// instance non-Ready if the loaded artifact's mandatory tensor
    /// inventory is not yet fully covered, or if Provider/Device readiness
    /// is not currently satisfied -- publishing bindings and evidence for
    /// what *was* successfully staged is not itself proof the instance is
    /// usable. A further audit of PR #36 found this exact gap: `commit`
    /// previously called `mark_ready` unconditionally right after minting
    /// evidence, so `materialize_model_instance_weights(..., &BTreeMap::
    /// new())` against an instance with a non-empty mandatory inventory
    /// could still reach `Ready` -- non-compliant with `model-loading`'s
    /// pre-existing "Model Loading Does Not Bypass Instance Readiness" and
    /// "Partial Loading Policy" requirements, which this fix makes the code
    /// actually honor rather than changing the spec to match the bug.
    /// Reached only once every weight in this attempt staged successfully.
    pub(super) fn commit(
        self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
    ) -> Result<(), InferenceApiError> {
        {
            let model_instance = runtime
                .model_instances_mut()
                .instance_mut(instance)
                .map_err(InferenceApiError::from)?;
            for staged in &self.staged {
                model_instance
                    .definition
                    .resource_bindings
                    .weights
                    .insert(staged.name.clone(), staged.resource_id.clone());
                model_instance
                    .definition
                    .resource_bindings
                    .memory_allocations
                    .insert(staged.allocation);
            }
        }
        let (artifact, bound_resources) = {
            let model_instance = runtime
                .model_instance(instance)
                .map_err(InferenceApiError::from)?;
            (
                model_instance.definition.artifact.clone(),
                model_instance
                    .definition
                    .resource_bindings
                    .weights
                    .values()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>(),
            )
        };
        runtime
            .model_instances_mut()
            .record_materialization_evidence(
                instance,
                MaterializationEvidence::new(artifact, bound_resources),
            );
        let effective_checks = derive_effective_readiness_checks(
            runtime,
            instance,
            &ModelInstanceReadinessChecks::default(),
        )?;
        let model_instance = runtime
            .model_instances_mut()
            .instance_mut(instance)
            .map_err(InferenceApiError::from)?;
        if model_instance
            .validate_readiness(&effective_checks)
            .is_err()
        {
            // Not yet fully ready (incomplete mandatory inventory, Provider
            // or Device not ready, ...): materialization of what was
            // staged this attempt still succeeded and its evidence still
            // stands, but the instance itself is correctly left non-Ready
            // rather than force-marked. A later call -- either another
            // `materialize_model_instance_weights` attempt completing the
            // inventory, or `warm_model_instance` once other conditions
            // clear -- re-derives from scratch.
            return Ok(());
        }
        runtime
            .model_instances_mut()
            .mark_ready(instance)
            .map_err(InferenceApiError::from)
    }
}

#[cfg(test)]
mod tests;
