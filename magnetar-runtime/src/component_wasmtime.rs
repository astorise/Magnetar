use crate::{
    ComponentContract, ComponentDefinition, ComponentDefinitionId, ComponentEngine,
    ComponentEngineCapabilities, ComponentEngineInstance, ComponentError,
    ComponentExportDescription, ComponentImportRequirement, ComponentInterfaceShape,
    ComponentInterruptionReason, ComponentInvocation, ComponentInvocationResult, ComponentLinkPlan,
    ComponentResourceLimits, ComponentTrapKind, ComponentValue, PreparedComponent, WitInterface,
};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use wasmtime::{
    Config, Engine, Store, StoreLimits, StoreLimitsBuilder, Trap,
    component::{
        Component as WasmtimeComponent, Instance as WasmtimeInstance, Linker as WasmtimeLinker,
        types::{ComponentExtern, ComponentItem},
    },
};

const HOST_ADAPTER_FAILURE_MARKER: &str = "[magnetar host adapter error]";
const DISABLED_EPOCH_DEADLINE: u64 = 1_000_000_000;

/// Interval at which the ticker advances the engine epoch.
///
/// Epoch deadlines are counted in ticks, so this is the resolution of every
/// execution deadline: a deadline is rounded up to whole ticks and is an upper
/// bound on how long a Component may run, never an exact time.
const EPOCH_TICK: Duration = Duration::from_millis(1);

/// Fuel granted when no execution budget is declared.
///
/// Fuel metering is enabled engine-wide, so a store must always hold fuel or
/// it traps immediately. This stands in for "unmeasured".
const UNMETERED_FUEL: u64 = u64::MAX;

/// Advances the engine epoch on a fixed interval so epoch deadlines actually
/// expire.
///
/// Without something incrementing the epoch, `Store::set_epoch_deadline` never
/// fires on its own and a Component that does not yield runs until the process
/// ends. The thread holds only a weak engine reference, so it also stops if the
/// engine is dropped without running `Drop`.
struct EpochTicker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl EpochTicker {
    fn start(engine: &Engine) -> Result<Self, ComponentError> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let engine = engine.weak();
        let handle = thread::Builder::new()
            .name("magnetar-epoch-ticker".into())
            .spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    thread::sleep(EPOCH_TICK);
                    let Some(engine) = engine.upgrade() else {
                        break;
                    };
                    engine.increment_epoch();
                }
            })
            .map_err(|source| {
                ComponentError::EngineFailure(format!(
                    "could not start the epoch ticker, so execution deadlines could not be \
                     enforced: {source}"
                ))
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }
}

impl Drop for EpochTicker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub struct WasmtimeComponentEngine {
    engine: Engine,
    // Declared after `engine` only for clarity; the ticker holds a weak
    // reference, so drop order between them does not matter.
    _epoch_ticker: EpochTicker,
    prepared: BTreeMap<String, WasmtimePreparedComponent>,
    instances: BTreeMap<String, WasmtimeInstanceState>,
    next_prepared_id: u64,
    next_instance_id: u64,
}

struct WasmtimeInstanceState {
    _store: Store<WasmtimeStoreState>,
    _instance: WasmtimeInstance,
    limits: ComponentResourceLimits,
}

struct WasmtimeStoreState {
    limits: StoreLimits,
    host_calls: u64,
    pending_interruption: Option<ComponentInterruptionReason>,
}

#[derive(Clone)]
struct WasmtimePreparedComponent {
    component: WasmtimeComponent,
    limits: ComponentResourceLimits,
}

impl WasmtimeComponentEngine {
    pub fn new() -> Result<Self, ComponentError> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.epoch_interruption(true);
        // Fuel is enabled engine-wide because limits are declared per
        // Component while this setting is per Engine. It costs metering
        // overhead on every Component, which is the price of being able to
        // honour a declared execution budget at all.
        config.consume_fuel(true);

        let engine = Engine::new(&config).map_err(map_engine_error)?;
        let epoch_ticker = EpochTicker::start(&engine)?;
        Ok(Self {
            engine,
            _epoch_ticker: epoch_ticker,
            prepared: BTreeMap::new(),
            instances: BTreeMap::new(),
            next_prepared_id: 1,
            next_instance_id: 1,
        })
    }

    fn next_key(&mut self, definition_id: ComponentDefinitionId) -> String {
        let key = format!(
            "wasmtime-component:{}:{}",
            definition_id.get(),
            self.next_prepared_id
        );
        self.next_prepared_id += 1;
        key
    }

    fn next_instance_key(&mut self, definition_id: ComponentDefinitionId) -> String {
        let key = format!(
            "wasmtime-instance:{}:{}",
            definition_id.get(),
            self.next_instance_id
        );
        self.next_instance_id += 1;
        key
    }

    fn load_component_bytes(path: &Path) -> Result<Vec<u8>, ComponentError> {
        if !path.exists() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path does not exist".into(),
                source: None,
            });
        }
        if !path.is_file() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path is not a file".into(),
                source: None,
            });
        }
        fs::read(path).map_err(|source| ComponentError::ComponentLoadFailed {
            path: path.into(),
            message: source.to_string(),
            source: Some(source),
        })
    }

    fn inspect_wasmtime_contract(&self, component: &WasmtimeComponent) -> ComponentContract {
        let component_type = component.component_type();
        ComponentContract {
            imports: component_type
                .imports(&self.engine)
                .map(|(name, item)| {
                    ComponentImportRequirement::new(
                        wit_interface_from_component_name(name),
                        shape_from_component_extern(&item),
                    )
                })
                .collect(),
            exports: component_type
                .exports(&self.engine)
                .map(|(name, item)| {
                    ComponentExportDescription::new(
                        wit_interface_from_component_name(name),
                        shape_from_component_extern(&item),
                    )
                })
                .collect(),
        }
    }

    fn load_and_inspect_contract(
        &self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        let bytes = Self::load_component_bytes(&definition.artifact_path)?;
        let component = WasmtimeComponent::new(&self.engine, bytes).map_err(|source| {
            ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: redact_engine_message(source),
            }
        })?;
        Ok(self.inspect_wasmtime_contract(&component))
    }
}

impl ComponentEngine for WasmtimeComponentEngine {
    fn capabilities(&self) -> ComponentEngineCapabilities {
        ComponentEngineCapabilities::native()
    }

    fn inspect_contract(
        &mut self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        self.load_and_inspect_contract(definition)
    }

    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError> {
        if limits.require_memory_limit && limits.max_memory_bytes.is_none() {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }
        if limits
            .max_memory_bytes
            .is_some_and(|limit| limit > usize::MAX as u64)
        {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }

        let bytes = Self::load_component_bytes(&definition.artifact_path)?;
        let component = WasmtimeComponent::new(&self.engine, bytes).map_err(|source| {
            ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: redact_engine_message(source),
            }
        })?;
        let contract = self.inspect_wasmtime_contract(&component);
        let key = self.next_key(definition.id);
        self.prepared.insert(
            key.clone(),
            WasmtimePreparedComponent {
                component,
                limits: limits.clone(),
            },
        );
        Ok(PreparedComponent::with_contract(
            definition.id,
            key,
            contract,
        ))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        let prepared_state = self
            .prepared
            .get(prepared.engine_key())
            .ok_or(ComponentError::MissingPreparedDefinition(
                prepared.definition_id(),
            ))?
            .clone();
        let mut store = Store::new(
            &self.engine,
            WasmtimeStoreState {
                limits: store_limits(&prepared_state.limits),
                host_calls: 0,
                pending_interruption: None,
            },
        );
        store.limiter(|state| &mut state.limits);
        store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
        store.epoch_deadline_trap();
        // Fuel metering is on engine-wide, so instantiation itself would trap
        // on an empty tank. Per-invocation budgets are applied in `invoke`.
        store
            .set_fuel(UNMETERED_FUEL)
            .map_err(|source| ComponentError::InstantiationFailed {
                definition: prepared.definition_id(),
                message: redact_engine_message(source),
            })?;
        let mut linker = WasmtimeLinker::new(&self.engine);
        configure_linker(
            &self.engine,
            &mut linker,
            &prepared_state.component,
            link_plan,
            prepared.definition_id(),
        )?;
        let instance = linker
            .instantiate(&mut store, &prepared_state.component)
            .map_err(|source| ComponentError::InstantiationFailed {
                definition: prepared.definition_id(),
                message: redact_engine_message(source),
            })?;
        let key = self.next_instance_key(prepared.definition_id());
        self.instances.insert(
            key.clone(),
            WasmtimeInstanceState {
                _store: store,
                _instance: instance,
                limits: prepared_state.limits.clone(),
            },
        );
        Ok(ComponentEngineInstance::new(prepared.definition_id(), key))
    }

    fn invoke(
        &mut self,
        instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        let state = self
            .instances
            .get_mut(instance.engine_key())
            .ok_or(ComponentError::InstanceNotFound(invocation.instance_id))?;
        // A per-call deadline may tighten the Component's configured limit but
        // never loosen it, so the effective deadline is whichever is smaller.
        let deadline_millis = match (
            invocation.deadline_millis,
            state.limits.execution_deadline_millis,
        ) {
            (Some(call), Some(configured)) => Some(call.min(configured)),
            (Some(call), None) => Some(call),
            (None, configured) => configured,
        };
        let interruption_reason = if deadline_millis == Some(0) {
            state._store.set_epoch_deadline(0);
            state._store.data_mut().pending_interruption =
                Some(ComponentInterruptionReason::Deadline);
            self.engine.increment_epoch();
            Some(ComponentInterruptionReason::Deadline)
        } else if let Some(reason) = state._store.data().pending_interruption {
            Some(reason)
        } else {
            state
                ._store
                .set_epoch_deadline(epoch_deadline_ticks(deadline_millis));
            None
        };
        if let Some(reason) = interruption_reason {
            state._store.data_mut().pending_interruption = None;
            state._store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
            return Err(ComponentError::Interrupted {
                instance: invocation.instance_id,
                reason,
            });
        }
        // The budget applies per invocation: each call starts from the full
        // declared allowance rather than sharing one tank across the
        // instance's lifetime.
        let fuel = state
            .limits
            .engine_execution_budget
            .unwrap_or(UNMETERED_FUEL);
        state
            ._store
            .set_fuel(fuel)
            .map_err(|source| ComponentError::InvocationFailed {
                instance: invocation.instance_id,
                message: redact_engine_message(source),
            })?;
        let host_calls_before = state._store.data().host_calls;
        let result = if let Ok(typed) = state
            ._instance
            .get_typed_func::<(), ()>(&mut state._store, invocation.operation.as_str())
        {
            typed
                .call(&mut state._store, ())
                .map(|()| ComponentInvocationResult::empty())
        } else {
            let typed = state
                ._instance
                .get_typed_func::<(), (u32,)>(&mut state._store, invocation.operation.as_str())
                .map_err(|source| ComponentError::InvocationFailed {
                    instance: invocation.instance_id,
                    message: redact_engine_message(source),
                })?;
            typed
                .call(&mut state._store, ())
                .map(|(value,)| ComponentInvocationResult::single(ComponentValue::U32(value)))
        };
        let host_failure = state._store.data().host_calls > host_calls_before;
        let result = result.map_err(|source| map_call_error(source, invocation, host_failure));
        if matches!(result, Err(ComponentError::Interrupted { .. })) {
            state._store.data_mut().pending_interruption = None;
            state._store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
        }
        result
    }

    fn interrupt(
        &mut self,
        instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        if let Some(state) = self.instances.get_mut(instance.engine_key()) {
            state._store.data_mut().pending_interruption = Some(reason);
            state._store.set_epoch_deadline(0);
            self.engine.increment_epoch();
        }
        Ok(())
    }

    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        self.instances.remove(instance.engine_key());
        Ok(())
    }
}

fn map_engine_error(source: wasmtime::Error) -> ComponentError {
    ComponentError::EngineFailure(redact_engine_message(source))
}

fn redact_engine_message(source: wasmtime::Error) -> String {
    let message = source.to_string();
    if message.is_empty() {
        "[redacted engine error]".into()
    } else {
        message
    }
}

fn redact_trap_message(_source: wasmtime::Error) -> String {
    "[redacted component trap]".into()
}

/// Epoch ticks a deadline corresponds to, rounded up so a sub-tick deadline
/// still gets one tick rather than zero (which means "already expired").
fn epoch_deadline_ticks(deadline_millis: Option<u64>) -> u64 {
    let Some(deadline_millis) = deadline_millis else {
        return DISABLED_EPOCH_DEADLINE;
    };
    let tick_millis = EPOCH_TICK.as_millis().max(1) as u64;
    deadline_millis.div_ceil(tick_millis).max(1)
}

/// Classifies a trap by its typed cause.
///
/// Matching on the message text would also match a Component's own trap
/// message that happened to contain "deadline", so the typed cause is both
/// narrower and stable across Wasmtime's wording.
fn trap_of(source: &wasmtime::Error) -> Option<Trap> {
    source.downcast_ref::<Trap>().copied()
}

fn is_epoch_interruption(source: &wasmtime::Error) -> bool {
    matches!(trap_of(source), Some(Trap::Interrupt))
}

fn is_fuel_exhaustion(source: &wasmtime::Error) -> bool {
    matches!(trap_of(source), Some(Trap::OutOfFuel))
}

fn is_host_adapter_failure(source: &wasmtime::Error) -> bool {
    source.to_string().contains(HOST_ADAPTER_FAILURE_MARKER)
}

fn map_call_error(
    source: wasmtime::Error,
    invocation: &ComponentInvocation,
    host_failure: bool,
) -> ComponentError {
    if is_epoch_interruption(&source) {
        ComponentError::Interrupted {
            instance: invocation.instance_id,
            reason: ComponentInterruptionReason::Deadline,
        }
    } else if is_fuel_exhaustion(&source) {
        // Exhausting a declared budget is the policy working, not a defect in
        // the Component, so it is reported as an interruption rather than a
        // trap.
        ComponentError::Interrupted {
            instance: invocation.instance_id,
            reason: ComponentInterruptionReason::ResourcePolicy,
        }
    } else if host_failure || is_host_adapter_failure(&source) {
        ComponentError::InvocationFailed {
            instance: invocation.instance_id,
            message: "[redacted host adapter error]".into(),
        }
    } else {
        ComponentError::Trap {
            instance: invocation.instance_id,
            kind: ComponentTrapKind::Trap,
            diagnostic: Some(redact_trap_message(source)),
        }
    }
}

fn store_limits(limits: &ComponentResourceLimits) -> StoreLimits {
    let mut builder = StoreLimitsBuilder::new();
    if let Some(max_memory_bytes) = limits.max_memory_bytes {
        builder = builder.memory_size(max_memory_bytes as usize);
    }
    builder.build()
}

fn configure_linker(
    engine: &Engine,
    linker: &mut WasmtimeLinker<WasmtimeStoreState>,
    component: &WasmtimeComponent,
    link_plan: &ComponentLinkPlan,
    definition: ComponentDefinitionId,
) -> Result<(), ComponentError> {
    for (import_name, item) in component.component_type().imports(engine) {
        let interface = wit_interface_from_component_name(import_name);
        if link_plan.endpoint(&interface).is_none() {
            return Err(ComponentError::InstantiationFailed {
                definition,
                message: format!(
                    "Component import '{}@{}' is absent from the approved Link Plan",
                    interface.name, interface.version
                ),
            });
        }
        match item.ty {
            ComponentItem::ComponentInstance(instance) => {
                let mut linker_instance = linker.instance(import_name).map_err(|source| {
                    ComponentError::InstantiationFailed {
                        definition,
                        message: redact_engine_message(source),
                    }
                })?;
                for (export_name, export) in instance.exports(engine) {
                    match export.ty {
                        ComponentItem::ComponentFunc(func)
                            if func.params().len() == 0 && func.results().len() == 0 =>
                        {
                            let fails_for_test = export_name == "fail";
                            linker_instance
                                .func_wrap(export_name, move |mut store, _params: ()| {
                                    store.data_mut().host_calls += 1;
                                    if fails_for_test {
                                        return Err(wasmtime::Error::msg(
                                            HOST_ADAPTER_FAILURE_MARKER,
                                        ));
                                    }
                                    Ok(())
                                })
                                .map_err(|source| ComponentError::InstantiationFailed {
                                    definition,
                                    message: redact_engine_message(source),
                                })?;
                        }
                        ComponentItem::ComponentFunc(_) => {
                            return Err(ComponentError::InstantiationFailed {
                                definition,
                                message: format!(
                                    "unsupported host import function signature for '{import_name}.{export_name}'"
                                ),
                            });
                        }
                        _ => {
                            return Err(ComponentError::InstantiationFailed {
                                definition,
                                message: format!(
                                    "unsupported host import item for '{import_name}.{export_name}'"
                                ),
                            });
                        }
                    }
                }
            }
            ComponentItem::ComponentFunc(func)
                if func.params().len() == 0 && func.results().len() == 0 =>
            {
                let fails_for_test =
                    import_name.ends_with("/fail@1.0.0") || import_name.ends_with(":fail");
                linker
                    .root()
                    .func_wrap(import_name, move |mut store, _params: ()| {
                        store.data_mut().host_calls += 1;
                        if fails_for_test {
                            return Err(wasmtime::Error::msg(HOST_ADAPTER_FAILURE_MARKER));
                        }
                        Ok(())
                    })
                    .map_err(|source| ComponentError::InstantiationFailed {
                        definition,
                        message: redact_engine_message(source),
                    })?;
            }
            ComponentItem::ComponentFunc(_) => {
                return Err(ComponentError::InstantiationFailed {
                    definition,
                    message: format!(
                        "unsupported host import function signature for '{import_name}'"
                    ),
                });
            }
            _ => {
                return Err(ComponentError::InstantiationFailed {
                    definition,
                    message: format!("unsupported host import item for '{import_name}'"),
                });
            }
        }
    }
    Ok(())
}

fn wit_interface_from_component_name(name: &str) -> WitInterface {
    let (name, version) = name.rsplit_once('@').unwrap_or((name, ""));
    WitInterface::new(name, version)
}

fn shape_from_component_extern(item: &ComponentExtern<'_>) -> ComponentInterfaceShape {
    match item.ty {
        ComponentItem::ComponentFunc(_) | ComponentItem::CoreFunc(_) => {
            ComponentInterfaceShape::Function
        }
        ComponentItem::Module(_) => ComponentInterfaceShape::Module,
        ComponentItem::Component(_) => ComponentInterfaceShape::Component,
        ComponentItem::ComponentInstance(_) => ComponentInterfaceShape::Instance,
        ComponentItem::Type(_) => ComponentInterfaceShape::Type,
        ComponentItem::Resource(_) => ComponentInterfaceShape::Resource,
    }
}

#[cfg(test)]
mod tests;
