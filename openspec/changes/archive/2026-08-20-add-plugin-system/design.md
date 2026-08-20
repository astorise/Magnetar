# Plugin System Design

## Architecture

`Plugin` is Magnetar's general extension contract:

```rust
trait Plugin {
    fn metadata(&self) -> PluginMetadata;
    fn register(&self, registry: &mut Registry) -> Result<(), PluginError>;
}
```

`Backend` remains a separate hardware-execution contract. A plugin may register a backend, but it is never required to do so.

`Registry` owns the registered contributions. The initial implementation supports backends; its `PluginCapabilities` metadata makes the intended categories explicit: backend, model loader, kernel provider, compiler pass, scheduler extension, and telemetry provider. Those future categories do not yet define runtime contracts or storage.

`PluginManager` owns the registry and plugins, validates the plugin API version, prevents duplicate plugin names, initializes compatible plugins, and shuts them down in reverse initialization order. `Runtime` exposes the registry's backends as its regular backend set.

## Dynamic Libraries

Discovery searches configured directories for platform dynamic-library filenames. Loading uses `libloading` and an exported `magnetar_plugin_create` factory. The factory returns a Rust `Box<dyn Plugin>`; therefore dynamic plugins must be built with the compatible Magnetar API and Rust toolchain. A future ABI-stabilization change can replace this factory with a C-compatible descriptor without changing the in-process `Plugin`/`Registry` model.

Loaded libraries are retained for at least as long as their plugin instances, so plugin code is never unloaded before the instance is dropped.

## Errors

Plugin API mismatches, duplicate names, loader failures, registration failures, and lifecycle failures are reported as `PluginError` values.
