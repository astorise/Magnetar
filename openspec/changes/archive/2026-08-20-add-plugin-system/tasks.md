# Tasks

## Plugin API

- [x] Define the general Plugin trait and its separation from Backend
- [x] Define PluginMetadata
- [x] Define PluginCapabilities and Registry extension categories

## Loader

- [x] Implement plugin discovery
- [x] Implement dynamic loading
- [x] Validate API version

## Registry

- [x] Register loaded plugins and their Registry contributions
- [x] Query loaded plugins
- [x] Prevent duplicate registration

## Lifecycle

- [x] Initialize plugins
- [x] Shutdown plugins

## Tests

- [x] Load a valid plugin
- [x] Reject incompatible API version
- [x] Reject duplicate plugin
