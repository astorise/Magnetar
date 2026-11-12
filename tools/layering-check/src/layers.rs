//! The layer map and the rule it enforces.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Deserialize;

use crate::analysis::{Analysis, ROOT_GLOB};

/// `quality/layers.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerMap {
    /// Layer name -> rank. A module may depend on its own layer and on
    /// layers of a strictly lower rank. Two layers of the same rank may not
    /// depend on each other.
    pub layers: BTreeMap<String, u32>,
    /// Top-level module -> layer.
    pub modules: BTreeMap<String, String>,
    /// Modules kept in a higher layer than their target, with the reason.
    /// Informational: the gate checks `modules`, and checks that every
    /// entry here names a mapped module and a known target layer.
    #[serde(default)]
    pub held: BTreeMap<String, Held>,
    /// Upward dependencies that exist today and are being removed. The gate
    /// accepts exactly these: a new one fails, and so does an entry that no
    /// longer matches a dependency, so the list can only shrink.
    #[serde(default)]
    pub debt: Debt,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Debt {
    /// `"module -> dependency"`.
    #[serde(default)]
    pub edges: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Held {
    pub target: String,
    pub reason: String,
}

impl LayerMap {
    pub fn parse(text: &str) -> Result<Self, String> {
        let map: LayerMap =
            toml::from_str(text).map_err(|error| format!("invalid layer map: {error}"))?;
        for (module, layer) in &map.modules {
            if !map.layers.contains_key(layer) {
                return Err(format!(
                    "invalid layer map: module `{module}` is assigned to unknown layer `{layer}`"
                ));
            }
        }
        for (module, held) in &map.held {
            if !map.modules.contains_key(module) {
                return Err(format!(
                    "invalid layer map: held module `{module}` has no layer"
                ));
            }
            if !map.layers.contains_key(&held.target) {
                return Err(format!(
                    "invalid layer map: held module `{module}` targets unknown layer `{}`",
                    held.target
                ));
            }
            if held.reason.trim().is_empty() {
                return Err(format!(
                    "invalid layer map: held module `{module}` has no reason"
                ));
            }
        }
        Ok(map)
    }

    fn layer_of(&self, module: &str) -> Option<(&str, u32)> {
        let layer = self.modules.get(module)?;
        Some((layer.as_str(), self.layers[layer]))
    }
}

/// Every violation of the layering rule, as one line each.
pub fn check(analysis: &Analysis, map: &LayerMap) -> Vec<String> {
    let mut violations = Vec::new();
    for module in &analysis.modules {
        if !map.modules.contains_key(module) {
            violations.push(format!(
                "module `{module}` has no layer: add it to the layer map"
            ));
        }
    }
    for module in map.modules.keys() {
        if !analysis.modules.contains(module) {
            violations.push(format!(
                "layer map names module `{module}`, which does not exist"
            ));
        }
    }
    for (module, names) in &analysis.unresolved {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        violations.push(format!(
            "module `{module}` uses crate-root names that no module exports: {}",
            names.join(", ")
        ));
    }
    let mut unused_debt: std::collections::BTreeSet<&str> =
        map.debt.edges.iter().map(String::as_str).collect();
    for (module, deps) in &analysis.deps {
        let Some((layer, rank)) = map.layer_of(module) else {
            continue;
        };
        for (dep, items) in deps {
            if dep == ROOT_GLOB {
                violations.push(format!(
                    "module `{module}` ({layer}) glob-imports the crate root: name the items instead"
                ));
                continue;
            }
            let Some((dep_layer, dep_rank)) = map.layer_of(dep) else {
                continue;
            };
            if dep_layer == layer || dep_rank < rank {
                continue;
            }
            if unused_debt.remove(format!("{module} -> {dep}").as_str()) {
                continue;
            }
            let items: Vec<&str> = items.iter().map(String::as_str).collect();
            violations.push(format!(
                "module `{module}` ({layer}, rank {rank}) depends on `{dep}` ({dep_layer}, rank {dep_rank}): {}",
                items.join(", ")
            ));
        }
    }
    for edge in unused_debt {
        violations.push(format!(
            "debt entry `{edge}` no longer matches an upward dependency: remove it from the layer map"
        ));
    }
    violations
}

/// The upward dependencies as `debt.edges` entries, for the initial list.
pub fn debt_edges(analysis: &Analysis, map: &LayerMap) -> Vec<String> {
    let mut edges = Vec::new();
    for (module, deps) in &analysis.deps {
        let Some((layer, rank)) = map.layer_of(module) else {
            continue;
        };
        for dep in deps.keys() {
            if let Some((dep_layer, dep_rank)) = map.layer_of(dep)
                && dep_layer != layer
                && dep_rank >= rank
            {
                edges.push(format!("{module} -> {dep}"));
            }
        }
    }
    edges
}

/// Every dependency edge with the items behind it, grouped by module.
pub fn report(analysis: &Analysis, map: &LayerMap) -> String {
    let mut out = String::new();
    for module in &analysis.modules {
        let layer = map.modules.get(module).map_or("?", String::as_str);
        let _ = writeln!(out, "{module} [{layer}]");
        if let Some(deps) = analysis.deps.get(module) {
            for (dep, items) in deps {
                let dep_layer = map.modules.get(dep).map_or("?", String::as_str);
                let items: Vec<&str> = items.iter().map(String::as_str).collect();
                let _ = writeln!(out, "  -> {dep} [{dep_layer}]: {}", items.join(", "));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyze_sources;

    const MAP: &str = r#"
[layers]
core = 1
graph = 2
kernel = 2
runtime = 3

[modules]
tensor = "core"
device = "core"
planning = "graph"
kernel = "kernel"
api = "runtime"
"#;

    fn violations(sources: &[(&str, &str)], map: &str) -> Vec<String> {
        let analysis = analyze_sources(sources).unwrap();
        check(&analysis, &LayerMap::parse(map).unwrap())
    }

    const LIB: &str =
        "pub mod tensor; pub mod device; pub mod planning; pub mod kernel; pub mod api;
        pub use tensor::*; pub use api::*;";

    #[test]
    fn downward_and_same_layer_dependencies_are_allowed() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                (
                    "tensor.rs",
                    "use crate::device::Device; pub struct Tensor(Device);",
                ),
                ("device.rs", "pub struct Device;"),
                (
                    "planning.rs",
                    "use crate::tensor::Tensor; pub fn plan(_: Tensor) {}",
                ),
                ("kernel.rs", "pub fn k(_: crate::device::Device) {}"),
                (
                    "api.rs",
                    "use crate::{kernel::k, planning::plan}; pub struct Api;",
                ),
            ],
            MAP,
        );
        assert_eq!(found, Vec::<String>::new());
    }

    #[test]
    fn upward_dependency_is_reported_with_module_item_and_layers() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                ("tensor.rs", "pub struct Tensor;"),
                ("device.rs", "pub struct Device;"),
                ("planning.rs", ""),
                ("kernel.rs", "pub fn k(_: crate::api::Api) {}"),
                ("api.rs", "pub struct Api;"),
            ],
            MAP,
        );
        assert_eq!(
            found,
            vec![
                "module `kernel` (kernel, rank 2) depends on `api` (runtime, rank 3): Api"
                    .to_string()
            ]
        );
    }

    #[test]
    fn upward_dependency_through_a_crate_root_reexport_is_resolved() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                ("tensor.rs", "use crate::Api; pub struct Tensor(Api);"),
                ("device.rs", ""),
                ("planning.rs", ""),
                ("kernel.rs", ""),
                ("api.rs", "pub struct Api;"),
            ],
            MAP,
        );
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with("module `tensor` (core, rank 1) depends on `api`"));
    }

    #[test]
    fn same_rank_layers_may_not_depend_on_each_other() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                ("tensor.rs", ""),
                ("device.rs", ""),
                ("planning.rs", "pub fn plan() { crate::kernel::k(); }"),
                ("kernel.rs", "pub fn k() {}"),
                ("api.rs", ""),
            ],
            MAP,
        );
        assert_eq!(found.len(), 1);
        assert!(
            found[0].contains("`planning` (graph, rank 2) depends on `kernel` (kernel, rank 2)")
        );
    }

    #[test]
    fn module_without_a_layer_is_reported_by_name() {
        let found = violations(
            &[
                ("lib.rs", "pub mod tensor; pub mod extra;"),
                ("tensor.rs", ""),
                ("extra.rs", ""),
            ],
            "[layers]\ncore = 1\n[modules]\ntensor = \"core\"\n",
        );
        assert_eq!(
            found,
            vec!["module `extra` has no layer: add it to the layer map".to_string()]
        );
    }

    #[test]
    fn map_entry_for_a_missing_module_is_reported() {
        let found = violations(
            &[("lib.rs", "pub mod tensor;"), ("tensor.rs", "")],
            "[layers]\ncore = 1\n[modules]\ntensor = \"core\"\ngone = \"core\"\n",
        );
        assert_eq!(
            found,
            vec!["layer map names module `gone`, which does not exist".to_string()]
        );
    }

    #[test]
    fn test_only_code_is_not_checked() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                (
                    "tensor.rs",
                    "pub struct Tensor;
                     #[cfg(test)] mod tests { use crate::api::Api; }
                     #[cfg(test)] fn helper(_: crate::api::Api) {}
                     #[cfg(test)] mod file_tests;",
                ),
                ("device.rs", ""),
                ("planning.rs", ""),
                ("kernel.rs", ""),
                ("api.rs", "pub struct Api;"),
            ],
            MAP,
        );
        assert_eq!(found, Vec::<String>::new());
    }

    #[test]
    fn crate_root_glob_import_is_a_violation() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                ("tensor.rs", "use crate::*;"),
                ("device.rs", ""),
                ("planning.rs", ""),
                ("kernel.rs", ""),
                ("api.rs", ""),
            ],
            MAP,
        );
        assert_eq!(found.len(), 1);
        assert!(found[0].contains("glob-imports the crate root"));
    }

    #[test]
    fn paths_inside_macros_and_submodules_are_seen() {
        let found = violations(
            &[
                ("lib.rs", LIB),
                ("tensor.rs", "mod inner; pub struct Tensor;"),
                (
                    "tensor/inner.rs",
                    "fn f() { let _ = format!(\"{:?}\", crate::api::NAME); let _ = super::super::api::Api; }",
                ),
                ("device.rs", ""),
                ("planning.rs", ""),
                ("kernel.rs", ""),
                ("api.rs", "pub struct Api; pub const NAME: &str = \"\";"),
            ],
            MAP,
        );
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("depends on `api` (runtime, rank 3): Api, NAME"));
    }

    #[test]
    fn recorded_debt_is_accepted_and_stale_or_new_debt_fails() {
        let sources = [
            ("lib.rs", LIB),
            ("tensor.rs", "pub struct Tensor;"),
            ("device.rs", "pub struct Device;"),
            ("planning.rs", ""),
            ("kernel.rs", "pub fn k(_: crate::api::Api) {}"),
            ("api.rs", "pub struct Api;"),
        ];
        let with_debt = format!("{MAP}\n[debt]\nedges = [\"kernel -> api\"]\n");
        assert_eq!(violations(&sources, &with_debt), Vec::<String>::new());

        let stale = format!("{MAP}\n[debt]\nedges = [\"kernel -> api\", \"tensor -> api\"]\n");
        assert_eq!(
            violations(&sources, &stale),
            vec![
                "debt entry `tensor -> api` no longer matches an upward dependency: remove it from the layer map"
                    .to_string()
            ]
        );

        let other = format!("{MAP}\n[debt]\nedges = [\"kernel -> api\"]\n");
        let mut more = sources.to_vec();
        more[3] = ("planning.rs", "use crate::api::Api;");
        let found = violations(&more, &other);
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with("module `planning` (graph, rank 2) depends on `api`"));
    }

    #[test]
    fn layer_map_rejects_unknown_layers_and_empty_reasons() {
        assert!(LayerMap::parse("[layers]\ncore = 1\n[modules]\na = \"nope\"\n").is_err());
        assert!(
            LayerMap::parse(
                "[layers]\ncore = 1\n[modules]\na = \"core\"\n[held.a]\ntarget = \"core\"\nreason = \" \"\n"
            )
            .is_err()
        );
    }
}
