//! Module dependency analysis of one crate.
//!
//! Walks the module tree from `lib.rs`, skips everything under
//! `#[cfg(test)]`, and records, for every top-level module, which other
//! top-level modules its code names: through `crate::module::…` paths,
//! `super::` chains that reach the crate root, crate-root re-exports
//! (`crate::Name`, resolved to the module that defines `Name`), and the
//! same paths inside macro invocations.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::Visit;

/// Pseudo-dependency recorded for `use crate::*`.
pub const ROOT_GLOB: &str = "*";

#[derive(Debug, Default)]
pub struct Analysis {
    /// Top-level modules that are not test-only.
    pub modules: BTreeSet<String>,
    /// module -> dependency -> items named.
    pub deps: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    /// module -> crate-root names that no module exports.
    pub unresolved: BTreeMap<String, BTreeSet<String>>,
}

impl Analysis {
    /// Strongly connected components of more than one module, each sorted,
    /// largest first.
    pub fn cycles(&self) -> Vec<Vec<String>> {
        let nodes: Vec<&String> = self.modules.iter().collect();
        let index_of: BTreeMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let edges: Vec<Vec<usize>> = nodes
            .iter()
            .map(|name| {
                self.deps
                    .get(*name)
                    .into_iter()
                    .flat_map(|deps| deps.keys())
                    .filter_map(|dep| index_of.get(dep.as_str()).copied())
                    .collect()
            })
            .collect();
        // Kosaraju: order by finish time, then sweep the reversed graph.
        let mut order = Vec::new();
        let mut seen = vec![false; nodes.len()];
        for start in 0..nodes.len() {
            if seen[start] {
                continue;
            }
            let mut stack = vec![(start, 0usize)];
            seen[start] = true;
            while let Some((node, next)) = stack.pop() {
                if next < edges[node].len() {
                    stack.push((node, next + 1));
                    let target = edges[node][next];
                    if !seen[target] {
                        seen[target] = true;
                        stack.push((target, 0));
                    }
                } else {
                    order.push(node);
                }
            }
        }
        let mut reversed = vec![Vec::new(); nodes.len()];
        for (from, targets) in edges.iter().enumerate() {
            for &to in targets {
                reversed[to].push(from);
            }
        }
        let mut assigned = vec![false; nodes.len()];
        let mut cycles = Vec::new();
        for &root in order.iter().rev() {
            if assigned[root] {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![root];
            assigned[root] = true;
            while let Some(node) = stack.pop() {
                component.push(nodes[node].clone());
                for &previous in &reversed[node] {
                    if !assigned[previous] {
                        assigned[previous] = true;
                        stack.push(previous);
                    }
                }
            }
            if component.len() > 1 {
                component.sort();
                cycles.push(component);
            }
        }
        cycles.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        cycles
    }
}

/// Source access, so that tests can analyze in-memory crates.
trait Sources {
    fn read(&self, path: &Path) -> Option<String>;
}

struct Disk;

impl Sources for Disk {
    fn read(&self, path: &Path) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }
}

#[cfg(test)]
struct InMemory(BTreeMap<PathBuf, String>);

#[cfg(test)]
impl Sources for InMemory {
    fn read(&self, path: &Path) -> Option<String> {
        self.0.get(path).cloned()
    }
}

/// Analyzes one or more crates as one module graph. Each `src_dir` is a
/// crate's `src` directory; the crate is named after the directory that
/// holds it (`magnetar-core/src` -> `magnetar_core`), so that a path through
/// another analyzed crate (`magnetar_core::tensor::X`) is followed like a
/// `crate::` path.
pub fn analyze(src_dirs: &[PathBuf]) -> Result<Analysis, String> {
    analyze_with(&Disk, src_dirs)
}

#[cfg(test)]
pub fn analyze_sources(files: &[(&str, &str)]) -> Result<Analysis, String> {
    let sources = InMemory(
        files
            .iter()
            .map(|(path, text)| (PathBuf::from(path), (*text).to_string()))
            .collect(),
    );
    analyze_with(&sources, &[PathBuf::new()])
}

#[cfg(test)]
pub fn analyze_crates(src_dirs: &[&str], files: &[(&str, &str)]) -> Result<Analysis, String> {
    let sources = InMemory(
        files
            .iter()
            .map(|(path, text)| (PathBuf::from(path), (*text).to_string()))
            .collect(),
    );
    let src_dirs: Vec<PathBuf> = src_dirs.iter().map(PathBuf::from).collect();
    analyze_with(&sources, &src_dirs)
}

/// One raw reference from a top-level module to the crate root.
struct RootReference {
    from: String,
    /// Path segments after the crate root; `*` for a glob.
    segments: Vec<String>,
}

/// What one module contributes to the names visible at its own root.
#[derive(Default)]
struct ModuleExports {
    /// `pub` items defined at the module root, and names re-exported there.
    names: BTreeSet<String>,
    /// Top-level modules whose whole public surface is re-exported
    /// (`pub use crate::other::*`).
    globs: BTreeSet<String>,
    /// Names re-exported one by one from another top-level module
    /// (`pub use crate::other::Name`): name -> that module.
    forwarded: BTreeMap<String, String>,
}

fn crate_ident(src_dir: &Path) -> Option<String> {
    let name = src_dir.parent()?.file_name()?.to_str()?;
    Some(name.replace('-', "_"))
}

fn analyze_with(sources: &dyn Sources, src_dirs: &[PathBuf]) -> Result<Analysis, String> {
    let mut analysis = Analysis::default();
    let mut references = Vec::new();
    let mut idents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut exports: BTreeMap<String, ModuleExports> = BTreeMap::new();
    let roots: BTreeSet<String> = src_dirs.iter().filter_map(|dir| crate_ident(dir)).collect();
    let mut libs = Vec::new();

    for src_dir in src_dirs {
        let lib = parse(sources, &src_dir.join("lib.rs"))?;
        for item in &lib.items {
            let syn::Item::Mod(module) = item else {
                continue;
            };
            if is_test_only(&module.attrs) {
                continue;
            }
            let name = module.ident.to_string();
            if !analysis.modules.insert(name.clone()) {
                return Err(format!(
                    "module `{name}` is defined by more than one analyzed crate"
                ));
            }
            let mut walker = Walker {
                sources,
                roots: &roots,
                top: name.clone(),
                depth: 1,
                references: &mut references,
                idents: idents.entry(name.clone()).or_default(),
                error: None,
            };
            let module_exports = walker.walk_module(module, src_dir, src_dir)?;
            if let Some(error) = walker.error {
                return Err(error);
            }
            exports.insert(name, module_exports);
        }
        libs.push(lib);
    }

    // name -> modules exporting it at their root.
    let mut owners: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for module in exports.keys() {
        let mut visited = BTreeSet::new();
        let mut names = BTreeSet::new();
        collect_names(module, &exports, &mut visited, &mut names);
        for name in names {
            owners.entry(name).or_default().insert(module.clone());
        }
    }

    let exported_by = |module: &str| {
        let mut visited = BTreeSet::new();
        let mut names = BTreeSet::new();
        collect_names(module, &exports, &mut visited, &mut names);
        names
    };
    for reference in references {
        let first = &reference.segments[0];
        if analysis.modules.contains(first)
            && *first != reference.from
            && reference.segments.get(1).is_some_and(|s| s == ROOT_GLOB)
        {
            // `use crate::module::*`: the items behind the edge are the
            // module's exports that this module's code names.
            let used: Vec<String> = exported_by(first)
                .intersection(&idents[&reference.from])
                .cloned()
                .collect();
            if used.is_empty() {
                add(&mut analysis, &reference.from, first, "*");
            }
            for item in used {
                // A name the module only passes through belongs to the
                // module that defines it.
                let definers: Vec<&String> = owners
                    .get(&item)
                    .into_iter()
                    .flatten()
                    .filter(|module| exports[*module].names.contains(&item))
                    .collect();
                if definers.is_empty() || definers.contains(&first) {
                    add(&mut analysis, &reference.from, first, &item);
                } else {
                    for definer in definers {
                        if *definer != reference.from {
                            add(&mut analysis, &reference.from, definer, &item);
                        }
                    }
                }
            }
            continue;
        }
        if first == ROOT_GLOB {
            add(&mut analysis, &reference.from, ROOT_GLOB, "*");
        } else if analysis.modules.contains(first) {
            if *first != reference.from {
                let item = reference.segments.get(1).unwrap_or(first);
                add(&mut analysis, &reference.from, first, item);
            }
        } else if let Some(modules) = owners.get(first) {
            if modules.contains(&reference.from) {
                continue;
            }
            // A name defined in one module and glob-re-exported by others
            // has several owners: depend on the modules that name it at
            // their own root, not on the ones that only pass it through.
            let definers: Vec<&String> = modules
                .iter()
                .filter(|module| exports[*module].names.contains(first))
                .collect();
            let targets: Vec<&String> = if definers.is_empty() {
                modules.iter().collect()
            } else {
                definers
            };
            for target in targets {
                add(&mut analysis, &reference.from, target, first);
            }
        } else if !libs.iter().any(|lib| is_test_only_module_name(first, lib)) {
            analysis
                .unresolved
                .entry(reference.from.clone())
                .or_default()
                .insert(first.clone());
        }
    }
    Ok(analysis)
}

fn add(analysis: &mut Analysis, from: &str, to: &str, item: &str) {
    analysis
        .deps
        .entry(from.to_string())
        .or_default()
        .entry(to.to_string())
        .or_default()
        .insert(item.to_string());
}

fn collect_names(
    module: &str,
    exports: &BTreeMap<String, ModuleExports>,
    visited: &mut BTreeSet<String>,
    names: &mut BTreeSet<String>,
) {
    if !visited.insert(module.to_string()) {
        return;
    }
    let Some(module_exports) = exports.get(module) else {
        return;
    };
    names.extend(module_exports.names.iter().cloned());
    names.extend(module_exports.forwarded.keys().cloned());
    for other in &module_exports.globs {
        collect_names(other, exports, visited, names);
    }
}

/// A crate-root name that is a `#[cfg(test)]` module of `lib.rs` (reachable
/// only from test code, which is not analyzed) is not an unresolved name.
fn is_test_only_module_name(name: &str, lib: &syn::File) -> bool {
    lib.items.iter().any(|item| match item {
        syn::Item::Mod(module) => module.ident == name && is_test_only(&module.attrs),
        _ => false,
    })
}

fn parse(sources: &dyn Sources, path: &Path) -> Result<syn::File, String> {
    let text = sources
        .read(path)
        .ok_or_else(|| format!("cannot read {}", path.display()))?;
    syn::parse_file(&text).map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

/// `#[cfg(test)]`, `#[cfg(all(test, …))]`: any `cfg` naming `test` outside
/// a `not(…)`.
fn is_test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        let syn::Meta::List(list) = &attr.meta else {
            return false;
        };
        names_test(list.tokens.clone())
    })
}

fn names_test(tokens: TokenStream) -> bool {
    let mut previous_was_not = false;
    for token in tokens {
        match token {
            TokenTree::Ident(ident) => {
                if ident == "test" {
                    return true;
                }
                previous_was_not = ident == "not";
            }
            TokenTree::Group(group) => {
                if !previous_was_not && names_test(group.stream()) {
                    return true;
                }
                previous_was_not = false;
            }
            _ => previous_was_not = false,
        }
    }
    false
}

fn path_attribute(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("path") {
            return None;
        }
        let syn::Meta::NameValue(value) = &attr.meta else {
            return None;
        };
        match &value.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(text),
                ..
            }) => Some(text.value()),
            _ => None,
        }
    })
}

struct Walker<'a> {
    sources: &'a dyn Sources,
    /// Names of the analyzed crates: a path starting with one of them is a
    /// path from that crate's root.
    roots: &'a BTreeSet<String>,
    /// The top-level module being walked.
    top: String,
    /// Module nesting below the crate root (1 = the top-level module).
    depth: usize,
    references: &'a mut Vec<RootReference>,
    /// Every identifier that starts a relative path or appears in a macro:
    /// what a glob import of another module may be resolving.
    idents: &'a mut BTreeSet<String>,
    error: Option<String>,
}

impl Walker<'_> {
    /// Walks `module` (inline or in a file under `dir`), and returns what it
    /// exports at its own root.
    ///
    /// `file_dir` is the directory of the file declaring the module (what a
    /// `#[path]` attribute is relative to); `dir` is where the declaring
    /// module's children live.
    fn walk_module(
        &mut self,
        module: &syn::ItemMod,
        file_dir: &Path,
        dir: &Path,
    ) -> Result<ModuleExports, String> {
        let name = module.ident.to_string();
        if let Some((_, items)) = &module.content {
            // An inline module's children live in `<dir>/<name>/`.
            return self.walk_items(items, file_dir, &dir.join(&name));
        }
        let (file, child_dir) = if let Some(path) = path_attribute(&module.attrs) {
            let file = file_dir.join(path);
            let child_dir = file.parent().map(Path::to_path_buf).unwrap_or_default();
            (file, child_dir)
        } else {
            let flat = dir.join(format!("{name}.rs"));
            if self.sources.read(&flat).is_some() {
                (flat, dir.join(&name))
            } else {
                (dir.join(&name).join("mod.rs"), dir.join(&name))
            }
        };
        let parsed = parse(self.sources, &file)?;
        let file_dir = file.parent().map(Path::to_path_buf).unwrap_or_default();
        self.walk_items(&parsed.items, &file_dir, &child_dir)
    }

    fn walk_items(
        &mut self,
        items: &[syn::Item],
        file_dir: &Path,
        child_dir: &Path,
    ) -> Result<ModuleExports, String> {
        let mut exports = ModuleExports::default();
        let mut children: BTreeMap<String, ModuleExports> = BTreeMap::new();
        for item in items {
            if item_is_test_only(item) {
                continue;
            }
            if let syn::Item::Mod(child) = item {
                self.depth += 1;
                let child_exports = self.walk_module(child, file_dir, child_dir)?;
                self.depth -= 1;
                if matches!(child.vis, syn::Visibility::Public(_)) {
                    exports.names.insert(child.ident.to_string());
                }
                children.insert(child.ident.to_string(), child_exports);
                continue;
            }
            self.visit_item(item);
            if let Some(name) = public_item_name(item) {
                exports.names.insert(name);
            }
        }
        // Re-exports at this module's root.
        for item in items {
            let syn::Item::Use(item_use) = item else {
                continue;
            };
            if item_is_test_only(item) || !matches!(item_use.vis, syn::Visibility::Public(_)) {
                continue;
            }
            let mut paths = Vec::new();
            flatten_use(&item_use.tree, &mut Vec::new(), &mut paths);
            for (mut path, exported) in paths {
                while path.first().is_some_and(|segment| segment == "self") {
                    path.remove(0);
                }
                let through_root = match path.first().map(String::as_str) {
                    Some("crate") => Some(1),
                    Some(first) if self.roots.contains(first) => Some(1),
                    Some("super") => {
                        let supers = path.iter().take_while(|s| *s == "super").count();
                        (supers == self.depth).then_some(supers)
                    }
                    _ => None,
                };
                if exported == ROOT_GLOB {
                    if let Some(skip) = through_root {
                        // `pub use crate::other::*`
                        if path.len() == skip + 2 {
                            exports.globs.insert(path[skip].clone());
                        }
                    } else if path.len() == 2 {
                        // `pub use child::*`
                        if let Some(child) = children.get(&path[0]) {
                            exports.names.extend(child.names.iter().cloned());
                            exports.globs.extend(child.globs.iter().cloned());
                        }
                    }
                } else if let Some(skip) = through_root
                    && path.len() == skip + 2
                {
                    // `pub use crate::other::Name`: exported here, defined
                    // (or forwarded again) there.
                    exports.forwarded.insert(exported, path[skip].clone());
                } else {
                    exports.names.insert(exported);
                }
            }
        }
        Ok(exports)
    }

    fn record(&mut self, segments: &[String]) {
        let rest = match segments.first().map(String::as_str) {
            Some("crate") => &segments[1..],
            Some("super") => {
                let supers = segments.iter().take_while(|s| *s == "super").count();
                if supers != self.depth {
                    return;
                }
                &segments[supers..]
            }
            Some("self") => return,
            Some(first) if self.roots.contains(first) => &segments[1..],
            Some(first) => {
                self.idents.insert(first.to_string());
                return;
            }
            None => return,
        };
        if rest.is_empty() {
            return;
        }
        self.references.push(RootReference {
            from: self.top.clone(),
            segments: rest.to_vec(),
        });
    }

    fn record_tokens(&mut self, tokens: TokenStream) {
        let tokens: Vec<TokenTree> = tokens.into_iter().collect();
        let mut index = 0;
        while index < tokens.len() {
            match &tokens[index] {
                TokenTree::Group(group) => {
                    self.record_tokens(group.stream());
                    index += 1;
                }
                TokenTree::Ident(ident)
                    if ident != "crate"
                        && ident != "super"
                        && !self.roots.contains(&ident.to_string()) =>
                {
                    self.idents.insert(ident.to_string());
                    index += 1;
                }
                TokenTree::Ident(ident) => {
                    let mut segments = vec![ident.to_string()];
                    index += 1;
                    while index + 2 < tokens.len()
                        && is_colon(&tokens[index])
                        && is_colon(&tokens[index + 1])
                    {
                        match &tokens[index + 2] {
                            TokenTree::Ident(next) => {
                                segments.push(next.to_string());
                                index += 3;
                            }
                            TokenTree::Punct(punct) if punct.as_char() == '*' => {
                                segments.push(ROOT_GLOB.to_string());
                                index += 3;
                                break;
                            }
                            _ => break,
                        }
                    }
                    self.record(&segments);
                }
                _ => index += 1,
            }
        }
    }
}

fn is_colon(token: &TokenTree) -> bool {
    matches!(token, TokenTree::Punct(punct) if punct.as_char() == ':')
}

fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

fn item_is_test_only(item: &syn::Item) -> bool {
    is_test_only(item_attrs(item))
}

fn public_item_name(item: &syn::Item) -> Option<String> {
    let (vis, ident) = match item {
        syn::Item::Const(item) => (&item.vis, &item.ident),
        syn::Item::Enum(item) => (&item.vis, &item.ident),
        syn::Item::Fn(item) => (&item.vis, &item.sig.ident),
        syn::Item::Static(item) => (&item.vis, &item.ident),
        syn::Item::Struct(item) => (&item.vis, &item.ident),
        syn::Item::Trait(item) => (&item.vis, &item.ident),
        syn::Item::TraitAlias(item) => (&item.vis, &item.ident),
        syn::Item::Type(item) => (&item.vis, &item.ident),
        syn::Item::Union(item) => (&item.vis, &item.ident),
        _ => return None,
    };
    matches!(vis, syn::Visibility::Public(_)).then(|| ident.to_string())
}

/// Flattens a use tree into `(path, exported name)` pairs; the exported
/// name is `*` for a glob.
fn flatten_use(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, String)>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            flatten_use(&path.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = prefix.clone();
            full.push(name.ident.to_string());
            out.push((full, name.ident.to_string()));
        }
        syn::UseTree::Rename(rename) => {
            let mut full = prefix.clone();
            full.push(rename.ident.to_string());
            out.push((full, rename.rename.to_string()));
        }
        syn::UseTree::Glob(_) => {
            let mut full = prefix.clone();
            full.push(ROOT_GLOB.to_string());
            out.push((full, ROOT_GLOB.to_string()));
        }
        syn::UseTree::Group(group) => {
            for tree in &group.items {
                flatten_use(tree, prefix, out);
            }
        }
    }
}

impl<'ast> Visit<'ast> for Walker<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if item_is_test_only(item) {
            return;
        }
        if let syn::Item::Mod(module) = item {
            // A module nested in a function body or another inline item.
            if let Some((_, items)) = &module.content {
                self.depth += 1;
                for item in items {
                    self.visit_item(item);
                }
                self.depth -= 1;
            }
            return;
        }
        syn::visit::visit_item(self, item);
    }

    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        let attrs = match item {
            syn::ImplItem::Const(item) => &item.attrs,
            syn::ImplItem::Fn(item) => &item.attrs,
            syn::ImplItem::Type(item) => &item.attrs,
            syn::ImplItem::Macro(item) => &item.attrs,
            _ => return syn::visit::visit_impl_item(self, item),
        };
        if !is_test_only(attrs) {
            syn::visit::visit_impl_item(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let mut paths = Vec::new();
        flatten_use(&item.tree, &mut Vec::new(), &mut paths);
        for (path, _) in paths {
            self.record(&path);
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let segments: Vec<String> = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.record(&segments);
        syn::visit::visit_path(self, path);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.record_tokens(mac.tokens.clone());
        syn::visit::visit_macro(self, mac);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycles_are_reported_as_sorted_components_largest_first() {
        let analysis = analyze_sources(&[
            ("lib.rs", "mod a; mod b; mod c; mod d; mod e;"),
            ("a.rs", "use crate::b::X;"),
            ("b.rs", "use crate::c::X;"),
            ("c.rs", "use crate::a::X;"),
            ("d.rs", "use crate::e::X; use crate::a::X as Y;"),
            ("e.rs", "use crate::d::X;"),
        ])
        .unwrap();
        assert_eq!(
            analysis.cycles(),
            vec![
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec!["d".to_string(), "e".to_string()],
            ]
        );
    }

    #[test]
    fn reexport_through_a_child_module_resolves_to_the_top_level_module() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod b; pub use a::*;"),
            ("a.rs", "mod inner; pub use inner::*;"),
            ("a/inner.rs", "pub struct Deep;"),
            ("b.rs", "use crate::Deep;"),
        ])
        .unwrap();
        assert_eq!(
            analysis.deps["b"]["a"],
            BTreeSet::from(["Deep".to_string()])
        );
        assert!(analysis.unresolved.is_empty());
    }

    #[test]
    fn reexport_of_another_top_level_module_points_at_the_definer() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod b; pub mod c; pub use b::*;"),
            ("a.rs", "pub struct Moved;"),
            ("b.rs", "pub use crate::a::*;"),
            ("c.rs", "use crate::Moved;"),
        ])
        .unwrap();
        assert_eq!(
            analysis.deps["c"].keys().collect::<Vec<_>>(),
            vec![&"a".to_string()]
        );
    }

    #[test]
    fn glob_import_of_a_module_is_resolved_to_the_items_used() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod b;"),
            (
                "a.rs",
                "pub struct Used; pub struct Unused; pub fn helper() {}",
            ),
            ("b.rs", "use crate::a::*; fn f(_: Used) { helper(); }"),
        ])
        .unwrap();
        assert_eq!(
            analysis.deps["b"]["a"],
            BTreeSet::from(["Used".to_string(), "helper".to_string()])
        );
    }

    #[test]
    fn named_reexport_of_another_module_points_at_the_definer() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod b; pub mod c; pub use b::*;"),
            ("a.rs", "pub struct Moved;"),
            ("b.rs", "pub use crate::a::Moved;"),
            ("c.rs", "use crate::Moved;"),
        ])
        .unwrap();
        assert_eq!(
            analysis.deps["c"].keys().collect::<Vec<_>>(),
            vec![&"a".to_string()]
        );
    }

    #[test]
    fn paths_through_another_analyzed_crate_are_followed() {
        let analysis = analyze_crates(
            &["low-crate/src", "high/src"],
            &[
                ("low-crate/src/lib.rs", "pub mod tensor; pub use tensor::*;"),
                (
                    "low-crate/src/tensor.rs",
                    "pub struct Tensor; pub struct Shape;",
                ),
                (
                    "high/src/lib.rs",
                    "pub use low_crate::tensor; pub use tensor::*; pub mod api; pub mod plan;",
                ),
                (
                    "high/src/api.rs",
                    "use crate::tensor::Tensor; use crate::Shape;",
                ),
                (
                    "high/src/plan.rs",
                    "use low_crate::tensor::Shape; use low_crate::Tensor;",
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            analysis.modules,
            BTreeSet::from(["api".to_string(), "plan".to_string(), "tensor".to_string()])
        );
        let both = BTreeSet::from(["Shape".to_string(), "Tensor".to_string()]);
        assert_eq!(analysis.deps["api"]["tensor"], both);
        assert_eq!(analysis.deps["plan"]["tensor"], both);
        assert!(analysis.unresolved.is_empty());
    }

    #[test]
    fn a_module_defined_by_two_crates_is_an_error() {
        let error = analyze_crates(
            &["a/src", "b/src"],
            &[
                ("a/src/lib.rs", "pub mod same;"),
                ("a/src/same.rs", ""),
                ("b/src/lib.rs", "pub mod same;"),
                ("b/src/same.rs", ""),
            ],
        )
        .unwrap_err();
        assert!(error.contains("`same` is defined by more than one analyzed crate"));
    }

    #[test]
    fn unknown_crate_root_name_is_unresolved() {
        let analysis =
            analyze_sources(&[("lib.rs", "pub mod a;"), ("a.rs", "use crate::Nowhere;")]).unwrap();
        assert_eq!(
            analysis.unresolved["a"],
            BTreeSet::from(["Nowhere".to_string()])
        );
    }

    #[test]
    fn path_attribute_and_mod_rs_layouts_are_followed() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod b; pub mod c;"),
            ("a/mod.rs", "mod inner;"),
            ("a/inner.rs", "use crate::c::X;"),
            ("b.rs", "#[path = \"b_extra.rs\"] mod extra;"),
            ("b_extra.rs", "use crate::c::Y;"),
            ("c.rs", ""),
        ])
        .unwrap();
        assert_eq!(analysis.deps["a"]["c"], BTreeSet::from(["X".to_string()]));
        assert_eq!(analysis.deps["b"]["c"], BTreeSet::from(["Y".to_string()]));
    }

    #[test]
    fn cfg_not_test_code_is_checked() {
        let analysis = analyze_sources(&[
            ("lib.rs", "pub mod a; pub mod c;"),
            ("a.rs", "#[cfg(not(test))] use crate::c::X;"),
            ("c.rs", ""),
        ])
        .unwrap();
        assert!(analysis.deps["a"].contains_key("c"));
    }
}
