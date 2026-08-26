//! CLI-owned friendly model aliases (§6 "Model Commands" / §22 "Model
//! Aliases" in the change proposal).
//!
//! `magnetar-cli` MAY maintain friendly model aliases; Runtime SHALL
//! receive a resolved `ModelRef`, never an alias name. This table is
//! in-memory and non-persistent -- persistent alias/config storage remains
//! out of scope for this change (see `commands::cmd_model_list`'s doc
//! comment for the same honesty note about the model registry). Aliasing
//! here can never bypass Runtime trust/loading/compatibility checks by
//! construction: [`resolve_alias`] only ever produces a plain `&str` that
//! then goes through the exact same `ModelRef::new` validation as any
//! literal reference typed by the user (see `commands::cmd_run` /
//! `commands::cmd_chat`, which call `ModelRef::new(resolve_alias(input))`
//! unconditionally).

/// Illustrative built-in alias table: friendly name -> literal `ModelRef`
/// string. Intentionally small and in-process; not user-configurable in
/// this increment.
pub const BUILT_IN_ALIASES: &[(&str, &str)] = &[
    ("qwen-small", "qwen-test"),
    ("cpu-default", "reference-cpu-model"),
];

/// Resolves `input` against [`BUILT_IN_ALIASES`], returning the mapped
/// literal reference if `input` matches a known alias, or `input` itself
/// unchanged otherwise (i.e. `input` is already treated as a literal
/// `ModelRef` string). Never touches Runtime; the result still has to pass
/// `ModelRef::new`'s own validation just like any other input.
pub fn resolve_alias(input: &str) -> &str {
    BUILT_IN_ALIASES
        .iter()
        .find(|(alias, _)| *alias == input)
        .map(|(_, target)| *target)
        .unwrap_or(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::ModelRef;

    #[test]
    fn known_alias_resolves_to_its_mapped_literal_reference() {
        assert_eq!(resolve_alias("qwen-small"), "qwen-test");
    }

    #[test]
    fn unknown_input_passes_through_unchanged() {
        assert_eq!(resolve_alias("not-an-alias"), "not-an-alias");
    }

    /// An alias and its literal expansion produce the exact same
    /// downstream `ModelRef`, proving alias resolution cannot bypass or
    /// alter `ModelRef::new`'s validation -- it is purely a string
    /// substitution that happens strictly before validation.
    #[test]
    fn alias_and_literal_expansion_produce_the_same_model_ref() {
        for (alias, literal) in BUILT_IN_ALIASES {
            let via_alias = ModelRef::new(resolve_alias(alias)).unwrap();
            let via_literal = ModelRef::new(*literal).unwrap();
            assert_eq!(via_alias, via_literal);
        }
    }
}
