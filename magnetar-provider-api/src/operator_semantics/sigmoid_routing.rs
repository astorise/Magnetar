//! Portable reference routing for selection-biased sigmoid experts.

use std::collections::BTreeMap;

use crate::operator::OperatorAttributeValue;
use crate::tensor::{ReferenceCpuError, ReferenceCpuErrorCode};

/// Whether the chosen unbiased sigmoid scores are normalized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SigmoidNormalization {
    Never,
    /// Olala normalizes only when more than one expert is selected.
    MultipleExperts,
    /// GLM normalizes even when only one expert is selected.
    Always,
}

/// Equal groups ranked by the sum of their two largest biased scores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertGroupSelection {
    pub count: usize,
    pub selected: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidRouting {
    pub top_k: usize,
    pub groups: Option<ExpertGroupSelection>,
    pub normalization: SigmoidNormalization,
    pub scale: f32,
}

/// Parse the explicitly selected sigmoid form of `moe-experts`. The absence
/// of these attributes preserves the original softmax operator. A partial or
/// unknown declaration is refused instead of falling back to softmax.
pub fn sigmoid_routing_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
    top_k: u64,
) -> Result<Option<SigmoidRouting>, ReferenceCpuError> {
    let invalid =
        |message: &str| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message);
    const OTHER_ATTRIBUTES: [&str; 4] = [
        "routing_normalization",
        "routing_scale",
        "routing_group_count",
        "routing_group_selected",
    ];
    match attributes.get("routing_scoring") {
        None if OTHER_ATTRIBUTES
            .iter()
            .all(|name| !attributes.contains_key(*name)) =>
        {
            return Ok(None);
        }
        Some(OperatorAttributeValue::String(value)) if value == "sigmoid" => {}
        _ => return Err(invalid("unsupported or incomplete expert routing scoring")),
    }
    let normalization = match attributes.get("routing_normalization") {
        Some(OperatorAttributeValue::String(value)) if value == "never" => {
            SigmoidNormalization::Never
        }
        Some(OperatorAttributeValue::String(value)) if value == "multiple" => {
            SigmoidNormalization::MultipleExperts
        }
        Some(OperatorAttributeValue::String(value)) if value == "always" => {
            SigmoidNormalization::Always
        }
        _ => return Err(invalid("invalid sigmoid routing normalization")),
    };
    if let Some(value) = attributes.get("normalize_top_k") {
        match value {
            OperatorAttributeValue::Boolean(value)
                if *value == (normalization != SigmoidNormalization::Never) => {}
            _ => return Err(invalid("conflicting top-k normalization attributes")),
        }
    }
    let scale = match attributes.get("routing_scale") {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite()
                && *value > 0.0
                && (*value as f32).is_finite()
                && (*value as f32) > 0.0 =>
        {
            *value as f32
        }
        _ => return Err(invalid("invalid sigmoid routing scale")),
    };
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => usize::try_from(*value).ok(),
        _ => None,
    };
    let groups = match (
        attributes.get("routing_group_count"),
        attributes.get("routing_group_selected"),
    ) {
        (None, None) => None,
        (Some(_), Some(_)) => Some(ExpertGroupSelection {
            count: positive("routing_group_count")
                .ok_or_else(|| invalid("invalid routing group count"))?,
            selected: positive("routing_group_selected")
                .ok_or_else(|| invalid("invalid selected group count"))?,
        }),
        _ => return Err(invalid("both routing group attributes must be provided")),
    };
    let top_k = usize::try_from(top_k).map_err(|_| invalid("invalid routing top-k"))?;
    Ok(Some(SigmoidRouting {
        top_k,
        groups,
        normalization,
        scale,
    }))
}

/// Select experts using `sigmoid(logit) + bias`, then combine them using
/// the unbiased sigmoid scores. Normalization uses a denominator epsilon of
/// `1e-20`, before the routed scaling factor is applied.
///
/// When grouping is enabled, each group's score is the sum of its two
/// highest biased scores; only the best selected groups are eligible.
/// Ties crossing a group or expert selection boundary are rejected because
/// upstream `topk` does not specify which tied candidate wins. Ties entirely
/// inside the selected set are ordered by index.
pub fn sigmoid_moe_route(
    logits: &[f32],
    bias: &[f32],
    options: SigmoidRouting,
) -> Result<Vec<(usize, f32)>, ReferenceCpuError> {
    let invalid =
        |message: &str| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message);
    if logits.is_empty()
        || logits.len() != bias.len()
        || options.top_k == 0
        || options.top_k > logits.len()
    {
        return Err(invalid("invalid expert count, bias length, or top-k"));
    }
    if !options.scale.is_finite()
        || options.scale <= 0.0
        || logits.iter().chain(bias).any(|value| !value.is_finite())
    {
        return Err(invalid("non-finite routing values or non-positive scale"));
    }
    let mut scores = Vec::with_capacity(logits.len());
    for (index, (&logit, &correction)) in logits.iter().zip(bias).enumerate() {
        let unbiased = 1.0 / (1.0 + (-logit).exp());
        let selection = unbiased + correction;
        if !selection.is_finite() {
            return Err(invalid("non-finite biased expert score"));
        }
        scores.push((index, unbiased, selection));
    }

    let eligible = if let Some(groups) = options.groups {
        if groups.count == 0
            || groups.selected == 0
            || groups.selected > groups.count
            || !logits.len().is_multiple_of(groups.count)
        {
            return Err(invalid("invalid expert group dimensions"));
        }
        let group_width = logits.len() / groups.count;
        if group_width < 2 || options.top_k > groups.selected * group_width {
            return Err(invalid(
                "selected groups cannot supply the requested experts",
            ));
        }
        let mut ranked = Vec::with_capacity(groups.count);
        for (index, group) in scores.chunks_exact(group_width).enumerate() {
            let mut best = [f32::NEG_INFINITY; 2];
            for (_, _, score) in group {
                if *score > best[0] {
                    best[1] = best[0];
                    best[0] = *score;
                } else if *score > best[1] {
                    best[1] = *score;
                }
            }
            let total = best[0] + best[1];
            if !total.is_finite() {
                return Err(invalid("non-finite expert group score"));
            }
            ranked.push((index, total));
        }
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        if groups.selected < ranked.len()
            && ranked[groups.selected - 1].1 == ranked[groups.selected].1
        {
            return Err(invalid("equal group scores cross the selection boundary"));
        }
        let mut allowed = vec![false; groups.count];
        for (group, _) in ranked.into_iter().take(groups.selected) {
            allowed[group] = true;
        }
        scores.retain(|(expert, _, _)| allowed[*expert / group_width]);
        scores
    } else {
        scores
    };
    let mut eligible = eligible;
    eligible.sort_by(|left, right| right.2.total_cmp(&left.2).then(left.0.cmp(&right.0)));
    if options.top_k < eligible.len() && eligible[options.top_k - 1].2 == eligible[options.top_k].2
    {
        return Err(invalid("equal expert scores cross the selection boundary"));
    }
    eligible.truncate(options.top_k);
    let normalize = match options.normalization {
        SigmoidNormalization::Never => false,
        SigmoidNormalization::MultipleExperts => options.top_k > 1,
        SigmoidNormalization::Always => true,
    };
    let denominator = if normalize {
        eligible.iter().map(|(_, score, _)| score).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    let selected: Vec<_> = eligible
        .into_iter()
        .map(|(index, score, _)| (index, score / denominator * options.scale))
        .collect();
    if selected.iter().any(|(_, weight)| !weight.is_finite()) {
        return Err(invalid("non-finite expert weight"));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests;
