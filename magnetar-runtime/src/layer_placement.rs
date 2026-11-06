//! Runtime-owned placement of a Model Instance's decoder layers across the
//! Devices of one Provider (`enforce-local-multigpu-and-expert-placement`
//! G05, staged layer segments first). The Runtime decides which contiguous
//! range of layers each Device holds, from the weights' sizes and each
//! Device's memory capacity, and hands the plan to the Provider
//! ([`crate::provider::ProviderExecutionApi::apply_layer_placement`]) before
//! any weight is written. The Provider stores each weight on its layer's
//! Device and runs each Kernel where its weights are, moving activations
//! across a segment boundary explicitly.
//!
//! Layers are identified by the Runtime's canonical weight names
//! (`layers.{N}.…`); the token embedding belongs to the first segment and
//! every other weight outside a layer (final norm, output head) to the
//! last. Nothing here is specific to a model family.

use crate::device::DeviceId;

/// One Device's contiguous range of decoder layers `[first_layer,
/// end_layer)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerSegment {
    pub device: DeviceId,
    pub first_layer: u32,
    pub end_layer: u32,
}

/// Ordered, contiguous segments covering every decoder layer once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerPlacement {
    segments: Vec<LayerSegment>,
}

/// How [`LayerPlacement::plan`] spreads the layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerPlacementPolicy {
    /// As few Devices as fit: fill the first Device, then the next. A
    /// model that fits one Device stays on it (no boundary movement).
    Fill,
    /// Every Device, each holding a share of the weights proportional to
    /// its budget.
    Balanced,
}

impl LayerPlacementPolicy {
    /// `MAGNETAR_LAYER_PLACEMENT=balanced` selects [`Self::Balanced`];
    /// anything else, or nothing, [`Self::Fill`].
    pub fn from_env() -> Self {
        match std::env::var("MAGNETAR_LAYER_PLACEMENT").as_deref() {
            Ok("balanced") => Self::Balanced,
            _ => Self::Fill,
        }
    }
}

/// Why no placement fits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayerPlacementError {
    NoDevices,
    NoLayers,
    /// The weights exceed every Device's budget combined, or a single
    /// layer exceeds the Device it would go to.
    DoesNotFit {
        required: u64,
        available: u64,
    },
    Invalid(String),
}

impl std::fmt::Display for LayerPlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevices => write!(f, "no Device to place layers on"),
            Self::NoLayers => write!(f, "no decoder layer to place"),
            Self::DoesNotFit {
                required,
                available,
            } => write!(
                f,
                "{required} weight bytes do not fit the Devices' {available}-byte budgets"
            ),
            Self::Invalid(reason) => write!(f, "invalid layer placement: {reason}"),
        }
    }
}

impl std::error::Error for LayerPlacementError {}

/// The bytes a placement must hold: each decoder layer's weights, those
/// only the first segment uses (token embedding) and those only the last
/// uses (final norm, output head).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayerWeightBytes {
    pub layers: Vec<u64>,
    pub first: u64,
    pub last: u64,
}

impl LayerWeightBytes {
    /// Sums `(canonical name, bytes)` pairs by [`weight_layer`].
    pub fn from_weights<'a>(
        layer_count: u32,
        weights: impl IntoIterator<Item = (&'a str, u64)>,
    ) -> Self {
        let mut bytes = Self {
            layers: vec![0; layer_count as usize],
            ..Self::default()
        };
        for (name, size) in weights {
            match weight_layer(name) {
                Some(layer) if (layer as usize) < bytes.layers.len() => {
                    bytes.layers[layer as usize] += size;
                }
                _ if name == FIRST_SEGMENT_WEIGHT => bytes.first += size,
                _ => bytes.last += size,
            }
        }
        bytes
    }

    fn total(&self) -> u64 {
        self.first + self.last + self.layers.iter().sum::<u64>()
    }
}

/// The one weight outside the decoder layers the first segment holds.
const FIRST_SEGMENT_WEIGHT: &str = "token_embedding";

/// The decoder layer a canonical weight name (`layers.{N}.…`, possibly
/// behind a resource prefix ending in `.`) belongs to.
pub fn weight_layer(name: &str) -> Option<u32> {
    let start = if name.starts_with("layers.") {
        0
    } else {
        name.find(".layers.")? + 1
    };
    let rest = &name[start + "layers.".len()..];
    rest.split('.').next()?.parse().ok()
}

impl LayerPlacement {
    /// One segment holding every layer on `device`.
    pub fn single(device: DeviceId, layers: u32) -> Self {
        Self {
            segments: vec![LayerSegment {
                device,
                first_layer: 0,
                end_layer: layers,
            }],
        }
    }

    /// Validates explicit segments: ordered, contiguous, non-empty, from
    /// layer 0 to `layers`.
    pub fn new(segments: Vec<LayerSegment>, layers: u32) -> Result<Self, LayerPlacementError> {
        if segments.is_empty() {
            return Err(LayerPlacementError::NoDevices);
        }
        let mut next = 0;
        for segment in &segments {
            if segment.first_layer != next || segment.end_layer <= segment.first_layer {
                return Err(LayerPlacementError::Invalid(format!(
                    "segment [{}, {}) does not continue at layer {next}",
                    segment.first_layer, segment.end_layer
                )));
            }
            next = segment.end_layer;
        }
        if next != layers {
            return Err(LayerPlacementError::Invalid(format!(
                "segments end at layer {next}, not {layers}"
            )));
        }
        Ok(Self { segments })
    }

    /// Places `bytes` on `devices` (in order, each with its byte budget)
    /// under `policy`, every segment within its Device's budget.
    pub fn plan(
        bytes: &LayerWeightBytes,
        devices: &[(DeviceId, u64)],
        policy: LayerPlacementPolicy,
    ) -> Result<Self, LayerPlacementError> {
        let layers = bytes.layers.len() as u32;
        if devices.is_empty() {
            return Err(LayerPlacementError::NoDevices);
        }
        if layers == 0 {
            return Err(LayerPlacementError::NoLayers);
        }
        let available: u64 = devices.iter().map(|(_, budget)| budget).sum();
        let total = bytes.total();
        let does_not_fit = LayerPlacementError::DoesNotFit {
            required: total,
            available,
        };
        if total > available {
            return Err(does_not_fit);
        }
        // How much each Device should take before the next one starts.
        let targets: Vec<u64> = match policy {
            LayerPlacementPolicy::Fill => devices.iter().map(|(_, budget)| *budget).collect(),
            LayerPlacementPolicy::Balanced => devices
                .iter()
                .map(|(_, budget)| {
                    ((u128::from(total) * u128::from(*budget)) / u128::from(available)) as u64
                })
                .collect(),
        };
        let mut segments: Vec<LayerSegment> = Vec::new();
        let mut layer = 0_u32;
        for (index, (device, budget)) in devices.iter().enumerate() {
            if layer == layers {
                break;
            }
            let last_device = index + 1 == devices.len();
            let mut held = if layer == 0 { bytes.first } else { 0 };
            let first_layer = layer;
            while layer < layers {
                let next = held + bytes.layers[layer as usize];
                let with_tail = if layer + 1 == layers {
                    next + bytes.last
                } else {
                    next
                };
                let over_target = with_tail > targets[index] && layer > first_layer;
                if !last_device && over_target {
                    break;
                }
                if with_tail > *budget {
                    if layer > first_layer && !last_device {
                        break;
                    }
                    return Err(does_not_fit);
                }
                held = next;
                layer += 1;
            }
            if layer > first_layer {
                segments.push(LayerSegment {
                    device: device.clone(),
                    first_layer,
                    end_layer: layer,
                });
            }
        }
        if layer != layers {
            return Err(does_not_fit);
        }
        Self::new(segments, layers)
    }

    pub fn segments(&self) -> &[LayerSegment] {
        &self.segments
    }

    /// The Device holding the weight `name` (a canonical weight name,
    /// possibly behind a resource prefix).
    pub fn device_for_weight(&self, name: &str) -> &DeviceId {
        let first = &self.segments[0];
        let last = &self.segments[self.segments.len() - 1];
        match weight_layer(name) {
            Some(layer) => self
                .segments
                .iter()
                .find(|segment| (segment.first_layer..segment.end_layer).contains(&layer))
                .map_or(&last.device, |segment| &segment.device),
            None if name == FIRST_SEGMENT_WEIGHT
                || name.ends_with(&format!(".{FIRST_SEGMENT_WEIGHT}")) =>
            {
                &first.device
            }
            None => &last.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(name: &str) -> DeviceId {
        DeviceId::new(name)
    }

    #[test]
    fn weight_layer_reads_the_canonical_layer_index() {
        assert_eq!(weight_layer("layers.12.self_attn.q_proj"), Some(12));
        assert_eq!(
            weight_layer("model.instance-1.weight.layers.3.moe.router_proj"),
            Some(3)
        );
        assert_eq!(weight_layer("token_embedding"), None);
        assert_eq!(weight_layer("model.i.weight.lm_head"), None);
    }

    #[test]
    fn fill_stays_on_one_device_when_it_fits_and_balanced_splits() {
        let bytes = LayerWeightBytes {
            layers: vec![10; 4],
            first: 5,
            last: 5,
        };
        let devices = [(device("gpu0"), 100), (device("gpu1"), 100)];
        let fill = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Fill).unwrap();
        assert_eq!(fill.segments().len(), 1);
        let balanced =
            LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Balanced).unwrap();
        assert_eq!(
            balanced.segments(),
            &[
                LayerSegment {
                    device: device("gpu0"),
                    first_layer: 0,
                    end_layer: 2,
                },
                LayerSegment {
                    device: device("gpu1"),
                    first_layer: 2,
                    end_layer: 4,
                },
            ]
        );
        assert_eq!(
            balanced.device_for_weight("model.i.weight.token_embedding"),
            &device("gpu0")
        );
        assert_eq!(
            balanced.device_for_weight("model.i.weight.layers.3.mlp.up_proj"),
            &device("gpu1")
        );
        assert_eq!(
            balanced.device_for_weight("model.i.weight.lm_head"),
            &device("gpu1")
        );
    }

    #[test]
    fn fill_spills_to_the_next_device_and_rejects_what_cannot_fit() {
        // 15 GB of layers on two 12 GB Devices.
        let gb = 1_u64 << 30;
        let bytes = LayerWeightBytes {
            layers: vec![gb / 2; 24],
            first: gb / 4,
            last: gb / 4,
        };
        let devices = [(device("gpu0"), 10 * gb), (device("gpu1"), 10 * gb)];
        let plan = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Fill).unwrap();
        assert_eq!(plan.segments().len(), 2);
        assert_eq!(plan.segments()[0].end_layer, 19);
        assert!(matches!(
            LayerPlacement::plan(&bytes, &devices[..1], LayerPlacementPolicy::Fill),
            Err(LayerPlacementError::DoesNotFit { .. })
        ));
        assert!(
            LayerPlacement::new(
                vec![LayerSegment {
                    device: device("gpu0"),
                    first_layer: 1,
                    end_layer: 24,
                }],
                24
            )
            .is_err()
        );
    }
}
