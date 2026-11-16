//! Runtime-owned placement of a Model Instance's decoder layers across the
//! Devices of one Provider (`enforce-local-multigpu-and-expert-placement`
//! G05, staged layer segments first). The Runtime decides which contiguous
//! range of layers each Device holds, from the weights' sizes and each
//! Device's memory capacity, and hands the plan to the Provider
//! ([`crate::ProviderExecutionApi::apply_layer_placement`]) before
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
    /// Whether layer `i` must stay in its predecessor's segment (see
    /// [`Self::keep_kv_sources_together`]); empty means no layer must.
    pub joined: Vec<bool>,
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

    /// Keeps each layer that reuses an earlier layer's K/V in that source
    /// layer's segment, for every `(layer, source)` pair: the layer reads
    /// the source's KV history at every step, which a segment boundary
    /// between them would copy across Devices each time.
    pub fn keep_kv_sources_together(&mut self, sources: impl IntoIterator<Item = (u32, u32)>) {
        self.joined.resize(self.layers.len(), false);
        for (layer, source) in sources {
            let end = (layer as usize + 1).min(self.joined.len());
            if let Some(joined) = self.joined.get_mut(source as usize + 1..end) {
                joined.fill(true);
            }
        }
    }

    /// Whether no segment may start at `layer`.
    fn joined_with_previous(&self, layer: u32) -> bool {
        self.joined.get(layer as usize).copied().unwrap_or(false)
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
    /// under `policy`, every segment within its Device's budget. A balanced
    /// share is a soft target: when stopping at the targets leaves too much
    /// for the remaining Devices, the placement fills the Devices instead,
    /// so a model is rejected only when no contiguous placement fits. No
    /// segment starts at a layer joined to its predecessor
    /// ([`LayerWeightBytes::joined`]): a segment reaching its target there
    /// grows to the next allowed start, and one reaching its budget ends at
    /// the last allowed start.
    pub fn plan(
        bytes: &LayerWeightBytes,
        devices: &[(DeviceId, u64)],
        policy: LayerPlacementPolicy,
    ) -> Result<Self, LayerPlacementError> {
        match Self::plan_once(bytes, devices, policy) {
            Err(LayerPlacementError::DoesNotFit { .. })
                if policy == LayerPlacementPolicy::Balanced =>
            {
                Self::plan_once(bytes, devices, LayerPlacementPolicy::Fill)
            }
            result => result,
        }
    }

    fn plan_once(
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
                let may_end = layer > first_layer && !bytes.joined_with_previous(layer);
                if !last_device && may_end && with_tail > targets[index] {
                    break;
                }
                if with_tail > *budget {
                    // End at the last layer a segment may start at.
                    match (first_layer + 1..=layer)
                        .rev()
                        .find(|&start| !bytes.joined_with_previous(start))
                    {
                        Some(start) if !last_device => {
                            layer = start;
                            break;
                        }
                        _ => return Err(does_not_fit),
                    }
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
mod tests;
