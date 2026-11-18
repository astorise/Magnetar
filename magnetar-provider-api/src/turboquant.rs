//! TurboQuant KV quantization (`turboquant-kv-history`), shared by every
//! Provider so that a quantized KV history means the same values
//! everywhere.
//!
//! Each head's row `x` of `d` values (`d` a power of two) is stored as its
//! norm `‖x‖` and, for every coordinate of the rotated unit vector
//! `y = Π x / ‖x‖`, the index of its nearest centroid in a `b`-bit codebook.
//! `Π = H D / √d` is the Walsh-Hadamard matrix `H` after fixed
//! pseudo-random signs `D`: orthogonal, applied in `O(d log d)`. The
//! codebook is the Lloyd-Max quantizer of one coordinate of a uniformly
//! random unit vector in `R^d`, whose density is proportional to
//! `(1 − t²)^((d − 3) / 2)` on `[−1, 1]` (Zandieh et al., "TurboQuant:
//! Online Vector Quantization with Near-optimal Distortion Rate"). The
//! stored row reads back as `‖x‖ · Πᵀ ĉ`.
//!
//! A store also keeps one `f32` offset per column of its rows
//! ([`turboquant_store_offsets`]): the mean of the rows of the write that
//! starts its history (the write at row 0). Each row is quantized as its
//! difference from the offsets and reads back with them added, so a
//! component every row shares (a key projection's bias, outlier channels)
//! is kept exactly instead of taking up the codes' precision. Offsets are
//! all zero until that write.
//!
//! The signs and the codebooks are functions of `d` and `b` alone,
//! computed deterministically in `f64`; only the offsets depend on data.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

/// A KV store's storage precision.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvStorePrecision {
    /// Every value as `f32` (the reference).
    #[default]
    F32,
    /// TurboQuant, 4-bit codes.
    Tq4,
    /// TurboQuant, 3-bit codes.
    Tq3,
}

impl KvStorePrecision {
    /// Parses `f32`, `tq4` or `tq3`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "f32" => Some(Self::F32),
            "tq4" => Some(Self::Tq4),
            "tq3" => Some(Self::Tq3),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Tq4 => "tq4",
            Self::Tq3 => "tq3",
        }
    }

    /// The code width of a TurboQuant precision, `None` for `f32`.
    pub fn bits(self) -> Option<u32> {
        match self {
            Self::F32 => None,
            Self::Tq4 => Some(4),
            Self::Tq3 => Some(3),
        }
    }

    /// Whether a head of `head_dimension` values can be stored in this
    /// precision: any width as `f32`; a power of two of at least 8 for
    /// TurboQuant (whole bytes of 3-bit codes).
    pub fn supports_head_dimension(self, head_dimension: u64) -> bool {
        match self {
            Self::F32 => head_dimension > 0,
            Self::Tq4 | Self::Tq3 => head_dimension >= 8 && head_dimension.is_power_of_two(),
        }
    }

    /// Bytes one head's row takes: `4 d` as `f32`; the packed codes and an
    /// `f32` norm in TurboQuant.
    pub fn head_row_bytes(self, head_dimension: u64) -> u64 {
        match self.bits() {
            None => 4 * head_dimension,
            Some(bits) => head_dimension * u64::from(bits) / 8 + 4,
        }
    }

    /// Bytes a store of `capacity` rows of `width` values takes, its heads
    /// `head_dimension` wide; `None` when `width` is not whole heads or the
    /// head width is unsupported.
    pub fn stored_bytes(self, capacity: u64, width: u64, head_dimension: u64) -> Option<u64> {
        if !self.supports_head_dimension(head_dimension) || !width.is_multiple_of(head_dimension) {
            return None;
        }
        capacity
            .checked_mul(width / head_dimension)?
            .checked_mul(self.head_row_bytes(head_dimension))
    }

    /// [`Self::stored_bytes`] of a store whose first `exact_rows` rows stay
    /// exact (`keep-turboquant-kv-head-exact`): the codes of every row plus
    /// those rows as `f32`. An `f32` store has no separate head.
    pub fn stored_bytes_with_head(
        self,
        capacity: u64,
        width: u64,
        head_dimension: u64,
        exact_rows: u64,
    ) -> Option<u64> {
        let codes = self.stored_bytes(capacity, width, head_dimension)?;
        if self.bits().is_none() {
            return Some(codes);
        }
        exact_rows
            .min(capacity)
            .checked_mul(width)?
            .checked_mul(4)?
            .checked_add(codes)
    }
}

/// The `b`-bit codebook for coordinates of unit vectors in `R^d`.
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantCodebook {
    pub dimension: u64,
    pub bits: u32,
    /// `2^b` centroids, ascending and symmetric about 0.
    pub centroids: Vec<f32>,
    /// The `2^b − 1` midpoints between consecutive centroids: a coordinate
    /// takes the index of the first boundary above it.
    pub boundaries: Vec<f32>,
    /// The expected squared error of one coordinate, from the density.
    pub coordinate_mse: f64,
}

impl TurboQuantCodebook {
    /// The index of the centroid nearest `value`.
    pub fn index(&self, value: f32) -> u8 {
        self.boundaries
            .partition_point(|boundary| *boundary < value) as u8
    }
}

/// The computed codebooks, by dimension and bits.
type CodebookCache = BTreeMap<(u64, u32), Arc<TurboQuantCodebook>>;

/// The codebook for `dimension` (a power of two of at least 8) and `bits`
/// (1 to 8), computed once per pair and shared.
pub fn turboquant_codebook(dimension: u64, bits: u32) -> Arc<TurboQuantCodebook> {
    static CACHE: OnceLock<Mutex<CodebookCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = cache.lock().unwrap();
    cache
        .entry((dimension, bits))
        .or_insert_with(|| Arc::new(lloyd_max_codebook(dimension, bits)))
        .clone()
}

/// Lloyd's algorithm on the coordinate density, discretized on a fine grid
/// over the interval holding all but a negligible tail of the mass.
fn lloyd_max_codebook(dimension: u64, bits: u32) -> TurboQuantCodebook {
    assert!(
        dimension >= 8 && dimension.is_power_of_two(),
        "TurboQuant dimension {dimension} must be a power of two of at least 8"
    );
    assert!(
        (1..=8).contains(&bits),
        "TurboQuant bits {bits} must be 1 to 8"
    );
    let d = dimension as f64;
    // 14 standard deviations (`1/√d`) hold all but ~1e-44 of the mass.
    let half_width = (14.0 / d.sqrt()).min(1.0);
    const POINTS: usize = 1 << 16;
    let step = 2.0 * half_width / POINTS as f64;
    let exponent = (d - 3.0) / 2.0;
    let grid: Vec<f64> = (0..POINTS)
        .map(|i| -half_width + (i as f64 + 0.5) * step)
        .collect();
    let mut weights: Vec<f64> = grid
        .iter()
        .map(|t| {
            let inside = 1.0 - t * t;
            if inside <= 0.0 {
                0.0
            } else {
                (exponent * inside.ln()).exp()
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= total);

    let levels = 1usize << bits;
    // Start at the levels' quantile midpoints.
    let mut centroids = Vec::with_capacity(levels);
    let mut cumulative = 0.0;
    let mut next = 0;
    for (t, weight) in grid.iter().zip(&weights) {
        cumulative += weight;
        while next < levels && cumulative >= (next as f64 + 0.5) / levels as f64 {
            centroids.push(*t);
            next += 1;
        }
    }
    while centroids.len() < levels {
        centroids.push(half_width);
    }
    let mut boundaries = vec![0.0_f64; levels - 1];
    for _ in 0..500 {
        for (boundary, pair) in boundaries.iter_mut().zip(centroids.windows(2)) {
            *boundary = 0.5 * (pair[0] + pair[1]);
        }
        let mut mass = vec![0.0_f64; levels];
        let mut moment = vec![0.0_f64; levels];
        let mut cell = 0;
        for (t, weight) in grid.iter().zip(&weights) {
            while cell < levels - 1 && *t > boundaries[cell] {
                cell += 1;
            }
            mass[cell] += weight;
            moment[cell] += weight * t;
        }
        let mut moved = 0.0_f64;
        for level in 0..levels {
            if mass[level] > 0.0 {
                let updated = moment[level] / mass[level];
                moved = moved.max((updated - centroids[level]).abs());
                centroids[level] = updated;
            }
        }
        if moved < 1e-13 {
            break;
        }
    }
    // Exact symmetry: average each centroid with its mirror.
    for level in 0..levels / 2 {
        let mirrored = levels - 1 - level;
        let magnitude = 0.5 * (centroids[mirrored] - centroids[level]);
        centroids[level] = -magnitude;
        centroids[mirrored] = magnitude;
    }
    for (boundary, pair) in boundaries.iter_mut().zip(centroids.windows(2)) {
        *boundary = 0.5 * (pair[0] + pair[1]);
    }
    let mut coordinate_mse = 0.0;
    let mut cell = 0;
    for (t, weight) in grid.iter().zip(&weights) {
        while cell < levels - 1 && *t > boundaries[cell] {
            cell += 1;
        }
        coordinate_mse += weight * (t - centroids[cell]).powi(2);
    }
    TurboQuantCodebook {
        dimension,
        bits,
        centroids: centroids.iter().map(|value| *value as f32).collect(),
        boundaries: boundaries.iter().map(|value| *value as f32).collect(),
        coordinate_mse,
    }
}

/// The fixed signs `D` for `dimension`: `±1.0`, from splitmix64 seeded
/// with a constant and the dimension.
pub fn turboquant_signs(dimension: u64) -> Vec<f32> {
    let mut state = 0x7475_7262_6f71_7561_u64 ^ dimension.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            if z & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// The unnormalized fast Walsh-Hadamard transform, in place (`values.len()`
/// a power of two). Butterflies from stride 1 upwards.
pub fn walsh_hadamard_in_place(values: &mut [f32]) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    let mut half = 1;
    while half < n {
        for start in (0..n).step_by(2 * half) {
            for i in start..start + half {
                let (a, b) = (values[i], values[i + half]);
                values[i] = a + b;
                values[i + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// `values ← Π values = H D values / √d`.
pub fn turboquant_rotate(values: &mut [f32], signs: &[f32]) {
    for (value, sign) in values.iter_mut().zip(signs) {
        *value *= sign;
    }
    walsh_hadamard_in_place(values);
    let scale = 1.0 / (values.len() as f32).sqrt();
    values.iter_mut().for_each(|value| *value *= scale);
}

/// `values ← Πᵀ values = D H values / √d`.
pub fn turboquant_rotate_back(values: &mut [f32], signs: &[f32]) {
    walsh_hadamard_in_place(values);
    let scale = 1.0 / (values.len() as f32).sqrt();
    for (value, sign) in values.iter_mut().zip(signs) {
        *value *= sign * scale;
    }
}

/// One head's quantized row: its norm and one centroid index per
/// coordinate (unpacked).
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantHead {
    pub norm: f32,
    pub indices: Vec<u8>,
}

/// Quantizes one head's row.
pub fn turboquant_quantize_head(
    head: &[f32],
    signs: &[f32],
    codebook: &TurboQuantCodebook,
) -> TurboQuantHead {
    let norm = head.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        // A zero row stores zero; a non-finite one keeps its norm, so it
        // reads back non-finite rather than as a plausible value.
        return TurboQuantHead {
            norm,
            indices: vec![0; head.len()],
        };
    }
    let mut rotated: Vec<f32> = head.iter().map(|value| value / norm).collect();
    turboquant_rotate(&mut rotated, signs);
    TurboQuantHead {
        norm,
        indices: rotated.iter().map(|value| codebook.index(*value)).collect(),
    }
}

/// The values a quantized head reads back as.
pub fn turboquant_dequantize_head(
    head: &TurboQuantHead,
    signs: &[f32],
    codebook: &TurboQuantCodebook,
) -> Vec<f32> {
    if head.norm == 0.0 {
        return vec![0.0; head.indices.len()];
    }
    let mut values: Vec<f32> = head
        .indices
        .iter()
        .map(|index| codebook.centroids[*index as usize])
        .collect();
    turboquant_rotate_back(&mut values, signs);
    values.iter_mut().for_each(|value| *value *= head.norm);
    values
}

/// `row` (whole heads of `head_dimension` values) as a store in `precision`
/// holds it: unchanged for `f32`, each head quantized and read back for
/// TurboQuant. `None` when the head width is unsupported or `row` is not
/// whole heads.
pub fn kv_store_round_trip(
    row: &[f32],
    head_dimension: u64,
    precision: KvStorePrecision,
) -> Option<Vec<f32>> {
    let Some(bits) = precision.bits() else {
        return Some(row.to_vec());
    };
    let d = head_dimension as usize;
    if !precision.supports_head_dimension(head_dimension) || !row.len().is_multiple_of(d) {
        return None;
    }
    let signs = turboquant_signs(head_dimension);
    let codebook = turboquant_codebook(head_dimension, bits);
    Some(
        row.chunks_exact(d)
            .flat_map(|head| {
                let quantized = turboquant_quantize_head(head, &signs, &codebook);
                turboquant_dequantize_head(&quantized, &signs, &codebook)
            })
            .collect(),
    )
}

/// A TurboQuant store's offsets (one per column of its `width`-wide rows):
/// the mean of `rows`, the rows of the write that starts the store's
/// history, summed in row order in `f32` and divided by their count. Zeros
/// for no rows.
pub fn turboquant_store_offsets(rows: &[f32], width: usize) -> Vec<f32> {
    let mut offsets = vec![0.0f32; width];
    if width == 0 || rows.len() < width {
        return offsets;
    }
    let count = rows.len() / width;
    for row in rows.chunks_exact(width) {
        for (offset, value) in offsets.iter_mut().zip(row) {
            *offset += value;
        }
    }
    offsets
        .iter_mut()
        .for_each(|offset| *offset /= count as f32);
    offsets
}

/// `rows` (whole heads of `head_dimension` values; `offsets.len()` values
/// per row) as a TurboQuant store with those offsets holds them: each head
/// of `row - offsets` quantized and read back, plus `offsets`. Unchanged
/// for `f32`. `None` when the head width is unsupported or the rows are not
/// whole heads of `offsets`' width.
pub fn kv_store_round_trip_with_offsets(
    rows: &[f32],
    head_dimension: u64,
    precision: KvStorePrecision,
    offsets: &[f32],
) -> Option<Vec<f32>> {
    if precision.bits().is_none() {
        return Some(rows.to_vec());
    }
    let width = offsets.len();
    if width == 0
        || head_dimension == 0
        || !width.is_multiple_of(head_dimension as usize)
        || !rows.len().is_multiple_of(width)
    {
        return None;
    }
    let centered: Vec<f32> = rows
        .chunks_exact(width)
        .flat_map(|row| {
            row.iter()
                .zip(offsets)
                .map(|(value, offset)| value - offset)
        })
        .collect();
    let mut stored = kv_store_round_trip(&centered, head_dimension, precision)?;
    for row in stored.chunks_exact_mut(width) {
        for (value, offset) in row.iter_mut().zip(offsets) {
            *value += offset;
        }
    }
    Some(stored)
}

/// What a Provider that keeps a TurboQuant store's rows as their round trip
/// (the CPU Providers) knows of the store: its precision, head width,
/// offsets and exact head.
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantStoreState {
    pub precision: KvStorePrecision,
    pub head_dimension: u64,
    /// See [`turboquant_store_offsets`]; zeros until the write at row 0.
    pub offsets: Vec<f32>,
    /// The rows below this index stay exact
    /// (`keep-turboquant-kv-head-exact`); 0 for none.
    pub exact_rows: u64,
}

impl TurboQuantStoreState {
    /// A store of `width`-wide rows, its offsets zero, without an exact
    /// head.
    pub fn new(precision: KvStorePrecision, head_dimension: u64, width: u64) -> Self {
        Self {
            precision,
            head_dimension,
            offsets: vec![0.0; width as usize],
            exact_rows: 0,
        }
    }

    /// This store with its first `exact_rows` rows kept exact.
    pub fn with_exact_rows(mut self, exact_rows: u64) -> Self {
        self.exact_rows = exact_rows;
        self
    }

    /// `rows` written at `row_offset`, as the store holds them: a row below
    /// the exact head's end unchanged, any other as its round trip. A write
    /// at row 0 starts the history: it first sets the offsets from its rows.
    /// `None` when the rows are not whole rows of the store.
    pub fn write(&mut self, rows: &[f32], row_offset: u64) -> Option<Vec<f32>> {
        let width = self.offsets.len();
        if width == 0 || !rows.len().is_multiple_of(width) {
            return None;
        }
        if row_offset == 0 && !rows.is_empty() {
            self.offsets = turboquant_store_offsets(rows, width);
        }
        let mut stored = kv_store_round_trip_with_offsets(
            rows,
            self.head_dimension,
            self.precision,
            &self.offsets,
        )?;
        let exact = self
            .exact_rows
            .saturating_sub(row_offset)
            .min((rows.len() / width) as u64);
        let exact_values = exact as usize * width;
        stored[..exact_values].copy_from_slice(&rows[..exact_values]);
        Some(stored)
    }
}

/// Packs `indices` (each below `2^bits`) into bytes: 4-bit codes two per
/// byte, low nibble first; 3-bit codes eight per three bytes,
/// little-endian bit order.
pub fn turboquant_pack(indices: &[u8], bits: u32) -> Vec<u8> {
    let total_bits = indices.len() * bits as usize;
    let mut packed = vec![0u8; total_bits.div_ceil(8)];
    for (position, index) in indices.iter().enumerate() {
        let bit = position * bits as usize;
        let value = u16::from(*index) << (bit % 8);
        packed[bit / 8] |= value as u8;
        if bit % 8 + bits as usize > 8 {
            packed[bit / 8 + 1] |= (value >> 8) as u8;
        }
    }
    packed
}

/// The inverse of [`turboquant_pack`] for `count` codes.
pub fn turboquant_unpack(packed: &[u8], bits: u32, count: usize) -> Vec<u8> {
    let mask = (1u16 << bits) - 1;
    (0..count)
        .map(|position| {
            let bit = position * bits as usize;
            let low = u16::from(packed[bit / 8]);
            let high = packed.get(bit / 8 + 1).copied().map_or(0, u16::from);
            (((low | (high << 8)) >> (bit % 8)) & mask) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests;
