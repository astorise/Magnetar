//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

fn pseudo_random_row(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // A sum of uniforms: roughly Gaussian, with an outlier
            // channel every 16 values as keys have.
            let uniform = |shift: u32| ((state >> shift) & 0xffff) as f32 / 65_535.0 - 0.5;
            uniform(16) + uniform(32) + uniform(48)
        })
        .enumerate()
        .map(|(index, value)| if index % 16 == 3 { value * 12.0 } else { value })
        .collect()
}

#[test]
fn the_rotation_is_orthogonal_and_undone() {
    for d in [8_u64, 64, 128, 512] {
        let signs = turboquant_signs(d);
        assert!(signs.iter().all(|sign| sign.abs() == 1.0));
        assert!(signs.iter().any(|sign| *sign < 0.0) && signs.iter().any(|sign| *sign > 0.0));
        let row = pseudo_random_row(d as usize, d);
        let mut rotated = row.clone();
        turboquant_rotate(&mut rotated, &signs);
        let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm(&rotated) - norm(&row)).abs() < 1e-4 * norm(&row));
        turboquant_rotate_back(&mut rotated, &signs);
        for (back, original) in rotated.iter().zip(&row) {
            assert!((back - original).abs() < 1e-5 * norm(&row), "{d}");
        }
    }
}

#[test]
fn codebooks_are_sorted_symmetric_and_reach_the_lloyd_max_error() {
    // Gaussian Lloyd-Max distortion per unit variance: 0.03454 at 3
    // bits, 0.009497 at 4; the sphere coordinate has variance 1/d.
    for (bits, unit_mse) in [(3_u32, 0.034_54), (4, 0.009_497)] {
        for d in [64_u64, 128, 256, 512] {
            let codebook = turboquant_codebook(d, bits);
            assert_eq!(codebook.centroids.len(), 1 << bits);
            assert!(codebook.centroids.windows(2).all(|pair| pair[0] < pair[1]));
            for (low, high) in codebook
                .centroids
                .iter()
                .zip(codebook.centroids.iter().rev())
            {
                assert_eq!(*low, -*high);
            }
            let per_unit = codebook.coordinate_mse * d as f64;
            assert!(
                (per_unit - unit_mse).abs() < 0.04 * unit_mse,
                "{bits} bits, d {d}: {per_unit}"
            );
        }
    }
}

#[test]
fn a_round_trip_loses_what_the_codebook_predicts() {
    for (precision, bound) in [
        (KvStorePrecision::Tq4, 0.012),
        (KvStorePrecision::Tq3, 0.042),
    ] {
        for d in [64_u64, 128, 256] {
            let mut error = 0.0_f64;
            let mut energy = 0.0_f64;
            for seed in 0..64 {
                let row = pseudo_random_row(4 * d as usize, seed * 31 + d);
                let back = kv_store_round_trip(&row, d, precision).unwrap();
                for (value, read) in row.iter().zip(&back) {
                    error += f64::from(value - read).powi(2);
                    energy += f64::from(*value).powi(2);
                }
            }
            let relative = error / energy;
            assert!(relative < bound, "{precision:?} d {d}: {relative}");
            assert!(relative > bound / 4.0, "{precision:?} d {d}: {relative}");
        }
    }
    let zero = vec![0.0_f32; 128];
    assert_eq!(
        kv_store_round_trip(&zero, 64, KvStorePrecision::Tq4).unwrap(),
        zero
    );
    let row = pseudo_random_row(96, 1);
    assert_eq!(kv_store_round_trip(&row, 96, KvStorePrecision::Tq4), None);
    assert_eq!(
        kv_store_round_trip(&row, 96, KvStorePrecision::F32),
        Some(row)
    );
}

/// A component every row shares is kept exactly by the offsets: rows
/// with a large common part read back far closer with them than
/// without, and a row equal to the offsets reads back exactly.
#[test]
fn offsets_keep_what_every_row_shares() {
    let d = 64usize;
    let width = 2 * d;
    let common: Vec<f32> = pseudo_random_row(width, 7)
        .iter()
        .map(|value| value * 20.0)
        .collect();
    let rows: Vec<f32> = (0..8)
        .flat_map(|row| {
            pseudo_random_row(width, 100 + row)
                .into_iter()
                .zip(&common)
                .map(|(value, common)| value + common)
                .collect::<Vec<_>>()
        })
        .collect();
    let error = |stored: &[f32]| -> f32 {
        stored
            .iter()
            .zip(&rows)
            .map(|(stored, row)| (stored - row).powi(2))
            .sum::<f32>()
            .sqrt()
    };
    let mut store = TurboQuantStoreState::new(KvStorePrecision::Tq4, d as u64, width as u64);
    let centered = store.write(&rows, 0).expect("whole rows");
    let plain = kv_store_round_trip(&rows, d as u64, KvStorePrecision::Tq4).unwrap();
    assert!(
        error(&centered) * 5.0 < error(&plain),
        "with offsets {} without {}",
        error(&centered),
        error(&plain)
    );
    // The offsets are the rows' mean, and stay for later writes.
    let mean = turboquant_store_offsets(&rows, width);
    assert_eq!(store.offsets, mean);
    assert_eq!(store.write(&mean, 8).unwrap(), mean);
    // A later write at row 0 starts a new history.
    store.write(&rows[..width], 0).unwrap();
    assert_eq!(store.offsets, rows[..width].to_vec());
    // f32 stores are unchanged.
    let mut f32_store = TurboQuantStoreState::new(KvStorePrecision::F32, d as u64, width as u64);
    assert_eq!(f32_store.write(&rows, 0).unwrap(), rows);
    assert_eq!(store.write(&rows[..d], 9), None, "not a whole row");
}

#[test]
fn codes_pack_and_unpack() {
    for bits in [3_u32, 4] {
        let indices: Vec<u8> = (0..128)
            .map(|i| ((i * 7 + 3) % (1 << bits)) as u8)
            .collect();
        let packed = turboquant_pack(&indices, bits);
        assert_eq!(packed.len(), 128 * bits as usize / 8);
        assert_eq!(turboquant_unpack(&packed, bits, indices.len()), indices);
    }
    assert_eq!(turboquant_pack(&[1, 2], 4), vec![0x21]);
}

#[test]
fn stored_sizes() {
    assert_eq!(KvStorePrecision::Tq4.head_row_bytes(128), 68);
    assert_eq!(KvStorePrecision::Tq3.head_row_bytes(128), 52);
    assert_eq!(KvStorePrecision::F32.head_row_bytes(128), 512);
    assert_eq!(
        KvStorePrecision::Tq4.stored_bytes(1024, 8 * 128, 128),
        Some(1024 * 8 * 68)
    );
    assert_eq!(KvStorePrecision::Tq4.stored_bytes(4, 100, 50), None);
    for name in ["f32", "tq4", "tq3"] {
        assert_eq!(KvStorePrecision::parse(name).unwrap().as_str(), name);
    }
    assert_eq!(KvStorePrecision::parse("q4"), None);
}

/// A store with an exact head (`keep-turboquant-kv-head-exact`) keeps the
/// rows below the head's end unchanged and stores the others as their
/// round trip with the offsets the write at row 0 set, across writes that
/// straddle the head's end.
#[test]
fn an_exact_head_keeps_its_rows_unchanged() {
    let (d, heads) = (16usize, 2usize);
    let width = d * heads;
    let rows: Vec<f32> = (0..6)
        .flat_map(|row| pseudo_random_row(width, row as u64 + 3))
        .collect();
    // The same writes without a head: the offsets come from the first.
    let mut codes_only = TurboQuantStoreState::new(KvStorePrecision::Tq4, d as u64, width as u64);
    let mut rounded = codes_only.write(&rows[..4 * width], 0).unwrap();
    rounded.extend(codes_only.write(&rows[4 * width..], 4).unwrap());
    let mut store =
        TurboQuantStoreState::new(KvStorePrecision::Tq4, d as u64, width as u64).with_exact_rows(3);
    // Rows 0..4, then 4..6: the head ends inside the first write.
    let mut stored = store.write(&rows[..4 * width], 0).unwrap();
    stored.extend(store.write(&rows[4 * width..], 4).unwrap());
    assert_eq!(store.offsets, codes_only.offsets);
    assert_eq!(&stored[..3 * width], &rows[..3 * width]);
    assert_eq!(&stored[3 * width..], &rounded[3 * width..]);
    assert_ne!(&stored[3 * width..], &rows[3 * width..]);
    // An f32 store has no separate head; a TurboQuant one adds its rows.
    assert_eq!(
        KvStorePrecision::F32.stored_bytes_with_head(8, 64, 16, 4),
        KvStorePrecision::F32.stored_bytes(8, 64, 16)
    );
    assert_eq!(
        KvStorePrecision::Tq4.stored_bytes_with_head(8, 64, 16, 4),
        KvStorePrecision::Tq4
            .stored_bytes(8, 64, 16)
            .map(|codes| codes + 4 * 64 * 4)
    );
    assert_eq!(
        KvStorePrecision::Tq4.stored_bytes_with_head(2, 64, 16, 4),
        KvStorePrecision::Tq4
            .stored_bytes(2, 64, 16)
            .map(|codes| codes + 2 * 64 * 4)
    );
}
