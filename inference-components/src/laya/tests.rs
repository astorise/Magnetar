use super::*;
use std::io::{Seek, SeekFrom, Write};

#[test]
fn graph_payload_transposes_authenticated_half_precision_and_rejects_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    let header = br#"{"encoder.matrix":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0, 0x3c, 0, 0x40, 0, 0x42, 0, 0x44, 0, 0x45, 0, 0x46]);
    std::fs::write(&path, &bytes).unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(path.clone()),
        directory.path().to_path_buf(),
    );
    let raw = inspect_authenticated_safetensors_file(
        &source,
        "weights.safetensors",
        &ModelDigest::sha256(&bytes),
    )
    .unwrap();
    let source = raw.tensors.into_iter().next().unwrap();
    let spec = WeightSpec {
        checkpoint_name: "encoder.matrix".into(),
        graph_name: "matrix".into(),
        checkpoint_shape: vec![2, 3],
        graph_shape: vec![3, 2],
        transpose: true,
    };
    let payload = GraphPayloads {
        source: raw.payloads,
        tensors: BTreeMap::from([(
            "matrix".into(),
            GraphTensor {
                source,
                spec,
                length: 24,
            },
        )]),
    };
    let range = ProductionPayloadRange {
        identity: "matrix".into(),
        offset: 0,
        length: 24,
        digest: None,
    };
    let expected: Vec<u8> = [1_f32, 4., 2., 5., 3., 6.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(payload.read_payload(&range).unwrap(), expected);
    let mut wrong = range.clone();
    wrong.offset = 1;
    assert!(matches!(
        payload.read_payload(&wrong),
        Err(ProductionIngestionError::PayloadOutOfBounds { .. })
    ));
    wrong = range.clone();
    wrong.digest = Some(ModelDigest::sha256(b"wrong"));
    assert!(matches!(
        payload.read_payload(&wrong),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(8 + header.len() as u64)).unwrap();
    file.write_all(&[0x42]).unwrap();
    drop(file);
    assert!(matches!(
        payload.read_payload(&range),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));
}

#[test]
fn pinned_configs_accept_crlf_but_reject_content_changes() {
    let bytes =
        include_bytes!("../../../components/modernbert/tests/fixtures/laya-encoder.config.json");
    let canonical = pinned_json(bytes, ENCODER_CONFIG_DIGEST).unwrap();
    let crlf = String::from_utf8(canonical.clone())
        .unwrap()
        .replace('\n', "\r\n");
    assert_eq!(
        pinned_json(crlf.as_bytes(), ENCODER_CONFIG_DIGEST).unwrap(),
        canonical
    );
    let mut altered = canonical;
    altered.push(b' ');
    assert!(pinned_json(&altered, ENCODER_CONFIG_DIGEST).is_err());
}
