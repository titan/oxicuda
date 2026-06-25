//! On-disk index (de)serialisation utilities.
//!
//! [`serializer`] defines a compact little-endian flat-binary container (magic +
//! version header + typed sections) for the core ANN payloads, built without any
//! `zip` / `bincode` / `serde` dependency. This module additionally bridges the
//! raw section codecs to the in-memory [`crate::pq::codebook::PqCodebook`] type
//! so a trained codebook can round-trip through bytes directly.

pub mod serializer;

pub use serializer::{
    ByteReader, ByteWriter, FORMAT_VERSION, FlatVectorBlob, IvfPostingsBlob, MAGIC, PqCodebookBlob,
    SectionKind, deserialize_flat, deserialize_ivf_postings, deserialize_pq_codebook,
    serialize_flat, serialize_ivf_postings, serialize_pq_codebook,
};

use crate::error::AnnResult;
use crate::pq::codebook::PqCodebook;

/// Serialise a trained [`PqCodebook`] to the flat-binary section format.
#[must_use]
pub fn pq_codebook_to_bytes(cb: &PqCodebook) -> Vec<u8> {
    // Shape is internally consistent by construction, so the length check inside
    // `serialize_pq_codebook` cannot fail here; fall back to an empty buffer
    // rather than panicking if that invariant is ever broken.
    serialize_pq_codebook(cb.m, cb.ksub, cb.dsub, cb.centroids_raw()).unwrap_or_default()
}

/// Reconstruct a [`PqCodebook`] from bytes produced by [`pq_codebook_to_bytes`]
/// or [`serialize_pq_codebook`].
///
/// # Errors
/// Propagates any header / shape error from [`deserialize_pq_codebook`].
pub fn pq_codebook_from_bytes(bytes: &[u8]) -> AnnResult<PqCodebook> {
    let blob = deserialize_pq_codebook(bytes)?;
    let mut cb = PqCodebook::new(blob.m, blob.ksub, blob.dsub);
    for s in 0..blob.m {
        for c in 0..blob.ksub {
            let off = (s * blob.ksub + c) * blob.dsub;
            cb.centroid_mut(s, c)
                .copy_from_slice(&blob.centroids[off..off + blob.dsub]);
        }
    }
    Ok(cb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::pq::train::train_pq;

    #[test]
    fn pq_codebook_byte_roundtrip() {
        let mut rng = LcgRng::new(123);
        let n = 200;
        let dim = 8;
        let m = 2;
        let ksub = 16;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cb = train_pq(&data, n, dim, m, ksub, 15, &mut rng).expect("train_pq");

        let bytes = pq_codebook_to_bytes(&cb);
        let cb2 = pq_codebook_from_bytes(&bytes).expect("from_bytes");

        assert_eq!(cb2.m, cb.m);
        assert_eq!(cb2.ksub, cb.ksub);
        assert_eq!(cb2.dsub, cb.dsub);
        assert_eq!(cb2.centroids_raw(), cb.centroids_raw());
    }

    #[test]
    fn pq_codebook_from_bad_bytes_errors() {
        assert!(pq_codebook_from_bytes(&[0u8; 4]).is_err());
    }
}
