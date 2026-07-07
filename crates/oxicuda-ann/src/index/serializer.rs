//! Pure-Rust flat-binary index serialiser (no `zip` / `bincode` / `serde`).
//!
//! Provides a compact, self-describing, little-endian on-disk format for the
//! core ANN index payloads — flat vector stores, PQ codebooks and IVF posting
//! lists. The layout is intentionally simple and `mmap`-friendly:
//!
//! ```text
//! ┌────────────┬──────────┬──────────┬───────────────── … ─────────────┐
//! │ magic (8B) │ ver  (u32)│ kind (u32)│      kind-specific body         │
//! └────────────┴──────────┴──────────┴───────────────── … ─────────────┘
//! ```
//!
//! * `magic`   — the 8 ASCII bytes `OXANNIDX`.
//! * `version` — `u32` format version ([`FORMAT_VERSION`]).
//! * `kind`    — a [`SectionKind`] discriminant identifying the payload.
//!
//! All multi-byte scalars are encoded little-endian. Slices are length-prefixed
//! with a `u64` element count. No external crates are used: the reader/writer
//! are built from `from_le_bytes` / `to_le_bytes` over `Vec<u8>` and `&[u8]`.
//!
//! The format carries enough structure (shape fields up front) for a consumer to
//! locate every section without scanning, supporting future memory-mapped reads.

use crate::error::{AnnError, AnnResult};

/// 8-byte file magic identifying an OxiCUDA-ANN serialized index.
pub const MAGIC: [u8; 8] = *b"OXANNIDX";

/// On-disk format version. Bumped on any incompatible layout change.
pub const FORMAT_VERSION: u32 = 1;

/// Discriminant for the payload that follows the shared header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SectionKind {
    /// Flat row-major `f32` vector store: `dim`, `n`, then `n*dim` floats.
    FlatVectors = 1,
    /// PQ codebook: `m`, `ksub`, `dsub`, then `m*ksub*dsub` floats.
    PqCodebook = 2,
    /// IVF posting lists: `n_lists`, then for each list a `u64`-prefixed run of
    /// `u32` ids.
    IvfPostings = 3,
}

impl SectionKind {
    /// Decode a raw discriminant.
    fn from_u32(v: u32) -> AnnResult<Self> {
        match v {
            1 => Ok(SectionKind::FlatVectors),
            2 => Ok(SectionKind::PqCodebook),
            3 => Ok(SectionKind::IvfPostings),
            other => Err(AnnError::Internal {
                msg: format!("unknown section kind {other}"),
            }),
        }
    }
}

// ─── byte writer ───────────────────────────────────────────────────────────

/// Append-only little-endian byte buffer builder.
#[derive(Debug, Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    /// Create an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consume the writer and return the underlying bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Current byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// `true` when nothing has been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Write raw bytes verbatim.
    pub fn put_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    /// Write a `u32` little-endian.
    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Write a `u64` little-endian.
    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Write a `usize` as a `u64` (portable across 32/64-bit hosts).
    pub fn put_usize(&mut self, v: usize) {
        self.put_u64(v as u64);
    }

    /// Write an `f32` little-endian.
    pub fn put_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Write a length-prefixed (`u64` count) `f32` slice.
    pub fn put_f32_slice(&mut self, s: &[f32]) {
        self.put_u64(s.len() as u64);
        for &x in s {
            self.put_f32(x);
        }
    }

    /// Write a length-prefixed (`u64` count) `u32` slice.
    pub fn put_u32_slice(&mut self, s: &[u32]) {
        self.put_u64(s.len() as u64);
        for &x in s {
            self.put_u32(x);
        }
    }
}

// ─── byte reader ───────────────────────────────────────────────────────────

/// Cursor-based little-endian byte reader with bounds checking.
#[derive(Debug)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Wrap a byte slice.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Current read offset.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes remaining.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn truncated() -> AnnError {
        AnnError::Internal {
            msg: "unexpected end of buffer while deserialising".to_string(),
        }
    }

    /// Read `n` raw bytes.
    pub fn get_bytes(&mut self, n: usize) -> AnnResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(Self::truncated)?;
        if end > self.buf.len() {
            return Err(Self::truncated());
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Read a `u32` little-endian.
    pub fn get_u32(&mut self) -> AnnResult<u32> {
        let b = self.get_bytes(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| Self::truncated())?;
        Ok(u32::from_le_bytes(arr))
    }

    /// Read a `u64` little-endian.
    pub fn get_u64(&mut self) -> AnnResult<u64> {
        let b = self.get_bytes(8)?;
        let arr: [u8; 8] = b.try_into().map_err(|_| Self::truncated())?;
        Ok(u64::from_le_bytes(arr))
    }

    /// Read a `u64`-encoded length as `usize`.
    pub fn get_usize(&mut self) -> AnnResult<usize> {
        Ok(self.get_u64()? as usize)
    }

    /// Read an `f32` little-endian.
    pub fn get_f32(&mut self) -> AnnResult<f32> {
        let b = self.get_bytes(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| Self::truncated())?;
        Ok(f32::from_le_bytes(arr))
    }

    /// Read a length-prefixed `f32` slice into a `Vec`.
    pub fn get_f32_slice(&mut self) -> AnnResult<Vec<f32>> {
        let n = self.get_u64()? as usize;
        // Guard against a corrupt length claiming more than the buffer holds.
        if n.saturating_mul(4) > self.remaining() {
            return Err(Self::truncated());
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.get_f32()?);
        }
        Ok(out)
    }

    /// Read a length-prefixed `u32` slice into a `Vec`.
    pub fn get_u32_slice(&mut self) -> AnnResult<Vec<u32>> {
        let n = self.get_u64()? as usize;
        if n.saturating_mul(4) > self.remaining() {
            return Err(Self::truncated());
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.get_u32()?);
        }
        Ok(out)
    }
}

// ─── header helpers ────────────────────────────────────────────────────────

fn write_header(w: &mut ByteWriter, kind: SectionKind) {
    w.put_bytes(&MAGIC);
    w.put_u32(FORMAT_VERSION);
    w.put_u32(kind as u32);
}

fn read_header(r: &mut ByteReader<'_>) -> AnnResult<SectionKind> {
    let magic = r.get_bytes(8)?;
    if magic != MAGIC {
        return Err(AnnError::Internal {
            msg: "bad magic: not an OxiCUDA-ANN index".to_string(),
        });
    }
    let ver = r.get_u32()?;
    if ver != FORMAT_VERSION {
        return Err(AnnError::Internal {
            msg: format!("unsupported format version {ver} (expected {FORMAT_VERSION})"),
        });
    }
    SectionKind::from_u32(r.get_u32()?)
}

// ─── flat vector store ─────────────────────────────────────────────────────

/// A decoded flat vector payload: `n` row-major vectors of dimension `dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatVectorBlob {
    /// Vector dimensionality.
    pub dim: usize,
    /// Number of stored vectors.
    pub n: usize,
    /// Row-major `[n × dim]` storage.
    pub data: Vec<f32>,
}

/// Serialise a flat `[n × dim]` vector store.
///
/// # Errors
/// [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
pub fn serialize_flat(dim: usize, n: usize, data: &[f32]) -> AnnResult<Vec<u8>> {
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }
    let mut w = ByteWriter::new();
    write_header(&mut w, SectionKind::FlatVectors);
    w.put_usize(dim);
    w.put_usize(n);
    w.put_f32_slice(data);
    Ok(w.into_bytes())
}

/// Deserialise a flat vector store produced by [`serialize_flat`].
///
/// # Errors
/// [`AnnError::Internal`] on bad magic / version / kind / truncation, or
/// [`AnnError::DimensionMismatch`] if the embedded shape and payload disagree.
pub fn deserialize_flat(bytes: &[u8]) -> AnnResult<FlatVectorBlob> {
    let mut r = ByteReader::new(bytes);
    let kind = read_header(&mut r)?;
    if kind != SectionKind::FlatVectors {
        return Err(AnnError::Internal {
            msg: format!("expected FlatVectors, got {kind:?}"),
        });
    }
    let dim = r.get_usize()?;
    let n = r.get_usize()?;
    let data = r.get_f32_slice()?;
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }
    Ok(FlatVectorBlob { dim, n, data })
}

// ─── PQ codebook ───────────────────────────────────────────────────────────

/// A decoded PQ codebook payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PqCodebookBlob {
    /// Number of subspaces.
    pub m: usize,
    /// Codewords per subspace.
    pub ksub: usize,
    /// Sub-vector dimension.
    pub dsub: usize,
    /// Flat `[m × ksub × dsub]` centroid storage.
    pub centroids: Vec<f32>,
}

/// Serialise a PQ codebook's raw centroid storage and shape.
///
/// # Errors
/// [`AnnError::DimensionMismatch`] if `centroids.len() != m * ksub * dsub`.
pub fn serialize_pq_codebook(
    m: usize,
    ksub: usize,
    dsub: usize,
    centroids: &[f32],
) -> AnnResult<Vec<u8>> {
    if centroids.len() != m * ksub * dsub {
        return Err(AnnError::DimensionMismatch {
            expected: m * ksub * dsub,
            got: centroids.len(),
        });
    }
    let mut w = ByteWriter::new();
    write_header(&mut w, SectionKind::PqCodebook);
    w.put_usize(m);
    w.put_usize(ksub);
    w.put_usize(dsub);
    w.put_f32_slice(centroids);
    Ok(w.into_bytes())
}

/// Deserialise a PQ codebook produced by [`serialize_pq_codebook`].
///
/// # Errors
/// As [`deserialize_flat`], plus a shape/payload mismatch check.
pub fn deserialize_pq_codebook(bytes: &[u8]) -> AnnResult<PqCodebookBlob> {
    let mut r = ByteReader::new(bytes);
    let kind = read_header(&mut r)?;
    if kind != SectionKind::PqCodebook {
        return Err(AnnError::Internal {
            msg: format!("expected PqCodebook, got {kind:?}"),
        });
    }
    let m = r.get_usize()?;
    let ksub = r.get_usize()?;
    let dsub = r.get_usize()?;
    let centroids = r.get_f32_slice()?;
    if centroids.len() != m * ksub * dsub {
        return Err(AnnError::DimensionMismatch {
            expected: m * ksub * dsub,
            got: centroids.len(),
        });
    }
    Ok(PqCodebookBlob {
        m,
        ksub,
        dsub,
        centroids,
    })
}

// ─── IVF posting lists ─────────────────────────────────────────────────────

/// A decoded IVF posting-list payload: `n_lists` runs of `u32` ids.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfPostingsBlob {
    /// Per-list id runs (`posting_lists[c]` is the ids of cluster `c`).
    pub posting_lists: Vec<Vec<u32>>,
}

/// Serialise IVF posting lists.
pub fn serialize_ivf_postings(posting_lists: &[Vec<u32>]) -> Vec<u8> {
    let mut w = ByteWriter::new();
    write_header(&mut w, SectionKind::IvfPostings);
    w.put_usize(posting_lists.len());
    for list in posting_lists {
        w.put_u32_slice(list);
    }
    w.into_bytes()
}

/// Deserialise IVF posting lists produced by [`serialize_ivf_postings`].
///
/// # Errors
/// As [`deserialize_flat`] for header / truncation problems.
pub fn deserialize_ivf_postings(bytes: &[u8]) -> AnnResult<IvfPostingsBlob> {
    let mut r = ByteReader::new(bytes);
    let kind = read_header(&mut r)?;
    if kind != SectionKind::IvfPostings {
        return Err(AnnError::Internal {
            msg: format!("expected IvfPostings, got {kind:?}"),
        });
    }
    let n_lists = r.get_usize()?;
    // Guard against a corrupt/malicious length claiming more lists than the
    // buffer could possibly hold: each posting list costs at least its own
    // 8-byte u64 length prefix (an empty list still occupies those 8 bytes),
    // so this bound is exact and rejects no valid file.
    if n_lists.saturating_mul(8) > r.remaining() {
        return Err(AnnError::Internal {
            msg: "unexpected end of buffer while deserialising".to_string(),
        });
    }
    let mut posting_lists = Vec::with_capacity(n_lists);
    for _ in 0..n_lists {
        posting_lists.push(r.get_u32_slice()?);
    }
    Ok(IvfPostingsBlob { posting_lists })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect()
    }

    // ── byte primitives ────────────────────────────────────────────────────

    #[test]
    fn writer_reader_roundtrip_scalars() {
        let mut w = ByteWriter::new();
        w.put_u32(0xDEAD_BEEF);
        w.put_u64(0x0123_4567_89AB_CDEF);
        w.put_f32(std::f32::consts::PI);
        w.put_usize(123_456);
        let bytes = w.into_bytes();

        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_u32().expect("u32"), 0xDEAD_BEEF);
        assert_eq!(r.get_u64().expect("u64"), 0x0123_4567_89AB_CDEF);
        assert!((r.get_f32().expect("f32") - std::f32::consts::PI).abs() < 1e-7);
        assert_eq!(r.get_usize().expect("usize"), 123_456);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn writer_reader_roundtrip_slices() {
        let f = vec![1.0_f32, -2.5, 3.25, 0.0, 7.5];
        let u = vec![9u32, 8, 7, 0, u32::MAX];
        let mut w = ByteWriter::new();
        w.put_f32_slice(&f);
        w.put_u32_slice(&u);
        let bytes = w.into_bytes();
        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_f32_slice().expect("f32 slice"), f);
        assert_eq!(r.get_u32_slice().expect("u32 slice"), u);
    }

    #[test]
    fn reader_detects_truncation() {
        let mut w = ByteWriter::new();
        w.put_u64(100); // claims 100 f32 elems but provides none
        let bytes = w.into_bytes();
        let mut r = ByteReader::new(&bytes);
        assert!(r.get_f32_slice().is_err());
    }

    // ── flat vectors ───────────────────────────────────────────────────────

    #[test]
    fn flat_roundtrip() {
        let dim = 7;
        let n = 13;
        let data = rand_f32(n * dim, 1);
        let bytes = serialize_flat(dim, n, &data).expect("serialize");
        // header (16) + dim(8) + n(8) + len(8) + payload
        assert_eq!(bytes.len(), 16 + 8 + 8 + 8 + n * dim * 4);
        let blob = deserialize_flat(&bytes).expect("deserialize");
        assert_eq!(blob.dim, dim);
        assert_eq!(blob.n, n);
        assert_eq!(blob.data, data);
    }

    #[test]
    fn flat_bad_shape_errors() {
        let res = serialize_flat(4, 3, &[1.0, 2.0]); // 2 != 12
        assert!(matches!(res, Err(AnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn flat_bad_magic_errors() {
        let mut bytes = serialize_flat(2, 1, &[1.0, 2.0]).expect("serialize");
        bytes[0] ^= 0xFF;
        assert!(deserialize_flat(&bytes).is_err());
    }

    #[test]
    fn flat_wrong_kind_errors() {
        let cb = serialize_pq_codebook(1, 1, 2, &[1.0, 2.0]).expect("serialize cb");
        // Reading a codebook blob as flat must fail on the kind tag.
        assert!(deserialize_flat(&cb).is_err());
    }

    #[test]
    fn flat_bad_version_errors() {
        let mut bytes = serialize_flat(2, 1, &[1.0, 2.0]).expect("serialize");
        // version sits right after the 8-byte magic.
        bytes[8] = 0xFE;
        assert!(deserialize_flat(&bytes).is_err());
    }

    // ── PQ codebook ────────────────────────────────────────────────────────

    #[test]
    fn pq_codebook_roundtrip() {
        let (m, ksub, dsub) = (4, 16, 3);
        let cents = rand_f32(m * ksub * dsub, 2);
        let bytes = serialize_pq_codebook(m, ksub, dsub, &cents).expect("serialize");
        let blob = deserialize_pq_codebook(&bytes).expect("deserialize");
        assert_eq!(blob.m, m);
        assert_eq!(blob.ksub, ksub);
        assert_eq!(blob.dsub, dsub);
        assert_eq!(blob.centroids, cents);
    }

    #[test]
    fn pq_codebook_bad_shape_errors() {
        let res = serialize_pq_codebook(2, 2, 2, &[1.0; 7]); // 7 != 8
        assert!(matches!(res, Err(AnnError::DimensionMismatch { .. })));
    }

    // ── IVF postings ───────────────────────────────────────────────────────

    #[test]
    fn ivf_postings_roundtrip() {
        let lists: Vec<Vec<u32>> = vec![vec![0, 5, 9], vec![], vec![3], vec![1, 2, 4, 8, 100]];
        let bytes = serialize_ivf_postings(&lists);
        let blob = deserialize_ivf_postings(&bytes).expect("deserialize");
        assert_eq!(blob.posting_lists, lists);
    }

    #[test]
    fn ivf_postings_empty_index() {
        let lists: Vec<Vec<u32>> = Vec::new();
        let bytes = serialize_ivf_postings(&lists);
        let blob = deserialize_ivf_postings(&bytes).expect("deserialize");
        assert!(blob.posting_lists.is_empty());
    }

    #[test]
    fn ivf_postings_wrong_kind_errors() {
        let flat = serialize_flat(2, 1, &[1.0, 2.0]).expect("serialize flat");
        assert!(deserialize_ivf_postings(&flat).is_err());
    }

    #[test]
    fn ivf_postings_corrupt_huge_n_lists_errors_without_oom() {
        // Craft a header followed by a wildly oversized `n_lists` claim with
        // no backing data. A vulnerable implementation would forward this
        // straight to `Vec::with_capacity`, aborting the process on capacity
        // overflow / OOM instead of returning a decode error.
        let lists: Vec<Vec<u32>> = vec![vec![1, 2, 3]];
        let mut bytes = serialize_ivf_postings(&lists);
        // The `n_lists` u64 sits immediately after the 8-byte magic + 4-byte
        // version + 4-byte kind header.
        let n_lists_off = 8 + 4 + 4;
        bytes[n_lists_off..n_lists_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(deserialize_ivf_postings(&bytes).is_err());
    }

    // ── header byte-exactness ──────────────────────────────────────────────

    #[test]
    fn header_layout_is_exact() {
        let bytes = serialize_flat(1, 1, &[42.0]).expect("serialize");
        assert_eq!(&bytes[0..8], &MAGIC);
        assert_eq!(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            FORMAT_VERSION
        );
        assert_eq!(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            SectionKind::FlatVectors as u32
        );
    }
}
