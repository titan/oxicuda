//! # GGUF Container File Format
//!
//! Pure-Rust reader and writer for the **GGUF** model container used by
//! `llama.cpp` / `ggml` to serialize quantized weights. This module implements
//! the *container* layer — the magic header, the typed metadata key-value
//! section, the tensor directory, and the aligned tensor-data region — as
//! opposed to the per-block quant codecs (`Q8_0`/`Q4_0`/`Q4_1`/`Q4_K`), which
//! live in [`crate::scheme::ggml`] and produce the tensor payloads stored here.
//!
//! ## On-disk layout (GGUF v3, little-endian)
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ HEADER                                                         │
//! │   u32   magic            = 0x4655_4747  ("GGUF" as LE bytes)   │
//! │   u32   version          = 3                                   │
//! │   u64   tensor_count                                           │
//! │   u64   metadata_kv_count                                      │
//! ├──────────────────────────────────────────────────────────────┤
//! │ METADATA  (metadata_kv_count entries)                          │
//! │   string  key            (u64 len + UTF-8 bytes)               │
//! │   u32     value_type      (see GgufValueType)                  │
//! │   <typed value>                                                │
//! ├──────────────────────────────────────────────────────────────┤
//! │ TENSOR DIRECTORY  (tensor_count entries)                       │
//! │   string  name           (u64 len + UTF-8 bytes)              │
//! │   u32     n_dims                                               │
//! │   u64×n   dims                                                 │
//! │   u32     ggml_type                                            │
//! │   u64     offset          (relative to data_start, aligned)    │
//! ├──────────────────────────────────────────────────────────────┤
//! │ <padding to `general.alignment` boundary>                      │
//! ├──────────────────────────────────────────────────────────────┤
//! │ TENSOR DATA                                                    │
//! │   for each tensor: bytes at data_start + offset                │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### Metadata value types
//!
//! | Tag | Type      | Encoding                                     |
//! |-----|-----------|---------------------------------------------|
//! | 0   | `u8`      | 1 byte                                      |
//! | 1   | `i8`      | 1 byte                                      |
//! | 2   | `u16`     | 2 bytes LE                                  |
//! | 3   | `i16`     | 2 bytes LE                                  |
//! | 4   | `u32`     | 4 bytes LE                                  |
//! | 5   | `i32`     | 4 bytes LE                                  |
//! | 6   | `f32`     | 4 bytes LE (IEEE-754)                        |
//! | 7   | `bool`    | 1 byte (`0`/`1`)                            |
//! | 8   | `string`  | u64 len + UTF-8 bytes                        |
//! | 9   | `array`   | u32 elem-type tag + u64 count + elements     |
//! | 10  | `u64`     | 8 bytes LE                                  |
//! | 11  | `i64`     | 8 bytes LE                                  |
//! | 12  | `f64`     | 8 bytes LE (IEEE-754)                        |
//!
//! Arrays carry a single element-type tag and may not directly contain another
//! array (matching the canonical `ggml` writer); this is enforced on both read
//! and write.
//!
//! The default alignment is **32 bytes**, overridable through the reserved
//! `general.alignment` metadata key (a `u32`).

use crate::error::{QuantError, QuantResult};

/// The four magic bytes `GGUF` interpreted as a little-endian `u32`.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// The GGUF format version emitted by [`write_gguf`].
pub const GGUF_VERSION: u32 = 3;

/// The default tensor-data alignment in bytes when `general.alignment` is unset.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// The reserved metadata key that overrides the tensor-data alignment.
pub const ALIGNMENT_KEY: &str = "general.alignment";

// ─── GGUF metadata value-type tags ──────────────────────────────────────────────

/// Discriminant tags for the typed metadata values, matching the `ggml`
/// `gguf_type` enum exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    /// 8-bit unsigned integer.
    U8 = 0,
    /// 8-bit signed integer.
    I8 = 1,
    /// 16-bit unsigned integer.
    U16 = 2,
    /// 16-bit signed integer.
    I16 = 3,
    /// 32-bit unsigned integer.
    U32 = 4,
    /// 32-bit signed integer.
    I32 = 5,
    /// 32-bit IEEE-754 float.
    F32 = 6,
    /// Boolean (one byte).
    Bool = 7,
    /// UTF-8 string (`u64` length prefix).
    String = 8,
    /// Homogeneous array (element tag + `u64` count).
    Array = 9,
    /// 64-bit unsigned integer.
    U64 = 10,
    /// 64-bit signed integer.
    I64 = 11,
    /// 64-bit IEEE-754 float.
    F64 = 12,
}

impl GgufValueType {
    /// Decode a raw `u32` tag into a [`GgufValueType`].
    ///
    /// # Errors
    ///
    /// [`QuantError::InvalidConfig`] if `tag` is not a known GGUF value type.
    pub fn from_tag(tag: u32) -> QuantResult<Self> {
        Ok(match tag {
            0 => GgufValueType::U8,
            1 => GgufValueType::I8,
            2 => GgufValueType::U16,
            3 => GgufValueType::I16,
            4 => GgufValueType::U32,
            5 => GgufValueType::I32,
            6 => GgufValueType::F32,
            7 => GgufValueType::Bool,
            8 => GgufValueType::String,
            9 => GgufValueType::Array,
            10 => GgufValueType::U64,
            11 => GgufValueType::I64,
            12 => GgufValueType::F64,
            other => {
                return Err(QuantError::InvalidConfig(format!(
                    "unknown GGUF value type tag {other}"
                )));
            }
        })
    }

    /// The raw `u32` tag for this value type.
    #[must_use]
    pub fn tag(self) -> u32 {
        self as u32
    }
}

/// An array of homogeneous metadata values.
///
/// GGUF arrays carry a single element-type tag and a count; the canonical
/// writer never nests arrays, so [`Array`](GgufMetadataValue::Array) is not a
/// permitted element type here.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufArray {
    /// Array of `u8`.
    U8(Vec<u8>),
    /// Array of `i8`.
    I8(Vec<i8>),
    /// Array of `u16`.
    U16(Vec<u16>),
    /// Array of `i16`.
    I16(Vec<i16>),
    /// Array of `u32`.
    U32(Vec<u32>),
    /// Array of `i32`.
    I32(Vec<i32>),
    /// Array of `f32`.
    F32(Vec<f32>),
    /// Array of `bool`.
    Bool(Vec<bool>),
    /// Array of UTF-8 strings.
    String(Vec<String>),
    /// Array of `u64`.
    U64(Vec<u64>),
    /// Array of `i64`.
    I64(Vec<i64>),
    /// Array of `f64`.
    F64(Vec<f64>),
}

impl GgufArray {
    /// The element-type tag stored before the array count.
    #[must_use]
    pub fn element_type(&self) -> GgufValueType {
        match self {
            GgufArray::U8(_) => GgufValueType::U8,
            GgufArray::I8(_) => GgufValueType::I8,
            GgufArray::U16(_) => GgufValueType::U16,
            GgufArray::I16(_) => GgufValueType::I16,
            GgufArray::U32(_) => GgufValueType::U32,
            GgufArray::I32(_) => GgufValueType::I32,
            GgufArray::F32(_) => GgufValueType::F32,
            GgufArray::Bool(_) => GgufValueType::Bool,
            GgufArray::String(_) => GgufValueType::String,
            GgufArray::U64(_) => GgufValueType::U64,
            GgufArray::I64(_) => GgufValueType::I64,
            GgufArray::F64(_) => GgufValueType::F64,
        }
    }

    /// The number of elements in the array.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            GgufArray::U8(v) => v.len(),
            GgufArray::I8(v) => v.len(),
            GgufArray::U16(v) => v.len(),
            GgufArray::I16(v) => v.len(),
            GgufArray::U32(v) => v.len(),
            GgufArray::I32(v) => v.len(),
            GgufArray::F32(v) => v.len(),
            GgufArray::Bool(v) => v.len(),
            GgufArray::String(v) => v.len(),
            GgufArray::U64(v) => v.len(),
            GgufArray::I64(v) => v.len(),
            GgufArray::F64(v) => v.len(),
        }
    }

    /// Whether the array has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    /// 8-bit unsigned integer.
    U8(u8),
    /// 8-bit signed integer.
    I8(i8),
    /// 16-bit unsigned integer.
    U16(u16),
    /// 16-bit signed integer.
    I16(i16),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit IEEE-754 float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array.
    Array(GgufArray),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// 64-bit IEEE-754 float.
    F64(f64),
}

impl GgufMetadataValue {
    /// The value-type tag written before the value bytes.
    #[must_use]
    pub fn value_type(&self) -> GgufValueType {
        match self {
            GgufMetadataValue::U8(_) => GgufValueType::U8,
            GgufMetadataValue::I8(_) => GgufValueType::I8,
            GgufMetadataValue::U16(_) => GgufValueType::U16,
            GgufMetadataValue::I16(_) => GgufValueType::I16,
            GgufMetadataValue::U32(_) => GgufValueType::U32,
            GgufMetadataValue::I32(_) => GgufValueType::I32,
            GgufMetadataValue::F32(_) => GgufValueType::F32,
            GgufMetadataValue::Bool(_) => GgufValueType::Bool,
            GgufMetadataValue::String(_) => GgufValueType::String,
            GgufMetadataValue::Array(_) => GgufValueType::Array,
            GgufMetadataValue::U64(_) => GgufValueType::U64,
            GgufMetadataValue::I64(_) => GgufValueType::I64,
            GgufMetadataValue::F64(_) => GgufValueType::F64,
        }
    }

    /// Interpret this value as a `u32` if it is an unsigned-integer scalar.
    ///
    /// Used to read the `general.alignment` override, which `ggml` writes as a
    /// `u32`; smaller unsigned widths are accepted as a convenience.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            GgufMetadataValue::U8(v) => Some(u32::from(v)),
            GgufMetadataValue::U16(v) => Some(u32::from(v)),
            GgufMetadataValue::U32(v) => Some(v),
            _ => None,
        }
    }
}

/// A single metadata key-value entry.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadataKv {
    /// The metadata key (a dotted-namespace UTF-8 string).
    pub key: String,
    /// The typed value.
    pub value: GgufMetadataValue,
}

/// The fixed-size GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    /// Magic number — must equal [`GGUF_MAGIC`].
    pub magic: u32,
    /// Format version (this writer emits [`GGUF_VERSION`]).
    pub version: u32,
    /// Number of tensors in the directory.
    pub tensor_count: u64,
    /// Number of metadata key-value entries.
    pub metadata_kv_count: u64,
}

/// One entry of the tensor directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// The tensor name (UTF-8).
    pub name: String,
    /// The per-dimension extents (row-major, innermost first as in `ggml`).
    pub dims: Vec<u64>,
    /// The `ggml` type id of the stored tensor data (e.g. `Q4_0` block type).
    pub ggml_type: u32,
    /// Byte offset of this tensor's data relative to the start of the
    /// tensor-data region; always a multiple of the file alignment.
    pub offset: u64,
}

/// A fully-parsed (or to-be-written) GGUF container.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    /// The header (its counts are recomputed by [`write_gguf`]).
    pub header: GgufHeader,
    /// Ordered metadata key-value entries.
    pub metadata: Vec<GgufMetadataKv>,
    /// Ordered tensor-directory entries.
    pub tensors: Vec<GgufTensorInfo>,
    /// The aligned tensor-data region, taken verbatim from the file. Tensor `i`
    /// occupies `tensor_data[tensors[i].offset ..]`.
    pub tensor_data: Vec<u8>,
    /// The resolved alignment in bytes (from `general.alignment` or the default).
    pub alignment: u64,
}

impl GgufFile {
    /// Construct a container from metadata, tensor infos and a packed data blob.
    ///
    /// The header counts are derived from the supplied vectors, so callers need
    /// not keep them in sync manually.
    #[must_use]
    pub fn new(
        metadata: Vec<GgufMetadataKv>,
        tensors: Vec<GgufTensorInfo>,
        tensor_data: Vec<u8>,
        alignment: u64,
    ) -> Self {
        let header = GgufHeader {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION,
            tensor_count: tensors.len() as u64,
            metadata_kv_count: metadata.len() as u64,
        };
        Self {
            header,
            metadata,
            tensors,
            tensor_data,
            alignment,
        }
    }

    /// Look up a metadata value by key, returning the first match.
    #[must_use]
    pub fn metadata_get(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }
}

// ─── Little-endian cursor over the input bytes ─────────────────────────────────

/// A bounds-checked forward cursor over a byte slice with little-endian readers.
///
/// Every accessor verifies the remaining length before reading, so a truncated
/// or malformed file yields a [`QuantError`] rather than a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> QuantResult<&'a [u8]> {
        if self.remaining() < n {
            return Err(QuantError::InvalidConfig(format!(
                "GGUF truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.remaining()
            )));
        }
        let start = self.pos;
        self.pos += n;
        Ok(&self.bytes[start..self.pos])
    }

    fn read_u8(&mut self) -> QuantResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_i8(&mut self) -> QuantResult<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> QuantResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> QuantResult<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> QuantResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> QuantResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> QuantResult<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_u64(&mut self) -> QuantResult<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> QuantResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f64(&mut self) -> QuantResult<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> QuantResult<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(QuantError::InvalidConfig(format!(
                "GGUF bool must be 0 or 1, got {other}"
            ))),
        }
    }

    /// Read a `u64`-prefixed UTF-8 string.
    fn read_string(&mut self) -> QuantResult<String> {
        let len = self.read_u64()?;
        // Guard against absurd lengths that would exceed the input.
        if len > self.remaining() as u64 {
            return Err(QuantError::InvalidConfig(format!(
                "GGUF string length {len} exceeds remaining {} bytes",
                self.remaining()
            )));
        }
        let raw = self.take(len as usize)?;
        String::from_utf8(raw.to_vec())
            .map_err(|e| QuantError::InvalidConfig(format!("GGUF string not valid UTF-8: {e}")))
    }
}

// ─── Little-endian byte writer ─────────────────────────────────────────────────

/// A growable little-endian byte writer used to serialize a container.
struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { out: Vec::new() }
    }

    fn write_u8(&mut self, v: u8) {
        self.out.push(v);
    }

    fn write_i8(&mut self, v: i8) {
        self.out.push(v as u8);
    }

    fn write_u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i16(&mut self, v: i16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i32(&mut self, v: i32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i64(&mut self, v: i64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f64(&mut self, v: f64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bool(&mut self, v: bool) {
        self.out.push(u8::from(v));
    }

    fn write_string(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.out.extend_from_slice(s.as_bytes());
    }

    /// Pad with zero bytes until the length is a multiple of `alignment`.
    fn pad_to(&mut self, alignment: u64) {
        if alignment == 0 {
            return;
        }
        let rem = (self.out.len() as u64) % alignment;
        if rem != 0 {
            let pad = (alignment - rem) as usize;
            self.out.resize(self.out.len() + pad, 0);
        }
    }
}

/// Round `value` up to the next multiple of `alignment` (no-op if already aligned
/// or if `alignment` is zero).
#[must_use]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

// ─── Value (de)serialization ───────────────────────────────────────────────────

/// Read a single scalar/array value of the given type (the tag has already been
/// consumed by the caller).
fn read_value(reader: &mut Reader<'_>, ty: GgufValueType) -> QuantResult<GgufMetadataValue> {
    Ok(match ty {
        GgufValueType::U8 => GgufMetadataValue::U8(reader.read_u8()?),
        GgufValueType::I8 => GgufMetadataValue::I8(reader.read_i8()?),
        GgufValueType::U16 => GgufMetadataValue::U16(reader.read_u16()?),
        GgufValueType::I16 => GgufMetadataValue::I16(reader.read_i16()?),
        GgufValueType::U32 => GgufMetadataValue::U32(reader.read_u32()?),
        GgufValueType::I32 => GgufMetadataValue::I32(reader.read_i32()?),
        GgufValueType::F32 => GgufMetadataValue::F32(reader.read_f32()?),
        GgufValueType::Bool => GgufMetadataValue::Bool(reader.read_bool()?),
        GgufValueType::String => GgufMetadataValue::String(reader.read_string()?),
        GgufValueType::U64 => GgufMetadataValue::U64(reader.read_u64()?),
        GgufValueType::I64 => GgufMetadataValue::I64(reader.read_i64()?),
        GgufValueType::F64 => GgufMetadataValue::F64(reader.read_f64()?),
        GgufValueType::Array => GgufMetadataValue::Array(read_array(reader)?),
    })
}

/// Read an array value (the outer `Array` tag has already been consumed).
fn read_array(reader: &mut Reader<'_>) -> QuantResult<GgufArray> {
    let elem_tag = reader.read_u32()?;
    let elem_ty = GgufValueType::from_tag(elem_tag)?;
    let count = reader.read_u64()? as usize;

    macro_rules! collect {
        ($read:ident) => {{
            let mut v = Vec::with_capacity(count.min(reader.remaining()));
            for _ in 0..count {
                v.push(reader.$read()?);
            }
            v
        }};
    }

    Ok(match elem_ty {
        GgufValueType::U8 => GgufArray::U8(collect!(read_u8)),
        GgufValueType::I8 => GgufArray::I8(collect!(read_i8)),
        GgufValueType::U16 => GgufArray::U16(collect!(read_u16)),
        GgufValueType::I16 => GgufArray::I16(collect!(read_i16)),
        GgufValueType::U32 => GgufArray::U32(collect!(read_u32)),
        GgufValueType::I32 => GgufArray::I32(collect!(read_i32)),
        GgufValueType::F32 => GgufArray::F32(collect!(read_f32)),
        GgufValueType::Bool => GgufArray::Bool(collect!(read_bool)),
        GgufValueType::String => GgufArray::String(collect!(read_string)),
        GgufValueType::U64 => GgufArray::U64(collect!(read_u64)),
        GgufValueType::I64 => GgufArray::I64(collect!(read_i64)),
        GgufValueType::F64 => GgufArray::F64(collect!(read_f64)),
        GgufValueType::Array => {
            return Err(QuantError::InvalidConfig(
                "GGUF arrays may not directly contain nested arrays".to_string(),
            ));
        }
    })
}

/// Write a single value's bytes (the caller has already written its type tag).
fn write_value(writer: &mut Writer, value: &GgufMetadataValue) {
    match value {
        GgufMetadataValue::U8(v) => writer.write_u8(*v),
        GgufMetadataValue::I8(v) => writer.write_i8(*v),
        GgufMetadataValue::U16(v) => writer.write_u16(*v),
        GgufMetadataValue::I16(v) => writer.write_i16(*v),
        GgufMetadataValue::U32(v) => writer.write_u32(*v),
        GgufMetadataValue::I32(v) => writer.write_i32(*v),
        GgufMetadataValue::F32(v) => writer.write_f32(*v),
        GgufMetadataValue::Bool(v) => writer.write_bool(*v),
        GgufMetadataValue::String(v) => writer.write_string(v),
        GgufMetadataValue::U64(v) => writer.write_u64(*v),
        GgufMetadataValue::I64(v) => writer.write_i64(*v),
        GgufMetadataValue::F64(v) => writer.write_f64(*v),
        GgufMetadataValue::Array(arr) => write_array(writer, arr),
    }
}

/// Write an array's element-type tag, count, and elements.
fn write_array(writer: &mut Writer, arr: &GgufArray) {
    writer.write_u32(arr.element_type().tag());
    writer.write_u64(arr.len() as u64);
    match arr {
        GgufArray::U8(v) => v.iter().for_each(|&x| writer.write_u8(x)),
        GgufArray::I8(v) => v.iter().for_each(|&x| writer.write_i8(x)),
        GgufArray::U16(v) => v.iter().for_each(|&x| writer.write_u16(x)),
        GgufArray::I16(v) => v.iter().for_each(|&x| writer.write_i16(x)),
        GgufArray::U32(v) => v.iter().for_each(|&x| writer.write_u32(x)),
        GgufArray::I32(v) => v.iter().for_each(|&x| writer.write_i32(x)),
        GgufArray::F32(v) => v.iter().for_each(|&x| writer.write_f32(x)),
        GgufArray::Bool(v) => v.iter().for_each(|&x| writer.write_bool(x)),
        GgufArray::String(v) => v.iter().for_each(|s| writer.write_string(s)),
        GgufArray::U64(v) => v.iter().for_each(|&x| writer.write_u64(x)),
        GgufArray::I64(v) => v.iter().for_each(|&x| writer.write_i64(x)),
        GgufArray::F64(v) => v.iter().for_each(|&x| writer.write_f64(x)),
    }
}

// ─── Top-level reader ──────────────────────────────────────────────────────────

/// Parse a complete GGUF v3 container from a byte slice.
///
/// Reads the header, the typed metadata section, and the tensor directory,
/// resolves the alignment from `general.alignment` (default
/// [`GGUF_DEFAULT_ALIGNMENT`]), and captures the aligned tensor-data region
/// verbatim into [`GgufFile::tensor_data`].
///
/// # Errors
///
/// Returns [`QuantError::InvalidConfig`] for a bad magic number, an
/// unsupported version, an unknown value/array type tag, a truncated input, a
/// non-UTF-8 string, or a tensor-data region that is shorter than the directory
/// requires. It never panics on malformed input.
pub fn read_gguf(bytes: &[u8]) -> QuantResult<GgufFile> {
    let mut reader = Reader::new(bytes);

    // ── Header ──────────────────────────────────────────────────────────────
    let magic = reader.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(QuantError::InvalidConfig(format!(
            "bad GGUF magic: expected {GGUF_MAGIC:#010x}, got {magic:#010x}"
        )));
    }
    let version = reader.read_u32()?;
    if version != GGUF_VERSION {
        return Err(QuantError::InvalidConfig(format!(
            "unsupported GGUF version {version} (this reader supports v{GGUF_VERSION})"
        )));
    }
    let tensor_count = reader.read_u64()?;
    let metadata_kv_count = reader.read_u64()?;

    // ── Metadata key-value section ──────────────────────────────────────────
    let mut metadata = Vec::with_capacity(metadata_kv_count.min(1 << 16) as usize);
    let mut alignment = GGUF_DEFAULT_ALIGNMENT;
    for _ in 0..metadata_kv_count {
        let key = reader.read_string()?;
        let value_tag = reader.read_u32()?;
        let value_ty = GgufValueType::from_tag(value_tag)?;
        let value = read_value(&mut reader, value_ty)?;
        if key == ALIGNMENT_KEY {
            if let Some(a) = value.as_u32() {
                if a == 0 {
                    return Err(QuantError::InvalidConfig(
                        "GGUF general.alignment must be non-zero".to_string(),
                    ));
                }
                alignment = u64::from(a);
            }
        }
        metadata.push(GgufMetadataKv { key, value });
    }

    // ── Tensor directory ────────────────────────────────────────────────────
    let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let n_dims = reader.read_u32()?;
        let mut dims = Vec::with_capacity(n_dims.min(8) as usize);
        for _ in 0..n_dims {
            dims.push(reader.read_u64()?);
        }
        let ggml_type = reader.read_u32()?;
        let offset = reader.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // ── Tensor data starts at the next alignment boundary ───────────────────
    let dir_end = reader.pos as u64;
    let data_start = align_up(dir_end, alignment) as usize;
    if data_start > bytes.len() {
        return Err(QuantError::InvalidConfig(format!(
            "GGUF tensor-data start {data_start} past end of {}-byte file",
            bytes.len()
        )));
    }
    let tensor_data = bytes[data_start..].to_vec();

    let header = GgufHeader {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
    };

    Ok(GgufFile {
        header,
        metadata,
        tensors,
        tensor_data,
        alignment,
    })
}

// ─── Top-level writer ──────────────────────────────────────────────────────────

/// Serialize a [`GgufFile`] into a spec-conformant GGUF v3 byte buffer.
///
/// The header's `tensor_count` / `metadata_kv_count` are recomputed from the
/// `tensors` / `metadata` vectors (the stored header counts are ignored), the
/// metadata and tensor directory are written in order, the output is padded to
/// the file alignment, and [`GgufFile::tensor_data`] is appended verbatim.
///
/// The resulting bytes round-trip through [`read_gguf`] to an equal
/// [`GgufFile`] whenever the tensor offsets are consistent with the alignment
/// and the packed data blob.
#[must_use]
pub fn write_gguf(file: &GgufFile) -> Vec<u8> {
    let mut writer = Writer::new();

    // ── Header ──────────────────────────────────────────────────────────────
    writer.write_u32(GGUF_MAGIC);
    writer.write_u32(GGUF_VERSION);
    writer.write_u64(file.tensors.len() as u64);
    writer.write_u64(file.metadata.len() as u64);

    // ── Metadata key-value section ──────────────────────────────────────────
    for kv in &file.metadata {
        writer.write_string(&kv.key);
        writer.write_u32(kv.value.value_type().tag());
        write_value(&mut writer, &kv.value);
    }

    // ── Tensor directory ────────────────────────────────────────────────────
    for t in &file.tensors {
        writer.write_string(&t.name);
        writer.write_u32(t.dims.len() as u32);
        for &d in &t.dims {
            writer.write_u64(d);
        }
        writer.write_u32(t.ggml_type);
        writer.write_u64(t.offset);
    }

    // ── Pad to alignment, then append tensor data verbatim ──────────────────
    writer.pad_to(file.alignment);
    writer.out.extend_from_slice(&file.tensor_data);

    writer.out
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a representative container: a string KV, an int KV, an array KV,
    /// the alignment override, and two tensor infos with aligned offsets.
    fn sample_file() -> GgufFile {
        let metadata = vec![
            GgufMetadataKv {
                key: "general.architecture".to_string(),
                value: GgufMetadataValue::String("llama".to_string()),
            },
            GgufMetadataKv {
                key: "llama.block_count".to_string(),
                value: GgufMetadataValue::U32(32),
            },
            GgufMetadataKv {
                key: "llama.attention.head_count".to_string(),
                value: GgufMetadataValue::I32(-1),
            },
            GgufMetadataKv {
                key: "tokenizer.ggml.tokens".to_string(),
                value: GgufMetadataValue::Array(GgufArray::String(vec![
                    "<s>".to_string(),
                    "</s>".to_string(),
                    "hello".to_string(),
                ])),
            },
            GgufMetadataKv {
                key: "llama.rope.freqs".to_string(),
                value: GgufMetadataValue::Array(GgufArray::F32(vec![1.0, 0.5, 0.25])),
            },
            GgufMetadataKv {
                key: ALIGNMENT_KEY.to_string(),
                value: GgufMetadataValue::U32(32),
            },
        ];

        // Two tensors; offsets must be multiples of the alignment (32).
        let tensors = vec![
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                dims: vec![4096, 32000],
                ggml_type: 8, // Q8_0 in ggml's tensor type table
                offset: 0,
            },
            GgufTensorInfo {
                name: "output_norm.weight".to_string(),
                dims: vec![4096],
                ggml_type: 0, // F32
                offset: 64,
            },
        ];

        // Packed data: first tensor occupies [0, 64), second [64, 96).
        let mut tensor_data = vec![0_u8; 96];
        for (i, b) in tensor_data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        GgufFile::new(metadata, tensors, tensor_data, 32)
    }

    #[test]
    fn magic_constant_is_gguf_ascii() {
        assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
    }

    #[test]
    fn value_type_tag_round_trip() {
        for tag in 0_u32..=12 {
            let ty = GgufValueType::from_tag(tag).expect("known tag");
            assert_eq!(ty.tag(), tag);
        }
    }

    #[test]
    fn unknown_value_type_tag_errors() {
        assert!(matches!(
            GgufValueType::from_tag(13),
            Err(QuantError::InvalidConfig(_))
        ));
        assert!(matches!(
            GgufValueType::from_tag(u32::MAX),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn round_trip_reproduces_file_exactly() {
        let original = sample_file();
        let bytes = write_gguf(&original);
        let parsed = read_gguf(&bytes).expect("read back the written container");

        // Header counts derived from the vectors.
        assert_eq!(parsed.header.magic, GGUF_MAGIC);
        assert_eq!(parsed.header.version, GGUF_VERSION);
        assert_eq!(parsed.header.tensor_count, original.tensors.len() as u64);
        assert_eq!(
            parsed.header.metadata_kv_count,
            original.metadata.len() as u64
        );

        // Metadata, tensors, data and alignment must all match exactly.
        assert_eq!(parsed.metadata, original.metadata);
        assert_eq!(parsed.tensors, original.tensors);
        assert_eq!(parsed.tensor_data, original.tensor_data);
        assert_eq!(parsed.alignment, original.alignment);

        // The whole `GgufFile` compares equal.
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_is_byte_stable() {
        let original = sample_file();
        let bytes_a = write_gguf(&original);
        let parsed = read_gguf(&bytes_a).expect("parse");
        let bytes_b = write_gguf(&parsed);
        assert_eq!(bytes_a, bytes_b, "re-serialization must be byte-identical");
    }

    #[test]
    fn metadata_get_finds_typed_values() {
        let file = sample_file();
        match file.metadata_get("general.architecture") {
            Some(GgufMetadataValue::String(s)) => assert_eq!(s, "llama"),
            other => panic!("unexpected: {other:?}"),
        }
        match file.metadata_get("llama.block_count") {
            Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, 32),
            other => panic!("unexpected: {other:?}"),
        }
        assert!(file.metadata_get("does.not.exist").is_none());
    }

    #[test]
    fn every_scalar_value_type_round_trips() {
        let metadata = vec![
            GgufMetadataKv {
                key: "a.u8".to_string(),
                value: GgufMetadataValue::U8(200),
            },
            GgufMetadataKv {
                key: "a.i8".to_string(),
                value: GgufMetadataValue::I8(-100),
            },
            GgufMetadataKv {
                key: "a.u16".to_string(),
                value: GgufMetadataValue::U16(40000),
            },
            GgufMetadataKv {
                key: "a.i16".to_string(),
                value: GgufMetadataValue::I16(-20000),
            },
            GgufMetadataKv {
                key: "a.u32".to_string(),
                value: GgufMetadataValue::U32(3_000_000_000),
            },
            GgufMetadataKv {
                key: "a.i32".to_string(),
                value: GgufMetadataValue::I32(-1_500_000_000),
            },
            GgufMetadataKv {
                key: "a.f32".to_string(),
                value: GgufMetadataValue::F32(core::f32::consts::PI),
            },
            GgufMetadataKv {
                key: "a.bool_t".to_string(),
                value: GgufMetadataValue::Bool(true),
            },
            GgufMetadataKv {
                key: "a.bool_f".to_string(),
                value: GgufMetadataValue::Bool(false),
            },
            GgufMetadataKv {
                key: "a.u64".to_string(),
                value: GgufMetadataValue::U64(12_000_000_000_000),
            },
            GgufMetadataKv {
                key: "a.i64".to_string(),
                value: GgufMetadataValue::I64(-9_000_000_000_000),
            },
            GgufMetadataKv {
                key: "a.f64".to_string(),
                value: GgufMetadataValue::F64(core::f64::consts::E),
            },
        ];
        let file = GgufFile::new(metadata.clone(), vec![], vec![], GGUF_DEFAULT_ALIGNMENT);
        let bytes = write_gguf(&file);
        let parsed = read_gguf(&bytes).expect("parse all scalar types");
        assert_eq!(parsed.metadata, metadata);
    }

    #[test]
    fn every_array_value_type_round_trips() {
        let metadata = vec![
            GgufMetadataKv {
                key: "arr.u8".to_string(),
                value: GgufMetadataValue::Array(GgufArray::U8(vec![1, 2, 3])),
            },
            GgufMetadataKv {
                key: "arr.i8".to_string(),
                value: GgufMetadataValue::Array(GgufArray::I8(vec![-1, 0, 1])),
            },
            GgufMetadataKv {
                key: "arr.u16".to_string(),
                value: GgufMetadataValue::Array(GgufArray::U16(vec![10, 20])),
            },
            GgufMetadataKv {
                key: "arr.i16".to_string(),
                value: GgufMetadataValue::Array(GgufArray::I16(vec![-10, 20])),
            },
            GgufMetadataKv {
                key: "arr.u32".to_string(),
                value: GgufMetadataValue::Array(GgufArray::U32(vec![100, 200])),
            },
            GgufMetadataKv {
                key: "arr.i32".to_string(),
                value: GgufMetadataValue::Array(GgufArray::I32(vec![-100, 200])),
            },
            GgufMetadataKv {
                key: "arr.f32".to_string(),
                value: GgufMetadataValue::Array(GgufArray::F32(vec![1.5, -2.5, 3.5])),
            },
            GgufMetadataKv {
                key: "arr.bool".to_string(),
                value: GgufMetadataValue::Array(GgufArray::Bool(vec![true, false, true])),
            },
            GgufMetadataKv {
                key: "arr.string".to_string(),
                value: GgufMetadataValue::Array(GgufArray::String(vec![
                    "x".to_string(),
                    "yy".to_string(),
                ])),
            },
            GgufMetadataKv {
                key: "arr.u64".to_string(),
                value: GgufMetadataValue::Array(GgufArray::U64(vec![1, 2, 3, 4])),
            },
            GgufMetadataKv {
                key: "arr.i64".to_string(),
                value: GgufMetadataValue::Array(GgufArray::I64(vec![-1, -2])),
            },
            GgufMetadataKv {
                key: "arr.f64".to_string(),
                value: GgufMetadataValue::Array(GgufArray::F64(vec![0.1, 0.2])),
            },
            GgufMetadataKv {
                key: "arr.empty".to_string(),
                value: GgufMetadataValue::Array(GgufArray::F32(vec![])),
            },
        ];
        let file = GgufFile::new(metadata.clone(), vec![], vec![], GGUF_DEFAULT_ALIGNMENT);
        let bytes = write_gguf(&file);
        let parsed = read_gguf(&bytes).expect("parse all array types");
        assert_eq!(parsed.metadata, metadata);
    }

    #[test]
    fn empty_container_round_trips() {
        let file = GgufFile::new(vec![], vec![], vec![], GGUF_DEFAULT_ALIGNMENT);
        let bytes = write_gguf(&file);
        // Header is exactly 24 bytes; with default alignment 32 the data region
        // starts at offset 32, so there are 8 padding bytes and no data.
        assert_eq!(bytes.len(), 32);
        let parsed = read_gguf(&bytes).expect("parse empty container");
        assert_eq!(parsed.header.tensor_count, 0);
        assert_eq!(parsed.header.metadata_kv_count, 0);
        assert!(parsed.metadata.is_empty());
        assert!(parsed.tensors.is_empty());
        assert!(parsed.tensor_data.is_empty());
        assert_eq!(parsed.alignment, GGUF_DEFAULT_ALIGNMENT);
    }

    #[test]
    fn tensor_data_region_respects_alignment() {
        // With a non-default alignment of 64, the data must start at the next
        // 64-byte boundary after the directory, and the first byte read back
        // must be the first byte of `tensor_data`.
        let metadata = vec![GgufMetadataKv {
            key: ALIGNMENT_KEY.to_string(),
            value: GgufMetadataValue::U32(64),
        }];
        let tensors = vec![GgufTensorInfo {
            name: "w".to_string(),
            dims: vec![10],
            ggml_type: 0,
            offset: 0,
        }];
        let tensor_data = vec![0xAB_u8; 64];
        let file = GgufFile::new(metadata, tensors, tensor_data.clone(), 64);
        let bytes = write_gguf(&file);
        // Data region must begin on a 64-byte boundary.
        let data_start = bytes.len() - tensor_data.len();
        assert_eq!(data_start % 64, 0, "data region not 64-aligned");
        let parsed = read_gguf(&bytes).expect("parse 64-aligned container");
        assert_eq!(parsed.alignment, 64);
        assert_eq!(parsed.tensor_data, tensor_data);
        // Offset 0 tensor begins exactly at the start of the data blob.
        assert_eq!(parsed.tensor_data[parsed.tensors[0].offset as usize], 0xAB);
    }

    #[test]
    fn default_alignment_applies_without_override() {
        // No general.alignment key → default 32, and a directory that ends
        // off-boundary still produces a 32-aligned data region.
        let tensors = vec![GgufTensorInfo {
            name: "tiny".to_string(),
            dims: vec![1],
            ggml_type: 0,
            offset: 0,
        }];
        let file = GgufFile::new(vec![], tensors, vec![7_u8; 32], GGUF_DEFAULT_ALIGNMENT);
        let bytes = write_gguf(&file);
        let data_start = bytes.len() - 32;
        assert_eq!(data_start % 32, 0);
        let parsed = read_gguf(&bytes).expect("parse");
        assert_eq!(parsed.alignment, 32);
    }

    #[test]
    fn bad_magic_errors_not_panics() {
        let mut bytes = write_gguf(&sample_file());
        bytes[0] ^= 0xFF; // corrupt the magic
        let result = read_gguf(&bytes);
        assert!(matches!(result, Err(QuantError::InvalidConfig(_))));
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn wrong_version_errors() {
        let mut bytes = write_gguf(&sample_file());
        // Bump the version field (bytes 4..8) to an unsupported value.
        bytes[4..8].copy_from_slice(&99_u32.to_le_bytes());
        assert!(matches!(
            read_gguf(&bytes),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn truncated_header_errors() {
        // Fewer than the 24 header bytes.
        for n in 0..24 {
            let bytes = vec![0_u8; n];
            assert!(
                read_gguf(&bytes).is_err(),
                "{n}-byte input should error, not panic"
            );
        }
    }

    #[test]
    fn truncated_in_metadata_errors() {
        let full = write_gguf(&sample_file());
        // Cut in the middle of the metadata section (just past the header).
        let bytes = &full[..30];
        assert!(matches!(
            read_gguf(bytes),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn truncated_before_data_region_errors() {
        // A header claiming one tensor but no directory bytes following.
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(1); // tensor_count
        writer.write_u64(0); // metadata_kv_count
        // Stop here — the tensor directory entry is missing entirely.
        assert!(matches!(
            read_gguf(&writer.out),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn bad_value_type_tag_errors() {
        // Header + one metadata entry whose value tag is invalid.
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0); // tensor_count
        writer.write_u64(1); // metadata_kv_count
        writer.write_string("key"); // key
        writer.write_u32(255); // invalid value type tag
        let result = read_gguf(&writer.out);
        assert!(matches!(result, Err(QuantError::InvalidConfig(_))));
        assert!(result.unwrap_err().to_string().contains("value type"));
    }

    #[test]
    fn bad_array_element_tag_errors() {
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        writer.write_string("arr");
        writer.write_u32(GgufValueType::Array.tag());
        writer.write_u32(200); // invalid array element tag
        writer.write_u64(3); // count
        assert!(matches!(
            read_gguf(&writer.out),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn nested_array_element_rejected() {
        // An array whose element type is itself `Array` (tag 9) is illegal.
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        writer.write_string("nested");
        writer.write_u32(GgufValueType::Array.tag());
        writer.write_u32(GgufValueType::Array.tag()); // element type = array
        writer.write_u64(1);
        let result = read_gguf(&writer.out);
        assert!(matches!(result, Err(QuantError::InvalidConfig(_))));
        assert!(result.unwrap_err().to_string().contains("nested"));
    }

    #[test]
    fn bad_bool_byte_errors() {
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        writer.write_string("flag");
        writer.write_u32(GgufValueType::Bool.tag());
        writer.write_u8(7); // not 0 or 1
        assert!(matches!(
            read_gguf(&writer.out),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn non_utf8_string_errors() {
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        // A 2-byte "string" containing an invalid UTF-8 sequence.
        writer.write_u64(2);
        writer.write_u8(0xFF);
        writer.write_u8(0xFE);
        let result = read_gguf(&writer.out);
        assert!(matches!(result, Err(QuantError::InvalidConfig(_))));
        assert!(result.unwrap_err().to_string().contains("UTF-8"));
    }

    #[test]
    fn absurd_string_length_errors() {
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        // A string claiming a length far exceeding the remaining bytes.
        writer.write_u64(u64::MAX);
        assert!(matches!(
            read_gguf(&writer.out),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn zero_alignment_override_errors() {
        let mut writer = Writer::new();
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);
        writer.write_u64(1);
        writer.write_string(ALIGNMENT_KEY);
        writer.write_u32(GgufValueType::U32.tag());
        writer.write_u32(0); // alignment 0 is illegal
        assert!(matches!(
            read_gguf(&writer.out),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn multi_dim_tensor_dims_preserved() {
        let tensors = vec![
            GgufTensorInfo {
                name: "scalar".to_string(),
                dims: vec![],
                ggml_type: 0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "vec".to_string(),
                dims: vec![7],
                ggml_type: 1,
                offset: 32,
            },
            GgufTensorInfo {
                name: "mat".to_string(),
                dims: vec![3, 5],
                ggml_type: 2,
                offset: 64,
            },
            GgufTensorInfo {
                name: "cube".to_string(),
                dims: vec![2, 3, 4],
                ggml_type: 3,
                offset: 96,
            },
        ];
        let file = GgufFile::new(vec![], tensors.clone(), vec![0_u8; 128], 32);
        let bytes = write_gguf(&file);
        let parsed = read_gguf(&bytes).expect("parse multi-dim tensors");
        assert_eq!(parsed.tensors, tensors);
    }

    #[test]
    fn integrates_with_ggml_block_payload() {
        // End-to-end: quantize a tensor with the ggml Q8_0 codec, pack the
        // blocks into the GGUF data region, write, read back, and verify the
        // payload bytes are preserved so they can be re-decoded.
        use crate::scheme::ggml::{BlockQ8_0, dequantize_q8_0, quantize_q8_0};

        // 64 weights → 2 Q8_0 blocks.
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 / 63.0) * 4.0 - 2.0).collect();
        let blocks = quantize_q8_0(&weights).expect("quantize");

        // Serialize the blocks to raw bytes (ggml block layout: f16 d + 32 i8).
        let mut payload = Vec::new();
        for b in &blocks {
            payload.extend_from_slice(&b.d.to_le_bytes());
            for &q in &b.qs {
                payload.push(q as u8);
            }
        }
        // Pad the data blob so its length is a multiple of the alignment.
        while payload.len() % 32 != 0 {
            payload.push(0);
        }

        let tensors = vec![GgufTensorInfo {
            name: "blk.0.ffn_down.weight".to_string(),
            dims: vec![64],
            ggml_type: 8, // Q8_0
            offset: 0,
        }];
        let file = GgufFile::new(
            vec![GgufMetadataKv {
                key: "general.name".to_string(),
                value: GgufMetadataValue::String("roundtrip".to_string()),
            }],
            tensors,
            payload.clone(),
            32,
        );

        let bytes = write_gguf(&file);
        let parsed = read_gguf(&bytes).expect("parse");
        assert_eq!(parsed.tensor_data, payload);

        // Decode the payload back into blocks and dequantize.
        let info = &parsed.tensors[0];
        let mut recovered = Vec::new();
        let block_bytes = 2 + 32; // f16 + 32 i8
        let mut cursor = info.offset as usize;
        for _ in 0..(blocks.len()) {
            let d =
                u16::from_le_bytes([parsed.tensor_data[cursor], parsed.tensor_data[cursor + 1]]);
            let mut qs = [0_i8; 32];
            for (k, q) in qs.iter_mut().enumerate() {
                *q = parsed.tensor_data[cursor + 2 + k] as i8;
            }
            recovered.push(BlockQ8_0 { d, qs });
            cursor += block_bytes;
        }
        assert_eq!(recovered, blocks);
        let deq = dequantize_q8_0(&recovered);
        assert_eq!(deq.len(), 64);
    }
}
