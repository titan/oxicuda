//! ASCII PCD (Point Cloud Data, PCL/Open3D) point-cloud reader (pure `std`).
//!
//! Supports the standard ASCII header
//!
//! ```text
//! # .PCD v0.7 - Point Cloud Data file format
//! VERSION 0.7
//! FIELDS x y z [normal_x normal_y normal_z …]
//! SIZE 4 4 4
//! TYPE F F F
//! COUNT 1 1 1
//! WIDTH N
//! HEIGHT 1
//! VIEWPOINT 0 0 0 1 0 0 0
//! POINTS N
//! DATA ascii
//! <N whitespace-separated point rows>
//! ```
//!
//! Per-field `COUNT > 1` is honoured when computing column offsets. Binary
//! payloads are rejected. Packed `rgb`/`rgba` float fields are not decoded
//! (the column is simply skipped), so [`PointCloud::colors`] is always `None`
//! for PCD; optional `normal_x/y/z` fields are read into
//! [`PointCloud::normals`].

use crate::error::{Geom3dError, Geom3dResult};
use crate::io::PointCloud;
use std::path::Path;

fn err(reason: &str) -> Geom3dError {
    Geom3dError::Internal(format!("PCD parse error: {reason}"))
}

/// Index of `name` within the field list, if present.
fn field_index(fields: &[String], name: &str) -> Option<usize> {
    fields.iter().position(|f| f == name)
}

/// Parse a single float token at column `idx`.
fn parse_at(tokens: &[&str], idx: usize) -> Geom3dResult<f32> {
    tokens
        .get(idx)
        .ok_or_else(|| err("data row has too few columns"))?
        .parse::<f32>()
        .map_err(|_| err("data row contains a non-numeric value"))
}

/// Advance `cursor` to the next non-empty line of `data`, returning it.
fn next_data_line<'a>(data: &[&'a str], cursor: &mut usize) -> Option<&'a str> {
    while *cursor < data.len() {
        let line = data[*cursor];
        *cursor += 1;
        if !line.trim().is_empty() {
            return Some(line);
        }
    }
    None
}

/// Parse an ASCII PCD document from a string.
///
/// # Errors
///
/// Returns [`Geom3dError::Internal`] for any malformed or unsupported (e.g.
/// binary) PCD input.
pub fn parse_pcd_str(text: &str) -> Geom3dResult<PointCloud> {
    let mut fields: Vec<String> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut points: Option<usize> = None;
    let mut width: Option<usize> = None;
    let mut height: Option<usize> = None;
    let mut data_ascii = false;
    let mut header_ended = false;

    let mut lines = text.lines();
    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let keyword = match tokens.first() {
            Some(k) => *k,
            None => continue,
        };
        match keyword {
            "FIELDS" | "COLUMNS" => {
                fields = tokens[1..].iter().map(|t| (*t).to_string()).collect();
            }
            "COUNT" => {
                counts.clear();
                for t in &tokens[1..] {
                    counts.push(
                        t.parse::<usize>()
                            .map_err(|_| err("COUNT contains a non-integer"))?,
                    );
                }
            }
            "POINTS" => {
                points = Some(
                    tokens
                        .get(1)
                        .ok_or_else(|| err("POINTS missing value"))?
                        .parse::<usize>()
                        .map_err(|_| err("POINTS is not an integer"))?,
                );
            }
            "WIDTH" => {
                width = Some(
                    tokens
                        .get(1)
                        .ok_or_else(|| err("WIDTH missing value"))?
                        .parse::<usize>()
                        .map_err(|_| err("WIDTH is not an integer"))?,
                );
            }
            "HEIGHT" => {
                height = Some(
                    tokens
                        .get(1)
                        .ok_or_else(|| err("HEIGHT missing value"))?
                        .parse::<usize>()
                        .map_err(|_| err("HEIGHT is not an integer"))?,
                );
            }
            "DATA" => {
                match tokens.get(1) {
                    Some(&"ascii") => data_ascii = true,
                    Some(other) => {
                        return Err(err(&format!(
                            "unsupported DATA '{other}', only ascii is supported"
                        )));
                    }
                    None => return Err(err("DATA missing format specifier")),
                }
                header_ended = true;
                break;
            }
            _ => {}
        }
    }

    if !header_ended || !data_ascii {
        return Err(err("missing 'DATA ascii' line"));
    }
    if fields.is_empty() {
        return Err(err("missing 'FIELDS' line"));
    }

    let n_points = points
        .or(match (width, height) {
            (Some(w), Some(h)) => Some(w * h),
            _ => None,
        })
        .ok_or_else(|| err("missing 'POINTS' (and WIDTH/HEIGHT)"))?;

    if counts.is_empty() {
        counts = vec![1; fields.len()];
    }
    if counts.len() != fields.len() {
        return Err(err("COUNT length does not match FIELDS length"));
    }

    // Column offset of each field = sum of preceding fields' counts.
    let mut offsets = Vec::with_capacity(fields.len());
    let mut acc = 0usize;
    for &c in &counts {
        offsets.push(acc);
        acc += c;
    }
    let min_columns = acc;

    let xi = offsets[field_index(&fields, "x").ok_or_else(|| err("missing field 'x'"))?];
    let yi = offsets[field_index(&fields, "y").ok_or_else(|| err("missing field 'y'"))?];
    let zi = offsets[field_index(&fields, "z").ok_or_else(|| err("missing field 'z'"))?];

    let normal_cols = match (
        field_index(&fields, "normal_x"),
        field_index(&fields, "normal_y"),
        field_index(&fields, "normal_z"),
    ) {
        (Some(a), Some(b), Some(c)) => Some([offsets[a], offsets[b], offsets[c]]),
        _ => None,
    };

    let data: Vec<&str> = lines.collect();
    let mut cursor = 0usize;
    let mut pts = Vec::with_capacity(n_points * 3);
    let mut normals = normal_cols.map(|_| Vec::with_capacity(n_points * 3));

    for _ in 0..n_points {
        let line = next_data_line(&data, &mut cursor)
            .ok_or_else(|| err("fewer data rows than declared POINTS"))?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < min_columns {
            return Err(err("data row has fewer columns than FIELDS/COUNT require"));
        }
        pts.push(parse_at(&tokens, xi)?);
        pts.push(parse_at(&tokens, yi)?);
        pts.push(parse_at(&tokens, zi)?);

        if let (Some([a, b, c]), Some(buf)) = (normal_cols, normals.as_mut()) {
            buf.push(parse_at(&tokens, a)?);
            buf.push(parse_at(&tokens, b)?);
            buf.push(parse_at(&tokens, c)?);
        }
    }

    Ok(PointCloud {
        points: pts,
        normals,
        colors: None,
    })
}

/// Read an ASCII PCD file from disk.
///
/// # Errors
///
/// Returns [`Geom3dError::Internal`] if the file cannot be read or is malformed.
pub fn read_pcd(path: impl AsRef<Path>) -> Geom3dResult<PointCloud> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| Geom3dError::Internal(format!("failed to read PCD file: {e}")))?;
    parse_pcd_str(&content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oxicuda_geom3d_pcd_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    const XYZ_PCD: &str = "# .PCD v0.7 - Point Cloud Data file format\n\
VERSION 0.7\n\
FIELDS x y z\n\
SIZE 4 4 4\n\
TYPE F F F\n\
COUNT 1 1 1\n\
WIDTH 3\n\
HEIGHT 1\n\
VIEWPOINT 0 0 0 1 0 0 0\n\
POINTS 3\n\
DATA ascii\n\
0.0 0.0 0.0\n\
1.0 2.0 3.0\n\
4.0 5.0 6.0\n";

    #[test]
    fn parse_basic_xyz() {
        let cloud = parse_pcd_str(XYZ_PCD).expect("parse_pcd_str should succeed");
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.point(1), Some([1.0, 2.0, 3.0]));
        assert_eq!(cloud.point(2), Some([4.0, 5.0, 6.0]));
        assert!(cloud.normals.is_none());
    }

    #[test]
    fn parse_with_normals() {
        let text = "VERSION 0.7\n\
FIELDS x y z normal_x normal_y normal_z\n\
SIZE 4 4 4 4 4 4\n\
TYPE F F F F F F\n\
COUNT 1 1 1 1 1 1\n\
WIDTH 2\n\
HEIGHT 1\n\
POINTS 2\n\
DATA ascii\n\
0 0 0 0 0 1\n\
1 1 1 1 0 0\n";
        let cloud = parse_pcd_str(text).expect("parse_pcd_str should succeed");
        assert_eq!(cloud.len(), 2);
        let normals = cloud.normals.expect("normals present");
        assert_eq!(normals, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn parse_skips_packed_rgb_column() {
        // An extra 'rgb' field sits between z and the normals; x/y/z still read.
        let text = "VERSION 0.7\n\
FIELDS x y z rgb\n\
SIZE 4 4 4 4\n\
TYPE F F F F\n\
COUNT 1 1 1 1\n\
WIDTH 2\n\
HEIGHT 1\n\
POINTS 2\n\
DATA ascii\n\
0 0 0 16711680\n\
1 2 3 255\n";
        let cloud = parse_pcd_str(text).expect("parse_pcd_str should succeed");
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.point(1), Some([1.0, 2.0, 3.0]));
        assert!(cloud.colors.is_none());
    }

    #[test]
    fn parse_width_height_fallback() {
        // No POINTS line: count derived from WIDTH * HEIGHT.
        let text = "VERSION 0.7\n\
FIELDS x y z\n\
WIDTH 2\n\
HEIGHT 1\n\
DATA ascii\n\
0 0 0\n\
1 1 1\n";
        let cloud = parse_pcd_str(text).expect("parse_pcd_str should succeed");
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn reject_binary_data() {
        let text = "FIELDS x y z\nPOINTS 1\nDATA binary\n";
        assert!(parse_pcd_str(text).is_err());
    }

    #[test]
    fn reject_missing_points() {
        let text = "FIELDS x y z\nDATA ascii\n0 0 0\n";
        assert!(parse_pcd_str(text).is_err());
    }

    #[test]
    fn reject_missing_xyz_field() {
        let text = "FIELDS a b c\nPOINTS 1\nDATA ascii\n0 0 0\n";
        assert!(parse_pcd_str(text).is_err());
    }

    #[test]
    fn reject_truncated_data() {
        let text = "FIELDS x y z\nPOINTS 3\nDATA ascii\n0 0 0\n1 1 1\n";
        assert!(parse_pcd_str(text).is_err());
    }

    #[test]
    fn read_pcd_roundtrip_from_disk() {
        let path = temp_path("roundtrip.pcd");
        {
            let mut f = std::fs::File::create(&path).expect("create should succeed");
            f.write_all(XYZ_PCD.as_bytes())
                .expect("value should be present");
        }
        let cloud = read_pcd(&path).expect("read_pcd should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.point(0), Some([0.0, 0.0, 0.0]));
        assert_eq!(cloud.point(2), Some([4.0, 5.0, 6.0]));
    }

    #[test]
    fn read_missing_file_errors() {
        let path = temp_path("does_not_exist.pcd");
        let _ = std::fs::remove_file(&path);
        assert!(read_pcd(&path).is_err());
    }
}
