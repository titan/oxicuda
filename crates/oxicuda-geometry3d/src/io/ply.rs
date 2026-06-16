//! ASCII PLY (Polygon File Format) point-cloud reader (pure `std`).
//!
//! Supports the standard ASCII header
//!
//! ```text
//! ply
//! format ascii 1.0
//! element vertex N
//! property float x
//! property float y
//! property float z
//! [property float nx] [property float ny] [property float nz]
//! [property uchar red] [property uchar green] [property uchar blue]
//! end_header
//! <N whitespace-separated vertex rows>
//! ```
//!
//! Multiple elements (e.g. a trailing `element face`) are tolerated: their
//! data rows are skipped. Binary formats are rejected.

use crate::error::{Geom3dError, Geom3dResult};
use crate::io::PointCloud;
use std::path::Path;

/// A single PLY property declaration.
struct PlyProperty {
    name: String,
    type_name: String,
    is_list: bool,
}

/// A single PLY element declaration (name, row count, and property list).
struct PlyElement {
    name: String,
    count: usize,
    props: Vec<PlyProperty>,
}

fn err(reason: &str) -> Geom3dError {
    Geom3dError::Internal(format!("PLY parse error: {reason}"))
}

/// `true` if the PLY scalar type stores an unsigned 8-bit channel.
fn is_uchar_type(t: &str) -> bool {
    matches!(t, "uchar" | "uint8" | "char" | "int8" | "u8" | "i8")
}

/// Index of the property named `name` within an element, if present.
fn prop_index(elem: &PlyElement, name: &str) -> Option<usize> {
    elem.props.iter().position(|p| p.name == name)
}

/// Parse a single float token at position `idx`.
fn parse_at(tokens: &[&str], idx: usize) -> Geom3dResult<f32> {
    tokens
        .get(idx)
        .ok_or_else(|| err("vertex row has too few columns"))?
        .parse::<f32>()
        .map_err(|_| err("vertex row contains a non-numeric value"))
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

/// Parse an ASCII PLY document from a string.
///
/// # Errors
///
/// Returns [`Geom3dError::Internal`] for any malformed or unsupported (e.g.
/// binary) PLY input.
pub fn parse_ply_str(text: &str) -> Geom3dResult<PointCloud> {
    let mut lines = text.lines();

    // First non-empty line must be the "ply" magic.
    let magic = loop {
        match lines.next() {
            Some(l) if l.trim().is_empty() => continue,
            Some(l) => break l,
            None => return Err(err("empty input")),
        }
    };
    if magic.trim() != "ply" {
        return Err(err("missing 'ply' magic on first line"));
    }

    let mut format_seen = false;
    let mut header_ended = false;
    let mut elements: Vec<PlyElement> = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let keyword = match tokens.first() {
            Some(k) => *k,
            None => continue,
        };
        match keyword {
            "format" => {
                if tokens.get(1) != Some(&"ascii") {
                    return Err(err("only 'format ascii' is supported"));
                }
                format_seen = true;
            }
            "comment" | "obj_info" => {}
            "element" => {
                let name = tokens.get(1).ok_or_else(|| err("element missing name"))?;
                let count = tokens
                    .get(2)
                    .ok_or_else(|| err("element missing count"))?
                    .parse::<usize>()
                    .map_err(|_| err("element count is not an integer"))?;
                elements.push(PlyElement {
                    name: (*name).to_string(),
                    count,
                    props: Vec::new(),
                });
            }
            "property" => {
                let elem = elements
                    .last_mut()
                    .ok_or_else(|| err("property declared before any element"))?;
                if tokens.get(1) == Some(&"list") {
                    // property list <count_type> <item_type> <name>
                    let name = tokens
                        .get(4)
                        .ok_or_else(|| err("list property missing name"))?;
                    elem.props.push(PlyProperty {
                        name: (*name).to_string(),
                        type_name: "list".to_string(),
                        is_list: true,
                    });
                } else {
                    let type_name = tokens.get(1).ok_or_else(|| err("property missing type"))?;
                    let name = tokens.get(2).ok_or_else(|| err("property missing name"))?;
                    elem.props.push(PlyProperty {
                        name: (*name).to_string(),
                        type_name: (*type_name).to_string(),
                        is_list: false,
                    });
                }
            }
            "end_header" => {
                header_ended = true;
                break;
            }
            _ => {}
        }
    }

    if !format_seen {
        return Err(err("missing 'format' line"));
    }
    if !header_ended {
        return Err(err("missing 'end_header'"));
    }

    // Remaining lines are element data, in declaration order.
    let data: Vec<&str> = lines.collect();
    let mut cursor = 0usize;
    let mut cloud = PointCloud::default();
    let mut vertex_seen = false;

    for elem in &elements {
        if elem.name == "vertex" {
            vertex_seen = true;
            parse_vertex_element(elem, &data, &mut cursor, &mut cloud)?;
        } else {
            // Skip this element's data rows.
            for _ in 0..elem.count {
                next_data_line(&data, &mut cursor)
                    .ok_or_else(|| err("unexpected end of data while skipping element"))?;
            }
        }
    }

    if !vertex_seen {
        return Err(err("no 'vertex' element in header"));
    }
    Ok(cloud)
}

/// Parse the rows belonging to the `vertex` element into `cloud`.
fn parse_vertex_element(
    elem: &PlyElement,
    data: &[&str],
    cursor: &mut usize,
    cloud: &mut PointCloud,
) -> Geom3dResult<()> {
    if elem.props.iter().any(|p| p.is_list) {
        return Err(err("list properties on vertex element are unsupported"));
    }

    let xi = prop_index(elem, "x").ok_or_else(|| err("vertex missing property 'x'"))?;
    let yi = prop_index(elem, "y").ok_or_else(|| err("vertex missing property 'y'"))?;
    let zi = prop_index(elem, "z").ok_or_else(|| err("vertex missing property 'z'"))?;

    let normal_idx = match (
        prop_index(elem, "nx"),
        prop_index(elem, "ny"),
        prop_index(elem, "nz"),
    ) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };
    let color_idx = match (
        prop_index(elem, "red"),
        prop_index(elem, "green"),
        prop_index(elem, "blue"),
    ) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };
    let color_scale = match color_idx {
        Some([r, _, _]) if is_uchar_type(&elem.props[r].type_name) => 1.0 / 255.0,
        _ => 1.0,
    };

    let mut points = Vec::with_capacity(elem.count * 3);
    let mut normals = normal_idx.map(|_| Vec::with_capacity(elem.count * 3));
    let mut colors = color_idx.map(|_| Vec::with_capacity(elem.count * 3));

    for _ in 0..elem.count {
        let line = next_data_line(data, cursor)
            .ok_or_else(|| err("fewer vertex rows than declared count"))?;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        points.push(parse_at(&tokens, xi)?);
        points.push(parse_at(&tokens, yi)?);
        points.push(parse_at(&tokens, zi)?);

        if let (Some([a, b, c]), Some(buf)) = (normal_idx, normals.as_mut()) {
            buf.push(parse_at(&tokens, a)?);
            buf.push(parse_at(&tokens, b)?);
            buf.push(parse_at(&tokens, c)?);
        }
        if let (Some([a, b, c]), Some(buf)) = (color_idx, colors.as_mut()) {
            buf.push(parse_at(&tokens, a)? * color_scale);
            buf.push(parse_at(&tokens, b)? * color_scale);
            buf.push(parse_at(&tokens, c)? * color_scale);
        }
    }

    cloud.points = points;
    cloud.normals = normals;
    cloud.colors = colors;
    Ok(())
}

/// Read an ASCII PLY file from disk.
///
/// # Errors
///
/// Returns [`Geom3dError::Internal`] if the file cannot be read or is malformed.
pub fn read_ply(path: impl AsRef<Path>) -> Geom3dResult<PointCloud> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| Geom3dError::Internal(format!("failed to read PLY file: {e}")))?;
    parse_ply_str(&content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oxicuda_geom3d_ply_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    const XYZ_PLY: &str = "ply\n\
format ascii 1.0\n\
comment generated by test\n\
element vertex 3\n\
property float x\n\
property float y\n\
property float z\n\
end_header\n\
0.0 0.0 0.0\n\
1.0 2.0 3.0\n\
-1.5 0.5 2.5\n";

    #[test]
    fn parse_basic_xyz() {
        let cloud = parse_ply_str(XYZ_PLY).expect("parse_ply_str should succeed");
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.point(1), Some([1.0, 2.0, 3.0]));
        assert_eq!(cloud.point(2), Some([-1.5, 0.5, 2.5]));
        assert!(cloud.normals.is_none());
        assert!(cloud.colors.is_none());
    }

    #[test]
    fn parse_with_normals_and_colors() {
        let text = "ply\n\
format ascii 1.0\n\
element vertex 2\n\
property float x\n\
property float y\n\
property float z\n\
property float nx\n\
property float ny\n\
property float nz\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
end_header\n\
0 0 0 0 0 1 255 0 0\n\
1 1 1 0 1 0 0 255 0\n";
        let cloud = parse_ply_str(text).expect("parse_ply_str should succeed");
        assert_eq!(cloud.len(), 2);
        let normals = cloud.normals.expect("normals present");
        assert_eq!(normals, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let colors = cloud.colors.expect("colors present");
        // uchar 255 -> 1.0
        assert!((colors[0] - 1.0).abs() < 1e-6);
        assert!(colors[1].abs() < 1e-6);
        assert!((colors[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_header_only_zero_vertices() {
        let text = "ply\n\
format ascii 1.0\n\
element vertex 0\n\
property float x\n\
property float y\n\
property float z\n\
end_header\n";
        let cloud = parse_ply_str(text).expect("parse_ply_str should succeed");
        assert!(cloud.is_empty());
        assert_eq!(cloud.len(), 0);
    }

    #[test]
    fn parse_with_trailing_face_element() {
        let text = "ply\n\
format ascii 1.0\n\
element vertex 3\n\
property float x\n\
property float y\n\
property float z\n\
element face 1\n\
property list uchar int vertex_indices\n\
end_header\n\
0 0 0\n\
1 0 0\n\
0 1 0\n\
3 0 1 2\n";
        let cloud = parse_ply_str(text).expect("parse_ply_str should succeed");
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.point(2), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn reject_missing_magic() {
        let text = "not_ply\nformat ascii 1.0\nend_header\n";
        assert!(parse_ply_str(text).is_err());
    }

    #[test]
    fn reject_binary_format() {
        let text = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
property float x\nproperty float y\nproperty float z\nend_header\n";
        assert!(parse_ply_str(text).is_err());
    }

    #[test]
    fn reject_truncated_data() {
        let text = "ply\nformat ascii 1.0\nelement vertex 3\n\
property float x\nproperty float y\nproperty float z\nend_header\n\
0 0 0\n1 1 1\n";
        assert!(parse_ply_str(text).is_err());
    }

    #[test]
    fn reject_non_numeric_row() {
        let text = "ply\nformat ascii 1.0\nelement vertex 1\n\
property float x\nproperty float y\nproperty float z\nend_header\n\
0 oops 0\n";
        assert!(parse_ply_str(text).is_err());
    }

    #[test]
    fn read_ply_roundtrip_from_disk() {
        let path = temp_path("roundtrip.ply");
        {
            let mut f = std::fs::File::create(&path).expect("create should succeed");
            f.write_all(XYZ_PLY.as_bytes())
                .expect("value should be present");
        }
        let cloud = read_ply(&path).expect("read_ply should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.point(0), Some([0.0, 0.0, 0.0]));
        assert_eq!(cloud.point(1), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn read_missing_file_errors() {
        let path = temp_path("does_not_exist.ply");
        let _ = std::fs::remove_file(&path);
        assert!(read_ply(&path).is_err());
    }
}
