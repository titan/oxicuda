//! Neural operators: FNO, DeepONet, MWT, GNO, WNO.

pub mod deeponet;
pub mod fft;
pub mod fno;
pub mod fno_3d;
pub mod gno;
pub mod mwt;
pub mod pi_deeponet;
pub mod point_fno;
pub mod wno;

pub use fno_3d::{Fno3d, Fno3dConfig};
pub use pi_deeponet::{PiDeepONet, PiDeepONetConfig};
pub use point_fno::{PointFno, PointFnoConfig};
pub use wno::{Wno, WnoConfig};
