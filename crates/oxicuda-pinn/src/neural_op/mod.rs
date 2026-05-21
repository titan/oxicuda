//! Neural operators: FNO, DeepONet, MWT, GNO, WNO.

pub mod deeponet;
pub mod fno;
pub mod fno_3d;
pub mod gno;
pub mod mwt;
pub mod wno;

pub use fno_3d::{Fno3d, Fno3dConfig};
pub use wno::{Wno, WnoConfig};
