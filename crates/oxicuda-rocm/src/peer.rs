//! Host-side xGMI / PCIe peer-access topology model.
//!
//! Models `hipDeviceCanAccessPeer` / `hipDeviceEnablePeerAccess` /
//! `hipMemcpyPeer` connectivity as a symmetric graph over device ordinals,
//! distinguishing high-bandwidth xGMI (Infinity Fabric) links from PCIe
//! fallbacks. Used to plan cross-GPU copies (e.g. on MI300X 8-GPU rings)
//! without any HIP runtime — fully CPU-testable.

use crate::error::{RocmError, RocmResult};

/// The physical interconnect between two GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// AMD xGMI / Infinity Fabric (high bandwidth, peer-DMA capable).
    XGmi,
    /// PCIe peer-to-peer (lower bandwidth, peer-DMA capable when the root
    /// complex permits it).
    Pcie,
    /// No direct peer access; transfers must stage through host memory.
    None,
}

impl LinkKind {
    /// Approximate one-directional bandwidth in GB/s for planning heuristics.
    ///
    /// xGMI3 (MI300) links run ~100 GB/s per link direction; PCIe Gen4 x16
    /// peer is ~25 GB/s; staging through host is modeled as 0 (no direct link).
    pub fn bandwidth_gbps(self) -> u32 {
        match self {
            LinkKind::XGmi => 100,
            LinkKind::Pcie => 25,
            LinkKind::None => 0,
        }
    }

    /// `true` if the link supports direct peer DMA (no host staging).
    pub fn supports_peer_dma(self) -> bool {
        !matches!(self, LinkKind::None)
    }
}

// ─── PeerTopology ───────────────────────────────────────────────────────────

/// A symmetric peer-access topology over `device_count` GPUs.
///
/// Internally a dense lower-triangular matrix of [`LinkKind`]; the diagonal is
/// implicitly "self" and never returned as a peer link.
#[derive(Debug, Clone)]
pub struct PeerTopology {
    device_count: usize,
    /// `links[i][j]` is the link between device `i` and `j` (symmetric).
    links: Vec<Vec<LinkKind>>,
}

impl PeerTopology {
    /// Create a topology of `device_count` GPUs with **no** peer links.
    pub fn new(device_count: usize) -> Self {
        let links = vec![vec![LinkKind::None; device_count]; device_count];
        Self {
            device_count,
            links,
        }
    }

    /// Create a fully-connected xGMI topology (e.g. an MI300X 8-GPU node where
    /// every GPU reaches every other over Infinity Fabric).
    pub fn fully_connected_xgmi(device_count: usize) -> Self {
        let mut t = Self::new(device_count);
        for i in 0..device_count {
            for j in 0..device_count {
                if i != j {
                    t.links[i][j] = LinkKind::XGmi;
                }
            }
        }
        t
    }

    /// Number of devices.
    pub fn device_count(&self) -> usize {
        self.device_count
    }

    /// Set the link between `a` and `b` (symmetrically).
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] if either ordinal is out of range or
    /// `a == b`.
    pub fn set_link(&mut self, a: usize, b: usize, kind: LinkKind) -> RocmResult<()> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err(RocmError::InvalidArgument(
                "a device cannot peer with itself".into(),
            ));
        }
        self.links[a][b] = kind;
        self.links[b][a] = kind;
        Ok(())
    }

    /// The link between `a` and `b` (`LinkKind::None` for self or unconnected).
    pub fn link(&self, a: usize, b: usize) -> LinkKind {
        if a == b || a >= self.device_count || b >= self.device_count {
            return LinkKind::None;
        }
        self.links[a][b]
    }

    /// `true` if `a` can directly access `b`'s memory
    /// (`hipDeviceCanAccessPeer`).
    pub fn can_access_peer(&self, a: usize, b: usize) -> bool {
        self.link(a, b).supports_peer_dma()
    }

    fn check(&self, d: usize) -> RocmResult<()> {
        if d >= self.device_count {
            return Err(RocmError::InvalidArgument(format!(
                "device {d} out of range (count {})",
                self.device_count
            )));
        }
        Ok(())
    }

    /// All peers `a` can directly reach, paired with the link kind.
    pub fn peers_of(&self, a: usize) -> Vec<(usize, LinkKind)> {
        if a >= self.device_count {
            return Vec::new();
        }
        (0..self.device_count)
            .filter(|&b| b != a && self.links[a][b].supports_peer_dma())
            .map(|b| (b, self.links[a][b]))
            .collect()
    }

    /// Plan a copy from `src` to `dst`: returns whether it can use direct peer
    /// DMA and the modeled link, or an error describing that host staging is
    /// required.
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] for out-of-range ordinals.
    /// [`RocmError::Unsupported`] when no direct link exists (caller must stage
    /// through host memory).
    pub fn plan_peer_copy(&self, src: usize, dst: usize) -> RocmResult<LinkKind> {
        self.check(src)?;
        self.check(dst)?;
        if src == dst {
            return Err(RocmError::InvalidArgument(
                "peer copy source and destination are the same device".into(),
            ));
        }
        let link = self.links[src][dst];
        if link.supports_peer_dma() {
            Ok(link)
        } else {
            Err(RocmError::Unsupported(format!(
                "no peer link between device {src} and {dst}; host staging required"
            )))
        }
    }

    /// `true` if every pair of devices is reachable by direct peer DMA — i.e.
    /// the peer graph is fully connected.
    pub fn is_fully_connected(&self) -> bool {
        for i in 0..self.device_count {
            for j in 0..self.device_count {
                if i != j && !self.links[i][j].supports_peer_dma() {
                    return false;
                }
            }
        }
        self.device_count > 0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_bandwidth_and_dma() {
        assert_eq!(LinkKind::XGmi.bandwidth_gbps(), 100);
        assert_eq!(LinkKind::Pcie.bandwidth_gbps(), 25);
        assert_eq!(LinkKind::None.bandwidth_gbps(), 0);
        assert!(LinkKind::XGmi.supports_peer_dma());
        assert!(LinkKind::Pcie.supports_peer_dma());
        assert!(!LinkKind::None.supports_peer_dma());
    }

    #[test]
    fn empty_topology_has_no_links() {
        let t = PeerTopology::new(4);
        assert_eq!(t.device_count(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(t.link(i, j), LinkKind::None);
                assert!(!t.can_access_peer(i, j));
            }
        }
        assert!(!t.is_fully_connected());
    }

    #[test]
    fn fully_connected_xgmi_ring() {
        let t = PeerTopology::fully_connected_xgmi(8);
        assert!(t.is_fully_connected());
        for i in 0..8 {
            assert_eq!(t.peers_of(i).len(), 7);
            for (_, kind) in t.peers_of(i) {
                assert_eq!(kind, LinkKind::XGmi);
            }
        }
        // Self is never a peer.
        assert!(!t.can_access_peer(3, 3));
    }

    #[test]
    fn set_link_is_symmetric() {
        let mut t = PeerTopology::new(3);
        t.set_link(0, 2, LinkKind::Pcie).expect("set");
        assert_eq!(t.link(0, 2), LinkKind::Pcie);
        assert_eq!(t.link(2, 0), LinkKind::Pcie);
        assert!(t.can_access_peer(0, 2));
        assert!(!t.can_access_peer(0, 1));
    }

    #[test]
    fn set_link_rejects_self_and_oob() {
        let mut t = PeerTopology::new(2);
        assert!(t.set_link(0, 0, LinkKind::XGmi).is_err());
        assert!(t.set_link(0, 5, LinkKind::XGmi).is_err());
    }

    #[test]
    fn plan_peer_copy_xgmi() {
        let t = PeerTopology::fully_connected_xgmi(4);
        let link = t.plan_peer_copy(0, 3).expect("xgmi copy");
        assert_eq!(link, LinkKind::XGmi);
    }

    #[test]
    fn plan_peer_copy_requires_host_staging() {
        let t = PeerTopology::new(4);
        let err = t.plan_peer_copy(0, 1).unwrap_err();
        assert!(matches!(err, RocmError::Unsupported(_)));
    }

    #[test]
    fn plan_peer_copy_same_device_errors() {
        let t = PeerTopology::fully_connected_xgmi(2);
        let err = t.plan_peer_copy(1, 1).unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn mixed_topology_peers_filtered() {
        // Device 0 has xGMI to 1, PCIe to 2, nothing to 3.
        let mut t = PeerTopology::new(4);
        t.set_link(0, 1, LinkKind::XGmi).unwrap();
        t.set_link(0, 2, LinkKind::Pcie).unwrap();
        let peers = t.peers_of(0);
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&(1, LinkKind::XGmi)));
        assert!(peers.contains(&(2, LinkKind::Pcie)));
        assert!(!t.is_fully_connected());
    }
}
