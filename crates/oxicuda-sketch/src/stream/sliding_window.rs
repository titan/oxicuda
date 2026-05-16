//! Sliding window count (basic implementation).
//!
//! Maintains a ring-buffer of size `w`. Each `add` records an item; the window
//! contains only the most-recent `w` items.

use crate::error::{SketchError, SketchResult};

/// Sliding window of fixed size `w` storing `u64` items.
#[derive(Debug, Clone)]
pub struct SlidingWindowCount {
    pub w: usize,
    pub buf: Vec<u64>,
    pub head: usize,
    pub filled: usize,
}

impl SlidingWindowCount {
    /// New sliding window of size `w`.
    pub fn new(w: usize) -> SketchResult<Self> {
        if w == 0 {
            return Err(SketchError::InvalidParameter {
                name: "w".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            w,
            buf: vec![0u64; w],
            head: 0,
            filled: 0,
        })
    }

    /// Add an item.
    pub fn add(&mut self, x: u64) {
        self.buf[self.head] = x;
        self.head = (self.head + 1) % self.w;
        if self.filled < self.w {
            self.filled += 1;
        }
    }

    /// Number of items in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Number of occurrences of `x` in the window.
    #[must_use]
    pub fn count(&self, x: u64) -> usize {
        if self.filled == 0 {
            return 0;
        }
        let mut s = 0usize;
        let start = if self.filled < self.w { 0 } else { self.head };
        let mut i = start;
        for _ in 0..self.filled {
            if self.buf[i] == x {
                s += 1;
            }
            i = (i + 1) % self.w;
        }
        s
    }

    /// Vector snapshot of the window (oldest → newest).
    #[must_use]
    pub fn snapshot(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.filled);
        let start = if self.filled < self.w { 0 } else { self.head };
        let mut i = start;
        for _ in 0..self.filled {
            out.push(self.buf[i]);
            i = (i + 1) % self.w;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swc_invalid_w() {
        assert!(SlidingWindowCount::new(0).is_err());
    }

    #[test]
    fn swc_count_correct() {
        let mut sw = SlidingWindowCount::new(5).expect("ok");
        sw.add(1);
        sw.add(2);
        sw.add(1);
        assert_eq!(sw.count(1), 2);
        assert_eq!(sw.count(2), 1);
        assert_eq!(sw.count(3), 0);
    }

    #[test]
    fn swc_oldest_evicted() {
        let mut sw = SlidingWindowCount::new(3).expect("ok");
        sw.add(1);
        sw.add(2);
        sw.add(3);
        sw.add(4);
        assert_eq!(sw.count(1), 0);
        assert_eq!(sw.snapshot(), vec![2, 3, 4]);
    }
}
