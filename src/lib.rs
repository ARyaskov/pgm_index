//! # PGM-Index: Ultra-Fast Learned Index
//!
//! A high-performance implementation of the Piecewise Geometric Model (PGM) Index,
//! a learned data structure for fast lookups in sorted arrays.
//!
//! ## Overview
//!
//! The PGM-Index builds a piecewise linear model of your sorted data and uses it
//! to predict element positions, significantly outperforming binary search for
//! large datasets.
//!
//! ## Features
//!
//! - **Ultra-fast lookups**: Often 3-10x faster than binary search
//! - **Memory efficient**: Low memory overhead per element
//! - **Parallel processing**: SIMD-optimized with multi-threading support
//!
//! ## Usage
//!
//! ```rust
//! use pgm_index::PGMIndex;
//!
//! fn main() {
//!     let data: Vec<u64> = (0..1_000_000).collect();
//!     let pgm = PGMIndex::new(data, 32);
//!     assert_eq!(pgm.get(123_456), Some(123_456));
//! }
//! ```
//!
//! ## Notes
//! - Keys must be sorted ascending.
//!
//! The epsilon parameter controls the trade-off between memory usage and query speed:
//! - **Smaller epsilon** → more segments → better predictions → faster queries
//! - **Larger epsilon** → fewer segments → coarser predictions → more memory efficient
//!
//! Recommended epsilon values: 16-128 for most use cases.

mod pgm_index;

// Re-export main types
pub use pgm_index::{Key, PGMIndex, PGMStats, Segment};

// --- rayon global pool init: at least 4 threads ---
use std::sync::Once;
static RAYON_INIT: Once = Once::new();

/// Ensure Rayon global thread-pool is initialized with at least 4 threads.
pub fn init_rayon_min_threads() {
    RAYON_INIT.call_once(|| {
        let n = std::cmp::max(4, num_cpus::get());
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let data: Vec<u64> = (0..1000).collect();
        let index = PGMIndex::new(data, 32);
        assert_eq!(index.get(123), Some(123));
        assert_eq!(index.get(999), Some(999));
        assert_eq!(index.get(1000), None);
    }

    #[test]
    fn test_edge_cases() {
        // Single element
        let data = vec![42u64];
        let index = PGMIndex::new(data, 1);
        assert_eq!(index.get(42), Some(0));
        assert_eq!(index.get(41), None);

        // Two elements
        let data = vec![10u64, 20u64];
        let index = PGMIndex::new(data, 1);
        assert_eq!(index.get(10), Some(0));
        assert_eq!(index.get(20), Some(1));
    }
}
