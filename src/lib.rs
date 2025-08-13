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
//! - **Type flexibility**: Works with any numeric key type
//! - **Zero dependencies**: Core functionality requires no external crates
//!
//! ## Quick Start
//!
//! ```rust
//! use pgm_index::PGMIndex;
//!
//! // Create sorted data
//! let data: Vec<u64> = (0..1_000_000).collect();
//!
//! // Build index with epsilon = 64
//! let index = PGMIndex::new(data, 64);
//!
//! // Fast lookups
//! if let Some(position) = index.get(123456) {
//!     println!("Found at position: {}", position);
//! }
//!
//! // Batch queries for better performance
//! let queries = vec![100, 500, 1000, 5000];
//! let results = index.batch_get(&queries);
//! ```
//!
//! ## Performance
//!
//! The epsilon parameter controls the trade-off between memory usage and query speed:
//! - **Smaller epsilon** → more segments → better predictions → faster queries
//! - **Larger epsilon** → fewer segments → coarser predictions → more memory efficient
//!
//! Recommended epsilon values: 16-128 for most use cases.

pub mod pgm_index;

// Re-export main types
pub use pgm_index::{Key, PGMIndex, Segment};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let data: Vec<u64> = (0..1000).collect();
        let index = PGMIndex::new(data, 32);

        // Test single lookups
        assert_eq!(index.get(0), Some(0));
        assert_eq!(index.get(500), Some(500));
        assert_eq!(index.get(999), Some(999));
        assert_eq!(index.get(1000), None);

        // Test batch lookups
        let queries = vec![10, 100, 500, 999];
        let results = index.batch_get(&queries);
        assert_eq!(results, vec![Some(10), Some(100), Some(500), Some(999)]);
    }

    #[test]
    fn test_different_key_types() {
        // Test with i32
        let data: Vec<i32> = (0..100).collect();
        let index = PGMIndex::new(data, 16);
        assert_eq!(index.get(50), Some(50));

        // Test with u32
        let data: Vec<u32> = (0..100).collect();
        let index = PGMIndex::new(data, 16);
        assert_eq!(index.get(50), Some(50));
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
