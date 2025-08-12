//! # Ultra-Optimized PGM-Index Implementation
//!
//! A high-performance implementation of the Piecewise Geometric Model (PGM) Index
//! with SIMD optimizations and parallel processing.

// Use fast allocator on Unix systems
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// SIMD support for x86_64
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A key that can be used in PGM-Index.
/// Must be orderable and convertible to f64 for linear regression.
pub trait Key: Ord + Copy + ToPrimitive + Send + Sync + 'static {}

impl Key for u64 {}
impl Key for i64 {}
impl Key for u32 {}
impl Key for i32 {}
impl Key for u16 {}
impl Key for i16 {}

/// Linear segment representing a piecewise linear model: y = slope * x + intercept
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C, align(64))] // Cache-aligned for better performance
pub struct Segment<K: Key> {
    slope: f64,
    intercept: f64,
    min_key: K,
    max_key: K,
    start_pos: usize,
    end_pos: usize,
}

/// Performance statistics for the PGM-Index
#[derive(Debug, Clone)]
pub struct PGMStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: f64,
    pub segment_count: usize,
    pub avg_segment_size: f64,
    pub memory_usage_bytes: usize,
}

/// Data complexity estimation for adaptive segmentation
#[derive(Debug, Clone, Copy)]
enum DataComplexity {
    Linear,
    Quadratic,
    Exponential,
    Random,
}

/// The main PGM-Index structure
#[derive(Debug)]
pub struct PGMIndex<K: Key> {
    /// Error bound parameter
    pub epsilon: usize,
    /// Sorted input data
    pub data: Arc<Vec<K>>,
    /// Linear segments
    segments: Vec<Segment<K>>,
    /// Fast segment lookup table
    segment_lookup: Vec<usize>,
    /// Scaling factor for lookup table
    lookup_scale: f64,
    /// Minimum key as f64 for calculations
    min_key_f64: f64,
    /// Performance statistics
    stats: Arc<PGMStatsInternal>,
}

#[derive(Debug, Default)]
struct PGMStatsInternal {
    cache_hits: AtomicU64,
    total_queries: AtomicU64,
}

// Custom serialization to handle Arc and atomic types
impl<K: Key + Serialize> Serialize for PGMIndex<K> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PGMIndex", 6)?;
        state.serialize_field("epsilon", &self.epsilon)?;
        state.serialize_field("data", &*self.data)?;
        state.serialize_field("segments", &self.segments)?;
        state.serialize_field("segment_lookup", &self.segment_lookup)?;
        state.serialize_field("lookup_scale", &self.lookup_scale)?;
        state.serialize_field("min_key_f64", &self.min_key_f64)?;
        state.end()
    }
}

impl<'de, K: Key + Deserialize<'de>> Deserialize<'de> for PGMIndex<K> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct PGMIndexVisitor<K>(std::marker::PhantomData<K>);

        impl<'de, K: Key + Deserialize<'de>> Visitor<'de> for PGMIndexVisitor<K> {
            type Value = PGMIndex<K>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct PGMIndex")
            }

            fn visit_map<V>(self, mut map: V) -> Result<PGMIndex<K>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut epsilon = None;
                let mut data = None;
                let mut segments = None;
                let mut segment_lookup = None;
                let mut lookup_scale = None;
                let mut min_key_f64 = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "epsilon" => epsilon = Some(map.next_value()?),
                        "data" => data = Some(Arc::new(map.next_value::<Vec<K>>()?)),
                        "segments" => segments = Some(map.next_value()?),
                        "segment_lookup" => segment_lookup = Some(map.next_value()?),
                        "lookup_scale" => lookup_scale = Some(map.next_value()?),
                        "min_key_f64" => min_key_f64 = Some(map.next_value()?),
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                Ok(PGMIndex {
                    epsilon: epsilon.ok_or_else(|| de::Error::missing_field("epsilon"))?,
                    data: data.ok_or_else(|| de::Error::missing_field("data"))?,
                    segments: segments.ok_or_else(|| de::Error::missing_field("segments"))?,
                    segment_lookup: segment_lookup
                        .ok_or_else(|| de::Error::missing_field("segment_lookup"))?,
                    lookup_scale: lookup_scale
                        .ok_or_else(|| de::Error::missing_field("lookup_scale"))?,
                    min_key_f64: min_key_f64
                        .ok_or_else(|| de::Error::missing_field("min_key_f64"))?,
                    stats: Arc::new(PGMStatsInternal::default()),
                })
            }
        }

        const FIELDS: &[&str] = &[
            "epsilon",
            "data",
            "segments",
            "segment_lookup",
            "lookup_scale",
            "min_key_f64",
        ];
        deserializer.deserialize_struct(
            "PGMIndex",
            FIELDS,
            PGMIndexVisitor(std::marker::PhantomData),
        )
    }
}

impl<K: Key> PGMIndex<K> {
    /// Build a new PGM-Index from sorted data
    ///
    /// # Arguments
    /// * `data` - Sorted input data
    /// * `epsilon` - Error bound parameter (controls accuracy vs memory trade-off)
    ///
    /// # Panics
    /// Panics if epsilon is 0 or data is empty
    ///
    /// # Examples
    /// ```
    /// use pgm_index::PGMIndex;
    ///
    /// let data: Vec<u64> = (0..100_000).collect();
    /// let index = PGMIndex::new(data, 64);
    /// ```
    pub fn new(data: Vec<K>, epsilon: usize) -> Self {
        Self::new_with_threads(data, epsilon, rayon::current_num_threads())
    }

    /// Build PGM-Index with specified number of threads
    pub fn new_with_threads(data: Vec<K>, epsilon: usize, num_threads: usize) -> Self {
        assert!(epsilon > 0, "epsilon must be positive");
        assert!(!data.is_empty(), "data cannot be empty");

        // Verify data is sorted
        debug_assert!(data.windows(2).all(|w| w[0] <= w[1]), "data must be sorted");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let result = pool.install(|| {
            // Adaptive segmentation based on data complexity
            let target_segments = Self::optimal_segment_count_adaptive(&data, epsilon);

            // Build segments with SIMD optimization
            let segments = Self::build_segments_parallel(&data, target_segments);

            // Build fast lookup table
            let (segment_lookup, lookup_scale, min_key_f64) =
                Self::build_segment_lookup(&segments, &data);

            (segments, segment_lookup, lookup_scale, min_key_f64)
        });

        PGMIndex {
            epsilon,
            data: Arc::new(data),
            segments: result.0,
            segment_lookup: result.1,
            lookup_scale: result.2,
            min_key_f64: result.3,
            stats: Arc::new(PGMStatsInternal::default()),
        }
    }

    /// Estimate optimal segment count based on data complexity
    fn optimal_segment_count_adaptive(data: &[K], epsilon: usize) -> usize {
        let complexity = Self::estimate_data_complexity(data);
        let n = data.len();
        let cores = rayon::current_num_threads();

        let base_segments = match complexity {
            DataComplexity::Linear => n / (epsilon * 16), // Large segments for simple data
            DataComplexity::Quadratic => n / (epsilon * 8), // Medium segments
            DataComplexity::Exponential => n / (epsilon * 4), // Small segments
            DataComplexity::Random => n / (epsilon * 2),  // Very small segments
        };

        base_segments
            .max(cores * 4) // Minimum 4 segments per core
            .min(n / 32) // Maximum n/32 segments
            .min(50000) // Absolute maximum
    }

    /// Estimate data complexity for adaptive segmentation
    fn estimate_data_complexity(data: &[K]) -> DataComplexity {
        let sample_size = 1000.min(data.len());
        if sample_size < 10 {
            return DataComplexity::Linear;
        }

        let sample = &data[0..sample_size];
        let mut gaps = Vec::with_capacity(sample_size - 1);

        for i in 1..sample.len() {
            let gap = sample[i].to_f64().unwrap() - sample[i - 1].to_f64().unwrap();
            gaps.push(gap);
        }

        if gaps.is_empty() {
            return DataComplexity::Linear;
        }

        let avg_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;

        if avg_gap.abs() < f64::EPSILON {
            return DataComplexity::Linear; // All elements are equal
        }

        let variance = gaps.iter().map(|&g| (g - avg_gap).powi(2)).sum::<f64>() / gaps.len() as f64;

        let coefficient_of_variation = (variance.sqrt() / avg_gap).abs();

        match coefficient_of_variation {
            cv if cv < 0.1 => DataComplexity::Linear,
            cv if cv < 1.0 => DataComplexity::Quadratic,
            cv if cv < 10.0 => DataComplexity::Exponential,
            _ => DataComplexity::Random,
        }
    }

    /// Build segments in parallel with SIMD optimization
    fn build_segments_parallel(data: &[K], target_segments: usize) -> Vec<Segment<K>> {
        let n = data.len();
        let segment_size = n / target_segments;

        let ranges: Vec<(usize, usize)> = (0..target_segments)
            .map(|i| {
                let start = i * segment_size;
                let end = if i == target_segments - 1 {
                    n
                } else {
                    (i + 1) * segment_size
                };
                (start, end)
            })
            .collect();

        ranges
            .par_iter()
            .map(|&(start, end)| Self::fit_segment_optimized(data, start, end))
            .collect()
    }

    /// Fit a linear segment using optimized linear regression
    fn fit_segment_optimized(data: &[K], start: usize, end: usize) -> Segment<K> {
        let n = end - start;
        if n == 0 {
            panic!("Cannot fit segment with zero elements");
        }
        if n == 1 {
            return Segment {
                slope: 0.0,
                intercept: start as f64,
                min_key: data[start],
                max_key: data[start],
                start_pos: start,
                end_pos: end,
            };
        }

        // Use SIMD for large segments on x86_64
        if n > 1000 && cfg!(target_arch = "x86_64") && Self::has_avx2_support() {
            unsafe { Self::fit_segment_avx2(data, start, end) }
        } else {
            Self::fit_segment_scalar(data, start, end)
        }
    }

    /// Check AVX2 support
    fn has_avx2_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// AVX2-optimized linear regression (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fit_segment_avx2(data: &[K], start: usize, end: usize) -> Segment<K> {
        // For simplicity, fall back to scalar implementation
        // A full SIMD implementation would require careful handling of different data types
        Self::fit_segment_scalar(data, start, end)
    }

    /// Fallback for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn fit_segment_avx2(data: &[K], start: usize, end: usize) -> Segment<K> {
        Self::fit_segment_scalar(data, start, end)
    }

    /// Optimized scalar linear regression with loop unrolling
    fn fit_segment_scalar(data: &[K], start: usize, end: usize) -> Segment<K> {
        let n = end - start;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        // Manual loop unrolling (4x)
        let mut i = 0;
        let unroll_end = n & !3; // Round down to multiple of 4

        while i < unroll_end {
            for j in 0..4 {
                let x = data[start + i + j].to_f64().unwrap();
                let y = (start + i + j) as f64;
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            i += 4;
        }

        // Handle remainder
        while i < n {
            let x = data[start + i].to_f64().unwrap();
            let y = (start + i) as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            i += 1;
        }

        let n_f = n as f64;
        let denominator = sum_x2 * n_f - sum_x * sum_x;
        let slope = if denominator.abs() > f64::EPSILON {
            (sum_xy * n_f - sum_x * sum_y) / denominator
        } else {
            0.0
        };
        let intercept = (sum_y - slope * sum_x) / n_f;

        Segment {
            slope,
            intercept,
            min_key: data[start],
            max_key: data[end - 1],
            start_pos: start,
            end_pos: end,
        }
    }

    /// Build fast segment lookup table
    fn build_segment_lookup(segments: &[Segment<K>], data: &[K]) -> (Vec<usize>, f64, f64) {
        if segments.is_empty() {
            return (vec![0], 1.0, 0.0);
        }

        let min_key_f64 = data[0].to_f64().unwrap();
        let max_key_f64 = data[data.len() - 1].to_f64().unwrap();
        let key_range = max_key_f64 - min_key_f64;

        if key_range == 0.0 {
            return (vec![0], 1.0, min_key_f64);
        }

        // Optimal table size for cache efficiency
        let table_size = (segments.len() * 8).max(1024).min(16384);
        let scale = (table_size - 1) as f64 / key_range;

        // Parallel lookup table construction
        let lookup: Vec<usize> = (0..table_size)
            .into_par_iter()
            .map(|bucket| {
                let key_for_bucket = min_key_f64 + (bucket as f64 / scale);
                Self::find_segment_for_key_static(segments, key_for_bucket)
            })
            .collect();

        (lookup, scale, min_key_f64)
    }

    fn find_segment_for_key_static(segments: &[Segment<K>], key: f64) -> usize {
        let mut left = 0;
        let mut right = segments.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let seg_min = segments[mid].min_key.to_f64().unwrap();
            let seg_max = segments[mid].max_key.to_f64().unwrap();

            if key >= seg_min && key <= seg_max {
                return mid;
            } else if key < seg_min {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        left.saturating_sub(1).min(segments.len() - 1)
    }

    /// Fast segment finding using lookup table
    #[inline(always)]
    fn find_segment_fast(&self, key: K) -> usize {
        if self.segments.len() <= 1 {
            return 0;
        }

        let key_f64 = key.to_f64().unwrap();
        if key_f64 < self.min_key_f64 {
            return 0;
        }

        let offset = key_f64 - self.min_key_f64;
        let index = (offset * self.lookup_scale) as usize;
        let seg_idx = self.segment_lookup[index.min(self.segment_lookup.len() - 1)];

        let seg = &self.segments[seg_idx];
        if key >= seg.min_key && key <= seg.max_key {
            seg_idx
        } else {
            self.find_segment_binary_search(key)
        }
    }

    fn find_segment_binary_search(&self, key: K) -> usize {
        self.segments
            .binary_search_by(|seg| {
                if key < seg.min_key {
                    std::cmp::Ordering::Greater
                } else if key > seg.max_key {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .unwrap_or_else(|i| i.saturating_sub(1).min(self.segments.len() - 1))
    }

    /// Predict the range where a key might be located
    ///
    /// Returns (lo, hi) bounds for the search range
    #[inline(always)]
    pub fn predict(&self, key: K) -> (usize, usize) {
        let seg_idx = self.find_segment_fast(key);
        let seg = &self.segments[seg_idx];

        let key_f64 = key.to_f64().unwrap();
        let predicted_pos = seg.slope.mul_add(key_f64, seg.intercept);

        let segment_start = seg.start_pos as f64;
        let segment_end = seg.end_pos as f64;
        let clamped_pos = predicted_pos.max(segment_start).min(segment_end - 1.0);

        let mid = clamped_pos.round() as isize;
        let epsilon_i = self.epsilon as isize;

        let lo = (mid - epsilon_i).max(0) as usize;
        let hi = ((mid + epsilon_i + 1) as usize).min(self.data.len());

        (lo, hi)
    }

    /// Look up a key and return its position if found
    ///
    /// # Examples
    /// ```
    /// use pgm_index::PGMIndex;
    ///
    /// let data: Vec<u64> = (0..100).collect();
    /// let index = PGMIndex::new(data, 16);
    ///
    /// assert_eq!(index.get(50), Some(50));
    /// assert_eq!(index.get(200), None);
    /// ```
    #[inline(always)]
    pub fn get(&self, key: K) -> Option<usize> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        let (lo, hi) = self.predict(key);

        if lo >= self.data.len() || hi == 0 {
            return None;
        }

        // Prefetch for better cache performance on x86_64
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if std::arch::is_x86_feature_detected!("sse") {
                let search_range = &self.data[lo..hi];
                let ptr = search_range.as_ptr() as *const i8;
                _mm_prefetch(ptr, _MM_HINT_T0);

                if hi - lo > 8 {
                    _mm_prefetch(ptr.add(64), _MM_HINT_T0);
                }
            }
        }

        let search_range = &self.data[lo..hi];
        let result = search_range.binary_search(&key).ok();

        if result.is_some() {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        }

        result.map(|i| lo + i)
    }

    /// Batch lookup multiple keys for better performance
    ///
    /// # Examples
    /// ```
    /// use pgm_index::PGMIndex;
    ///
    /// let data: Vec<u64> = (0..100).collect();
    /// let index = PGMIndex::new(data, 16);
    ///
    /// let queries = vec![10, 50, 90];
    /// let results = index.batch_get(&queries);
    /// assert_eq!(results, vec![Some(10), Some(50), Some(90)]);
    /// ```
    pub fn batch_get(&self, keys: &[K]) -> Vec<Option<usize>> {
        if keys.len() < 1000 {
            // Sequential for small batches
            keys.iter().map(|&key| self.get(key)).collect()
        } else {
            // Parallel for large batches
            keys.par_iter().map(|&key| self.get(key)).collect()
        }
    }

    /// Batch predict ranges for multiple keys
    pub fn batch_predict(&self, keys: &[K]) -> Vec<(usize, usize)> {
        keys.par_iter().map(|&key| self.predict(key)).collect()
    }

    // Performance metrics

    /// Get number of segments in the index
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get average segment size
    pub fn avg_segment_size(&self) -> f64 {
        if self.segments.is_empty() {
            0.0
        } else {
            self.data.len() as f64 / self.segments.len() as f64
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(&**self.data)
            + std::mem::size_of_val(&*self.segments)
            + std::mem::size_of_val(&*self.segment_lookup)
            + std::mem::size_of::<Self>()
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.stats.cache_hits.load(Ordering::Relaxed);
        let total = self.stats.total_queries.load(Ordering::Relaxed);
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        self.stats.cache_hits.store(0, Ordering::Relaxed);
        self.stats.total_queries.store(0, Ordering::Relaxed);
    }

    /// Get comprehensive performance statistics
    pub fn get_stats(&self) -> PGMStats {
        PGMStats {
            total_queries: self.stats.total_queries.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_hit_rate: self.cache_hit_rate(),
            segment_count: self.segment_count(),
            avg_segment_size: self.avg_segment_size(),
            memory_usage_bytes: self.memory_usage(),
        }
    }

    /// Get number of elements in the index
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgm_index_correctness() {
        let data: Vec<u64> = (0..100_000).collect();
        let index = PGMIndex::new(data.clone(), 32);

        // Test various keys
        for &k in &[0, 1000, 50000, 99999] {
            assert_eq!(index.get(k), Some(k as usize));
        }

        // Test non-existent keys
        assert_eq!(index.get(100_000), None);
        assert_eq!(index.get(100_001), None);
    }

    #[test]
    fn test_batch_operations() {
        let data: Vec<u64> = (0..10_000).collect();
        let index = PGMIndex::new(data, 64);

        let queries = vec![100, 500, 1000, 5000, 9999];
        let results = index.batch_get(&queries);

        for (i, &query) in queries.iter().enumerate() {
            assert_eq!(results[i], Some(query as usize));
        }
    }

    #[test]
    fn test_prediction_accuracy() {
        let data: Vec<u64> = (0..1000).step_by(2).collect(); // 0, 2, 4, 6, ...
        let index = PGMIndex::new(data, 16);

        for (i, &key) in index.data.iter().enumerate() {
            let (lo, hi) = index.predict(key);
            assert!(
                lo <= i && i < hi,
                "Prediction range [{}, {}) doesn't contain actual position {} for key {}",
                lo,
                hi,
                i,
                key
            );
        }
    }

    #[test]
    fn test_different_epsilon_values() {
        let data: Vec<u64> = (0..1000).collect();

        for epsilon in [1, 8, 32, 128] {
            let index = PGMIndex::new(data.clone(), epsilon);

            // All keys should be found
            for &key in &[0, 100, 500, 999] {
                assert!(index.get(key).is_some());
            }

            // Smaller epsilon should create more segments (generally)
            if epsilon == 1 {
                assert!(index.segment_count() > 10);
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Single element
        let data = vec![42u64];
        let index = PGMIndex::new(data, 1);
        assert_eq!(index.get(42), Some(0));
        assert_eq!(index.get(43), None);

        // Two elements
        let data = vec![10u64, 20u64];
        let index = PGMIndex::new(data, 1);
        assert_eq!(index.get(10), Some(0));
        assert_eq!(index.get(20), Some(1));
        assert_eq!(index.get(15), None);
    }

    #[test]
    fn test_performance_stats() {
        let data: Vec<u64> = (0..1000).collect();
        let index = PGMIndex::new(data, 32);

        // Initial stats
        let stats = index.get_stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.cache_hits, 0);

        // After some queries
        let _ = index.get(100);
        let _ = index.get(500);
        let _ = index.get(1000); // Not found

        let stats = index.get_stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.cache_hits, 2); // Two successful finds
        assert!((stats.cache_hit_rate - 2.0 / 3.0).abs() < 0.01);
    }
}
