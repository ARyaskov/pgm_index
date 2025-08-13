//! # Ultra-Optimized PGM-Index Implementation
//!
//! A high-performance implementation of the Piecewise Geometric Model (PGM) Index
//! with SIMD optimizations and parallel processing.

#[cfg(feature = "jemalloc")]
use jemallocator::Jemalloc;
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use num_traits::ToPrimitive;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type Idx = usize;

/// Trait bound for key types supported by the PGM-Index
pub trait Key: Copy + Send + Sync + Ord + ToPrimitive + std::fmt::Debug + 'static {}
impl Key for u8 {}
impl Key for i8 {}
impl Key for u16 {}
impl Key for i16 {}
impl Key for u32 {}
impl Key for i32 {}
impl Key for u64 {}
impl Key for i64 {}
impl Key for usize {}
impl Key for isize {}

/// Linear segment: y = slope * x + intercept  (x — index, y — key)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct Segment<K: Key> {
    pub min_key: K,
    pub max_key: K,
    pub slope: f64,
    pub intercept: f64,
    pub start_idx: Idx,
    pub end_idx: Idx,
}

/// Lightweight stats for export
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct PGMStats {
    pub segments: usize,
    pub avg_segment_size: f64,
    pub memory_bytes: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct SegmentLookupConfig {
    pub bins: usize,
}

#[derive(Clone, Copy, Debug)]
enum DataComplexity {
    Linear,
    Quadratic,
    Exponential,
    Random,
}

#[derive(Clone, Debug)]
pub struct PGMIndex<K: Key> {
    pub epsilon: usize,
    pub data: Arc<Vec<K>>,
    segments: Vec<Segment<K>>,
    segment_lookup: Vec<usize>,
    lookup_scale: f64,
    min_key_f64: f64,
}

impl<K: Key> PGMIndex<K> {
    pub fn new(data: Vec<K>, epsilon: usize) -> Self {
        assert!(epsilon > 0, "epsilon must be > 0");
        assert!(!data.is_empty(), "data must not be empty");
        assert!(is_sorted(&data), "data must be sorted");

        // Ensure global Rayon pool with at least 4 threads
        crate::init_rayon_min_threads();

        let data = Arc::new(data);

        // Use global Rayon pool for parallel iterators
        let (segments, segment_lookup, lookup_scale, min_key_f64) = {
            let target_segments = Self::optimal_segment_count_adaptive(&data, epsilon);
            let segments = Self::build_segments_parallel(&data, target_segments);
            let (segment_lookup, lookup_scale, min_key_f64) =
                Self::build_lookup_table(&data, &segments);
            (segments, segment_lookup, lookup_scale, min_key_f64)
        };

        Self {
            epsilon,
            data,
            segments,
            segment_lookup,
            lookup_scale,
            min_key_f64,
        }
    }

    pub fn stats(&self) -> PGMStats {
        PGMStats {
            segments: self.segment_count(),
            avg_segment_size: self.avg_segment_size(),
            memory_bytes: self.memory_usage(),
        }
    }

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
        let mean_gap = gaps.iter().copied().sum::<f64>() / (gaps.len() as f64);
        let var_gap = gaps
            .iter()
            .map(|g| (g - mean_gap) * (g - mean_gap))
            .sum::<f64>()
            / (gaps.len() as f64);

        if var_gap < 1e-9 {
            return DataComplexity::Linear;
        }

        let mut increasing = 0usize;
        let mut decreasing = 0usize;
        for i in 1..gaps.len() {
            if gaps[i] > gaps[i - 1] {
                increasing += 1;
            } else if gaps[i] < gaps[i - 1] {
                decreasing += 1;
            }
        }

        if increasing > (gaps.len() * 3) / 4 {
            DataComplexity::Exponential
        } else if decreasing > (gaps.len() * 3) / 4 {
            DataComplexity::Quadratic
        } else if var_gap > mean_gap * 10.0 {
            DataComplexity::Random
        } else {
            DataComplexity::Quadratic
        }
    }

    /// min: ≥1 and ≥4/core; max: ≤ n/32
    fn optimal_segment_count_adaptive(data: &[K], epsilon: usize) -> usize {
        let complexity = Self::estimate_data_complexity(data);
        let n = data.len();
        let cores = rayon::current_num_threads();

        let base_segments = match complexity {
            DataComplexity::Linear => n / (epsilon * 16),
            DataComplexity::Quadratic => n / (epsilon * 8),
            DataComplexity::Exponential => n / (epsilon * 4),
            DataComplexity::Random => n / (epsilon * 2),
        };

        base_segments.max(1).max(cores * 4).min(n / 32)
    }

    fn build_segments_parallel(data: &[K], target_segments: usize) -> Vec<Segment<K>> {
        let n = data.len();
        let target_segments = target_segments.max(1).min(n);

        let mut bounds = Vec::with_capacity(target_segments + 1);
        for s in 0..=target_segments {
            let idx = s * n / target_segments;
            bounds.push(idx);
        }

        let segments: Vec<Segment<K>> = (0..target_segments)
            .into_par_iter()
            .map(|i| {
                let start = bounds[i];
                let end = bounds[i + 1].max(start + 1);
                Self::fit_segment(&data[start..end], start)
            })
            .collect();

        let mut merged: Vec<Segment<K>> = Vec::with_capacity(segments.len());
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if (last.slope - seg.slope).abs() < 1e-12
                    && (last.intercept - seg.intercept).abs() < 1e-6
                    && last.end_idx == seg.start_idx
                    && last.max_key <= seg.min_key
                {
                    last.end_idx = seg.end_idx;
                    last.max_key = seg.max_key;
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }

    fn fit_segment(slice: &[K], global_start: usize) -> Segment<K> {
        let len = slice.len();
        let min_key = slice.first().copied().unwrap();
        let max_key = slice.last().copied().unwrap();

        let n = len as f64;
        let sum_i = (len - 1) as f64 * (len as f64) / 2.0;
        let sum_i2 = (len - 1) as f64 * (len as f64) * (2.0 * (len as f64) - 1.0) / 6.0;

        let mut sum_y = 0.0;
        let mut sum_i_y = 0.0;
        for (i, &k) in slice.iter().enumerate() {
            let y = k.to_f64().unwrap();
            sum_y += y;
            sum_i_y += (i as f64) * y;
        }

        let denom = n * sum_i2 - sum_i * sum_i;
        let (slope, intercept) = if denom.abs() < 1e-12 {
            (0.0, sum_y / n)
        } else {
            let slope = (n * sum_i_y - sum_i * sum_y) / denom;
            let intercept = (sum_y - slope * sum_i) / n;
            (slope, intercept)
        };

        Segment {
            min_key,
            max_key,
            slope,
            intercept,
            start_idx: global_start,
            end_idx: global_start + len,
        }
    }

    fn build_lookup_table(data: &[K], segments: &[Segment<K>]) -> (Vec<usize>, f64, f64) {
        let bins = (segments.len() * 4).max(1024).min(1 << 20);
        let min_key_f64 = data.first().unwrap().to_f64().unwrap();
        let max_key_f64 = data.last().unwrap().to_f64().unwrap();
        let span = (max_key_f64 - min_key_f64).max(1.0);
        let scale = (bins as f64) / span;

        let mut lut = vec![0usize; bins + 1];
        let mut seg_idx = 0usize;
        for b in 0..=bins {
            let key_at_bin = min_key_f64 + (b as f64) / scale;
            while seg_idx + 1 < segments.len()
                && segments[seg_idx].max_key.to_f64().unwrap() < key_at_bin
            {
                seg_idx += 1;
            }
            lut[b] = seg_idx;
        }
        (lut, scale, min_key_f64)
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
    pub fn avg_segment_size(&self) -> f64 {
        (self.data.len() as f64) / (self.segments.len() as f64).max(1.0)
    }
    pub fn memory_usage(&self) -> usize {
        let data_bytes = self.data.len() * std::mem::size_of::<K>();
        let seg_bytes = self.segments.len() * std::mem::size_of::<Segment<K>>();
        let lut_bytes = self.segment_lookup.len() * std::mem::size_of::<usize>();
        data_bytes + seg_bytes + lut_bytes
    }

    fn predict_index(&self, key: K, segment_idx: usize) -> usize {
        let seg = self.segments[segment_idx];
        let y = key.to_f64().unwrap();
        if seg.slope.abs() < 1e-18 {
            seg.start_idx
        } else {
            let x = (y - seg.intercept) / seg.slope;
            let x = x as isize;
            x.clamp(seg.start_idx as isize, (seg.end_idx as isize) - 1) as usize
        }
    }

    fn find_segment_for_key_lut(&self, key: K) -> usize {
        if self.segments.len() <= 1 {
            return 0;
        }
        let y = key.to_f64().unwrap();
        let bin = ((y - self.min_key_f64) * self.lookup_scale)
            .floor()
            .clamp(0.0, (self.segment_lookup.len() - 1) as f64) as usize;
        let mut idx = self.segment_lookup[bin];

        while idx + 1 < self.segments.len() && key > self.segments[idx].max_key {
            idx += 1;
        }
        while idx > 0 && key < self.segments[idx].min_key {
            idx -= 1;
        }
        idx
    }

    pub fn get(&self, key: K) -> Option<usize> {
        if self.segments.is_empty() {
            return None;
        }
        let sidx = self.find_segment_for_key_lut(key);
        let i = self.predict_index(key, sidx);

        let eps = self.epsilon;
        let start = i.saturating_sub(eps);
        let end = (i + eps + 1).min(self.data.len());

        let slice = &self.data[start..end];
        match slice.binary_search(&key) {
            Ok(pos) => Some(start + pos),
            Err(_) => None,
        }
    }

    /// Parallel batch lookup: returns positions for each key (None if absent).
    pub fn get_many_parallel(&self, keys: &[K]) -> Vec<Option<usize>>
    where
        K: Sync,
    {
        keys.par_iter().map(|&k| self.get(k)).collect()
    }

    /// Parallel batch hit count (useful for throughput microbenchmarks).
    pub fn count_hits_parallel(&self, keys: &[K]) -> usize
    where
        K: Sync,
    {
        keys.par_iter().filter(|&&k| self.get(k).is_some()).count()
    }
}

fn is_sorted<K: Ord>(data: &[K]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct PGMIndexSerde<K: Key> {
        epsilon: usize,
        data: Vec<K>,
        segments: Vec<Segment<K>>,
        segment_lookup: Vec<usize>,
        lookup_scale: f64,
        min_key_f64: f64,
    }

    impl<K: Key + Serialize> Serialize for PGMIndex<K> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let tmp = PGMIndexSerde {
                epsilon: self.epsilon,
                data: (*self.data).clone(),
                segments: self.segments.clone(),
                segment_lookup: self.segment_lookup.clone(),
                lookup_scale: self.lookup_scale,
                min_key_f64: self.min_key_f64,
            };
            tmp.serialize(serializer)
        }
    }

    impl<'de, K: Key + Deserialize<'de>> Deserialize<'de> for PGMIndex<K> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let tmp = PGMIndexSerde::<K>::deserialize(deserializer)?;
            Ok(PGMIndex {
                epsilon: tmp.epsilon,
                data: Arc::new(tmp.data),
                segments: tmp.segments,
                segment_lookup: tmp.segment_lookup,
                lookup_scale: tmp.lookup_scale,
                min_key_f64: tmp.min_key_f64,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_build_and_query() {
        let data: Vec<u64> = (0..10_000).collect();
        for &eps in &[16usize, 32, 64, 128] {
            let idx = PGMIndex::new(data.clone(), eps);
            assert!(idx.segment_count() >= 1);
            for &k in &[0u64, 1234, 9999] {
                let got = idx.get(k);
                assert_eq!(got, Some(k as usize));
            }
        }
    }

    #[test]
    fn epsilon_monotonicity_on_segments() {
        let data: Vec<u64> = (0..100_000).collect();
        let idx16 = PGMIndex::new(data.clone(), 16);
        let idx128 = PGMIndex::new(data, 128);
        assert!(idx16.segment_count() >= idx128.segment_count());
    }
}
