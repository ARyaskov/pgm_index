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

pub(crate) trait WindowSearch: Key {
    fn search_window(slice: &[Self], key: Self) -> Option<usize>;
}

impl WindowSearch for u64 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_simd_u64(slice, key)
    }
}

impl WindowSearch for u32 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_simd_u32(slice, key)
    }
}

impl WindowSearch for u8 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for i8 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for u16 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for i16 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for i32 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for i64 {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for usize {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}
impl WindowSearch for isize {
    #[inline(always)]
    fn search_window(slice: &[Self], key: Self) -> Option<usize> {
        search_window_scalar_unrolled(slice, key)
    }
}

/// Linear segment: y = slope * x + intercept  (x — key, y — index)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
#[repr(C)]
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
    pub hits_in_window_queries: usize,
    pub window_probes_total: usize,
    pub fallback_used: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct SegmentLookupConfig {
    pub bins: usize,
}

#[derive(Debug)]
pub struct PGMIndex<K: Key> {
    pub epsilon: usize,
    pub data: Arc<[K]>,
    #[cfg(any(feature = "serde", debug_assertions))]
    segments: Vec<Segment<K>>,
    segment_min_key: Vec<K>,
    segment_max_key: Vec<K>,
    segment_slope: Vec<f64>,
    segment_intercept: Vec<f64>,
    segment_start_idx: Vec<Idx>,
    segment_end_idx: Vec<Idx>,
    segment_max_err: Vec<usize>,
    segment_lookup: Vec<usize>,
    lookup_scale: f64,
    min_key_i128: i128,
    key_scale_shift: u32,
    #[cfg(feature = "metrics")]
    metrics: std::sync::Mutex<MetricsTotals>,
}

impl<K: Key> Clone for PGMIndex<K> {
    fn clone(&self) -> Self {
        Self {
            epsilon: self.epsilon,
            data: self.data.clone(),
            #[cfg(any(feature = "serde", debug_assertions))]
            segments: self.segments.clone(),
            segment_min_key: self.segment_min_key.clone(),
            segment_max_key: self.segment_max_key.clone(),
            segment_slope: self.segment_slope.clone(),
            segment_intercept: self.segment_intercept.clone(),
            segment_start_idx: self.segment_start_idx.clone(),
            segment_end_idx: self.segment_end_idx.clone(),
            segment_max_err: self.segment_max_err.clone(),
            segment_lookup: self.segment_lookup.clone(),
            lookup_scale: self.lookup_scale,
            min_key_i128: self.min_key_i128,
            key_scale_shift: self.key_scale_shift,
            #[cfg(feature = "metrics")]
            metrics: std::sync::Mutex::new(self.metrics.lock().unwrap().clone()),
        }
    }
}

#[allow(private_bounds)]
impl<K: Key + WindowSearch> PGMIndex<K> {
    pub fn new(data: Vec<K>, epsilon: usize) -> Self {
        assert!(epsilon > 0, "epsilon must be > 0");
        assert!(!data.is_empty(), "data must not be empty");
        assert!(is_sorted(&data), "data must be sorted");

        // Ensure global Rayon pool with at least 4 threads
        crate::init_rayon_min_threads();

        let data: Arc<[K]> = Arc::from(data.into_boxed_slice());

        // Use global Rayon pool for parallel iterators
        let (
            segments,
            segment_max_err,
            segment_lookup,
            lookup_scale,
            min_key_i128,
            key_scale_shift,
        ) = {
            let min_key_i128 = Self::key_to_i128(data.first().copied().unwrap());
            let max_key_i128 = Self::key_to_i128(data.last().copied().unwrap());
            let key_scale_shift = Self::compute_scale_shift(min_key_i128, max_key_i128);
            let (segments, segment_max_err) =
                Self::build_segments_pgm(&data, epsilon, min_key_i128, key_scale_shift);
            let (segment_lookup, lookup_scale) =
                Self::build_lookup_table(&data, &segments, min_key_i128, key_scale_shift);
            (
                segments,
                segment_max_err,
                segment_lookup,
                lookup_scale,
                min_key_i128,
                key_scale_shift,
            )
        };

        let (
            segment_min_key,
            segment_max_key,
            segment_slope,
            segment_intercept,
            segment_start_idx,
            segment_end_idx,
        ) = Self::segment_soa_from(&segments);

        let index = Self {
            epsilon,
            data,
            #[cfg(any(feature = "serde", debug_assertions))]
            segments,
            segment_min_key,
            segment_max_key,
            segment_slope,
            segment_intercept,
            segment_start_idx,
            segment_end_idx,
            segment_max_err,
            segment_lookup,
            lookup_scale,
            min_key_i128,
            key_scale_shift,
            #[cfg(feature = "metrics")]
            metrics: std::sync::Mutex::new(MetricsTotals::default()),
        };

        #[cfg(debug_assertions)]
        index.verify_segments_debug();

        index
    }

    pub fn stats(&self) -> PGMStats {
        #[cfg(feature = "metrics")]
        let totals = self.metrics.lock().unwrap().clone();
        #[cfg(not(feature = "metrics"))]
        let totals = MetricsTotals::default();
        PGMStats {
            segments: self.segment_count(),
            avg_segment_size: self.avg_segment_size(),
            memory_bytes: self.memory_usage(),
            hits_in_window_queries: totals.hits_in_window_queries,
            window_probes_total: totals.window_probes_total,
            fallback_used: totals.fallback_used,
        }
    }

    const PAR_ERR_THRESHOLD: usize = 50_000;

    fn compute_scale_shift(min_key_i128: i128, max_key_i128: i128) -> u32 {
        let range = (max_key_i128 - min_key_i128).abs() as u128;
        if range == 0 {
            return 0;
        }
        let bits = 128 - range.leading_zeros();
        if bits <= 52 { 0 } else { bits - 52 }
    }

    fn build_segments_pgm(
        data: &[K],
        epsilon: usize,
        min_key_i128: i128,
        key_scale_shift: u32,
    ) -> (Vec<Segment<K>>, Vec<usize>) {
        Self::build_segments_pgm_range(data, 0, data.len(), epsilon, min_key_i128, key_scale_shift)
    }

    fn build_segments_pgm_range(
        data: &[K],
        start: usize,
        end: usize,
        epsilon: usize,
        min_key_i128: i128,
        key_scale_shift: u32,
    ) -> (Vec<Segment<K>>, Vec<usize>) {
        let mut segments = Vec::new();
        let mut max_errs = Vec::new();

        if start >= end {
            return (segments, max_errs);
        }

        let eps_f = epsilon as f64;
        let mut seg_start = start;
        let mut x0 = Self::key_delta_f64_with_min(data[seg_start], min_key_i128, key_scale_shift);
        let mut y0 = seg_start as f64;
        let mut slope_low = f64::NEG_INFINITY;
        let mut slope_high = f64::INFINITY;
        let mut i = seg_start + 1;

        while i < end {
            let mut violate = false;
            let x = Self::key_delta_f64_with_min(data[i], min_key_i128, key_scale_shift);
            let y = i as f64;

            let prev_low = slope_low;
            let prev_high = slope_high;

            if x == x0 {
                if (y - y0).abs() > eps_f {
                    violate = true;
                }
            } else {
                let dx = x - x0;
                let low = (y - eps_f - y0) / dx;
                let high = (y + eps_f - y0) / dx;
                slope_low = slope_low.max(low);
                slope_high = slope_high.min(high);
                if slope_low > slope_high {
                    violate = true;
                    slope_low = prev_low;
                    slope_high = prev_high;
                }
            }

            if violate {
                let seg_end = i - 1;
                let (seg, max_err) = Self::finalize_segment(
                    data,
                    seg_start,
                    seg_end,
                    slope_low,
                    slope_high,
                    min_key_i128,
                    key_scale_shift,
                );
                segments.push(seg);
                max_errs.push(max_err);

                if seg_end == seg_start {
                    seg_start = i;
                    if seg_start >= end {
                        break;
                    }
                    x0 = Self::key_delta_f64_with_min(
                        data[seg_start],
                        min_key_i128,
                        key_scale_shift,
                    );
                    y0 = seg_start as f64;
                    slope_low = f64::NEG_INFINITY;
                    slope_high = f64::INFINITY;
                    i = seg_start + 1;
                    continue;
                } else {
                    seg_start = seg_end;
                    x0 = Self::key_delta_f64_with_min(
                        data[seg_start],
                        min_key_i128,
                        key_scale_shift,
                    );
                    y0 = seg_start as f64;
                    slope_low = f64::NEG_INFINITY;
                    slope_high = f64::INFINITY;
                    // Re-evaluate current point with the new segment start.
                    continue;
                }
            }

            i += 1;
        }

        let (seg, max_err) = Self::finalize_segment(
            data,
            seg_start,
            end - 1,
            slope_low,
            slope_high,
            min_key_i128,
            key_scale_shift,
        );
        segments.push(seg);
        max_errs.push(max_err);

        (segments, max_errs)
    }

    fn finalize_segment(
        data: &[K],
        start: usize,
        end: usize,
        slope_low: f64,
        slope_high: f64,
        min_key_i128: i128,
        key_scale_shift: u32,
    ) -> (Segment<K>, usize) {
        let min_key = data[start];
        let max_key = data[end];
        let x0 = Self::key_delta_f64_with_min(min_key, min_key_i128, key_scale_shift);
        let y0 = start as f64;
        let x1 = Self::key_delta_f64_with_min(max_key, min_key_i128, key_scale_shift);
        let y1 = end as f64;

        let slope = if slope_low.is_finite() && slope_high.is_finite() {
            (slope_low + slope_high) * 0.5
        } else if x1 != x0 {
            (y1 - y0) / (x1 - x0)
        } else {
            0.0
        };
        let intercept = y0 - slope * x0;

        let len = end - start + 1;
        let max_err = if len >= Self::PAR_ERR_THRESHOLD {
            (start..=end)
                .into_par_iter()
                .map(|idx| {
                    let x = Self::key_delta_f64_with_min(data[idx], min_key_i128, key_scale_shift);
                    let pred = slope * x + intercept;
                    (pred - (idx as f64)).abs().ceil() as usize
                })
                .max()
                .unwrap_or(0)
        } else {
            let mut max_err = 0usize;
            for idx in start..=end {
                let x = Self::key_delta_f64_with_min(data[idx], min_key_i128, key_scale_shift);
                let pred = slope * x + intercept;
                let err = (pred - (idx as f64)).abs();
                max_err = max_err.max(err.ceil() as usize);
            }
            max_err
        };

        let seg = Segment {
            min_key,
            max_key,
            slope,
            intercept,
            start_idx: start,
            end_idx: end + 1,
        };
        (seg, max_err)
    }

    fn build_lookup_table(
        data: &[K],
        segments: &[Segment<K>],
        min_key_i128: i128,
        key_scale_shift: u32,
    ) -> (Vec<usize>, f64) {
        let bins = (segments.len() * 4).max(1024).min(1 << 20);
        let min_key_f64 = Self::key_delta_f64_with_min(
            data.first().copied().unwrap(),
            min_key_i128,
            key_scale_shift,
        );
        let max_key_f64 = Self::key_delta_f64_with_min(
            data.last().copied().unwrap(),
            min_key_i128,
            key_scale_shift,
        );
        let span = (max_key_f64 - min_key_f64).max(1.0);
        let scale = (bins as f64) / span;

        let mut lut = vec![0usize; bins + 1];
        let mut seg_idx = 0usize;
        for b in 0..=bins {
            let key_at_bin = min_key_f64 + (b as f64) / scale;
            while seg_idx + 1 < segments.len() {
                let max_key_f64 = Self::key_delta_f64_with_min(
                    segments[seg_idx].max_key,
                    min_key_i128,
                    key_scale_shift,
                );
                if max_key_f64 < key_at_bin {
                    seg_idx += 1;
                    continue;
                }
                break;
            }
            lut[b] = seg_idx;
        }
        (lut, scale)
    }

    fn segment_soa_from(
        segs: &[Segment<K>],
    ) -> (Vec<K>, Vec<K>, Vec<f64>, Vec<f64>, Vec<Idx>, Vec<Idx>) {
        let n = segs.len();
        let mut segment_min_key = Vec::with_capacity(n);
        let mut segment_max_key = Vec::with_capacity(n);
        let mut segment_slope = Vec::with_capacity(n);
        let mut segment_intercept = Vec::with_capacity(n);
        let mut segment_start_idx = Vec::with_capacity(n);
        let mut segment_end_idx = Vec::with_capacity(n);
        for seg in segs {
            segment_min_key.push(seg.min_key);
            segment_max_key.push(seg.max_key);
            segment_slope.push(seg.slope);
            segment_intercept.push(seg.intercept);
            segment_start_idx.push(seg.start_idx);
            segment_end_idx.push(seg.end_idx);
        }
        (
            segment_min_key,
            segment_max_key,
            segment_slope,
            segment_intercept,
            segment_start_idx,
            segment_end_idx,
        )
    }

    pub fn segment_count(&self) -> usize {
        self.segment_min_key.len()
    }
    pub fn avg_segment_size(&self) -> f64 {
        (self.data.len() as f64) / (self.segment_min_key.len() as f64).max(1.0)
    }
    pub fn memory_usage(&self) -> usize {
        let data_bytes = self.data.len() * std::mem::size_of::<K>();
        #[cfg(any(feature = "serde", debug_assertions))]
        let seg_bytes = self.segment_min_key.len() * std::mem::size_of::<Segment<K>>();
        #[cfg(not(any(feature = "serde", debug_assertions)))]
        let seg_bytes = 0usize;
        let soa_bytes = self.segment_min_key.len() * std::mem::size_of::<K>()
            + self.segment_max_key.len() * std::mem::size_of::<K>()
            + self.segment_slope.len() * std::mem::size_of::<f64>()
            + self.segment_intercept.len() * std::mem::size_of::<f64>()
            + self.segment_start_idx.len() * std::mem::size_of::<Idx>()
            + self.segment_end_idx.len() * std::mem::size_of::<Idx>();
        let err_bytes = self.segment_max_err.len() * std::mem::size_of::<usize>();
        let lut_bytes = self.segment_lookup.len() * std::mem::size_of::<usize>();
        data_bytes + soa_bytes + err_bytes + lut_bytes + seg_bytes
    }

    #[inline(always)]
    fn predict_index(&self, key: K, segment_idx: usize) -> usize {
        let x = self.key_delta_f64(key);
        let pred = self.segment_slope[segment_idx] * x + self.segment_intercept[segment_idx];
        let x = pred as isize;
        x.clamp(
            self.segment_start_idx[segment_idx] as isize,
            (self.segment_end_idx[segment_idx] as isize) - 1,
        ) as usize
    }

    #[inline(always)]
    fn find_segment_for_key_lut(&self, key: K) -> usize {
        if self.segment_min_key.len() <= 1 {
            return 0;
        }
        let x = self.key_delta_f64(key);
        let max_bin = self.segment_lookup.len().saturating_sub(2);
        let bin = (x * self.lookup_scale).floor().clamp(0.0, max_bin as f64) as usize;
        let seg_len = self.segment_min_key.len();
        let start = self.segment_lookup[bin].min(seg_len.saturating_sub(1));
        let mut end = self.segment_lookup[bin + 1].min(seg_len);
        if end <= start {
            end = (start + 1).min(seg_len);
        }
        let slice_max = &self.segment_max_key[start..end];
        let pos = slice_max.partition_point(|&max_key| max_key < key);
        let mut idx = start + pos;
        if idx >= seg_len {
            idx = seg_len - 1;
        }
        if key < self.segment_min_key[idx] && idx > 0 {
            idx -= 1;
        }
        idx
    }

    #[inline(always)]
    pub fn get(&self, key: K) -> Option<usize> {
        self.get_internal(key, None)
    }

    #[inline(always)]
    fn get_internal(&self, key: K, mut metrics: Option<&mut LocalMetrics>) -> Option<usize> {
        self.get_internal_fast(key, &mut metrics)
    }

    #[inline(always)]
    fn get_internal_fast(&self, key: K, metrics: &mut Option<&mut LocalMetrics>) -> Option<usize> {
        if self.segment_min_key.is_empty() {
            return None;
        }
        let sidx = self.find_segment_for_key_lut(key);
        let i = self.predict_index(key, sidx);

        let eps0 = self.epsilon;
        let eps1 = self.segment_max_err[sidx].max(eps0);

        let start0 = i.saturating_sub(eps0);
        let end0 = (i + eps0 + 1).min(self.data.len());
        if let Some(m) = metrics.as_mut() {
            m.window_probes_total += end0 - start0;
        }
        if let Some(pos) = self.search_slice(&self.data[start0..end0], key) {
            if let Some(m) = metrics.as_mut() {
                m.hits_in_window_queries += 1;
            }
            return Some(start0 + pos);
        }

        if eps1 == eps0 {
            return None;
        }

        let start1 = i.saturating_sub(eps1);
        let end1 = (i + eps1 + 1).min(self.data.len());
        if let Some(m) = metrics.as_mut() {
            m.fallback_used += 1;
        }
        if let Some(pos) = self.search_slice(&self.data[start1..end1], key) {
            return Some(start1 + pos);
        }
        None
    }

    /// Parallel batch lookup: returns positions for each key (None if absent).
    pub fn get_many_parallel(&self, keys: &[K]) -> Vec<Option<usize>>
    where
        K: Sync,
    {
        #[cfg(feature = "metrics")]
        let (_start_idx, results, metrics) = keys
            .par_iter()
            .enumerate()
            .fold(
                || (usize::MAX, Vec::new(), LocalMetrics::default()),
                |mut acc, (idx, &k)| {
                    if acc.0 == usize::MAX {
                        acc.0 = idx;
                    }
                    let res = self.get_internal_fast(k, &mut Some(&mut acc.2));
                    acc.1.push(res);
                    acc
                },
            )
            .reduce(
                || (usize::MAX, Vec::new(), LocalMetrics::default()),
                |a, b| {
                    if a.0 == usize::MAX {
                        return b;
                    }
                    if b.0 == usize::MAX {
                        return a;
                    }
                    if a.0 <= b.0 {
                        let mut v = a.1;
                        v.extend(b.1);
                        (
                            a.0,
                            v,
                            LocalMetrics {
                                hits_in_window_queries: a.2.hits_in_window_queries
                                    + b.2.hits_in_window_queries,
                                window_probes_total: a.2.window_probes_total
                                    + b.2.window_probes_total,
                                fallback_used: a.2.fallback_used + b.2.fallback_used,
                            },
                        )
                    } else {
                        let mut v = b.1;
                        v.extend(a.1);
                        (
                            b.0,
                            v,
                            LocalMetrics {
                                hits_in_window_queries: a.2.hits_in_window_queries
                                    + b.2.hits_in_window_queries,
                                window_probes_total: a.2.window_probes_total
                                    + b.2.window_probes_total,
                                fallback_used: a.2.fallback_used + b.2.fallback_used,
                            },
                        )
                    }
                },
            );
        #[cfg(feature = "metrics")]
        {
            self.add_metrics(metrics);
            results
        }
        #[cfg(not(feature = "metrics"))]
        {
            keys.par_iter()
                .map(|&k| self.get_internal_fast(k, &mut None))
                .collect()
        }
    }

    pub fn get_many(&self, keys: &[K]) -> Vec<Option<usize>> {
        let mut out = Vec::with_capacity(keys.len());
        #[cfg(feature = "metrics")]
        let mut local = LocalMetrics::default();
        for &k in keys {
            #[cfg(feature = "metrics")]
            let res = self.get_internal_fast(k, &mut Some(&mut local));
            #[cfg(not(feature = "metrics"))]
            let res = self.get_internal_fast(k, &mut None);
            out.push(res);
        }
        #[cfg(feature = "metrics")]
        {
            self.add_metrics(local);
        }
        out
    }

    /// Parallel batch hit count (useful for throughput microbenchmarks).
    pub fn count_hits_parallel(&self, keys: &[K]) -> usize
    where
        K: Sync,
    {
        #[cfg(feature = "metrics")]
        let (hits, metrics) = keys
            .par_iter()
            .fold(
                || (0usize, LocalMetrics::default()),
                |mut acc, &k| {
                    if self.get_internal_fast(k, &mut Some(&mut acc.1)).is_some() {
                        acc.0 += 1;
                    }
                    acc
                },
            )
            .reduce(
                || (0usize, LocalMetrics::default()),
                |a, b| {
                    (
                        a.0 + b.0,
                        LocalMetrics {
                            hits_in_window_queries: a.1.hits_in_window_queries
                                + b.1.hits_in_window_queries,
                            window_probes_total: a.1.window_probes_total + b.1.window_probes_total,
                            fallback_used: a.1.fallback_used + b.1.fallback_used,
                        },
                    )
                },
            );
        #[cfg(feature = "metrics")]
        {
            self.add_metrics(metrics);
            hits
        }
        #[cfg(not(feature = "metrics"))]
        {
            keys.par_iter()
                .filter(|&&k| self.get_internal_fast(k, &mut None).is_some())
                .count()
        }
    }
}

fn is_sorted<K: Ord>(data: &[K]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

#[derive(Clone, Default, Debug)]
struct MetricsTotals {
    hits_in_window_queries: usize,
    window_probes_total: usize,
    fallback_used: usize,
}

#[derive(Clone, Default)]
struct LocalMetrics {
    hits_in_window_queries: usize,
    window_probes_total: usize,
    fallback_used: usize,
}

#[allow(private_bounds)]
impl<K: Key + WindowSearch> PGMIndex<K> {
    #[inline]
    fn key_to_i128(key: K) -> i128 {
        key.to_i128().unwrap()
    }

    #[inline]
    fn key_delta_f64_with_min(key: K, min_key_i128: i128, key_scale_shift: u32) -> f64 {
        let delta = Self::key_to_i128(key) - min_key_i128;
        let scaled = if key_scale_shift == 0 {
            delta
        } else {
            delta >> key_scale_shift
        };
        scaled as f64
    }

    #[inline(always)]
    fn key_delta_f64(&self, key: K) -> f64 {
        Self::key_delta_f64_with_min(key, self.min_key_i128, self.key_scale_shift)
    }

    #[inline(always)]
    fn search_slice(&self, slice: &[K], key: K) -> Option<usize> {
        let len = slice.len();
        if len <= 32 {
            return search_window_scalar_unrolled(slice, key);
        }
        if len <= 256 {
            return K::search_window(slice, key);
        }
        slice.binary_search(&key).ok()
    }

    #[cfg(feature = "metrics")]
    fn add_metrics(&self, local: LocalMetrics) {
        let mut totals = self.metrics.lock().unwrap();
        totals.hits_in_window_queries += local.hits_in_window_queries;
        totals.window_probes_total += local.window_probes_total;
        totals.fallback_used += local.fallback_used;
    }

    #[cfg(debug_assertions)]
    fn verify_segments_debug(&self) {
        let eps = self.epsilon;
        for (seg_idx, seg) in self.segments.iter().enumerate() {
            let max_err = self.segment_max_err[seg_idx];
            debug_assert!(
                max_err <= eps,
                "segment_max_err {} exceeds epsilon {} at segment {}",
                max_err,
                eps,
                seg_idx
            );

            let start = seg.start_idx;
            let end = seg.end_idx.saturating_sub(1);
            if start > end {
                continue;
            }

            let len = end - start + 1;
            let samples = 8usize.min(len);
            for s in 0..samples {
                let idx = start + (s * len / samples);
                let key = self.data[idx];
                let x = self.key_delta_f64(key);
                let pred = seg.slope * x + seg.intercept;
                let pred = pred.clamp(seg.start_idx as f64, (seg.end_idx - 1) as f64);
                let err = (pred - (idx as f64)).abs();
                debug_assert!(
                    err <= (eps as f64 + 1e-6),
                    "segment {} violates epsilon: err={} eps={} idx={}",
                    seg_idx,
                    err,
                    eps,
                    idx
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simd_search_u64_avx2(slice: &[u64], key: u64) -> Option<usize> {
    use std::arch::x86_64::*;
    let ptr = slice.as_ptr();
    let len = slice.len();
    let needle = _mm256_set1_epi64x(std::mem::transmute::<u64, i64>(key));
    let mut i = 0usize;
    while i + 4 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi64(v, needle);
        let mask = _mm256_movemask_pd(std::mem::transmute::<__m256i, __m256d>(cmp));
        if mask != 0 {
            for j in 0..4 {
                if *ptr.add(i + j) == key {
                    return Some(i + j);
                }
            }
        }
        i += 4;
    }
    for j in i..len {
        if *ptr.add(j) == key {
            return Some(j);
        }
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simd_search_u32_avx2(slice: &[u32], key: u32) -> Option<usize> {
    use std::arch::x86_64::*;
    let ptr = slice.as_ptr();
    let len = slice.len();
    let needle = _mm256_set1_epi32(std::mem::transmute::<u32, i32>(key));
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi32(v, needle);
        let mask = _mm256_movemask_ps(std::mem::transmute::<__m256i, __m256>(cmp));
        if mask != 0 {
            for j in 0..8 {
                if *ptr.add(i + j) == key {
                    return Some(i + j);
                }
            }
        }
        i += 8;
    }
    for j in i..len {
        if *ptr.add(j) == key {
            return Some(j);
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simd_search_u64_neon(slice: &[u64], key: u64) -> Option<usize> {
    use std::arch::aarch64::*;
    let ptr = slice.as_ptr();
    let len = slice.len();
    let needle = vdupq_n_u64(key);
    let mut i = 0usize;
    while i + 2 <= len {
        let v = vld1q_u64(ptr.add(i));
        let cmp = vceqq_u64(v, needle);
        let lane0 = vgetq_lane_u64(cmp, 0);
        let lane1 = vgetq_lane_u64(cmp, 1);
        if (lane0 | lane1) != 0 {
            for j in 0..2 {
                if *ptr.add(i + j) == key {
                    return Some(i + j);
                }
            }
        }
        i += 2;
    }
    for j in i..len {
        if *ptr.add(j) == key {
            return Some(j);
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simd_search_u32_neon(slice: &[u32], key: u32) -> Option<usize> {
    use std::arch::aarch64::*;
    let ptr = slice.as_ptr();
    let len = slice.len();
    let needle = vdupq_n_u32(key);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_u32(ptr.add(i));
        let cmp = vceqq_u32(v, needle);
        let lane0 = vgetq_lane_u32(cmp, 0);
        let lane1 = vgetq_lane_u32(cmp, 1);
        let lane2 = vgetq_lane_u32(cmp, 2);
        let lane3 = vgetq_lane_u32(cmp, 3);
        if (lane0 | lane1 | lane2 | lane3) != 0 {
            for j in 0..4 {
                if *ptr.add(i + j) == key {
                    return Some(i + j);
                }
            }
        }
        i += 4;
    }
    for j in i..len {
        if *ptr.add(j) == key {
            return Some(j);
        }
    }
    None
}

#[inline(always)]
fn search_window_simd_u64(slice: &[u64], key: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { simd_search_u64_avx2(slice, key) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_search_u64_neon(slice, key) };
        }
    }
    let mut i = 0usize;
    let len = slice.len();
    while i < len {
        if slice[i] == key {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline(always)]
fn search_window_simd_u32(slice: &[u32], key: u32) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { simd_search_u32_avx2(slice, key) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_search_u32_neon(slice, key) };
        }
    }
    let mut i = 0usize;
    let len = slice.len();
    while i < len {
        if slice[i] == key {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline(always)]
fn search_window_scalar_unrolled<K: Key>(slice: &[K], key: K) -> Option<usize> {
    let len = slice.len();
    let mut i = 0usize;
    while i + 8 <= len {
        if slice[i] == key {
            return Some(i);
        }
        if slice[i + 1] == key {
            return Some(i + 1);
        }
        if slice[i + 2] == key {
            return Some(i + 2);
        }
        if slice[i + 3] == key {
            return Some(i + 3);
        }
        if slice[i + 4] == key {
            return Some(i + 4);
        }
        if slice[i + 5] == key {
            return Some(i + 5);
        }
        if slice[i + 6] == key {
            return Some(i + 6);
        }
        if slice[i + 7] == key {
            return Some(i + 7);
        }
        i += 8;
    }
    while i < len {
        if slice[i] == key {
            return Some(i);
        }
        i += 1;
    }
    None
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
        segment_max_err: Vec<usize>,
        segment_lookup: Vec<usize>,
        lookup_scale: f64,
        min_key_i128: i128,
        key_scale_shift: u32,
    }

    impl<K: Key + WindowSearch + Serialize> Serialize for PGMIndex<K> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let tmp = PGMIndexSerde {
                epsilon: self.epsilon,
                data: self.data.to_vec(),
                segments: self.segments.clone(),
                segment_max_err: self.segment_max_err.clone(),
                segment_lookup: self.segment_lookup.clone(),
                lookup_scale: self.lookup_scale,
                min_key_i128: self.min_key_i128,
                key_scale_shift: self.key_scale_shift,
            };
            tmp.serialize(serializer)
        }
    }

    impl<'de, K: Key + WindowSearch + Deserialize<'de>> Deserialize<'de> for PGMIndex<K> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let tmp = PGMIndexSerde::<K>::deserialize(deserializer)?;
            let (
                segment_min_key,
                segment_max_key,
                segment_slope,
                segment_intercept,
                segment_start_idx,
                segment_end_idx,
            ) = PGMIndex::<K>::segment_soa_from(&tmp.segments);
            Ok(PGMIndex {
                epsilon: tmp.epsilon,
                data: Arc::from(tmp.data.into_boxed_slice()),
                segments: tmp.segments,
                segment_min_key,
                segment_max_key,
                segment_slope,
                segment_intercept,
                segment_start_idx,
                segment_end_idx,
                segment_max_err: tmp.segment_max_err,
                segment_lookup: tmp.segment_lookup,
                lookup_scale: tmp.lookup_scale,
                min_key_i128: tmp.min_key_i128,
                key_scale_shift: tmp.key_scale_shift,
                #[cfg(feature = "metrics")]
                metrics: std::sync::Mutex::new(MetricsTotals::default()),
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
