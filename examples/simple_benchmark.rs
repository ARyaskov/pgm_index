//! Simple benchmark demonstrating PGM-Index performance vs binary search
//!
//! This example creates a dataset and sweeps epsilon values to show
//! how segments grow/shrink and how speed↔memory tradeoff behaves.

use pgm_index::PGMIndex;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("=== PGM-Index Simple Benchmark ===\n");
    // Initialize Rayon global pool with at least 4 threads
    pgm_index::init_rayon_min_threads();
    println!("Rayon threads: {}", rayon::current_num_threads());

    const N: usize = 1_000_000_00;
    const QUERY_COUNT: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(42);

    println!("Creating dataset with {} elements...", N);
    // Noisy-but-sorted data to exercise segmentation on non-linear distributions.
    // This preserves sorted order without an expensive global sort.
    let mut data: Vec<u64> = Vec::with_capacity(N);
    let mut cur = 0u64;
    for _ in 0..N {
        let gap: u64 = rng.gen_range(1..=10_000);
        cur = cur.saturating_add(gap);
        data.push(cur);
    }

    let mut queries = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        let idx = rng.gen_range(0..N);
        queries.push(data[idx]);
    }
    queries.shuffle(&mut rng);

    print_table_header();
    for &epsilon in &[16usize, 32, 64, 128] {
        test_epsilon(epsilon, &data, &queries);
    }

    println!(
        "\nTip: smaller ε ⇒ more segments (faster queries, higher memory); \
         larger ε ⇒ fewer segments (slower, lower memory)."
    );
}

fn test_epsilon(epsilon: usize, data: &[u64], queries: &[u64]) {
    let build_start = Instant::now();
    let index = PGMIndex::new(data.to_vec(), epsilon);
    let build_time = build_start.elapsed();

    // Single query smoke (10)
    let single_start = Instant::now();
    let mut hits = 0usize;
    for &q in &queries[0..queries.len().min(10)] {
        let q = black_box(q);
        if black_box(index.get(q)).is_some() {
            hits += 1;
        }
    }
    let single_time = single_start.elapsed();

    // Batch throughput (sequential)
    let batch_start = Instant::now();
    let results = index.get_many(queries);
    let hits = results.iter().filter(|r| r.is_some()).count();
    let _ = black_box(&results);
    let batch_time = batch_start.elapsed();
    let ns_per_query = batch_time.as_nanos() as f64 / (queries.len() as f64);

    // Parallel batch throughput (rayon)
    let par_start = Instant::now();
    let par_hits = queries
        .par_iter()
        .filter(|&&q| black_box(index.get(black_box(q))).is_some())
        .count();
    let par_time = par_start.elapsed();
    let par_ns_per_query = par_time.as_nanos() as f64 / (queries.len() as f64);

    #[cfg(feature = "metrics")]
    {
        // Populate metrics without adding overhead to the hot path.
        let _ = index.get_many_parallel(queries);
    }

    // Edge keys
    let test_keys = vec![data[0], data[data.len() / 2], data[data.len() - 1]];
    let start = Instant::now();
    for &key in &test_keys {
        let key = black_box(key);
        let _ = black_box(index.get(key));
    }
    let query_time = start.elapsed();
    let stats = index.stats();

    print_table_row(
        epsilon,
        build_time.as_millis(),
        index.segment_count(),
        index.avg_segment_size(),
        index.memory_usage() as f64 / 1024.0 / 1024.0,
        (index.memory_usage() as f64 / (data.len() * 8) as f64 - 1.0) * 100.0,
        single_time.as_nanos() as f64 / 10.0,
        ns_per_query,
        par_ns_per_query,
        hits,
        par_hits,
        query_time.as_nanos(),
        stats.hits_in_window_queries,
        stats.fallback_used,
        stats.window_probes_total,
    );
}

fn print_table_header() {
    println!(
        "{:>5} | {:>8} | {:>8} | {:>10} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>7} | {:>7} | {:>10} | {:>7} | {:>7} | {:>10}",
        "eps",
        "build ms",
        "segs",
        "avg_seg",
        "mem MB",
        "over%",
        "single ns",
        "batch ns",
        "par ns",
        "hits",
        "phits",
        "edge ns",
        "win_hit",
        "fb",
        "win_probe"
    );
    println!(
        "{:>5}-+-{:>8}-+-{:>8}-+-{:>10}-+-{:>8}-+-{:>8}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>7}-+-{:>7}-+-{:>10}-+-{:>7}-+-{:>7}-+-{:>10}",
        "-----",
        "--------",
        "--------",
        "----------",
        "--------",
        "--------",
        "----------",
        "----------",
        "----------",
        "-------",
        "-------",
        "----------",
        "-------",
        "-------",
        "----------"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_table_row(
    epsilon: usize,
    build_ms: u128,
    segments: usize,
    avg_seg: f64,
    mem_mb: f64,
    mem_over: f64,
    single_ns: f64,
    batch_ns: f64,
    par_ns: f64,
    hits: usize,
    par_hits: usize,
    edge_ns: u128,
    win_hit: usize,
    fallback: usize,
    win_probe: usize,
) {
    println!(
        "{:>5} | {:>8} | {:>8} | {:>10.1} | {:>8.2} | {:>8.2} | {:>10.1} | {:>10.1} | {:>10.1} | {:>7} | {:>7} | {:>10} | {:>7} | {:>7} | {:>10}",
        epsilon,
        build_ms,
        segments,
        avg_seg,
        mem_mb,
        mem_over,
        single_ns,
        batch_ns,
        par_ns,
        hits,
        par_hits,
        edge_ns,
        win_hit,
        fallback,
        win_probe
    );
}
