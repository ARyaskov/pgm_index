//! Simple benchmark demonstrating PGM-Index performance vs binary search
//!
//! This example creates a dataset and sweeps epsilon values to show
//! how segments grow/shrink and how speed↔memory tradeoff behaves.

use pgm_index::PGMIndex;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn main() {
    println!("=== PGM-Index Simple Benchmark ===\n");

    const N: usize = 1_000_000_000;
    const QUERY_COUNT: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(42);

    println!("Creating dataset with {} elements...", N);
    let data: Vec<u64> = (0..N as u64).collect();

    let mut queries = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        queries.push(rng.gen_range(0..N as u64));
    }

    for &epsilon in &[16usize, 32, 64, 128] {
        test_epsilon(epsilon, &data, &queries);
    }

    println!(
        "\nTip: smaller ε ⇒ more segments (faster queries, higher memory); \
         larger ε ⇒ fewer segments (slower, lower memory)."
    );
}

fn test_epsilon(epsilon: usize, data: &[u64], queries: &[u64]) {
    println!("=== PGM-Index (ε = {}) ===", epsilon);

    let build_start = Instant::now();
    let index = PGMIndex::new(data.to_vec(), epsilon);
    let build_time = build_start.elapsed();

    println!("Build time: {:?}", build_time);
    println!("Segments: {}", index.segment_count());
    println!("Avg segment size: {:.1}", index.avg_segment_size());
    println!(
        "Memory usage: {:.2} MB",
        index.memory_usage() as f64 / 1024.0 / 1024.0
    );
    println!(
        "Memory overhead: {:.2}%",
        (index.memory_usage() as f64 / (data.len() * 8) as f64 - 1.0) * 100.0
    );

    // Single query smoke (10)
    let single_start = Instant::now();
    let mut hits = 0usize;
    for &q in &queries[0..queries.len().min(10)] {
        if index.get(q).is_some() {
            hits += 1;
        }
    }
    let single_time = single_start.elapsed();
    println!("Single query time ({} hits / 10): {:?}", hits, single_time);

    // Batch throughput
    let batch_start = Instant::now();
    let mut hits = 0usize;
    for &q in queries {
        if index.get(q).is_some() {
            hits += 1;
        }
    }
    let batch_time = batch_start.elapsed();
    let ns_per_query = batch_time.as_nanos() as f64 / (queries.len() as f64);
    println!("Batch query time: {:?}", batch_time);
    println!(
        "Batch throughput: {:.0} queries/sec",
        (queries.len() as f64) / batch_time.as_secs_f64()
    );
    println!("Batch average: {:.1} ns/query", ns_per_query);
    println!("Hits: {}/{}", hits, queries.len());

    // Edge keys
    let test_keys = vec![data[0], data[data.len() / 2], data[data.len() - 1]];
    let start = Instant::now();
    for &key in &test_keys {
        let _ = index.get(key);
    }
    let query_time = start.elapsed();
    println!("Query time (3 edge keys): {:?}", query_time);
}
