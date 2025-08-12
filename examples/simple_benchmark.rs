//! Simple benchmark demonstrating PGM-Index performance vs binary search
//!
//! This example creates a dataset of 1 million integers and compares
//! the performance of PGM-Index against standard binary search.

use pgm_index::PGMIndex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("=== PGM-Index Simple Benchmark ===\n");

    // Create dataset with 1 million integers
    const N: usize = 1_000_000;
    const QUERY_COUNT: usize = 100_000;

    println!("Creating dataset with {} elements...", N);
    let data: Vec<u64> = (0..N as u64).collect();

    // Generate random queries
    let mut rng = StdRng::seed_from_u64(42);
    let queries: Vec<u64> = (0..QUERY_COUNT)
        .map(|_| rng.gen_range(0..N as u64))
        .collect();

    println!("Generated {} random queries", QUERY_COUNT);
    println!();

    // Test different epsilon values
    for epsilon in [16, 32, 64, 128] {
        test_epsilon(epsilon, &data, &queries);
    }

    // Compare with binary search
    println!("=== Binary Search Baseline ===");
    let start = Instant::now();
    let mut hits = 0;
    for &query in &queries {
        if data.binary_search(&query).is_ok() {
            hits += 1;
        }
    }
    let binary_time = start.elapsed();
    let binary_ns_per_query = binary_time.as_nanos() as f64 / QUERY_COUNT as f64;

    println!("Time: {:?}", binary_time);
    println!(
        "Throughput: {:.0} queries/sec",
        QUERY_COUNT as f64 / binary_time.as_secs_f64()
    );
    println!("Average: {:.1} ns/query", binary_ns_per_query);
    println!("Hits: {}/{}", hits, QUERY_COUNT);
    println!();

    // Summary comparison
    println!("=== Performance Summary ===");
    println!("Dataset size: {} elements", N);
    println!(
        "Memory usage: {:.2} MB",
        std::mem::size_of_val(&*data) as f64 / 1024.0 / 1024.0
    );
    println!();
    println!("Key takeaways:");
    println!("• Smaller epsilon = more segments = better accuracy = faster queries");
    println!("• PGM-Index typically 2-10x faster than binary search");
    println!("• Memory overhead is typically 1-5% of original data size");
    println!("• Best performance on uniformly distributed sequential data");
}

fn test_epsilon(epsilon: usize, data: &[u64], queries: &[u64]) {
    println!("=== PGM-Index (ε = {}) ===", epsilon);

    // Build index
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

    // Single queries
    let start = Instant::now();
    let mut hits = 0;
    for &query in queries {
        if index.get(query).is_some() {
            hits += 1;
        }
    }
    let single_time = start.elapsed();
    let single_ns_per_query = single_time.as_nanos() as f64 / queries.len() as f64;

    println!("Single query time: {:?}", single_time);
    println!(
        "Throughput: {:.0} queries/sec",
        queries.len() as f64 / single_time.as_secs_f64()
    );
    println!("Average: {:.1} ns/query", single_ns_per_query);

    // Batch queries
    let start = Instant::now();
    let batch_results = index.batch_get(queries);
    let batch_time = start.elapsed();
    let batch_hits = batch_results.iter().filter(|r| r.is_some()).count();
    let batch_ns_per_query = batch_time.as_nanos() as f64 / queries.len() as f64;

    println!("Batch query time: {:?}", batch_time);
    println!(
        "Batch throughput: {:.0} queries/sec",
        queries.len() as f64 / batch_time.as_secs_f64()
    );
    println!("Batch average: {:.1} ns/query", batch_ns_per_query);

    assert_eq!(hits, batch_hits);
    println!("Hits: {}/{}", hits, queries.len());

    // Performance stats
    let stats = index.get_stats();
    println!("Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);

    println!();
}

/// Additional test with different data patterns
#[allow(dead_code)]
fn test_data_patterns() {
    println!("=== Testing Different Data Patterns ===");

    let patterns = vec![
        ("Sequential", (0..100_000).collect::<Vec<u64>>()),
        (
            "Gaps (every 10th)",
            (0..100_000).map(|i| i * 10).collect::<Vec<u64>>(),
        ),
        ("Random sorted", {
            let mut rng = StdRng::seed_from_u64(42);
            let mut v: Vec<u64> = (0..100_000).map(|_| rng.gen_range(0..1_000_000)).collect();
            v.sort();
            v
        }),
    ];

    for (name, data) in patterns {
        println!("\n{} ({} elements):", name, data.len());

        let index = PGMIndex::new(data.clone(), 64);
        println!("  Segments: {}", index.segment_count());
        println!("  Avg segment size: {:.1}", index.avg_segment_size());
        println!(
            "  Memory overhead: {:.2}%",
            (index.memory_usage() as f64 / (data.len() * 8) as f64 - 1.0) * 100.0
        );

        // Test a few queries
        let test_keys = vec![data[0], data[data.len() / 2], data[data.len() - 1]];
        let start = Instant::now();
        for &key in &test_keys {
            let _ = index.get(key);
        }
        let query_time = start.elapsed();
        println!("  Query time (3 queries): {:?}", query_time);
    }
}
