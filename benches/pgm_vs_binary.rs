//! Criterion benchmark comparing PGM-Index vs binary search

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pgm_index::PGMIndex;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn bench_single_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_lookups");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let data: Vec<u64> = (0..*size as u64).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let queries: Vec<u64> = (0..1000).map(|_| rng.gen_range(0..*size as u64)).collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        // Binary search baseline
        group.bench_with_input(
            BenchmarkId::new("binary_search", size),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    for &query in queries.iter() {
                        black_box(data.binary_search(&query));
                    }
                })
            },
        );

        // PGM-Index with different epsilon values
        for epsilon in [32, 64, 128] {
            let index = PGMIndex::new(data.clone(), epsilon);
            group.bench_with_input(
                BenchmarkId::new(format!("pgm_eps_{}", epsilon), size),
                &(&index, &queries),
                |b, (index, queries)| {
                    b.iter(|| {
                        for &query in queries.iter() {
                            black_box(index.get(query));
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_batch_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_lookups");

    let data: Vec<u64> = (0..1_000_000).collect();
    let mut rng = StdRng::seed_from_u64(42);

    for batch_size in [100, 1_000, 10_000].iter() {
        let queries: Vec<u64> = (0..*batch_size)
            .map(|_| rng.gen_range(0..1_000_000))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        // Sequential binary search
        group.bench_with_input(
            BenchmarkId::new("binary_search_sequential", batch_size),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let results: Vec<_> = queries.iter().map(|&q| data.binary_search(&q)).collect();
                    black_box(results);
                })
            },
        );

        // PGM-Index batch
        let index = PGMIndex::new(data.clone(), 64);
        group.bench_with_input(
            BenchmarkId::new("pgm_batch", batch_size),
            &(&index, &queries),
            |b, (index, queries)| {
                b.iter(|| {
                    let results = index.batch_get(queries);
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

fn bench_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_time");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let data: Vec<u64> = (0..*size as u64).collect();

        group.throughput(Throughput::Elements(*size as u64));

        for epsilon in [32, 64, 128] {
            group.bench_with_input(
                BenchmarkId::new(format!("pgm_eps_{}", epsilon), size),
                &(&data, epsilon),
                |b, (data, &epsilon)| {
                    b.iter(|| {
                        let index = PGMIndex::new(data.clone(), epsilon);
                        black_box(index);
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_different_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_patterns");
    group.sample_size(20); // Fewer samples for slower benchmarks

    let size = 100_000;
    let mut rng = StdRng::seed_from_u64(42);
    let queries: Vec<u64> = (0..1000).map(|_| rng.gen_range(0..size)).collect();

    // Different data patterns
    let patterns = vec![
        ("sequential", (0..size).collect::<Vec<u64>>()),
        ("gaps_10x", (0..size).map(|i| i * 10).collect::<Vec<u64>>()),
        ("random_sorted", {
            let mut v: Vec<u64> = (0..size).map(|_| rng.gen_range(0..size * 10)).collect();
            v.sort();
            v.dedup();
            v.truncate(size as usize);
            v
        }),
    ];

    for (pattern_name, data) in patterns {
        group.throughput(Throughput::Elements(queries.len() as u64));

        // Binary search
        group.bench_with_input(
            BenchmarkId::new("binary_search", pattern_name),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    for &query in queries.iter() {
                        black_box(data.binary_search(&query));
                    }
                })
            },
        );

        // PGM-Index
        let index = PGMIndex::new(data.clone(), 64);
        group.bench_with_input(
            BenchmarkId::new("pgm_index", pattern_name),
            &(&index, &queries),
            |b, (index, queries)| {
                b.iter(|| {
                    for &query in queries.iter() {
                        black_box(index.get(query));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_lookups,
    bench_batch_lookups,
    bench_build_time,
    bench_different_data_patterns
);
criterion_main!(benches);
