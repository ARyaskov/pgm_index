# 📐 pgm_index — Learned Index for Sorted Keys

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/pgm_index)

> PGM-Index is a space-efficient data structure for fast lookup in sorted sequences.  
> It approximates the distribution of keys with piecewise linear models, allowing searches in **O(log ε)** with a guaranteed error bound.

---

## 📄 Algorithm

Based on the work by Paolo Ferragina & Giorgio Vinciguerra:  
> *The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds* (2020)  
🔗 [Paper](https://doi.org/10.1145/3373718.3394764) · 🌐 [Official site](https://pgm.di.unipi.it/)

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pgm_index = ">=0.3.4"
````

---

## 🛠 Usage

```rust
use pgm_index::PGMIndex;

fn main() {
    let data: Vec<u64> = (0..1_000_000).collect();
    let pgm = PGMIndex::new(&data, 32); // ε = 32

    let key = 123456;
    if let Some(pos) = pgm.search(key) {
        println!("Found at position {}", pos);
    } else {
        println!("Not found");
    }
}
```

---

## 🏎 Benchmarks by ε

Dataset: **1,000,000 elements**, 100,000 random queries
CPU: Intel Core i7 12700, Windows 11, single-threaded

| ε   | Build Time | Mem Usage | Segments | Single Lookup | Batch Lookup | Avg ns/query |
| --- | ---------: | --------: | -------: | ------------: | -----------: | -----------: |
| 16  |    2.19 ms |   7.99 MB |     3906 |     20.84 M/s |    25.35 M/s |  48.0 / 39.4 |
| 32  |    2.22 ms |   7.87 MB |     1953 |     24.09 M/s |    24.98 M/s |  41.5 / 40.0 |
| 64  |    2.05 ms |   7.75 MB |      976 |     21.96 M/s |    26.06 M/s |  45.5 / 38.4 |
| 128 |    2.07 ms |   7.69 MB |      488 |     19.64 M/s |    25.56 M/s |  50.9 / 39.1 |

**Binary Search (baseline)**:
4.32 ms, 23.13 M/s, 43.2 ns/query.

---

## 🍎 Apple M1 Benchmark (PGM-Index Simple Benchmark)

Dataset: **100,000,000 elements**, 100,000 random queries  
CPU: Apple M1 (8 threads), macOS

```
  eps | build ms |     segs |    avg_seg |   mem MB |    over% |  single ns |   batch ns |     par ns |    hits |   phits |    edge ns | win_hit |      fb |  win_probe
------+----------+----------+------------+----------+----------+------------+------------+------------+---------+---------+------------+---------+---------+-----------
   16 |     1942 |    49122 |     2035.7 |   767.06 |     0.54 |     1579.1 |      904.1 |       57.8 |  100000 |  100000 |       1417 |       0 |       0 |          0
   32 |     2539 |    12484 |     8010.3 |   763.99 |     0.14 |      795.9 |      559.7 |       40.2 |  100000 |  100000 |        708 |       0 |       0 |          0
   64 |     1453 |     3191 |    31338.1 |   763.21 |     0.04 |     1600.0 |     1841.9 |       56.8 |  100000 |  100000 |       1542 |       0 |       0 |          0
  128 |     1466 |      811 |   123304.6 |   763.01 |     0.01 |      945.9 |      643.7 |       60.7 |  100000 |  100000 |       1000 |       0 |       0 |          0
```

---

## 📊 Comparison to Other Indexes (1M elements) *

Using **ε = 32** as a balanced configuration:

| Structure            | Memory Usage |   Build Time | Lookup Speed (single) |   Batch Lookup Speed |
| -------------------- | -----------: | -----------: | --------------------: | -------------------: |
| **PGM-Index (ε=32)** |  **7.87 MB** |  **2.22 ms** |         **24.09 M/s** |        **24.98 M/s** |
| **Binary Search**    |     \~8.0 MB | — (no build) |   23.13 M/s *(0.96×)* |  23.13 M/s *(0.93×)* |
| **BTreeMap**         |      \~24 MB |      \~50 ms |   \~4.0 M/s *(0.17×)* |  \~4.0 M/s *(0.16×)* |
| **HashMap**          |      \~64 MB |      \~15 ms |  \~40.0 M/s *(1.66×)* | \~40.0 M/s *(1.60×)* |

* our benchmarks
---

### 📈 Relative Performance *

| Metric        | vs Binary Search | vs BTreeMap      | vs HashMap       |
| ------------- | ---------------- | ---------------- | ---------------- |
| Memory        | **1.02× better** | **3.05× better** | **8.13× better** |
| Build Time    | —                | **22.5× faster** | **6.8× faster**  |
| Single Lookup | **1.04× faster** | **6.0× faster**  | 0.6× slower      |
| Batch Lookup  | **1.08× faster** | **6.2× faster**  | 0.62× slower     |

* our benchmarks
---

## 📌 Potential Use Cases

* Indexing large sorted numeric datasets
* Time-series databases
* Read-optimized storage engines
* Scientific & bioinformatics data search
* Columnar store secondary indexes

---

## In case you want a single-threaded environment

There is a more optimized for single-threaded execution implementation of the PGM algo by @i18nsite [https://crates.io/crates/jdb_pgm](https://crates.io/crates/jdb_pgm) https://github.com/ARyaskov/pgm_index/issues/2

---

## 📜 License

MIT
