//! Cache-blocking candidates (the crate performance workplan's K7, K8) for the crate's two dense products, bit-checked against today's
//! kernels and timed on one machine, with no crate build: the kernels are copied from
//! `rust/core/src/linalg.rs` (AVX2+FMA only, so x86_64 machines). `a @ b` is `tiled_row_range`'s
//! family (each output one FMA chain over `k`), `a @ b.T` is `matmul_nt`'s (each output
//! `dot_product`'s 4 lanes, combined at the end).
//!
//!     rustc -C opt-level=3 -C codegen-units=1 -C llvm-args=-x86-branches-within-32B-boundaries \
//!         --edition 2024 -o blocking_probe blocking_probe.rs
//!     blocking_probe [TARGET_MS]
//!     blocking_probe --one MxKxN KIND:VARIANT CALLS        (for perf stat: one variant, CALLS calls)
//!
//! Prints one JSON object per (product, variant), its median and best µs a call, and whether its
//! bits equal the current kernel's. Variants:
//!
//!     mm:current      today's matmul_2d: rows per block from 16 KB of `a`, each tile over all of k
//!     mm:rows=R       R rows per block (0: all of `a`), so one K x 16 panel of b serves R / 2 tiles
//!     mm:slab=S       all rows per block, k in slabs of S rows of b, resuming (matmul_long_k's way)
//!     mm:pack         all rows per block, each K x 16 panel copied to a contiguous buffer first
//!     mm:packslab=S   the same, in slabs of S rows of b
//!     mm:threads=T    today's row split over T threads (spawned per call)
//!     mm:cols=T       column ranges over T threads, all rows per block
//!     mm:colspack=T   column ranges over T threads, all rows per block, panels packed
//!     nt:current      today's matmul_nt: 4 x 2 register tile over all of k
//!     nt:slab=S       k in slabs of S (a multiple of 4), the 4 lanes of every output stored between
//!     nt:threads=T    today's row split over T threads (spawned per call)
//!     nt:threadslab=S the row split over 8 threads, k in slabs of S
//!     nt:group=G      `b` by groups of G rows, every row of `a` past each group, and no stored lanes
//!                     (0: one group of all of `b`, which is today's loop order)
//!
//!     blocking_probe --groups [TARGET_MS] runs only nt:group, on the shapes it was added for.

use std::arch::x86_64::*;
use std::time::Instant;

const TILE_ROWS: usize = 2;

fn random_values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

// ---- a @ b: each output one FMA chain over k

/// The panel of one 16-wide column tile: `b_at(kk)` gives `k`'s row of the tile's 16 columns.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile16(
    a: *const f64,
    k: usize,
    rows: (usize, usize),
    k_range: (usize, usize),
    b_at: impl Fn(usize) -> *const f64,
    out_at: impl Fn(usize, usize) -> *mut f64,
) {
    unsafe {
        let mut row = rows.0;
        while row + TILE_ROWS <= rows.1 {
            let mut acc = [[_mm256_setzero_pd(); 4]; TILE_ROWS];
            for r in 0..TILE_ROWS {
                for j in 0..4 {
                    acc[r][j] = _mm256_loadu_pd(out_at(row + r, 4 * j));
                }
            }
            for kk in k_range.0..k_range.1 {
                let b_row = b_at(kk);
                let b_tile = [
                    _mm256_loadu_pd(b_row),
                    _mm256_loadu_pd(b_row.add(4)),
                    _mm256_loadu_pd(b_row.add(8)),
                    _mm256_loadu_pd(b_row.add(12)),
                ];
                for r in 0..TILE_ROWS {
                    let a_vec = _mm256_set1_pd(*a.add((row + r) * k + kk));
                    for j in 0..4 {
                        acc[r][j] = _mm256_fmadd_pd(a_vec, b_tile[j], acc[r][j]);
                    }
                }
            }
            for r in 0..TILE_ROWS {
                for j in 0..4 {
                    _mm256_storeu_pd(out_at(row + r, 4 * j), acc[r][j]);
                }
            }
            row += TILE_ROWS;
        }
        for row in row..rows.1 {
            let mut acc = [0, 4, 8, 12].map(|c| _mm256_loadu_pd(out_at(row, c)));
            for kk in k_range.0..k_range.1 {
                let a_vec = _mm256_set1_pd(*a.add(row * k + kk));
                let b_row = b_at(kk);
                for j in 0..4 {
                    acc[j] = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(b_row.add(4 * j)), acc[j]);
                }
            }
            for j in 0..4 {
                _mm256_storeu_pd(out_at(row, 4 * j), acc[j]);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Blocking {
    Rows(usize),
    Slab(usize),
    Pack(usize), // slab rows; 0: all of k
}

/// Columns `cols` of `rows` of `a @ b`, into `out` (zeroed), `stride` wide, its column 0 `cols.0`.
#[target_feature(enable = "avx2,fma")]
unsafe fn product_part(
    a: &[f64],
    b: &[f64],
    (k, n): (usize, usize),
    rows: (usize, usize),
    cols: (usize, usize),
    out: &mut [f64],
    stride: usize,
    blocking: Blocking,
) {
    let tiles = (cols.1 - cols.0) / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let out_at = |col: usize| {
        move |row: usize, offset: usize| unsafe {
            out_ptr.add((row - rows.0) * stride + col - cols.0 + offset)
        }
    };
    match blocking {
        Blocking::Rows(rows_per_block) => {
            let block = if rows_per_block == 0 {
                rows.1 - rows.0
            } else {
                rows_per_block
            };
            let mut block_start = rows.0;
            while block_start < rows.1 {
                let block_end = (block_start + block).min(rows.1);
                for tile in 0..tiles {
                    let col = cols.0 + tile * 16;
                    unsafe {
                        tile16(
                            a_ptr,
                            k,
                            (block_start, block_end),
                            (0, k),
                            |kk| b_ptr.add(kk * n + col),
                            out_at(col),
                        );
                    }
                }
                block_start = block_end;
            }
        }
        Blocking::Slab(slab) => {
            let mut k_start = 0;
            while k_start < k {
                let k_end = (k_start + slab).min(k);
                for tile in 0..tiles {
                    let col = cols.0 + tile * 16;
                    unsafe {
                        tile16(
                            a_ptr,
                            k,
                            rows,
                            (k_start, k_end),
                            |kk| b_ptr.add(kk * n + col),
                            out_at(col),
                        );
                    }
                }
                k_start = k_end;
            }
        }
        Blocking::Pack(slab) => {
            let slab = if slab == 0 { k } else { slab };
            let mut panel = vec![0.0f64; slab.min(k) * 16];
            let panel_ptr = panel.as_mut_ptr();
            for tile in 0..tiles {
                let col = cols.0 + tile * 16;
                let mut k_start = 0;
                while k_start < k {
                    let k_end = (k_start + slab).min(k);
                    for kk in k_start..k_end {
                        unsafe {
                            let src = b_ptr.add(kk * n + col);
                            let dst = panel_ptr.add((kk - k_start) * 16);
                            for j in 0..4 {
                                _mm256_storeu_pd(dst.add(4 * j), _mm256_loadu_pd(src.add(4 * j)));
                            }
                        }
                    }
                    let packed = panel_ptr as *const f64;
                    unsafe {
                        tile16(
                            a_ptr,
                            k,
                            rows,
                            (k_start, k_end),
                            |kk| packed.add((kk - k_start) * 16),
                            out_at(col),
                        );
                    }
                    k_start = k_end;
                }
            }
        }
    }
    let done = cols.0 + tiles * 16;
    if done < cols.1 {
        for row in rows.0..rows.1 {
            for col in done..cols.1 {
                let at = (row - rows.0) * stride + col - cols.0;
                let mut sum = out[at];
                for kk in 0..k {
                    sum = a[row * k + kk].mul_add(b[kk * n + col], sum);
                }
                out[at] = sum;
            }
        }
    }
}

fn current_rows_per_block(k: usize) -> usize {
    (16 * 1024 / (k * 8).max(1)).max(1)
}

fn mm(a: &[f64], b: &[f64], (m, k, n): (usize, usize, usize), out: &mut [f64], variant: &str) {
    out.fill(0.0);
    let (name, value) = variant
        .split_once('=')
        .map_or((variant, 0), |(l, r)| (l, r.parse().unwrap()));
    let serial = |out: &mut [f64], blocking| unsafe {
        product_part(a, b, (k, n), (0, m), (0, n), out, n, blocking)
    };
    match name {
        "current" => serial(out, Blocking::Rows(current_rows_per_block(k))),
        "rows" => serial(out, Blocking::Rows(value)),
        "slab" => serial(out, Blocking::Slab(value)),
        "pack" => serial(out, Blocking::Pack(0)),
        "packslab" => serial(out, Blocking::Pack(value)),
        "threads" => {
            let per = m.div_ceil(value);
            std::thread::scope(|scope| {
                for (chunk, part) in out.chunks_mut(per * n).enumerate() {
                    let row_start = chunk * per;
                    let row_end = (row_start + per).min(m);
                    scope.spawn(move || unsafe {
                        product_part(
                            a,
                            b,
                            (k, n),
                            (row_start, row_end),
                            (0, n),
                            part,
                            n,
                            Blocking::Rows(current_rows_per_block(k)),
                        )
                    });
                }
            });
        }
        "cols" | "colspack" => {
            let blocking = if name == "cols" {
                Blocking::Rows(0)
            } else {
                Blocking::Pack(0)
            };
            let chunks = n.div_ceil(16);
            let per = chunks.div_ceil(value);
            let ranges: Vec<(usize, usize)> = (0..chunks)
                .step_by(per)
                .map(|c| (c * 16, ((c + per) * 16).min(n)))
                .collect();
            let parts: Vec<Vec<f64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = ranges
                    .iter()
                    .map(|&range| {
                        scope.spawn(move || {
                            let width = range.1 - range.0;
                            let mut part = vec![0.0; m * width];
                            unsafe {
                                product_part(
                                    a,
                                    b,
                                    (k, n),
                                    (0, m),
                                    range,
                                    &mut part,
                                    width,
                                    blocking,
                                )
                            };
                            part
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (&(start, end), part) in ranges.iter().zip(&parts) {
                let width = end - start;
                for row in 0..m {
                    out[row * n + start..row * n + end]
                        .copy_from_slice(&part[row * width..(row + 1) * width]);
                }
            }
        }
        _ => panic!("unknown variant {variant}"),
    }
}

// ---- a @ b.T: each output dot_product's 4 lanes

fn combine(l: [f64; 4]) -> f64 {
    (l[0] + l[1]) + (l[2] + l[3])
}

/// Lanes of rows `row..row + 4` x `b` rows `col..col + 2`, over groups `i_range` (each 4 wide).
#[target_feature(enable = "avx2,fma")]
unsafe fn nt_tile(
    a: *const f64,
    b: *const f64,
    k: usize,
    (row, col): (usize, usize),
    i_range: (usize, usize),
    lanes: *mut f64,
    n: usize,
) {
    unsafe {
        let lane_at = |r: usize, c: usize| lanes.add(((row + r) * n + col + c) * 4);
        let mut acc = [[_mm256_setzero_pd(); 2]; 4];
        for r in 0..4 {
            for c in 0..2 {
                acc[r][c] = _mm256_loadu_pd(lane_at(r, c));
            }
        }
        let mut i = i_range.0;
        while i < i_range.1 {
            let b_vecs = [
                _mm256_loadu_pd(b.add(col * k + i)),
                _mm256_loadu_pd(b.add((col + 1) * k + i)),
            ];
            for r in 0..4 {
                let a_vec = _mm256_loadu_pd(a.add((row + r) * k + i));
                for c in 0..2 {
                    acc[r][c] = _mm256_fmadd_pd(b_vecs[c], a_vec, acc[r][c]);
                }
            }
            i += 4;
        }
        for r in 0..4 {
            for c in 0..2 {
                _mm256_storeu_pd(lane_at(r, c), acc[r][c]);
            }
        }
    }
}

/// Rows `rows` of `a @ b.T`, `lanes` holding 4 doubles an output of all `m x n`.
#[target_feature(enable = "avx2,fma")]
unsafe fn nt_part(
    a: &[f64],
    b: &[f64],
    (k, n): (usize, usize),
    rows: (usize, usize),
    lanes: &mut [f64],
    out: &mut [f64],
    slab: usize,
) {
    let groups = k / 4 * 4;
    let slab = if slab == 0 { groups.max(4) } else { slab };
    let lanes_ptr = lanes.as_mut_ptr();
    let shifted = unsafe { lanes_ptr.sub(rows.0 * n * 4) };
    let mut i0 = 0;
    while i0 < groups {
        let i1 = (i0 + slab).min(groups);
        let mut row = rows.0;
        while row + 4 <= rows.1 {
            let mut col = 0;
            while col + 2 <= n {
                unsafe { nt_tile(a.as_ptr(), b.as_ptr(), k, (row, col), (i0, i1), shifted, n) };
                col += 2;
            }
            row += 4;
        }
        // the rows and columns past the tiles, lane by lane
        for r in rows.0..rows.1 {
            let tiled_col = if r < row { n / 2 * 2 } else { 0 };
            for col in tiled_col..n {
                let at = ((r - rows.0) * n + col) * 4;
                let mut i = i0;
                while i < i1 {
                    for lane in 0..4 {
                        lanes[at + lane] =
                            b[col * k + i + lane].mul_add(a[r * k + i + lane], lanes[at + lane]);
                    }
                    i += 4;
                }
            }
        }
        i0 = i1;
    }
    for r in rows.0..rows.1 {
        for col in 0..n {
            let at = ((r - rows.0) * n + col) * 4;
            let mut sum = combine([lanes[at], lanes[at + 1], lanes[at + 2], lanes[at + 3]]);
            for i in groups..k {
                sum = b[col * k + i].mul_add(a[r * k + i], sum);
            }
            out[(r - rows.0) * n + col] = sum;
        }
    }
}

/// `b` rows `col..col + 2` against `a` rows `row..row + 4`, written straight to `out`, as the crate's
/// `dot_product_tile_avx2_fma` does.
#[target_feature(enable = "avx2,fma")]
unsafe fn nt_direct_tile(
    a: *const f64,
    b: *const f64,
    (k, n): (usize, usize),
    (row, col): (usize, usize),
    out: *mut f64,
) {
    unsafe {
        let groups = k / 4 * 4;
        let mut acc = [[_mm256_setzero_pd(); 2]; 4];
        let mut i = 0;
        while i < groups {
            let b_vecs = [
                _mm256_loadu_pd(b.add(col * k + i)),
                _mm256_loadu_pd(b.add((col + 1) * k + i)),
            ];
            for (r, acc_row) in acc.iter_mut().enumerate() {
                let a_vec = _mm256_loadu_pd(a.add((row + r) * k + i));
                for (lane, b_vec) in acc_row.iter_mut().zip(&b_vecs) {
                    *lane = _mm256_fmadd_pd(*b_vec, a_vec, *lane);
                }
            }
            i += 4;
        }
        for (r, acc_row) in acc.iter().enumerate() {
            for (c, lane) in acc_row.iter().enumerate() {
                let mut l = [0.0; 4];
                _mm256_storeu_pd(l.as_mut_ptr(), *lane);
                let mut sum = combine(l);
                for t in groups..k {
                    sum = (*b.add((col + c) * k + t)).mul_add(*a.add((row + r) * k + t), sum);
                }
                *out.add((row + r) * n + col + c) = sum;
            }
        }
    }
}

/// One output of `a @ b.T` with no tile.
fn nt_one(a: &[f64], b: &[f64], (k, n): (usize, usize), (r, col): (usize, usize), out: &mut [f64]) {
    let groups = k / 4 * 4;
    let mut l = [0.0; 4];
    let mut i = 0;
    while i < groups {
        for (lane, value) in l.iter_mut().enumerate() {
            *value = b[col * k + i + lane].mul_add(a[r * k + i + lane], *value);
        }
        i += 4;
    }
    let mut sum = combine(l);
    for t in groups..k {
        sum = b[col * k + t].mul_add(a[r * k + t], sum);
    }
    out[r * n + col] = sum;
}

/// `a @ b.T` with `b` by groups of `group` rows (even; 0 for all of them) and every row of `a` past each.
#[target_feature(enable = "avx2,fma")]
unsafe fn nt_grouped(
    a: &[f64],
    b: &[f64],
    (m, k, n): (usize, usize, usize),
    out: &mut [f64],
    group: usize,
) {
    let group = if group == 0 { n.max(1) } else { group };
    let tiled_rows = m / 4 * 4;
    let tiled_cols = n / 2 * 2;
    let mut g0 = 0;
    while g0 < tiled_cols {
        let g1 = (g0 + group).min(tiled_cols);
        let mut row = 0;
        while row < tiled_rows {
            let mut col = g0;
            while col < g1 {
                unsafe {
                    nt_direct_tile(a.as_ptr(), b.as_ptr(), (k, n), (row, col), out.as_mut_ptr())
                };
                col += 2;
            }
            row += 4;
        }
        g0 = g1;
    }
    for r in 0..m {
        let from = if r < tiled_rows { tiled_cols } else { 0 };
        for col in from..n {
            nt_one(a, b, (k, n), (r, col), out);
        }
    }
}

fn nt(
    a: &[f64],
    b: &[f64],
    (m, k, n): (usize, usize, usize),
    out: &mut [f64],
    lanes: &mut [f64],
    variant: &str,
) {
    lanes.fill(0.0);
    let (name, value) = variant
        .split_once('=')
        .map_or((variant, 0), |(l, r)| (l, r.parse().unwrap()));
    match name {
        "current" => unsafe { nt_part(a, b, (k, n), (0, m), lanes, out, 0) },
        "group" => unsafe { nt_grouped(a, b, (m, k, n), out, value) },
        "slab" => unsafe { nt_part(a, b, (k, n), (0, m), lanes, out, value) },
        "threads" | "threadslab" => {
            let (threads, slab) = if name == "threads" {
                (value, 0)
            } else {
                (8, value)
            };
            let per = m.div_ceil(threads);
            std::thread::scope(|scope| {
                for (chunk, (part, lane_part)) in out
                    .chunks_mut(per * n)
                    .zip(lanes.chunks_mut(per * n * 4))
                    .enumerate()
                {
                    let row_start = chunk * per;
                    let row_end = (row_start + per).min(m);
                    scope.spawn(move || unsafe {
                        nt_part(a, b, (k, n), (row_start, row_end), lane_part, part, slab)
                    });
                }
            });
        }
        _ => panic!("unknown variant {variant}"),
    }
}

// ---- the runs

struct Case {
    shape: (usize, usize, usize),
    a: Vec<f64>,
    b: Vec<f64>,
    out: Vec<f64>,
    lanes: Vec<f64>,
}

fn case(kind: &str, shape: (usize, usize, usize)) -> Case {
    let (m, k, n) = shape;
    let b_len = if kind == "mm" { k * n } else { n * k };
    Case {
        shape,
        a: random_values(m * k, 1),
        b: random_values(b_len, 2),
        out: vec![0.0; m * n],
        lanes: vec![0.0; if kind == "nt" { m * n * 4 } else { 0 }],
    }
}

fn call(kind: &str, c: &mut Case, variant: &str) {
    match kind {
        "mm" => mm(&c.a, &c.b, c.shape, &mut c.out, variant),
        _ => nt(&c.a, &c.b, c.shape, &mut c.out, &mut c.lanes, variant),
    }
}

fn parse_shape(text: &str) -> (usize, usize, usize) {
    let parts: Vec<usize> = text.split('x').map(|p| p.parse().unwrap()).collect();
    (parts[0], parts[1], parts[2])
}

fn timed(kind: &str, c: &mut Case, variant: &str, target_ms: f64) -> (f64, f64, usize) {
    call(kind, c, variant); // warm-up
    let mut times = Vec::new();
    let started = Instant::now();
    while started.elapsed().as_secs_f64() * 1e3 < target_ms || times.len() < 5 {
        let t = Instant::now();
        call(kind, c, variant);
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0], times.len())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--one") {
        let (kind, variant) = args[2].split_once(':').unwrap();
        let mut c = case(kind, parse_shape(&args[1]));
        for _ in 0..args[3].parse::<usize>().unwrap() {
            call(kind, &mut c, variant);
        }
        println!("{}", c.out[0]);
        return;
    }
    if args.first().map(String::as_str) == Some("--groups") {
        let target_ms: f64 = args.get(1).map_or(300.0, |t| t.parse().unwrap());
        for shape in [
            "2048x2048x512",
            "2048x512x2048",
            "2048x512x512",
            "2048x64x256",
            "2048x64x64",
            "512x784x30",
            "32x784x30",
            "32x5408x32",
            "96x16x16",
        ] {
            let mut c = case("nt", parse_shape(shape));
            call("nt", &mut c, "current");
            let reference: Vec<u64> = c.out.iter().map(|v| v.to_bits()).collect();
            for round in 0..2 {
                for variant in [
                    "group=0",
                    "group=2",
                    "group=4",
                    "group=8",
                    "group=16",
                    "group=32",
                    "group=64",
                    "group=128",
                    "group=256",
                ] {
                    let (median, best, calls) = timed("nt", &mut c, variant, target_ms);
                    let same = c
                        .out
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(reference.iter().copied());
                    println!(
                        "{{\"kind\":\"nt\",\"shape\":\"{shape}\",\"variant\":\"{variant}\",\"round\":{round},\"median_us\":{median:.1},\"best_us\":{best:.1},\"calls\":{calls},\"same_bits\":{same}}}"
                    );
                }
            }
        }
        return;
    }
    let target_ms: f64 = args.first().map_or(300.0, |t| t.parse().unwrap());
    let plans: &[(&str, &[&str], &[&str])] = &[
        (
            "mm",
            &[
                "32x512x5408",
                "30x512x784",
                "32x32x5408",
                "30x32x784",
                "64x256x1024",
                "8x18432x64",
                "8x2048x64",
            ],
            &[
                "current",
                "rows=0",
                "slab=64",
                "pack",
                "packslab=64",
                "packslab=128",
                "packslab=256",
                "packslab=512",
                "threads=8",
                "cols=8",
                "colspack=8",
            ],
        ),
        (
            "nt",
            &[
                "512x5408x32",
                "512x784x30",
                "32x5408x32",
                "32x784x30",
                "64x128x256",
                "64x1024x64",
            ],
            &[
                "current",
                "slab=64",
                "slab=128",
                "slab=256",
                "slab=512",
                "slab=1024",
                "slab=2048",
                "slab=4096",
                "threads=8",
                "threadslab=256",
                "threadslab=512",
            ],
        ),
    ];
    for (kind, shapes, variants) in plans {
        for shape in *shapes {
            let mut c = case(kind, parse_shape(shape));
            call(kind, &mut c, "current");
            let reference: Vec<u64> = c.out.iter().map(|v| v.to_bits()).collect();
            for round in 0..2 {
                for variant in *variants {
                    let (median, best, calls) = timed(kind, &mut c, variant, target_ms);
                    let same = c
                        .out
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(reference.iter().copied());
                    println!(
                        "{{\"kind\":\"{kind}\",\"shape\":\"{shape}\",\"variant\":\"{variant}\",\"round\":{round},\"median_us\":{median:.1},\"best_us\":{best:.1},\"calls\":{calls},\"same_bits\":{same}}}"
                    );
                }
            }
        }
    }
}
