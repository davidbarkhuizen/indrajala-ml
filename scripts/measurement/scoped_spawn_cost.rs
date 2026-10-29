//! The cost of one threaded call's thread starts in the crate, with no work in the threads: a
//! `std::thread::scope` that spawns one thread per chunk and joins them, as `for_each_row_range`,
//! `matmul_long_k` and conv's per-example downstream do (the crate kernels workplan, stage 3, D3).
//! Built and run by `thread_cost_probe.py`; prints one JSON object per thread count.
//!
//!     scoped_spawn_cost MAX_THREADS LOOPS CALLS

use std::hint::black_box;
use std::time::Instant;

fn call(threads: usize, out: &mut [u64]) {
    std::thread::scope(|scope| {
        for slot in out.iter_mut().take(threads) {
            scope.spawn(move || *slot = black_box(*slot) + 1);
        }
    });
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("MAX_THREADS LOOPS CALLS, each a count"))
        .collect();
    let [max_threads, loops, calls] = args[..] else {
        panic!("usage: scoped_spawn_cost MAX_THREADS LOOPS CALLS");
    };
    let mut out = vec![0u64; max_threads];
    for threads in 1..=max_threads {
        for _ in 0..calls {
            call(threads, &mut out); // warm-up
        }
        let mut per_call: Vec<f64> = (0..loops)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..calls {
                    call(threads, &mut out);
                }
                start.elapsed().as_secs_f64() * 1e6 / calls as f64
            })
            .collect();
        per_call.sort_by(f64::total_cmp);
        println!(
            "{{\"threads\": {threads}, \"median_us\": {}, \"min_us\": {}, \"max_us\": {}}}",
            per_call[loops / 2],
            per_call[0],
            per_call[loops - 1]
        );
    }
}
