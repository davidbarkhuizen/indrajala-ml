//! What handing one call to threads costs, with the work a call does fixed: `std::thread::scope`, as
//! the crate's threaded products do today, against a parked pool, the prototype of the crate
//! performance workplan's D3 (no crate code: it prices the design, and its spin, before stage 9).
//! Built and run by `thread_cost_probe.py calls`, one process a cell, in a rotating order.
//!
//!     thread_call_cost MECHANISM THREADS IDLE_US WORK_K SPIN_US LOOPS TARGET_MS
//!
//! - MECHANISM: `scoped` spawns THREADS threads and joins them; `pool` keeps THREADS - 1 workers, and
//!   the caller works too. Chunks are claimed from an atomic counter, as D3 has it.
//! - IDLE_US: sleep this long before each call, timing only the call, so the workers park and the cores
//!   cool as they do between a training step's products. 0 is back-to-back, the warm reading.
//! - WORK_K: thousands of steps of arithmetic a call does, split over its THREADS chunks; no memory
//!   traffic, so this prices the hand-off and the clocks, not the memory bus. 0 is an empty call.
//! - SPIN_US: how long a pool worker polls for the next job before it parks (`pool` only).
//!
//! It prints one JSON object: median, min and max over LOOPS of the µs per call.

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

fn work(steps: u64) {
    let mut x = 1.0f64;
    for _ in 0..steps {
        x = black_box(x * 1.000_000_1 + 1e-9);
    }
    black_box(x);
}

struct Shared {
    generation: AtomicU64,
    chunks: AtomicUsize,
    steps: AtomicU64,
    next_chunk: AtomicUsize,
    done: AtomicUsize,
    sleepers: AtomicUsize,
    stop: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    fn claim_chunks(&self) {
        while self.next_chunk.fetch_add(1, SeqCst) < self.chunks.load(SeqCst) {
            work(self.steps.load(SeqCst));
            self.done.fetch_add(1, SeqCst);
        }
    }

    /// Back when the generation has moved from `seen`: polled for `spin`, then parked.
    fn wait_for_job(&self, seen: u64, spin: Duration) -> u64 {
        let start = Instant::now();
        while start.elapsed() < spin {
            let generation = self.generation.load(SeqCst);
            if generation != seen {
                return generation;
            }
            std::hint::spin_loop();
        }
        let mut guard = self.lock.lock().unwrap();
        self.sleepers.fetch_add(1, SeqCst);
        while self.generation.load(SeqCst) == seen {
            guard = self.wake.wait(guard).unwrap();
        }
        self.sleepers.fetch_sub(1, SeqCst);
        self.generation.load(SeqCst)
    }

    fn publish(&self) {
        self.generation.fetch_add(1, SeqCst);
        if self.sleepers.load(SeqCst) > 0 {
            drop(self.lock.lock().unwrap()); // a worker past its check is in wait() before this returns
            self.wake.notify_all();
        }
    }
}

struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(workers: usize, spin: Duration) -> Pool {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            chunks: AtomicUsize::new(0),
            steps: AtomicU64::new(0),
            next_chunk: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (0..workers)
            .map(|_| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        seen = shared.wait_for_job(seen, spin);
                        if shared.stop.load(SeqCst) {
                            return;
                        }
                        shared.claim_chunks();
                    }
                })
            })
            .collect();
        Pool { shared, workers }
    }

    fn call(&self, chunks: usize, steps: u64) {
        let shared = &self.shared;
        shared.chunks.store(chunks, SeqCst);
        shared.steps.store(steps, SeqCst);
        shared.done.store(0, SeqCst);
        shared.next_chunk.store(0, SeqCst);
        shared.publish();
        shared.claim_chunks();
        while shared.done.load(SeqCst) < chunks {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        self.shared.publish();
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

fn scoped_call(threads: usize, steps: u64) {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(move || work(steps));
        }
    });
}

/// The µs per call of each of `loops` loops, each loop's length sized to about `target`.
fn timed(call: &dyn Fn(), idle: Duration, loops: usize, target: Duration) -> Vec<f64> {
    let one_call = |total: &mut Duration| {
        if !idle.is_zero() {
            std::thread::sleep(idle);
        }
        let start = Instant::now();
        call();
        *total += start.elapsed();
    };
    let mut warm_up = Duration::ZERO;
    for _ in 0..20 {
        one_call(&mut warm_up);
    }
    let cost = (warm_up / 20 + idle).as_secs_f64();
    let calls = ((target.as_secs_f64() / cost) as usize).max(5);
    (0..loops)
        .map(|_| {
            let mut total = Duration::ZERO;
            for _ in 0..calls {
                one_call(&mut total);
            }
            total.as_secs_f64() * 1e6 / calls as f64
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mechanism, numbers @ ..] = &args[..] else {
        panic!("usage: thread_call_cost MECHANISM THREADS IDLE_US WORK_K SPIN_US LOOPS TARGET_MS");
    };
    let numbers: Vec<u64> = numbers.iter().map(|arg| arg.parse().expect("a count")).collect();
    let [threads, idle_us, work_k, spin_us, loops, target_ms] = numbers[..] else {
        panic!("usage: thread_call_cost MECHANISM THREADS IDLE_US WORK_K SPIN_US LOOPS TARGET_MS");
    };
    let chunks = threads as usize;
    let steps = work_k * 1000 / threads;
    let idle = Duration::from_micros(idle_us);
    let target = Duration::from_millis(target_ms);

    let mut per_call = match mechanism.as_str() {
        "scoped" => timed(&|| scoped_call(chunks, steps), idle, loops as usize, target),
        "pool" => {
            let pool = Pool::new(chunks - 1, Duration::from_micros(spin_us));
            timed(&|| pool.call(chunks, steps), idle, loops as usize, target)
        }
        other => panic!("{other}: scoped or pool"),
    };
    per_call.sort_by(f64::total_cmp);
    println!(
        "{{\"median_us\": {}, \"min_us\": {}, \"max_us\": {}}}",
        per_call[per_call.len() / 2],
        per_call[0],
        per_call[per_call.len() - 1]
    );
}
