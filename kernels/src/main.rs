//! Hand-written reference kernels for Phibi, built with cuda-oxide.
//!
//! These are not the compiler's output. They are the baseline that generated
//! kernels are checked against, both for numerics and for speed. Every kernel
//! here is validated against a CPU oracle in `reference.rs` that follows the
//! Hugging Face module semantics.
//!
//! Run with `cargo oxide run` from this directory.

mod bf16;
mod reference;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

/// Threads per block for row-wise kernels. One block owns one row.
const BLOCK: u32 = 256;

#[cuda_module]
mod kernels {
    use cuda_device::convert::{cvt_bf16x2_f32, cvt_f32x2_bf16x2};
    use cuda_device::cooperative_groups::{block_reduce, ops::Sum, this_thread_block};
    use cuda_device::float::sqrt_rn_f32;
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread};

    const BLOCK: usize = super::BLOCK as usize;
    const WARPS: usize = BLOCK / 32;

    /// RMSNorm over rows of packed bf16x2, one block per row.
    ///
    /// `x` and `out` are `rows * n_pairs` words, `weight` is `n_pairs` words.
    /// Accumulates in f32 and rounds the normalized value to bf16 before the
    /// weight multiply, matching `Qwen2RMSNorm`.
    #[kernel]
    #[launch_bounds(256)]
    pub fn rms_norm_bf16(
        x: &[u32],
        weight: &[u32],
        eps: f32,
        n_pairs: u32,
        mut out: DisjointSlice<u32>,
    ) {
        static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let block = this_thread_block();
        let n = n_pairs as usize;
        let base = thread::blockIdx_x() as usize * n;
        let tid = thread::threadIdx_x() as usize;

        let mut sum_sq = 0.0f32;
        let mut j = tid;
        while j < n {
            let (a, b) = cvt_f32x2_bf16x2(x[base + j]);
            sum_sq += a * a + b * b;
            j += BLOCK;
        }

        // SAFETY: every thread in the block calls this once with the same scratch.
        let total = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum_sq, &raw mut SMEM) };
        let inv_rms = 1.0 / sqrt_rn_f32(total / (2 * n) as f32 + eps);

        let mut j = tid;
        while j < n {
            let (a, b) = cvt_f32x2_bf16x2(x[base + j]);
            let (wa, wb) = cvt_f32x2_bf16x2(weight[j]);
            let (na, nb) = cvt_f32x2_bf16x2(cvt_bf16x2_f32(a * inv_rms, b * inv_rms));
            // SAFETY: each (row, j) is written by exactly one thread.
            unsafe { *out.get_unchecked_mut(base + j) = cvt_bf16x2_f32(wa * na, wb * nb) };
            j += BLOCK;
        }
    }
}

/// Small deterministic generator so the harness needs no extra dependencies.
struct XorShift(u64);

impl XorShift {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }

    fn bf16_vec(&mut self, len: usize, lo: f32, hi: f32) -> Vec<u16> {
        (0..len)
            .map(|_| bf16::from_f32(self.next_f32(lo, hi)))
            .collect()
    }
}

/// Maximum bf16 ULP distance tolerated between GPU and CPU. Summation order
/// differs, so values sitting on a rounding boundary can land one ULP apart
/// before the weight multiply, which can widen to two after it.
const MAX_ULP: u32 = 2;

fn check_rms_norm(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    rows: usize,
    hidden: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let eps = 1e-6;
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15 ^ (rows * hidden) as u64);
    let x = rng.bf16_vec(rows * hidden, -4.0, 4.0);
    let weight = rng.bf16_vec(hidden, 0.5, 1.5);
    let expected = reference::rms_norm(&x, &weight, eps);

    let x_dev = DeviceBuffer::from_host(&stream, &bf16::pack(&x))?;
    let w_dev = DeviceBuffer::from_host(&stream, &bf16::pack(&weight))?;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, rows * hidden / 2)?;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one 256-thread block per row; buffers cover rows * hidden bf16 values.
    unsafe {
        module.rms_norm_bf16(
            &stream,
            cfg,
            &x_dev,
            &w_dev,
            eps,
            (hidden / 2) as u32,
            &mut out_dev,
        )
    }?;
    let got = bf16::unpack(&out_dev.to_host_vec(&stream)?);

    let ulps: Vec<u32> = got
        .iter()
        .zip(&expected)
        .map(|(&g, &e)| bf16::ulp_distance(g, e))
        .collect();
    let max_ulp = ulps.iter().copied().max().unwrap_or(0);
    let mismatched = ulps.iter().filter(|&&u| u > 0).count();
    let ok = max_ulp <= MAX_ULP;

    println!(
        "{} rms_norm_bf16 rows={rows:<3} hidden={hidden:<5} max_ulp={max_ulp} inexact={mismatched}/{}",
        if ok { "PASS" } else { "FAIL" },
        got.len(),
    );
    Ok(ok)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;

    // Qwen2.5-1.5B hidden size, decode (1 row) and a few rows to cover the grid.
    // 8960 exercises a size that is not a multiple of the block stride.
    let cases = [(1, 1536), (7, 1536), (1, 8960), (3, 2)];
    let mut all_ok = true;
    for (rows, hidden) in cases {
        all_ok &= check_rms_norm(&ctx, &module, rows, hidden)?;
    }

    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
