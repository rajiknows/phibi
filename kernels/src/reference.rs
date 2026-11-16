//! CPU reference implementations used as numerical oracles for the GPU
//! kernels. They follow the Hugging Face module semantics op for op, including
//! where intermediate values are rounded to bf16.

use crate::bf16;

/// `Qwen2RMSNorm` / `LlamaRMSNorm`:
///
/// ```text
/// x = x.to(f32)
/// x = x * rsqrt(mean(x^2) + eps)
/// return weight * x.to(bf16)
/// ```
pub fn rms_norm(x: &[u16], weight: &[u16], eps: f32) -> Vec<u16> {
    let hidden = weight.len();
    assert!(x.len().is_multiple_of(hidden));

    x.chunks_exact(hidden)
        .flat_map(|row| {
            let sum_sq: f64 = row
                .iter()
                .map(|&v| {
                    let v = bf16::to_f32(v) as f64;
                    v * v
                })
                .sum();
            let inv_rms = 1.0 / ((sum_sq / hidden as f64) as f32 + eps).sqrt();

            row.iter().zip(weight).map(move |(&v, &w)| {
                let normed = bf16::round(bf16::to_f32(v) * inv_rms);
                bf16::from_f32(bf16::to_f32(w) * normed)
            })
        })
        .collect()
}
