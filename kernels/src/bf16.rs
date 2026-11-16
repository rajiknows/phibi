//! Host-side bf16 helpers. Values are carried as raw `u16` bit patterns and
//! packed two per `u32` (low half first), the same layout the device kernels
//! read through `cuda_device::convert`.

/// Round an `f32` to the nearest bf16, ties to even (matches `torch.to(bfloat16)`).
pub fn from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

pub fn to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-trip through bf16, i.e. what a `.to(bfloat16)` does to an f32 value.
pub fn round(value: f32) -> f32 {
    to_f32(from_f32(value))
}

pub fn pack(values: &[u16]) -> Vec<u32> {
    assert!(
        values.len().is_multiple_of(2),
        "bf16x2 packing needs an even length"
    );
    values
        .chunks_exact(2)
        .map(|pair| pair[0] as u32 | ((pair[1] as u32) << 16))
        .collect()
}

pub fn unpack(packed: &[u32]) -> Vec<u16> {
    packed
        .iter()
        .flat_map(|&word| [word as u16, (word >> 16) as u16])
        .collect()
}

/// Distance between two bf16 values in units in the last place.
pub fn ulp_distance(a: u16, b: u16) -> u32 {
    // Map sign-magnitude onto a monotonic integer line.
    let ordered = |bits: u16| {
        if bits & 0x8000 != 0 {
            -((bits & 0x7FFF) as i32)
        } else {
            bits as i32
        }
    };
    ordered(a).abs_diff(ordered(b))
}
