# Phibi agent notes

## Layout and toolchains

- `compiler/` – Rust compiler crate, normal stable toolchain.
  - Test: `cargo test --manifest-path compiler/Cargo.toml`
- `kernels/` – hand-written reference GPU kernels in cuda-oxide (NVIDIA cuda-rust).
  Standalone workspace pinned to Rust `1.99.1` via `kernels/rust-toolchain.toml`.
  - Build + run checks (from `kernels/`): `cargo oxide run`
  - Environment check: `cargo oxide doctor`
  - Print generated PTX: `cargo oxide inspect`
  - Each kernel is checked against a CPU oracle in `kernels/src/reference.rs` that follows
    Hugging Face semantics (including intermediate bf16 rounding). Keep new kernels bit-exact
    or within the documented ULP tolerance.
- `python/` – graph exporter (`python/export_graph.py`), run with `.venv/bin/python`
  (there is no bare `python` on PATH).

## Machine

- GPU: RTX 5050 Laptop, compute capability 12.0 (cuda-oxide detects `sm_120a`).
  No WGMMA/tcgen05; use `mma.sync`/`ldmatrix` for tensor cores.
- CUDA 13.3 at `/usr/local/cuda-13.3`. cuda-oxide uses the Rust toolchain's bundled `llc`.
- `cargo-oxide` installed with `cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide`.
  The cuda-rust revision is pinned in `kernels/Cargo.toml`; bump all three crates together.
