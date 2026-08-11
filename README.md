# Phibi

Phibi is a compiler-oriented inference project for decoder-only transformer models.

The goal is not to start by hand-writing CUDA kernels. The goal is to build a real compiler pipeline that can import a transformer graph, represent it in a high-level transformer IR, optimize that IR, lower it to GPU-specific IR, and eventually generate CUDA megakernels for fast single-token decode.

## Initial Scope

Phibi is intentionally narrow at the start.

Supported first:

- Decoder-only transformer models
- Batch size `1`
- Decode only, not prefill
- `BF16`
- RTX 5050 / Blackwell-class NVIDIA GPUs
- Gemma 3/4-style architectures

Out of scope initially:

- Training
- Prefill optimization
- Multi-batch serving
- Quantization
- Multi-GPU execution
- General-purpose tensor compilation
- Support for every Hugging Face model

The first target is specific:

> Compile decoder-only transformer graphs into optimized CUDA megakernels.

## Core Idea

Phibi should be built like a compiler, not like a CUDA kernel collection.

The project pipeline is:

```text
Exported model graph
        |
        v
Frontend
        |
        v
Transformer IR
        |
        v
Optimization passes
        |
        v
GPU IR
        |
        v
CUDA codegen
        |
        v
Runtime
```

CUDA is one backend of the compiler. It should not define the architecture of the whole project.

## Frontend

Phibi should not depend directly on Hugging Face Python module internals as its primary input format.

Instead, the frontend should consume an exported computation graph and translate it into Phibi's own transformer IR.

Preferred path:

```text
Hugging Face model
        |
        v
torch.export()
        |
        v
ATen graph
        |
        v
Phibi frontend
        |
        v
Transformer IR
```

Alternative path:

```text
Hugging Face model
        |
        v
ONNX
        |
        v
Phibi frontend
        |
        v
Transformer IR
```

This keeps Phibi closer to a production compiler: it consumes a stable graph representation and owns the compiler IR itself.

## Transformer Graph

The frontend should recover the structure of a decoder-only model:

```text
Embedding
    |
    v
Layer 0
    |
    v
Layer 1
    |
    v
...
    |
    v
LM Head
```

Each transformer layer should be recognized at the concept level:

```text
TransformerLayer
├── RMSNorm
├── QKV Projection
├── RoPE
├── Attention
├── Output Projection
├── Residual
├── RMSNorm
├── Gate Projection
├── Up Projection
├── SiLU
├── Down Projection
└── Residual
```

This graph is the compiler input. No CUDA is needed at this stage.

## Transformer IR

The most important part of Phibi is the high-level IR.

The IR should describe transformer concepts, not low-level memory operations.

Examples of IR nodes:

- `Module`
- `TransformerLayer`
- `Attention`
- `Linear`
- `RMSNorm`
- `Residual`
- `MLP`
- `RoPE`
- `LMHead`

The high-level IR should not be built from nodes like:

- `MatMul`
- `Load`
- `Store`
- `Memcpy`

Those belong to later lowering stages.

Example IR shape:

```text
Attention {
    hidden: 3072,
    heads: 24,
    kv_heads: 8,
    dtype: BF16,
}
```

The IR should know enough about transformer structure to make compiler passes meaningful.

## Optimization Passes

Optimization should happen as IR-to-IR transformations.

Example pass pipeline:

```text
PassManager
    |
    v
FusionPass
    |
    v
MemoryPass
    |
    v
SchedulingPass
    |
    v
LayoutPass
    |
    v
KernelFusionPass
```

Each pass transforms the IR without generating CUDA.

Example fusion:

```text
RMSNorm
    |
    v
Linear
```

becomes:

```text
RMSNormLinear
```

The purpose of this stage is to produce an optimized execution plan while staying independent of a specific backend.

## GPU Lowering

Only after high-level optimization should Phibi lower into GPU-specific IR.

Example:

```text
Attention
    |
    v
KernelIR
```

GPU IR may contain operations such as:

- `LoadTile`
- `TensorCoreMMA`
- `WarpReduce`
- `StoreTile`

This is where GPU execution details become explicit.

## Codegen

The CUDA backend should generate kernels from GPU IR.

Example output:

```cuda
__global__
void attention(...) {
    // generated kernel body
}
```

Initial codegen can use templates and compile generated CUDA with NVRTC.

The backend should evolve gradually from simple kernels to fused megakernels.

## Runtime

The runtime should stay small.

Responsibilities:

- Load weights
- Allocate and manage KV cache
- Launch generated kernels
- Sample tokens

The runtime should execute compiler output. It should not contain the compiler logic itself.

## Kernel Evolution

Phibi should replace kernels incrementally.

Initial target:

```text
Attention -> one generated kernel
```

Next:

```text
Attention + MLP -> one generated megakernel
```

Then:

```text
Whole transformer block -> one generated megakernel
```

Eventually:

```text
Whole decode step -> one generated megakernel
```

This keeps the project testable while moving toward the final architecture.

## Project Layout

Planned structure:

```text
phibi/
├── python/
├── compiler/
│   ├── frontend/
│   ├── ir/
│   ├── passes/
│   ├── lowering/
│   ├── scheduler/
│   ├── codegen/
│   └── backend/
├── runtime/
├── kernels/
├── benchmarks/
└── examples/
```

There is intentionally no top-level `cuda/` directory. CUDA is a backend, not the project identity.

## Milestones

### Milestone 1: Import A Transformer

Build a frontend that imports an exported decoder-only transformer graph and produces clean transformer IR.

Success means Phibi can identify embeddings, transformer layers, attention blocks, MLPs, norms, residuals, and the LM head from an exported graph.

### Milestone 2: Optimize The IR

Build compiler passes that transform transformer IR into an optimized execution plan.

Success means Phibi can run fusion, memory, scheduling, layout, and kernel-fusion passes without relying on CUDA codegen.

### Milestone 3: Generate CUDA

Build a CUDA backend that lowers optimized IR into GPU IR and generates one or more CUDA kernels.

Success means Phibi can compile a supported decode graph and execute generated kernels through the runtime.

## Design Principle

Adding support for Gemma, Qwen, Llama, or future decoder-only models should mostly mean teaching the frontend how to recognize their exported graphs.

It should not require rewriting the compiler.
