"""Export a decoder-only transformer to a Phibi raw FX graph.

Two modes:

  decode  (default) -- batch 1, seq_len 1, static KV cache. This is the graph
          Phibi actually targets: a single decode step, matching the README
          scope (BF16, batch 1, decode only).

  prefill -- batch 1, seq_len N, no KV cache. Cheaper to produce and easier to
          trace, but it contains no cache reads/writes. Useful only as a
          frontend parsing fixture.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = REPO_ROOT / "examples" / "fx_debug"
GRAPH_DIR = REPO_ROOT / "examples" / "graphs"

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static KV cache capacity, decode mode only.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Prompt length, prefill mode only.",
    )
    parser.add_argument(
        "--cache-position",
        type=int,
        default=16,
        help="Which decode step to trace, decode mode only. Must be > 0 so the "
        "graph contains a non-empty cache read.",
    )
    return parser.parse_args()


# --- graph serialization ---------------------------------------------------


def jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    try:
        return int(value)
    except (TypeError, ValueError):
        return repr(value)


def node_ref(value):
    if isinstance(value, torch.fx.Node):
        return {
            "kind": "node",
            "name": value.name,
            "op": value.op,
            "target": str(value.target),
        }

    if isinstance(value, (list, tuple)):
        return [node_ref(v) for v in value]

    if isinstance(value, dict):
        return {str(k): node_ref(v) for k, v in value.items()}

    return jsonable(value)


def tensor_meta_to_dict(meta):
    if meta is None:
        return None

    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    requires_grad = getattr(meta, "requires_grad", None)
    stride = getattr(meta, "stride", None)

    return {
        "shape": jsonable(list(shape)) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "requires_grad": jsonable(requires_grad),
        "stride": jsonable(list(stride)) if stride is not None else None,
    }


def serialize_nodes(fx_graph):
    return [
        {
            "index": idx,
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": node_ref(node.args),
            "kwargs": node_ref(node.kwargs),
            "users": [user.name for user in node.users],
            "tensor_meta": tensor_meta_to_dict(node.meta.get("tensor_meta")),
        }
        for idx, node in enumerate(fx_graph.nodes)
    ]


def write_debug_dumps(exported_program, fx_graph, slug):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    with (DEBUG_DIR / f"{slug}_nodes.txt").open("w") as f:
        for idx, node in enumerate(fx_graph.nodes):
            f.write(f"[{idx}] name={node.name}\n")
            f.write(f"    op={node.op}\n")
            f.write(f"    target={node.target}\n")
            f.write(f"    args={node_ref(node.args)}\n")
            f.write(f"    kwargs={node_ref(node.kwargs)}\n")
            f.write(f"    users={[user.name for user in node.users]}\n")
            f.write(f"    meta={tensor_meta_to_dict(node.meta.get('tensor_meta'))}\n")
            f.write("\n")

    with (DEBUG_DIR / f"{slug}_graph_code.py").open("w") as f:
        f.write(exported_program.graph_module.code)


# --- tracing ---------------------------------------------------------------


def export_prefill(model, opts, dtype):
    """Trace a prompt-length forward pass with caching disabled."""
    tokens = torch.randint(
        0, model.config.vocab_size, (1, opts.seq_len), dtype=torch.long
    ).to(opts.device)

    with torch.no_grad():
        exported_program = torch.export.export(
            model, args=(tokens,), kwargs={"use_cache": False}, strict=True
        )

    return exported_program, {
        "batch_size": 1,
        "seq_len": opts.seq_len,
        "use_cache": False,
    }


def build_static_cache_module(model, opts, dtype):
    """Wrap the model so its KV cache lives in buffers, not a Cache object.

    torch.export cannot trace Hugging Face's dynamic Cache classes directly.
    TorchExportableModuleWithStaticCache holds the cache as registered buffers,
    which keeps the traced graph pure-tensor and gives us real cache reads and
    writes in the FX graph.
    """
    try:
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )
    except ImportError as exc:  # pragma: no cover - depends on transformers build
        raise RuntimeError(
            "Could not import TorchExportableModuleWithStaticCache from "
            "transformers.integrations.executorch. Use --mode prefill, or pin a "
            "transformers version that ships the executorch integration."
        ) from exc

    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_length = opts.max_cache_len
    if hasattr(model.generation_config, "cache_config"):
        model.generation_config.cache_config = {
            "batch_size": 1,
            "max_cache_len": opts.max_cache_len,
            "device": opts.device,
            "dtype": dtype,
        }

    # The wrapper's signature has churned across transformers releases; pass
    # only the keyword arguments this build actually accepts.
    optional = {
        "max_batch_size": 1,
        "batch_size": 1,
        "max_cache_len": opts.max_cache_len,
    }
    accepted = inspect.signature(TorchExportableModuleWithStaticCache).parameters
    kwargs = {k: v for k, v in optional.items() if k in accepted}

    return TorchExportableModuleWithStaticCache(model, **kwargs)


def export_decode(model, opts, dtype):
    """Trace a single decode step reading from and writing to a static cache."""
    if opts.cache_position <= 0:
        raise ValueError("--cache-position must be > 0 to get a non-empty cache read")

    wrapped = build_static_cache_module(model, opts, dtype)

    token = torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.long).to(
        opts.device
    )
    cache_position = torch.tensor([opts.cache_position], dtype=torch.long).to(
        opts.device
    )

    with torch.no_grad():
        exported_program = torch.export.export(
            wrapped, args=(token, cache_position), strict=True
        )

    return exported_program, {
        "batch_size": 1,
        "seq_len": 1,
        "use_cache": True,
        "cache_kind": "static",
        "max_cache_len": opts.max_cache_len,
        "cache_position": opts.cache_position,
    }


def main():
    opts = parse_args()
    dtype = DTYPES[opts.dtype]

    print(f"Loading {opts.model_id} ({opts.dtype}) onto {opts.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        opts.model_id,
        dtype=dtype,
        device_map=opts.device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Tracing {opts.mode} step into a functional ATen FX graph...")
    tracer = export_decode if opts.mode == "decode" else export_prefill
    exported_program, inputs = tracer(model, opts, dtype)

    fx_graph = exported_program.graph_module.graph
    nodes = serialize_nodes(fx_graph)

    model_slug = opts.model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    slug = f"{model_slug}_{opts.mode}"

    write_debug_dumps(exported_program, fx_graph, slug)

    raw_graph = {
        "format": "phibi.raw_fx_graph.v0",
        "model_id": opts.model_id,
        "source": "torch.export",
        "mode": opts.mode,
        "dtype": opts.dtype,
        "inputs": inputs,
        "nodes": nodes,
    }

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    raw_graph_path = GRAPH_DIR / f"{slug}_fx_graph.json"
    with raw_graph_path.open("w") as f:
        json.dump(raw_graph, f, indent=2)

    print(f"Wrote FX debug files to {DEBUG_DIR}")
    print(f"Wrote raw graph to {raw_graph_path}")
    print(f"Total nodes: {len(nodes)}")

    print("\n=== First 30 nodes ===")
    for node in nodes[:30]:
        print(
            f"Node: {node['name']:<15} | Op: {node['op']:<13} "
            f"| Target: {node['target']:<35} | Users: {len(node['users'])}"
        )


if __name__ == "__main__":
    main()
