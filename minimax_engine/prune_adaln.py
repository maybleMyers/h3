#!/usr/bin/env python
"""Prune a diffusers-layout MiniMax-H3 transformer's AdaLN time projections into a curve basis.

`adaln_proj.linear` reads only `temb`, so its output is a function of `t` alone: a smooth 1-D
curve. Sampling that curve on a uniform grid over `t in [0, 1]` (H3's convention, `t = 1 - sigma`)
and truncating its SVD to `rank` folds the basis into every projection, turning the 51
`[96768, 2688]` bf16 matrices (13.0B params, ~26 GB) into `[96768, rank]` float32 ones and
replacing the timestep MLP with an `adaln_t_table` the transformer interpolates directly.

Usage:

    python minimax_engine/prune_adaln.py \
        --source /media/mayble/External/MiniMax-H3-diffusers/transformer \
        --output /media/mayble/External/MiniMax-H3-pruned/transformer-rank8.safetensors

Point the h3.py "DiT Override" box at the result. Ref2VA needs its own export from
`transformer_ref/`.
"""

import argparse
import glob
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TABLE_KEY = "adaln_t_table"
ADALN_WEIGHT_SUFFIX = ".linear.weight"
ADALN_BIAS_SUFFIX = ".linear.bias"
TIME_PREFIX = "time_embedder."
DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _is_adaln(key: str) -> bool:
    """The 51 projections driven by `temb`: one per block plus the final norm."""
    return (".adaln_proj" + ADALN_WEIGHT_SUFFIX in key or ".adaln_proj" + ADALN_BIAS_SUFFIX in key
            or key in ("norm_out.linear.weight", "norm_out.linear.bias"))


def _is_adaln_weight(key: str) -> bool:
    return _is_adaln(key) and key.endswith(".weight")


def _read_header(path):
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return 8 + header_size, header


def _source_files(source: Path):
    if source.is_dir():
        files = sorted(glob.glob(os.path.join(source, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors shards in {source}")
        return [Path(f) for f in files]
    return [source]


def _catalog(files):
    """{key: (path, data_start, entry)} in shard order, plus the config dir for `config.json`."""
    entries = {}
    order = []
    for path in files:
        data_start, header = _read_header(path)
        header.pop("__metadata__", None)
        for key, entry in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if key in entries:
                raise ValueError(f"duplicate key {key} across shards")
            entries[key] = (path, data_start, entry)
            order.append(key)
    return entries, order


def _config(source: Path, config_path):
    if config_path:
        path = Path(config_path)
    elif source.is_dir():
        path = source / "config.json"
    else:
        path = source.parent / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"no config.json at {path}; pass --config")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    return config, path


def _time_curve(entries, config, grid, device):
    """Sample `silu(time_embedder(time_proj(t)))` on a uniform grid over t in [0, 1].

    Built from the checkpoint's own `time_proj`/`time_embedder` modules rather than a
    re-derived sinusoid, so the sampled curve cannot drift from what the model computes. The
    trailing silu is the one `MiniMaxH3AdaLayerNormModulation` applies to `temb`; curve-form
    checkpoints run with `apply_silu=False` and consume the activated curve directly.
    """
    from minimax_video.transformer import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        skeleton = MiniMaxH3Transformer3DModel.from_config(config)
    time_proj, time_embedder = skeleton.time_proj, skeleton.time_embedder
    if time_embedder is None:
        raise ValueError("source checkpoint is already curve-pruned")

    sd = {k[len(TIME_PREFIX):]: _read_tensor(*entries[k]) for k in entries if k.startswith(TIME_PREFIX)}
    if not sd:
        raise KeyError(f"no {TIME_PREFIX}* weights in the source checkpoint")
    time_embedder.load_state_dict(sd, strict=True, assign=True)
    time_embedder = time_embedder.to(device=device, dtype=torch.float32)

    t = torch.linspace(0.0, 1.0, grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        return F.silu(time_embedder(time_proj(t).to(torch.float32)))


def _factor_curve(curve, rank):
    """Mean-centered rank-`rank` SVD. The mean folds into the projection biases."""
    curve64 = curve.double()
    mean64 = curve64.mean(dim=0)
    u, singular, vh = torch.linalg.svd(curve64 - mean64, full_matrices=False)
    table = (u[:, :rank] * singular[:rank]).float()
    basis = vh[:rank].float()
    mean = mean64.float()
    error = table @ basis + mean - curve
    metrics = {
        "curve_rmse": error.square().mean().sqrt().item(),
        "curve_max_abs": error.abs().max().item(),
        "retained_energy": (singular[:rank].square().sum() / singular.square().sum()).item(),
    }
    return table, basis, mean, metrics


_DTYPE_FROM_NAME = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def _read_tensor(path, data_start, entry):
    start, end = entry["data_offsets"]
    with open(path, "rb") as stream:
        stream.seek(data_start + start)
        buffer = bytearray(stream.read(end - start))
    dtype = _DTYPE_FROM_NAME.get(entry["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported source dtype {entry['dtype']}")
    return torch.frombuffer(buffer, dtype=dtype).reshape(entry["shape"])


def _compress(entries, base, basis, curve_mean, device, row_chunk):
    """`W @ V_r.T` and `b + W @ mean` for one projection, in row chunks (each source weight is
    ~520 MB bf16, so it is never held on the accelerator whole)."""
    weight_path, weight_start, weight_entry = entries[base + ".weight"]
    bias_path, bias_start, bias_entry = entries[base + ".bias"]
    source_weight = _read_tensor(weight_path, weight_start, weight_entry)
    source_bias = _read_tensor(bias_path, bias_start, bias_entry)
    rows = source_weight.shape[0]
    weight = torch.empty(rows, basis.shape[0], dtype=torch.float32)
    bias = torch.empty(rows, dtype=torch.float32)
    basis_t = basis.T.contiguous()
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        chunk = source_weight[start:stop].to(device=device, dtype=torch.float32)
        weight[start:stop].copy_((chunk @ basis_t).cpu())
        bias[start:stop].copy_((source_bias[start:stop].to(device=device, dtype=torch.float32)
                                + chunk @ curve_mean).cpu())
    return weight, bias


def _output_layout(entries, order, grid, rank):
    """Output keys in source (shard, offset) order so reads and writes both stay sequential."""
    names = [k for k in order if not k.startswith(TIME_PREFIX)]
    names.append(TABLE_KEY)
    out = {}
    offset = 0
    for name in names:
        if name == TABLE_KEY:
            entry = {"dtype": "F32", "shape": [grid, rank]}
        elif _is_adaln_weight(name):
            entry = {"dtype": "F32", "shape": [entries[name][2]["shape"][0], rank]}
        elif _is_adaln(name):
            entry = {"dtype": "F32", "shape": entries[name][2]["shape"]}
        else:
            source = entries[name][2]
            entry = {"dtype": source["dtype"], "shape": list(source["shape"])}
        size = math.prod(entry["shape"]) * DTYPE_SIZES[entry["dtype"]]
        out[name] = {**entry, "data_offsets": [offset, offset + size]}
        offset += size
    return names, out, offset


def _write_tensor(stream, tensor):
    data = memoryview(tensor.contiguous().numpy()).cast("B")
    if stream.write(data) != len(data):
        raise OSError("incomplete write")


def _copy_through(source_path, data_start, entry, target, progress, chunk_size=32 << 20):
    start, end = entry["data_offsets"]
    with open(source_path, "rb") as stream:
        stream.seek(data_start + start)
        remaining = end - start
        while remaining:
            data = stream.read(min(remaining, chunk_size))
            if not data:
                raise OSError(f"unexpected end of {source_path}")
            target.write(data)
            progress.update(len(data))
            remaining -= len(data)


def prune(source, output, grid, rank, device, row_chunk, config_path):
    files = _source_files(source)
    entries, order = _catalog(files)
    config, resolved_config = _config(source, config_path)
    if TABLE_KEY in entries:
        raise ValueError("source checkpoint is already curve-pruned")

    curve = _time_curve(entries, config, grid, device)
    table, basis, curve_mean, metrics = _factor_curve(curve, rank)
    names, out_entries, out_size = _output_layout(entries, order, grid, rank)

    header = json.dumps(
        {
            "__metadata__": {
                "format": "pt",
                "adaln_curve_grid": str(grid),
                "adaln_curve_rank": str(rank),
                "adaln_curve_centered": "true",
                "source_config": str(resolved_config),
            },
            **out_entries,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)

    partial = output.with_suffix(output.suffix + ".partial")
    pending = {}
    with open(partial, "wb") as target:
        target.write(struct.pack("<Q", len(header)))
        target.write(header)
        with tqdm(total=out_size, unit="B", unit_scale=True, desc="writing pruned H3") as progress:
            for name in names:
                if name == TABLE_KEY:
                    _write_tensor(target, table.cpu())
                    progress.update(table.numel() * 4)
                elif _is_adaln(name):
                    base = name.rsplit(".", 1)[0]
                    if base not in pending:
                        weight, bias = _compress(entries, base, basis, curve_mean, device, row_chunk)
                        pending[base] = {"weight": weight, "bias": bias}
                    tensor = pending[base].pop(name.rsplit(".", 1)[1])
                    if not pending[base]:
                        del pending[base]
                    _write_tensor(target, tensor)
                    progress.update(tensor.numel() * 4)
                else:
                    path, data_start, entry = entries[name]
                    _copy_through(path, data_start, entry, target, progress)
    if pending:
        raise RuntimeError(f"unwritten pruned tensors: {sorted(pending)}")
    os.replace(partial, output)
    return metrics


def _interpolate(table, timesteps):
    """The transformer's own table lookup (transformer.py `use_adaln_curves` branch)."""
    position = timesteps.clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))


def verify(source, output, device, row_chunk, config_path, test_timesteps=97):
    """Compare exact modulation against the pruned form on timesteps *off* the grid, so the
    reported error includes table interpolation and not just SVD truncation."""
    files = _source_files(source)
    entries, _ = _catalog(files)
    config, _ = _config(source, config_path)
    out_entries, _ = _catalog([output])

    if any(k.startswith(TIME_PREFIX) for k in out_entries):
        raise RuntimeError("pruned checkpoint still carries the timestep MLP")
    path, data_start, entry = out_entries[TABLE_KEY]
    table = _read_tensor(path, data_start, entry).to(device)
    projections = sorted(k for k in out_entries if _is_adaln_weight(k))
    expected = config["num_layers"] + 1  # one per block, plus norm_out
    if len(projections) != expected:
        raise RuntimeError(f"expected {expected} pruned AdaLN projections, found {len(projections)}")
    if any(out_entries[k][2]["shape"][1] != table.shape[1] for k in projections):
        raise RuntimeError("a pruned projection has the wrong input width")

    timesteps = torch.linspace(0.0, 1.0, test_timesteps, device=device)
    exact_curve = _time_curve(entries, config, test_timesteps, device)
    coordinates = _interpolate(table, timesteps)

    squared_error = squared_reference = 0.0
    count = 0
    max_abs = 0.0
    for name in tqdm(projections, desc="verifying AdaLN projections"):
        base = name[: -len(".weight")]
        source_weight = _read_tensor(*entries[name])
        source_bias = _read_tensor(*entries[base + ".bias"])
        pruned_weight = _read_tensor(*out_entries[name])
        pruned_bias = _read_tensor(*out_entries[base + ".bias"])
        for start in range(0, source_weight.shape[0], row_chunk):
            stop = min(start + row_chunk, source_weight.shape[0])
            expected = F.linear(
                exact_curve,
                source_weight[start:stop].to(device=device, dtype=torch.float32),
                source_bias[start:stop].to(device=device, dtype=torch.float32),
            )
            actual = F.linear(
                coordinates, pruned_weight[start:stop].to(device), pruned_bias[start:stop].to(device)
            )
            error = actual - expected
            squared_error += error.double().square().sum().item()
            squared_reference += expected.double().square().sum().item()
            count += error.numel()
            max_abs = max(max_abs, error.abs().max().item())
    return {
        "tensor_count": len(out_entries),
        "projection_count": len(projections),
        "grid": table.shape[0],
        "rank": table.shape[1],
        "modulation_rmse": math.sqrt(squared_error / count),
        "modulation_relative_rmse": math.sqrt(squared_error / squared_reference),
        "modulation_max_abs": max_abs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="diffusers-layout transformer dir or merged file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default=None, help="config.json override (default: alongside --source)")
    parser.add_argument("--grid", type=int, default=1025)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    device = torch.device(args.device)
    if args.verify_only:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        print(json.dumps(verify(args.source, args.output, device, args.row_chunk, args.config), indent=2))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    factorization = prune(args.source, args.output, args.grid, args.rank, device, args.row_chunk, args.config)
    verification = verify(args.source, args.output, device, args.row_chunk, args.config)
    print(json.dumps({"curve_factorization": factorization, "checkpoint_verification": verification}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
