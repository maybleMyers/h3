"""Static (CPU) tests for the int8 convrot support (minimax_video/int8_quant.py + the
adaln-curve transformer form + the model_loader int8 path).

Run from the repo root with the project venv:

    env/bin/python minimax_engine/tests/run_static_tests.py test_minimax_int8
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
for _p in (_ENGINE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _env_compat  # noqa: F401,E402  (works around the local bitsandbytes/triton breakage)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from minimax_video import int8_quant  # noqa: E402
from minimax_video.int8_quant import (  # noqa: E402
    apply_int8_monkey_patch,
    build_regular_hadamard,
    collect_quant_markers,
    convert_int8_dit_tensor,
    dequantize_int8_weight,
    marker_groupsize,
    parse_quant_marker,
    quantize_int8_convrot_weight,
    quantize_int8_rowwise,
    rotate_activation,
    rotate_weight_rows,
)

GS = 4  # smallest legal convrot group size (powers of 4)


def _marker(convrot: bool = True, groupsize: int = GS) -> torch.Tensor:
    info = {"format": "int8_tensorwise"}
    if convrot:
        info.update({"convrot": True, "convrot_groupsize": groupsize})
    return torch.tensor(list(json.dumps(info).encode("utf-8")), dtype=torch.uint8)


# --------------------------------------------------------------------------- rotation math


def test_hadamard_properties():
    for size in (4, 16, 64, 256):
        h = build_regular_hadamard(size, dtype=torch.float64)
        assert h.shape == (size, size)
        # symmetric + orthonormal => involutory, which is what makes de-rotation the same op
        assert torch.allclose(h, h.T)
        assert torch.allclose(h @ h.T, torch.eye(size, dtype=torch.float64), atol=1e-12)
        # regular Hadamard: every entry is +/- 1/sqrt(size)
        assert torch.allclose(h.abs(), torch.full_like(h, size**-0.5))


def test_hadamard_rejects_non_power_of_4():
    for bad in (2, 8, 32, 100):
        with pytest.raises(ValueError):
            build_regular_hadamard(bad)


def test_rotation_is_involutory_and_preserves_linear():
    torch.manual_seed(0)
    w = torch.randn(6, 8, dtype=torch.float64)
    x = torch.randn(5, 8, dtype=torch.float64)
    w_rot = rotate_weight_rows(w, GS)
    # applying the grouped transform twice returns the original
    assert torch.allclose(rotate_weight_rows(w_rot, GS), w, atol=1e-12)
    # rotating both operands preserves the GEMM: (x @ H_blk) @ (W @ H_blk^T)^T == x @ W^T
    x_rot = rotate_activation(x, GS)
    assert torch.allclose(x_rot @ w_rot.T, x @ w.T, atol=1e-10)


def test_quantize_matches_reference_formulas():
    torch.manual_seed(1)
    w = torch.randn(6, 8)
    q, scale = quantize_int8_convrot_weight(w, GS)
    # reference math straight from the int8_tensorwise format spec
    h = build_regular_hadamard(GS)
    w_rot = (w.reshape(6, 2, GS) @ h.T).reshape(6, 8)
    ref_scale = w_rot.abs().amax(dim=-1, keepdim=True).float().div(127.0).clamp(min=1e-30)
    ref_q = torch.round(w_rot / ref_scale).clamp(-128, 127).to(torch.int8)
    assert torch.equal(scale, ref_scale)
    assert torch.equal(q, ref_q)


def test_dequantize_roundtrip():
    torch.manual_seed(2)
    w = torch.randn(9, 12)
    q, scale = quantize_int8_convrot_weight(w, GS)
    back = dequantize_int8_weight(q, scale, GS, out_dtype=torch.float32)
    # worst case per element after un-rotation: half a quantization step spread by H (orthonormal)
    assert (back - w).abs().max() <= scale.max() * 0.5 * (GS**0.5) + 1e-6
    # non-rotated path
    q2, scale2 = quantize_int8_rowwise(w)
    back2 = dequantize_int8_weight(q2, scale2, 0, out_dtype=torch.float32)
    assert (back2 - w).abs().max() <= scale2.max() * 0.5 + 1e-6
    # row chunking must not change values
    assert torch.equal(back, dequantize_int8_weight(q, scale, GS, torch.float32, row_chunk=2))


def test_marker_parsing():
    info = parse_quant_marker(_marker())
    assert info == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GS}
    assert marker_groupsize(info) == GS
    assert marker_groupsize(parse_quant_marker(_marker(convrot=False))) == 0
    with pytest.raises(ValueError):
        marker_groupsize({"format": "float8_something"})
    sd = {"a.weight": torch.zeros(1), "a.comfy_quant": _marker()}
    markers = collect_quant_markers(sd)
    assert list(markers) == ["a"] and "a.comfy_quant" not in sd


# --------------------------------------------------------------------------- key conversion


def test_convert_dit_keys_renames():
    cases = {
        "video_patch_proj.weight": "proj_in.weight",
        "audio_patch_proj.bias": "audio_proj_in.bias",
        "condition_proj.weight": "context_embedder.weight",
        "adaln_t_table": "adaln_t_table",
        "final_layer.norm.weight": "norm_out.norm.weight",
        "final_layer.adaln_proj.linear.bias": "norm_out.linear.bias",
        "final_layer.video_out.weight": "proj_out.weight",
        "final_layer.audio_out.bias": "audio_proj_out.bias",
        "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
        "blocks.3.norm1.weight": "transformer_blocks.3.norm1.weight",
        "blocks.3.attn.q_norm.weight": "transformer_blocks.3.attn.norm_q.weight",
        "blocks.3.attn.out_proj.weight": "transformer_blocks.3.attn.to_out.0.weight",
        "blocks.3.attn.out_proj.weight_scale": "transformer_blocks.3.attn.to_out.0.weight_scale",
        "blocks.3.mlp.fc2.weight": "transformer_blocks.3.ff.net.2.weight",
        "blocks.3.adaln_proj.linear.weight": "transformer_blocks.3.adaln_proj.linear.weight",
        "token_refiner.blocks.1.mlp.fc2.weight": "token_refiner.refiner_blocks.1.ff.net.2.weight",
        "token_refiner.blocks.0.attn.k_norm.weight": "token_refiner.refiner_blocks.0.attn.norm_k.weight",
    }
    for source, expected in cases.items():
        out = convert_int8_dit_tensor(source, torch.zeros(2))
        assert [k for k, _ in out] == [expected], (source, out)
    assert convert_int8_dit_tensor("rope.inv_freq", torch.zeros(2)) == []
    with pytest.raises(KeyError):
        convert_int8_dit_tensor("blocks.0.attn.bogus.weight", torch.zeros(2))
    with pytest.raises(KeyError):
        convert_int8_dit_tensor("something.unknown", torch.zeros(2))


def test_convert_dit_qkv_split_and_fc1_swap():
    inner, hidden, ffn = 6, 4, 5
    qkv = torch.arange(3 * inner * hidden, dtype=torch.float32).reshape(3 * inner, hidden)
    out = dict(convert_int8_dit_tensor("blocks.0.attn.qkv_proj.weight", qkv))
    assert torch.equal(out["transformer_blocks.0.attn.to_q.weight"], qkv[:inner])
    assert torch.equal(out["transformer_blocks.0.attn.to_k.weight"], qkv[inner : 2 * inner])
    assert torch.equal(out["transformer_blocks.0.attn.to_v.weight"], qkv[2 * inner :])
    # per-row scales split identically, and the marker fans out to all three
    scales = torch.arange(3 * inner, dtype=torch.float32).reshape(-1, 1)
    out = dict(convert_int8_dit_tensor("blocks.0.attn.qkv_proj.weight_scale", scales))
    assert torch.equal(out["transformer_blocks.0.attn.to_k.weight_scale"], scales[inner : 2 * inner])
    out = dict(convert_int8_dit_tensor("blocks.0.attn.qkv_proj.comfy_quant", _marker()))
    assert set(out) == {
        "transformer_blocks.0.attn.to_q.comfy_quant",
        "transformer_blocks.0.attn.to_k.comfy_quant",
        "transformer_blocks.0.attn.to_v.comfy_quant",
    }
    # fc1: the export stores [gate; value], diffusers SwiGLU wants [value; gate]
    fc1 = torch.cat([torch.zeros(ffn, hidden), torch.ones(ffn, hidden)])
    out = dict(convert_int8_dit_tensor("blocks.0.mlp.fc1.weight", fc1))
    converted = out["transformer_blocks.0.ff.net.0.proj.weight"]
    assert torch.equal(converted[:ffn], torch.ones(ffn, hidden))
    assert torch.equal(converted[ffn:], torch.zeros(ffn, hidden))


# --------------------------------------------------------------------------- runtime patch


def test_int8_linear_patch_matches_dequant_reference():
    torch.manual_seed(3)
    w = torch.randn(6, 8)
    bias = torch.randn(6)
    q, scale = quantize_int8_convrot_weight(w, GS)

    layer = nn.Linear(8, 6, bias=True)
    layer.requires_grad_(False)
    apply_int8_monkey_patch(
        nn.ModuleDict({"lin": layer}), {"lin": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GS}}
    )
    layer.load_state_dict({"weight": q, "bias": bias, "weight_scale": scale}, assign=True)
    assert layer.weight.dtype == torch.int8

    x = torch.randn(3, 8)
    got = layer(x)
    expected = F.linear(x, dequantize_int8_weight(q, scale, GS, x.dtype), bias)
    assert torch.equal(got, expected)
    # and it lands within quantization distance of the unquantized layer
    assert (got - F.linear(x, w, bias)).abs().max() < 0.1


def test_int8_embedding_patch():
    torch.manual_seed(4)
    table = torch.randn(10, 8)
    q, scale = quantize_int8_rowwise(table)
    for stored_scale in (scale, scale.mean().reshape(())):  # per-row and tensor-wise scalar
        emb = nn.Embedding(10, 8)
        emb.requires_grad_(False)
        sd = {"weight": q, "weight_scale": stored_scale}
        apply_int8_monkey_patch(
            nn.ModuleDict({"emb": emb}),
            {"emb": {"format": "int8_tensorwise"}},
            embedding_output_dtype=torch.float32,
            state_dict={"emb.weight": q, "emb.weight_scale": stored_scale},
        )
        emb.load_state_dict(sd, assign=True)
        ids = torch.tensor([0, 3, 9])
        got = emb(ids)
        expected = q[ids].to(torch.float32) * (
            stored_scale[ids].to(torch.float32) if stored_scale.dim() >= 2 else stored_scale
        )
        assert torch.equal(got, expected)


def test_int8_mm_chunked_matches_full_math():
    """The chunked int8-GEMM epilogue (int32 accumulator + fp32 scaling only chunk-at-a-time)
    must be value-identical to the one-shot computation, including the fold of a <=16-row
    tail into the previous chunk (torch._int_mm rejects tiny row counts)."""
    torch.manual_seed(8)
    n = 24
    for m in (100, 70, 33, 48):  # exercises tail-fold (100 -> 32/32/36) and exact splits
        xq = torch.randint(-128, 127, (m, 16), dtype=torch.int8)
        xs = torch.rand(m, 1) + 0.5
        wq = torch.randint(-128, 127, (n, 16), dtype=torch.int8)
        ws = torch.rand(n, 1) + 0.5
        expected = (xq.float() @ wq.float().T) * (xs * ws.reshape(1, -1))
        old_budget = int8_quant._INT8_MM_ACC_BYTES
        int8_quant._INT8_MM_ACC_BYTES = 1  # force the minimum chunk size (32 rows)
        try:
            got = int8_quant._int8_mm_chunked(xq, xs, wq.T.contiguous(), ws, torch.float32)
        finally:
            int8_quant._INT8_MM_ACC_BYTES = old_budget
        assert torch.allclose(got, expected), m


def test_patched_modules_are_freed_by_gc():
    """The monkey patch stores a bound forward on each module, creating a module <-> method
    reference cycle: `del` alone cannot free a patched model (which is why the engine's
    teardown sites call gc.collect()). The cycle must at least be collectable."""
    import gc
    import weakref

    torch.manual_seed(9)
    w = torch.randn(6, 8)
    q, scale = quantize_int8_convrot_weight(w, GS)
    model = nn.ModuleDict({"lin": nn.Linear(8, 6, bias=False)})
    model.requires_grad_(False)
    apply_int8_monkey_patch(
        model, {"lin": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GS}}
    )
    model.load_state_dict({"lin.weight": q, "lin.weight_scale": scale}, assign=True)
    ref = weakref.ref(model["lin"])
    del model
    gc.collect()
    assert ref() is None, "patched module survived del + gc.collect — the release path would leak"


def test_convrot_embedding_rejected():
    emb = nn.Embedding(4, 8)
    with pytest.raises(ValueError):
        apply_int8_monkey_patch(
            nn.ModuleDict({"emb": emb}),
            {"emb": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GS}},
        )


# --------------------------------------------------------------------------- curve transformer

TINY_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}

NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8
CURVE_GRID = 9
CURVE_DIM = 8


def _packed_inputs():
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)

    video_patch_dim = TINY_CONFIG["in_channels"] * 4
    return dict(
        hidden_states=torch.randn(1, NUM_VIDEO_TOKENS, video_patch_dim),
        audio_hidden_states=torch.randn(1, NUM_AUDIO_TOKENS, TINY_CONFIG["audio_in_channels"]),
        encoder_hidden_states=torch.randn(1, NUM_TEXT_TOKENS, TINY_CONFIG["text_dim"]),
        timestep=torch.tensor([0.35, 0.62]),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )


def _curve_model(seed: int = 5):
    from minimax_video.transformer import MiniMaxH3Transformer3DModel

    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(
        **{**TINY_CONFIG, "time_embed_dim": CURVE_DIM}, adaln_curve_grid=CURVE_GRID
    )
    with torch.no_grad():
        model.adaln_t_table.copy_(torch.randn(CURVE_GRID, CURVE_DIM))
    return model.eval().requires_grad_(False)


def test_curve_transformer_structure_and_forward():
    model = _curve_model()
    assert model.time_embedder is None and model.time_proj is None
    assert model.adaln_t_table.shape == (CURVE_GRID, CURVE_DIM)
    assert not model.transformer_blocks[0].adaln_proj.apply_silu
    assert not model.norm_out.apply_silu

    inputs = _packed_inputs()
    with torch.no_grad():
        out = model(**inputs)
    video_patch_dim = TINY_CONFIG["in_channels"] * 4
    assert out.sample.shape == (1, NUM_VIDEO_TOKENS, video_patch_dim)
    assert out.audio_sample.shape == (1, NUM_AUDIO_TOKENS, TINY_CONFIG["audio_in_channels"])
    assert torch.isfinite(out.sample).all() and torch.isfinite(out.audio_sample).all()


def test_curve_temb_interpolation_semantics():
    """The forward's table lookup must match the export runtime: fractional index over t in [0,1],
    clamped ends, t = 1.0 served by the last interval."""
    model = _curve_model()
    table = model.adaln_t_table
    captured = {}

    original = model.transformer_blocks[0].adaln_proj.forward

    def spy(temb, out_dtype=None):
        captured["temb"] = temb.detach().clone()
        return original(temb, out_dtype=out_dtype)

    model.transformer_blocks[0].adaln_proj.forward = spy
    inputs = _packed_inputs()
    for t, expected in [
        (0.0, table[0]),
        (1.0, table[CURVE_GRID - 1]),
        (1.5, table[CURVE_GRID - 1]),  # out of range clamps to the curve end
        (-0.2, table[0]),
        (0.5, torch.lerp(table[4], table[5], 0.5 * (CURVE_GRID - 1) - 4)),
    ]:
        inputs["timestep"] = torch.tensor([t, t])
        inputs["timestep_indices"] = torch.zeros_like(inputs["timestep_indices"])
        with torch.no_grad():
            model(**inputs)
        assert torch.allclose(captured["temb"][0], expected, atol=1e-6), t


# --------------------------------------------------------------------------- end-to-end load


def _build_tiny_int8_checkpoint(tmp_dir: str) -> tuple[str, str]:
    """Write a tiny single-file int8 convrot checkpoint + a matching diffusers config.json.
    Returns (ckpt_dir, checkpoint file path)."""
    import safetensors.torch

    torch.manual_seed(6)
    cfg = TINY_CONFIG
    hidden = cfg["hidden_size"]
    inner = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    ffn = cfg["ffn_dim"]
    video_patch_dim = cfg["in_channels"] * 4

    sd = {
        "adaln_t_table": torch.randn(CURVE_GRID, CURVE_DIM),
        "video_patch_proj.weight": torch.randn(hidden, video_patch_dim),
        "video_patch_proj.bias": torch.randn(hidden),
        "audio_patch_proj.weight": torch.randn(hidden, cfg["audio_in_channels"]),
        "audio_patch_proj.bias": torch.randn(hidden),
        "condition_proj.weight": torch.randn(hidden, cfg["text_dim"]).to(torch.bfloat16),
        "condition_proj.bias": torch.randn(hidden).to(torch.bfloat16),
        "rope.inv_freq": torch.randn(cfg["rope_freq_dim"]),
        "token_refiner.final_norm.weight": torch.randn(hidden).to(torch.bfloat16),
        "final_layer.norm.weight": torch.randn(hidden).to(torch.bfloat16),
        "final_layer.adaln_proj.linear.weight": (0.1 * torch.randn(2 * hidden, CURVE_DIM)).to(torch.float16),
        "final_layer.adaln_proj.linear.bias": (0.1 * torch.randn(2 * hidden)).to(torch.float16),
        "final_layer.video_out.weight": torch.randn(video_patch_dim, hidden),
        "final_layer.video_out.bias": torch.randn(video_patch_dim),
        "final_layer.audio_out.weight": torch.randn(cfg["audio_in_channels"], hidden),
        "final_layer.audio_out.bias": torch.randn(cfg["audio_in_channels"]),
    }
    for i in range(cfg["num_layers"]):
        p = f"blocks.{i}."
        sd[p + "norm1.weight"] = torch.randn(hidden).to(torch.bfloat16)
        sd[p + "norm2.weight"] = torch.randn(hidden).to(torch.bfloat16)
        sd[p + "attn.q_norm.weight"] = torch.randn(cfg["attention_head_dim"]).to(torch.bfloat16)
        sd[p + "attn.k_norm.weight"] = torch.randn(cfg["attention_head_dim"]).to(torch.bfloat16)
        sd[p + "adaln_proj.linear.weight"] = (0.1 * torch.randn(6 * hidden * 3, CURVE_DIM)).to(torch.float16)
        sd[p + "adaln_proj.linear.bias"] = (0.1 * torch.randn(6 * hidden * 3)).to(torch.float16)
        for name, shape in (
            ("attn.qkv_proj", (3 * inner, hidden)),
            ("attn.out_proj", (hidden, inner)),
            ("mlp.fc1", (2 * ffn, hidden)),
            ("mlp.fc2", (hidden, ffn)),
        ):
            q, scale = quantize_int8_convrot_weight(torch.randn(shape) / shape[1] ** 0.5, GS)
            sd[p + name + ".weight"] = q
            sd[p + name + ".weight_scale"] = scale
            sd[p + name + ".comfy_quant"] = _marker()
    for i in range(cfg["num_refiner_layers"]):
        p = f"token_refiner.blocks.{i}."
        sd[p + "norm1.weight"] = torch.randn(hidden).to(torch.bfloat16)
        sd[p + "norm2.weight"] = torch.randn(hidden).to(torch.bfloat16)
        sd[p + "attn.q_norm.weight"] = torch.randn(cfg["attention_head_dim"]).to(torch.bfloat16)
        sd[p + "attn.k_norm.weight"] = torch.randn(cfg["attention_head_dim"]).to(torch.bfloat16)
        sd[p + "attn.qkv_proj.weight"] = (torch.randn(3 * inner, hidden) / hidden**0.5).to(torch.bfloat16)
        sd[p + "attn.out_proj.weight"] = (torch.randn(hidden, inner) / inner**0.5).to(torch.bfloat16)
        sd[p + "mlp.fc1.weight"] = (torch.randn(2 * ffn, hidden) / hidden**0.5).to(torch.bfloat16)
        sd[p + "mlp.fc2.weight"] = (torch.randn(hidden, ffn) / ffn**0.5).to(torch.bfloat16)

    ckpt_dir = os.path.join(tmp_dir, "snapshot")
    os.makedirs(os.path.join(ckpt_dir, "transformer"), exist_ok=True)
    with open(os.path.join(ckpt_dir, "transformer", "config.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "MiniMaxH3Transformer3DModel", **{k: list(v) if isinstance(v, tuple) else v for k, v in TINY_CONFIG.items()}}, f)
    file_path = os.path.join(tmp_dir, "tiny_int8_convrot.safetensors")
    safetensors.torch.save_file(sd, file_path)
    return ckpt_dir, file_path


def test_tiny_int8_checkpoint_end_to_end():
    from minimax_video.model_loader import load_transformer

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir, file_path = _build_tiny_int8_checkpoint(tmp)
        model = load_transformer(
            ckpt_dir, device=torch.device("cpu"), task="fl2va", dit_dtype=torch.float32, dit_path=file_path
        )

        # quantized layers landed as int8 with their scales; curve form is active
        block = model.transformer_blocks[0]
        assert block.attn.to_q.weight.dtype == torch.int8
        assert block.attn.to_q.weight_scale.dtype == torch.float32
        assert block.ff.net[0].proj.weight.dtype == torch.int8
        assert model.adaln_t_table.dtype == torch.float32
        assert block.adaln_proj.linear.weight.dtype == torch.float32  # promoted from f16
        assert model.time_embedder is None
        # refiner stayed unquantized
        assert model.token_refiner.refiner_blocks[0].attn.to_q.weight.dtype == torch.float32

        inputs = _packed_inputs()
        with torch.no_grad():
            out = model(**inputs)
        assert torch.isfinite(out.sample).all() and torch.isfinite(out.audio_sample).all()

        # reference model: identical architecture with every quantized weight dequantized to
        # float32 up front — the dequant runtime path must produce the same numbers
        ref = load_transformer(
            ckpt_dir, device=torch.device("cpu"), task="fl2va", dit_dtype=torch.float32, dit_path=file_path
        )
        for blk in list(ref.transformer_blocks):
            for lin in (blk.attn.to_q, blk.attn.to_k, blk.attn.to_v, blk.attn.to_out[0], blk.ff.net[0].proj, blk.ff.net[2]):
                w = dequantize_int8_weight(lin.weight, lin.weight_scale, lin.int8_convrot_groupsize, torch.float32)
                lin.forward = nn.Linear.forward.__get__(lin, nn.Linear)
                lin.weight = nn.Parameter(w, requires_grad=False)
        with torch.no_grad():
            ref_out = ref(**inputs)
        assert torch.allclose(out.sample, ref_out.sample, atol=1e-5)
        assert torch.allclose(out.audio_sample, ref_out.audio_sample, atol=1e-5)


def test_tiny_int8_checkpoint_lora_merge():
    from minimax_video.model_loader import load_transformer

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir, file_path = _build_tiny_int8_checkpoint(tmp)
        base = load_transformer(
            ckpt_dir, device=torch.device("cpu"), task="fl2va", dit_dtype=torch.float32, dit_path=file_path
        )

        torch.manual_seed(7)
        rank, hidden = 2, TINY_CONFIG["hidden_size"]
        inner = TINY_CONFIG["num_attention_heads"] * TINY_CONFIG["attention_head_dim"]
        down = torch.randn(rank, hidden) * 0.05
        up = torch.randn(inner, rank) * 0.05
        lora_sd = {
            "transformer_blocks.0.attn.to_q.lora_down.weight": down,
            "transformer_blocks.0.attn.to_q.lora_up.weight": up,
        }
        merged = load_transformer(
            ckpt_dir,
            device=torch.device("cpu"),
            task="fl2va",
            dit_dtype=torch.float32,
            dit_path=file_path,
            lora_weights_list=[lora_sd],
            lora_multipliers=[1.0],
        )

        lin_base = base.transformer_blocks[0].attn.to_q
        lin_merged = merged.transformer_blocks[0].attn.to_q
        assert lin_merged.weight.dtype == torch.int8  # still quantized after the merge
        w_base = dequantize_int8_weight(lin_base.weight, lin_base.weight_scale, GS, torch.float32)
        w_merged = dequantize_int8_weight(lin_merged.weight, lin_merged.weight_scale, GS, torch.float32)
        expected = w_base + (up @ down)  # alpha defaults to dim -> scale 1.0
        # exact up to one int8 re-quantization of the merged weight
        assert (w_merged - expected).abs().max() <= lin_merged.weight_scale.max() * 0.5 * (GS**0.5) + 1e-6
        # untouched layers keep bit-identical quantized data
        assert torch.equal(
            base.transformer_blocks[1].attn.to_q.weight, merged.transformer_blocks[1].attn.to_q.weight
        )
