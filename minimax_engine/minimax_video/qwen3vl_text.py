# Standalone port of transformers' Qwen3VLTextModel (models/qwen3_vl/modeling_qwen3_vl.py,
# Apache-2.0, referenced at v4.57.0) for the MiniMax-H3 conditioner. Vendored because the
# environment pins transformers 4.46.x, which predates qwen3_vl.
#
# MiniMax-H3 reads the *unnormalized* hidden state after the 50th decoder layer
# (`hidden_states[50]` with `output_hidden_states=True`, embeddings being index 0). Layers 50+,
# the final norm and the language-model head are never used, so this port instantiates only
# `layers[0..num_read_layers-1]` and returns the raw stream after the last kept layer.
# Attribute names match the upstream module exactly (`embed_tokens`, `layers.N.self_attn.q_proj`,
# `q_norm`, `input_layernorm`, `mlp.gate_proj`, ...) so the checkpoint's text_encoder state dict
# loads by prefix-stripping only. SDPA-only, single unpadded sequence, no KV cache (the
# conditioner runs exactly one forward per request).

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    # q/k: [bs, heads, T, head_dim]; cos/sin: [bs, T, head_dim] -> unsqueeze over heads.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Interleaved-mrope rotary embedding (Qwen3VLTextRotaryEmbedding parity, default rope type)."""

    def __init__(self, head_dim: int, rope_theta: float, mrope_section: list[int]):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(mrope_section)
        self.attention_scaling = 1.0

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """(3, bs, T, head_dim//2) chunked [TTT...HHH...WWW] -> interleaved [THWTHW...TT] -> (bs, T, head_dim//2)."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [3, bs, T] (or [bs, T] for text-only, expanded here).
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.to(position_ids.device) @ position_ids_expanded).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype), sin.to(dtype)


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", hidden_size // num_heads)
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5
        bias = config.get("attention_bias", False)

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        eps = config.get("rms_norm_eps", 1e-6)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # config.hidden_act is "silu" for every released Qwen3-VL text stack.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        eps = config.get("rms_norm_eps", 1e-6)
        self.input_layernorm = Qwen3VLTextRMSNorm(config["hidden_size"], eps=eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config["hidden_size"], eps=eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class _StreamSlot:
    """One reusable GPU buffer holding a single decoder layer's tensors during streaming."""

    def __init__(self):
        self.tensors: list[torch.Tensor] | None = None
        self.ready = torch.cuda.Event()  # upload finished (recorded on the copy stream)
        self.free = torch.cuda.Event()  # last compute out of this slot finished (recorded on the compute stream)


class _LayerStreamer:
    """Double-buffered CUDA prefetcher for CPU-resident decoder layers.

    The CPU weights are the permanent masters, moved into pinned memory on first use so
    host-to-device copies can run asynchronously (pinning falls back to pageable memory
    if the allocation fails; streaming still works, just without overlap). Two GPU slots
    alternate: while layer N computes out of one slot on the current stream, layer N+1
    uploads into the other on a side copy stream. Weights are read-only during the
    forward, so nothing is ever copied back to the CPU.

    Slot memory (two layers, ~2 GB for the bf16 conditioner) is released at the end of
    every forward; the pinned CPU masters persist so repeat encodes skip the pinning
    cost. Layers whose tensor layout differs from the first streamed layer (never the
    case for a real checkpoint) fall back to the synchronous `.to()` path.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        self.slots = (_StreamSlot(), _StreamSlot())
        self._pin_ok = True
        self._signature = None
        self._owners: dict[int, list[torch.Tensor]] = {}  # layer_idx -> param/buffer objects
        self._masters: dict[int, list[torch.Tensor]] = {}  # layer_idx -> their CPU (pinned) data
        self._skip: set[int] = set()
        self._active: dict[int, _StreamSlot] = {}  # uploaded (or uploading) layers
        self._pending: list[int] = []

    def _pin(self, owners: list[torch.Tensor]) -> None:
        for t in owners:
            if not self._pin_ok or t.data.is_pinned():
                continue
            try:
                pinned = torch.empty_like(t.data, pin_memory=True)
                pinned.copy_(t.data)
                t.data = pinned
            except RuntimeError:
                self._pin_ok = False

    def begin(self, layers: nn.ModuleList, streamed: list[int]) -> None:
        for idx in streamed:
            if idx in self._owners or idx in self._skip:
                continue
            owners = [p for _, p in layers[idx].named_parameters()] + [b for _, b in layers[idx].named_buffers()]
            signature = tuple((tuple(t.shape), t.dtype) for t in owners)
            if self._signature is None:
                self._signature = signature
            if signature != self._signature:
                self._skip.add(idx)
                continue
            self._pin(owners)
            self._owners[idx] = owners
            self._masters[idx] = [t.data for t in owners]
        self._pending = [i for i in streamed if i in self._owners]
        for slot in self.slots:
            if self._pending:
                self._upload(self._pending.pop(0), slot)

    def _upload(self, idx: int, slot: _StreamSlot) -> None:
        masters = self._masters[idx]
        if slot.tensors is None:
            slot.tensors = [torch.empty_like(t, device=self.device) for t in masters]
        self.copy_stream.wait_event(slot.free)
        with torch.cuda.stream(self.copy_stream):
            for dst, src in zip(slot.tensors, masters):
                dst.copy_(src, non_blocking=True)
        slot.ready.record(self.copy_stream)
        self._active[idx] = slot

    def holds(self, idx: int) -> bool:
        return idx in self._active

    def attach(self, idx: int) -> None:
        """Point layer `idx`'s tensors at its uploaded slot (after the upload lands)."""
        slot = self._active[idx]
        torch.cuda.current_stream().wait_event(slot.ready)
        for owner, gpu in zip(self._owners[idx], slot.tensors):
            owner.data = gpu

    def detach(self, idx: int) -> None:
        """Restore layer `idx`'s CPU tensors and start uploading the next pending layer."""
        slot = self._active.pop(idx)
        slot.free.record(torch.cuda.current_stream())
        for owner, master in zip(self._owners[idx], self._masters[idx]):
            owner.data = master
        if self._pending:
            self._upload(self._pending.pop(0), slot)

    def end(self) -> None:
        """Release slot memory once all queued work is done; also restores every master
        (a no-op unless an exception interrupted the forward mid-layer)."""
        torch.cuda.current_stream().synchronize()
        for idx, owners in self._owners.items():
            for owner, master in zip(owners, self._masters[idx]):
                owner.data = master
        self._active.clear()
        self._pending = []
        for slot in self.slots:
            slot.tensors = None


class Qwen3VLTruncatedTextModel(nn.Module):
    """The first `num_read_layers` decoder layers of a Qwen3-VL text stack, no norm, no lm_head.

    Returns the raw hidden state after layer `num_read_layers - 1`, which for
    `num_read_layers = 50` is exactly the `hidden_states[50]` MiniMax-H3 conditions on.
    """

    def __init__(self, text_config: dict, num_read_layers: int):
        super().__init__()
        num_hidden_layers = text_config["num_hidden_layers"]
        if num_hidden_layers < num_read_layers:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{num_read_layers}] of its Qwen3-VL conditioner, which "
                f"needs at least {num_read_layers} decoder layers, but the config declares {num_hidden_layers}."
            )
        self.num_read_layers = num_read_layers
        self.embed_tokens = nn.Embedding(text_config["vocab_size"], text_config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(text_config, i) for i in range(num_read_layers)])
        rope_scaling = text_config.get("rope_scaling") or {}
        head_dim = text_config.get("head_dim", text_config["hidden_size"] // text_config["num_attention_heads"])
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=text_config.get("rope_theta", 1000000.0),
            mrope_section=rope_scaling.get("mrope_section", [24, 20, 20]),
        )
        self._streamer = None

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        stream_device: torch.device | None = None,
    ) -> torch.Tensor:
        """One prefill pass. `inputs_embeds` is `[1, T, hidden]` with vision features already scattered in.

        `stream_device`: when set, CPU-resident layers run there instead of on the CPU — the
        whole ~30B conditioner never has to fit on the accelerator at once. On CUDA the
        layers are double-buffer prefetched from pinned CPU masters (`_LayerStreamer`): layer
        N+1 uploads on a side stream while layer N computes, and nothing is copied back. On
        other accelerators each layer is moved over and back synchronously.
        """
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(position_ids.to(inputs_embeds.device), inputs_embeds.dtype)

        streamer = None
        if stream_device is not None and stream_device.type == "cuda":
            streamed = [i for i, l in enumerate(self.layers) if l.input_layernorm.weight.device.type == "cpu"]
            if streamed:
                if self._streamer is None or self._streamer.device != stream_device:
                    self._streamer = _LayerStreamer(stream_device)
                streamer = self._streamer
                streamer.begin(self.layers, streamed)

        try:
            for layer_idx, layer in enumerate(self.layers):
                prefetched = streamer is not None and streamer.holds(layer_idx)
                home = layer.input_layernorm.weight.device
                stream = stream_device is not None and home.type == "cpu" and stream_device.type != "cpu"
                run_device = stream_device if stream else home
                if stream and not prefetched:
                    layer.to(run_device)
                if hidden_states.device != run_device:
                    hidden_states = hidden_states.to(run_device)
                    position_embeddings = tuple(t.to(run_device) for t in position_embeddings)

                if prefetched:
                    streamer.attach(layer_idx)
                hidden_states = layer(hidden_states, position_embeddings)
                if prefetched:
                    streamer.detach(layer_idx)

                # Deepstack: add merged ViT features at visual rows after the first
                # len(deepstack_visual_embeds) layers (Qwen3VLTextModel._deepstack_process parity).
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    mask = visual_pos_masks.to(hidden_states.device)
                    embeds = deepstack_visual_embeds[layer_idx].to(hidden_states.device, hidden_states.dtype)
                    local = hidden_states[mask, :].clone() + embeds
                    hidden_states = hidden_states.clone()
                    hidden_states[mask, :] = local

                if stream and not prefetched:
                    layer.to(home)
        finally:
            if streamer is not None:
                streamer.end()

        return hidden_states
