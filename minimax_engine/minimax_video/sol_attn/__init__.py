# Sol-Attn backend package: local wrapper around the vendored NVIDIA kernel (see VENDORED.md).
#
# `sol_attention` is called from minimax_video/attention.py after all gating has passed. It always
# uses the portable Triton implementation (`triton_ref`) because the CuTe DSL kernels are not
# vendored; on SM90+ with a TMA-capable triton the triton_ref path upgrades itself to descriptors.

import functools

import torch
import torch.nn.functional as F

from .context import SOL_CTX, SolContext

__all__ = ["SOL_CTX", "SolContext", "sol_attention", "is_sol_available"]


@functools.lru_cache(maxsize=1)
def is_sol_available() -> tuple[bool, str]:
    """Whether the sol kernel can run in this environment: (ok, reason-if-not)."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        return False, f"requires compute capability >= 8.0, got SM{capability[0]}{capability[1]}"
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "the `triton` package is not installed"
    try:
        from .triton_ref import sol_attn  # noqa: F401
    except Exception as e:  # triton too old / kernel source issue
        return False, f"sol kernel import failed: {e!r}"
    return True, ""


def _dense_rows(query, key, value):
    """SDPA over [B, S, H, D] slices, same layout out (used for the sink query rows)."""
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    )
    return out.transpose(1, 2)


def sol_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tau: float,
    sink_len: int = 0,
) -> torch.Tensor:
    """Sol-Attn over one packed [B, S, H, D] bf16 sequence.

    Rows [0, sink_len) (the text/condition/audio prefix) are passed to the kernel as an exact KV
    sink — their key blocks are never routed away for any query — and their own query rows are
    then recomputed densely, per the upstream MMDiT integration contract: only video query rows
    should route sparsely.
    """
    from .triton_ref import sol_attn as _triton_sol_attn

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    sink = min(int(sink_len), q.shape[1]) if sink_len and sink_len > 0 else 0

    out = _triton_sol_attn(
        q, k, v,
        tau=float(tau),
        thresh_type="diag",
        sink_tokens=sink,
        sink_start=0 if sink else None,
    )
    if sink:
        out[:, :sink] = _dense_rows(q[:, :sink], k, v)
    return out
