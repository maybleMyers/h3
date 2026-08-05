# Runtime state for the "sol" attention backend. Import-light on purpose: pipeline.py and
# transformer.py write these attributes unconditionally every forward, and that must stay free
# (no triton / kernel imports) when the backend is sdpa.

from dataclasses import dataclass, field


def _fresh_stats() -> dict:
    return {"sparse": 0, "dense_recipe": 0, "dense_fallback": 0}


@dataclass
class SolContext:
    tau: float = 1.0            # threshold scale: tau_i = mu_i + tau * sigma_i; NVIDIA H3 recipe = 1.0
    dense_steps: int = 10       # first N denoising steps fully dense (recipe default)
    dense_blocks: int = 2       # first N transformer blocks fully dense (recipe default)
    min_tokens: int = 4096      # below this, dense; also keeps the token-refiner blocks dense
    sink_len: int = 0           # rows [0, sink_len) = text/condition/audio prefix, exact KV sink
    current_step: int = -1      # set by MiniMaxH3Pipeline.denoise; -1 = unknown caller -> dense
    current_block: int = -1     # set by the transformer block loop; -1 (token refiner) -> dense
    strict: bool = False        # raise instead of silently falling back on capability failures
    stats: dict = field(default_factory=_fresh_stats)

    def reset_stats(self) -> None:
        self.stats = _fresh_stats()


SOL_CTX = SolContext()
