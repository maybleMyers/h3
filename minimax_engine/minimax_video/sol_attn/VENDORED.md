# Vendored Sol-Attn kernel

Source: https://github.com/NVlabs/Sana, branch `sol-engine`,
commit `8a26fb0ec9e353125ead798cb2e312d5ce48cded`,
directory `techniques/sparse_backends/sol_attn/`. License: Apache-2.0.

Vendored files (unmodified unless noted):

- `interface.py` — public `sol_attn` entry point and validation helpers.
- `preprocess.py` — TMA/descriptor preprocessing (only imported on SM90+ with a
  triton that provides `triton.tools.tensor_descriptor`; never imported on the
  pointer path used by SM80/SM89).
- `triton_ref/__init__.py`, `triton_ref/fwd.py`, `triton_ref/preprocess.py` —
  portable Triton implementation (pointer path needs no TMA support).
  Modification: `triton_ref/fwd.py` imports `..interface` instead of the
  upstream absolute `sol_attn.interface`.
- `THIRD_PARTY_NOTICES.md` — upstream notices. It references
  `_vendor/flash_attn/cute/` and the `sm90/`/`sm100/`/`sm120/` CuTe DSL
  kernels, which are NOT vendored here; this copy contains only the Triton
  path, so `interface.py`'s CuTe dispatch is unreachable (callers use
  `triton_ref.sol_attn` directly via `minimax_video/sol_attn/__init__.py`).

Local (non-vendored) files: `__init__.py`, `context.py`, `VENDORED.md`.
