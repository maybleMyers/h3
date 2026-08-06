# MiniMax-H3 checked-in component configs (no weights)

Component configs for `MiniMaxAI/MiniMax-H3`, derived from the diffusers PR
huggingface/diffusers#14355 @ e1b518d (`scripts/convert_minimax_h3_to_diffusers.py`).
They let the loader and the dummy-weight tests run before any checkpoint download.

`vae/config.json` and `audio_vae/config.json` carry the real per-channel
`latents_mean` / `latents_std` from the official MiniMax-H3 release.

Still not checked in:

- `*.safetensors.index.json` shard indexes and the `text_encoder/`, `tokenizer/`,
  `processor/` configs; copy them from the HF snapshot (or point `--ckpt_dir` at the
  official repo clone, which ships all of them).
- `transformer_ref/` shares the architecture config with `transformer/`; only the weights
  differ.
