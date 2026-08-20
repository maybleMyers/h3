#!/usr/bin/env python
# minimax_generate_video.py — MiniMax-H3 joint video + audio generation for H1111.
#
# Native integration built on the vendored minimax_video/ package (port of
# huggingface/diffusers#14355 @ e1b518d) plus H1111 shared infra: block swap, fp8, LoRA
# merge-at-load, latent previews, latent save / decode-only.
#
# Tasks (inferred from inputs, or forced with --task):
#   t2va   — text to video + audio
#   fl2va  — first and/or last keyframe conditioning (--image_path / --last_image_path)
#   ref2va — ordered image/video/audio references (--reference ..., served by the
#            checkpoint's separate transformer_ref/ partition)
#
# The checkpoint is guidance-distilled: no negative prompt, no CFG, one forward per step.
# Execution is staged so the ~30B conditioner, the VAEs and the 33B transformer never
# coexist on the GPU: (A) conditioner -> prompt embeds, (B) VAEs -> condition latents,
# (C) transformer -> denoise, (D) VAEs -> decode + save.

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime

# Fragmentation guard: the denoise loop cycles many differently-sized multi-GB activation
# blocks; expandable segments keep the caching allocator from stranding reserved memory.
# Must be set before torch initializes CUDA; an explicit user setting wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Runnable from anywhere: the H1111 repo root provides utils/, modules/, and
# blissful_tuner/; this directory (minimax_engine/) provides minimax_video/.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Set the global compile flag BEFORE importing minimax_video model modules: the
# @maybe_compile decorators read it at import time (same pattern as wan2_generate_video.py).
from minimax_video import compile_config as _compile_config

if "--compile" in sys.argv:
    _compile_config.USE_TORCH_COMPILE = True
    print("torch.compile() enabled for optimized inference (function-level compilation)")

from utils.device_utils import clean_memory_on_device
from utils.model_utils import str_to_dtype
from utils.safetensors_utils import load_safetensors, mem_eff_save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


class _StopGeneration(Exception):
    """Raised by the step callback when the UI asks to stop and decode what exists."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 video+audio generation (H1111)")

    # Model
    parser.add_argument("--ckpt_dir", type=str, default=None, help="MiniMax-H3 HF snapshot dir (diffusers layout)")
    parser.add_argument("--dit", type=str, default=None, help="override transformer dir or merged .safetensors")
    parser.add_argument("--vae", type=str, default=None, help="override video VAE dir")
    parser.add_argument("--audio_vae", type=str, default=None, help="override audio VAE dir")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "t2va", "fl2va", "ref2va"])

    # Request (no --negative_prompt / --guidance_scale: the checkpoint is guidance-distilled)
    parser.add_argument("--prompt", type=str, default=None, help="prompt text or path to a .txt file")
    parser.add_argument("--image_path", type=str, default=None, help="first keyframe (stretched onto the canvas)")
    parser.add_argument("--last_image_path", type=str, default=None,
                        help="last keyframe (cover-cropped); while chaining it pins the end of the final segment")
    parser.add_argument("--reference", type=str, action="append", default=None,
                        help="ref2va reference (repeatable, ordered; kind sniffed from the file). "
                             "Up to 9 images / 3 videos / 3 audio clips, 12 total.")
    parser.add_argument("--reference_strip_audio", action="store_true",
                        help="ignore the soundtrack of video references")
    parser.add_argument("--video_size", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="explicit canvas, multiples of 32 (default: 768-short-edge auto canvas)")
    parser.add_argument("--aspect_ratio", type=str, default=None,
                        help="W:H used for the auto canvas when no keyframe binds it (e.g. 16:9)")
    parser.add_argument("--video_length", type=int, default=124,
                        help="frames @ 24 fps, snapped up to 17n+5; 5-15 s. 0 = derive from the single "
                             "audio-bearing reference (ref2va only)")
    parser.add_argument("--infer_steps", type=int, default=50,
                        help="sigma grid points, terminal 0 included (model evaluations = steps - 1)")
    parser.add_argument("--flow_shift", type=float, default=None, help="video sigma shift (checkpoint: 12.0)")
    parser.add_argument("--audio_flow_shift", type=float, default=None, help="audio sigma shift (checkpoint: 3.0)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--chain_count", type=int, default=1,
                        help=">1: generate that many segments and join them into one video")
    parser.add_argument("--chain_mode", type=str, default="motion", choices=["motion", "last_frame", "video"],
                        help="carry-over conditioning for segments after the first. motion: pin the previous "
                             "segment's tail on the new segment's own timeline, so it continues rather than cuts. "
                             "last_frame/video: carry the previous segment as a reference instead, which locks "
                             "identity and style but reads as a matched cut")
    parser.add_argument("--chain_context_length", type=int, default=22,
                        help="--chain_mode motion: frames of the previous segment pinned into the next one. Snapped "
                             "down to the video VAE grid (1, 5, 22, 39, ... i.e. 1 and every 17 * n + 5); each "
                             "segment delivers that many frames fewer")
    parser.add_argument("--chain_context_encode", type=str, default="video", choices=["video", "frames"],
                        help="--chain_mode motion: encode the pinned run in one VAE call so the motion lives inside "
                             "the temporal latent (video), or one call per frame (frames, a diagnostic: three times "
                             "the rows for less information)")
    parser.add_argument("--chain_context_crop", type=str, default="stretch", choices=["stretch", "center"],
                        help="--chain_mode motion: how to fit an off-canvas tail, only reached by --chain_extend")
    parser.add_argument("--chain_audio_context", type=int, default=0,
                        help="--chain_mode motion: soundtrack window in frames; 0 follows --chain_context_length. "
                             "A window wider than the video context risks landing on the text rows")
    parser.add_argument("--chain_audio_mode", type=str, default="timeline", choices=["timeline", "off"],
                        help="--chain_mode motion: pin the previous segment's soundtrack end-aligned with its "
                             "picture (timeline), or let audio restart each segment (off)")
    parser.add_argument("--chain_keep_segments", action="store_true",
                        help="also write each segment as its own mp4 next to the joined video")
    parser.add_argument("--chain_extend", type=str, default=None,
                        help="extension mode: seed the chain from this existing video instead of generating "
                             "the first segment cold, and prepend it to the joined output")

    # Performance
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_mode", type=str, default="sdpa",
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers",
                                 "sol"])
    parser.add_argument("--sol_tau", type=float, default=1.0,
                        help="Sol-Attn routing threshold scale (tau_i = mu_i + tau*sigma_i); useful range 0.8-1.25, "
                             "lower = denser/safer")
    parser.add_argument("--sol_dense_steps", type=int, default=10,
                        help="run the first N denoising steps fully dense (NVIDIA H3 recipe: 10)")
    parser.add_argument("--sol_dense_blocks", type=int, default=2,
                        help="run the first N transformer blocks of every step fully dense (NVIDIA H3 recipe: 2)")
    parser.add_argument("--sol_min_tokens", type=int, default=4096,
                        help="fall back to dense attention below this packed sequence length")
    parser.add_argument("--sol_strict", action="store_true",
                        help="raise instead of silently falling back to dense when the sol kernel cannot engage")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile with function-level decorators (mode: max-autotune-no-cudagraphs, "
                             "dynamic: True). Compatible with all dtypes and block swap; first run is slower while "
                             "kernels compile.")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                        help="0-49 transformer blocks kept on CPU (pinned) and streamed through the GPU")
    parser.add_argument("--classic_block_swap", action="store_true",
                        help="use the legacy rolling block swap instead of pinned sub-block weight streaming")
    parser.add_argument("--offload_engine", choices=["blockswap", "diffusers_mm"], default="blockswap",
                        help="DiT offload backend: 'blockswap' (built-in) or 'diffusers_mm' (diffusers-mm package)")
    parser.add_argument("--mm_strategy", default="auto",
                        choices=["auto", "no_offload", "model_offload", "block_pin", "group_offload"],
                        help="diffusers-mm offload strategy (only with --offload_engine diffusers_mm)")
    parser.add_argument("--act_chunk_rows", type=int, default=32768,
                        help="process row-wise ops (AdaLN, rotary, FF, output heads) in slices of this many rows "
                             "of the packed sequence to bound activation peaks; 0 = off")
    parser.add_argument("--fp8", action="store_true", help="cast transformer block weights to e4m3")
    parser.add_argument("--fp8_scaled", action="store_true", help="scaled fp8 quantization with monkey patch")
    parser.add_argument("--fp8_fast", action="store_true", help="use scaled_mm fp8 matmul (with --fp8_scaled)")
    parser.add_argument("--fp8_exclude_adaln", action="store_true",
                        help="keep the AdaLN projections out of fp8 (+~13 GB resident, higher fidelity)")
    parser.add_argument("--int8_fast", action="store_true",
                        help="run int8 convrot Linears through torch._int_mm (dynamic per-row activation "
                             "quantization) instead of dequantize-per-forward")
    parser.add_argument("--text_encoder", type=str, default=None,
                        help="single-file text encoder override (e.g. an int8 convrot export; the "
                             "'ultra_p' export also carries the vision tower)")
    parser.add_argument("--text_encoder_gpu_layers", type=int, default=-1,
                        help="conditioner decoder layers kept on the GPU (-1 = all, 0 = none/streamed)")
    parser.add_argument("--text_encoder_stream", action="store_true",
                        help="stream CPU-resident conditioner layers through the GPU one at a time")
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16")
    parser.add_argument("--prompt_cache", type=str, default=None,
                        help="cache file for prompt embeddings; reused when inputs match, skipping the conditioner")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16")
    parser.add_argument("--vae_dtype", type=str, default="float32")
    parser.add_argument("--vae_tiling", action="store_true",
                        help="(re-)enable spatial VAE tiling; the release ships with tiling on")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None)

    # Output
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_type", type=str, default="video", choices=["video", "latent", "both"])
    parser.add_argument("--output_filename", type=str, default=None, help="explicit output file path (queue)")
    parser.add_argument("--latent_path", type=str, default=None,
                        help="decode-only: a *_latent.safetensors with 'latent' and 'audio_latent' keys")
    parser.add_argument("--audio_save_path", type=str, default=None, help="also keep the .wav here")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--preview", type=int, default=None, metavar="N", help="write latent preview every N steps")
    parser.add_argument("--preview_suffix", type=str, default=None)
    parser.add_argument(
        "--preview_vae", type=str, default=None,
        help="path to a TAEHV checkpoint (e.g. weights/taeh3.safetensors) for full-resolution TAE previews; "
             "known madebyollin/taehv checkpoints are downloaded automatically when missing. "
             "default = fast latent2rgb",
    )
    parser.add_argument("--fps", type=int, default=24, help=argparse.SUPPRESS)  # fixed by the model; kept for tooling

    args = parser.parse_args()
    if args.fps != 24:
        logger.warning("MiniMax-H3 generates at a fixed 24 fps; ignoring --fps")
        args.fps = 24
    return args


def detect_task(args) -> str:
    if args.task and args.task != "auto":
        return args.task
    if args.reference:
        return "ref2va"
    if args.image_path or args.last_image_path:
        return "fl2va"
    return "t2va"


def read_text_or_path(value: str) -> str:
    if value and os.path.isfile(value) and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value or ""


def build_references(args):
    """Sniff each --reference file's kind from its extension and build MiniMaxH3Reference
    objects (which decode the media as they are constructed)."""
    from minimax_video.packing_ref2va import MiniMaxH3Reference

    references = []
    for path in args.reference or []:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            references.append(MiniMaxH3Reference(image=path))
        elif ext in AUDIO_EXTENSIONS:
            references.append(MiniMaxH3Reference(audio=path))
        else:
            reference = MiniMaxH3Reference(video=path)
            if args.reference_strip_audio:
                reference.audio = None
            references.append(reference)
    return references


def resolve_canvas_args(args):
    if args.video_size is not None:
        return int(args.video_size[0]), int(args.video_size[1])
    return None, None


def build_pipeline_shell(args, device):
    """The pipeline with only geometry facts + schedulers: enough for setup and layout."""
    from minimax_video.model_loader import load_schedulers, read_component_geometry
    from minimax_video.pipeline import MiniMaxH3Pipeline

    geometry = read_component_geometry(args.ckpt_dir)
    scheduler, audio_scheduler = load_schedulers(
        args.ckpt_dir, flow_shift=args.flow_shift, audio_flow_shift=args.audio_flow_shift
    )
    return MiniMaxH3Pipeline(
        scheduler=scheduler,
        audio_scheduler=audio_scheduler,
        device=device,
        **geometry,
    )


# ---------------------------------------------------------------------------
# Stage A: conditioner
# ---------------------------------------------------------------------------


def _prompt_cache_key(args, task, plan, prompt) -> str:
    import hashlib

    payload = {
        "prompt": prompt,
        "task": task,
        "canvas": [plan.height, plan.width, plan.num_frames],
        "image": args.image_path,
        "last_image": args.last_image_path,
        "references": args.reference,
        "strip_audio": bool(args.reference_strip_audio),
        # different encoder weights produce different embeddings for the same prompt
        "text_encoder": args.text_encoder,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def encode_prompt_stage(args, task, plan, prompt, device):
    """Load the conditioner, encode the presentation, free it. Returns (embeds, tags)."""
    cache_key = None
    if args.prompt_cache:
        cache_key = _prompt_cache_key(args, task, plan, prompt)
        if os.path.exists(args.prompt_cache):
            try:
                cached = load_safetensors(args.prompt_cache, device="cpu")
                with open(args.prompt_cache + ".json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("key") == cache_key:
                    logger.info(f"prompt cache hit: {args.prompt_cache}")
                    if task == "ref2va":
                        # block_timestamps and latent geometry are re-derived by the VAE stage;
                        # only the conditioner forward is skipped.
                        for reference, stamps in zip(plan.prepared_references, meta.get("block_timestamps", [])):
                            reference.block_timestamps = stamps
                    return cached["prompt_embeds"], cached["text_token_tags"]
            except Exception as e:
                logger.warning(f"prompt cache unusable ({e}); re-encoding")

    logger.info("Loading text encoder (Qwen3-VL conditioner)...")
    from minimax_video.conditioner import MiniMaxH3Conditioner

    start = time.time()
    conditioner = MiniMaxH3Conditioner(
        args.ckpt_dir,
        device=device if args.text_encoder_gpu_layers != 0 else "cpu",
        dtype=str_to_dtype(args.text_encoder_dtype),
        gpu_layers=args.text_encoder_gpu_layers,
        stream_device=device if args.text_encoder_stream else None,
        text_encoder_path=args.text_encoder,
        int8_use_int_mm=args.int8_fast,
    )
    logger.info(f"conditioner loaded in {time.time() - start:.1f}s")

    if task == "ref2va":
        embeds, tags = conditioner.encode_prompt_ref2va(prompt, plan.prepared_references)
    else:
        embeds, tags = conditioner.encode_prompt(prompt, plan.keyframes)
    embeds = embeds.to("cpu", torch.float32)

    if args.prompt_cache and cache_key is not None:
        mem_eff_save_file(
            {"prompt_embeds": embeds.contiguous(), "text_token_tags": tags.contiguous()}, args.prompt_cache
        )
        with open(args.prompt_cache + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": cache_key,
                    "block_timestamps": [
                        list(reference.block_timestamps) for reference in plan.prepared_references
                    ],
                },
                f,
            )

    del conditioner
    # The int8/fp8 monkey patches store a bound forward on each patched module, creating
    # module <-> method reference cycles that plain refcounting cannot free: without a
    # collect here, every GPU-resident conditioner layer survives the del and the
    # transformer stage starts ~20 GB short.
    gc.collect()
    clean_memory_on_device(device)
    return embeds, tags


# ---------------------------------------------------------------------------
# Stage C helpers: transformer + preview callback
# ---------------------------------------------------------------------------


def _apply_diffusers_mm(transformer, strategy, device, seq_len):
    """Offload the DiT via the diffusers-mm ModelManager instead of the built-in block swap.

    The activation fit is diffusers-mm's own measurement for MiniMax-H3, passed through the
    ctor overrides because its model profile is only adopted by whole-pipeline registration.
    `seq_len` is the packed sequence the resolver budgets the working set against, so this
    runs after the layout is built rather than at load time.
    """
    try:
        from diffusers_mm import ModelManager
    except ImportError as e:
        raise RuntimeError(
            "diffusers-mm offload engine selected but not installed: pip install diffusers-mm"
        ) from e
    mm = ModelManager(
        strategy=strategy,
        auto_block_pin_act_slope_gb_per_ktoken=0.158,
        auto_block_pin_act_safety_factor=1.2,
    )
    mm.register_component("transformer", transformer)
    mm.set_block_pin_workload(seq_len, 1)
    resolved = mm.apply_offload_strategy(device)
    logger.info(f"diffusers-mm offload engine active (strategy={resolved}, seq_len={seq_len})")
    return mm


def load_transformer_stage(args, task, device):
    from minimax_video import attention as minimax_attention
    from minimax_video import transformer as minimax_transformer
    from minimax_video.model_loader import load_transformer

    # "sol" is scoped to the DiT only (set per-processor after load); the module-level default
    # also serves the video/audio VAEs, which stay on SDPA.
    minimax_attention.set_attention_backend("sdpa" if args.attn_mode == "sol" else args.attn_mode)
    minimax_transformer.set_act_chunk_rows(args.act_chunk_rows)
    if args.act_chunk_rows:
        logger.info(f"row-chunked activations enabled: {args.act_chunk_rows} rows per slice")

    if args.compile:
        # Function-level compilation is handled via @maybe_compile decorators in minimax_video
        # (flag set at import time from sys.argv, before the model modules were imported).
        logger.info("torch.compile enabled via function-level decorators (mode: max-autotune-no-cudagraphs, dynamic: True)")
        # Enable persistent disk caching for compiled kernels
        try:
            from torch._inductor import config as inductor_config

            inductor_config.fx_graph_cache = True
            logger.info("Inductor disk cache enabled - compiled kernels will be cached for faster subsequent runs")
        except (ImportError, AttributeError):
            logger.warning("Could not enable inductor cache (requires PyTorch 2.1+)")
        torch._dynamo.config.cache_size_limit = 32

    lora_weights_list = None
    lora_multipliers = None
    if args.lora_weight:
        from utils.lora_utils import filter_lora_state_dict

        lora_weights_list = []
        lora_multipliers = list(args.lora_multiplier or [])
        for i, lora_path in enumerate(args.lora_weight):
            sd = load_safetensors(lora_path, device="cpu")
            include = args.include_patterns[i] if args.include_patterns and i < len(args.include_patterns) else None
            exclude = args.exclude_patterns[i] if args.exclude_patterns and i < len(args.exclude_patterns) else None
            if include or exclude:
                sd = filter_lora_state_dict(sd, include, exclude)
            lora_weights_list.append(sd)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)

    logger.info("Loading DiT weights...")
    start = time.time()
    transformer = load_transformer(
        args.ckpt_dir,
        device=device,
        task=task,
        dit_dtype=str_to_dtype(args.dit_dtype),
        fp8=args.fp8,
        fp8_scaled=args.fp8_scaled,
        fp8_fast=args.fp8_fast,
        fp8_exclude_adaln=args.fp8_exclude_adaln,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        dit_path=args.dit,
        int8_use_int_mm=args.int8_fast,
    )
    if args.attn_mode == "sol":
        from minimax_video.sol_attn import SOL_CTX, is_sol_available

        SOL_CTX.tau = args.sol_tau
        SOL_CTX.dense_steps = args.sol_dense_steps
        SOL_CTX.dense_blocks = args.sol_dense_blocks
        SOL_CTX.min_tokens = args.sol_min_tokens
        SOL_CTX.strict = args.sol_strict
        transformer.set_attention_backend("sol")
        ok, why = is_sol_available()
        if not ok:
            if args.sol_strict:
                raise RuntimeError(f"--attn_mode sol with --sol_strict, but the kernel is unavailable: {why}")
            logger.warning(f"sol-attn unavailable ({why}); the DiT will run dense SDPA")
        else:
            logger.info(
                f"sol-attn enabled: tau={args.sol_tau}, dense_steps={args.sol_dense_steps}, "
                f"dense_blocks={args.sol_dense_blocks}, min_tokens={args.sol_min_tokens}"
            )
    if args.offload_engine == "diffusers_mm":
        # the DiT stays on CPU: diffusers-mm places it in run_one, once the packed
        # sequence length the strategy is budgeted against is known
        if args.blocks_to_swap and args.blocks_to_swap > 0:
            logger.warning(
                f"--offload_engine diffusers_mm manages the DiT: --blocks_to_swap "
                f"{args.blocks_to_swap} is ignored"
            )
    elif args.blocks_to_swap and args.blocks_to_swap > 0:
        transformer.enable_block_swap(
            args.blocks_to_swap, device, supports_backward=False, streaming=not args.classic_block_swap
        )
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
    else:
        transformer.to(device)
    logger.info(f"transformer loaded in {time.time() - start:.1f}s")
    return transformer


def make_step_callback(args, pipe, plan, layout, progress_bar, previewer_holder):
    """Preview (x0 estimate of the generated video rows) + stop-file protocol."""
    stop_base = args.output_filename
    num_condition_rows = layout.num_condition_video_rows

    def callback(step, total, latents, audio_latents, noise_pred):
        progress_bar.update(1)
        if args.preview and (step + 1) % args.preview == 0 and step + 1 < total:
            try:
                # x0 = x_t + sigma * v (data-ward velocity), for the generated video rows only.
                t = float(pipe.scheduler.timesteps[step])
                sigma = 1.0 - t
                x0_rows = latents[num_condition_rows:] + sigma * noise_pred[0, num_condition_rows:].float()
                x0 = pipe.unpack_video_latents(x0_rows, plan)  # denormalized (1, C, T, H, W)
                if args.preview_vae:
                    # TAEHV (taeh3) mimics the real video VAE and decodes raw latents.
                    preview_latents = x0[0]  # [C, T, H, W]
                else:
                    # The latent2rgb factors were fit on *normalized* latents; re-normalize.
                    mean = torch.tensor(pipe.vae.config.latents_mean, device=x0.device).view(1, -1, 1, 1, 1)
                    std = torch.tensor(pipe.vae.config.latents_std, device=x0.device).view(1, -1, 1, 1, 1)
                    preview_latents = ((x0 - mean) / std)[0]  # [C, T, H, W]
                if previewer_holder.get("previewer") is None:
                    from blissful_tuner.latent_preview import LatentPreviewer

                    previewer_holder["previewer"] = LatentPreviewer(
                        args, None, None, latents.device, torch.float32, model_type="minimax"
                    )
                previewer_holder["previewer"].preview(
                    preview_latents.float(), step, preview_suffix=args.preview_suffix
                )
            except Exception as e:  # previews must never kill a run
                logger.warning(f"preview failed at step {step}: {e}")
        if stop_base is not None and os.path.exists(stop_base + ".stop_decode"):
            try:
                os.remove(stop_base + ".stop_decode")
            except OSError:
                pass
            raise _StopGeneration()

    return callback


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_video(frames: np.ndarray, path: str, fps: int):
    try:
        import av

        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except ImportError:
        import imageio

        imageio.mimwrite(path, list(frames), fps=fps, quality=8)


def save_audio_wav(sound: torch.Tensor, path: str, sample_rate: int):
    import wave

    audio = sound.detach().float().cpu().clamp(-1, 1)
    if audio.ndim == 3:
        audio = audio[0]
    pcm = (audio.transpose(0, 1).numpy() * 32767.0).astype(np.int16)  # [N, ch]
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def mux_audio(video_path: str, audio_path: str, out_path: str):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg mux failed: {result.stderr.strip()}; keeping silent video")
        return video_path
    return out_path


def build_metadata(args, task: str, seed: int, plan) -> dict:
    return {
        "model": "minimax-h3",
        "task": task,
        "prompt": str(args.prompt),
        "seed": str(seed),
        "height": str(plan.height),
        "width": str(plan.width),
        "video_length": str(plan.num_frames),
        "fps": "24",
        "infer_steps": str(args.infer_steps),
        "flow_shift": str(args.flow_shift if args.flow_shift is not None else 12.0),
        "audio_flow_shift": str(args.audio_flow_shift if args.audio_flow_shift is not None else 3.0),
        "num_latent_frames": str(plan.num_latent_frames),
        "latent_height": str(plan.latent_height),
        "latent_width": str(plan.latent_width),
        "num_audio_latents": str(plan.num_audio_latents),
    }


def output_base(args, task: str, seed: int, index: int) -> str:
    os.makedirs(args.save_path, exist_ok=True)
    if args.output_filename:
        base = os.path.splitext(args.output_filename)[0]
        if index > 0:
            base = f"{base}_{index}"
        return base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(args.save_path, f"minimax_{task}_{stamp}_{seed}")


def save_output(args, task, seed, plan, video_latents, audio_latents, frames, waveform, sample_rate, index=0):
    """Write latents and/or the muxed video+audio mp4. `frames`/`waveform` may be None in
    latent-only mode."""
    base = output_base(args, task, seed, index)
    metadata = None if args.no_metadata else build_metadata(args, task, seed, plan)
    saved = []

    if args.output_type in ("latent", "both") and video_latents is not None:
        latent_path = base + "_latent.safetensors"
        mem_eff_save_file(
            {
                "latent": video_latents.detach().cpu().contiguous(),
                "audio_latent": audio_latents.detach().cpu().contiguous(),
            },
            latent_path,
            metadata=metadata,
        )
        saved.append(latent_path)

    if args.output_type in ("video", "both") and frames is not None:
        video_path = base + ".mp4"
        save_video(frames, video_path, fps=24)
        if waveform is not None:
            wav_path = args.audio_save_path or (base + ".wav")
            save_audio_wav(waveform, wav_path, sample_rate)
            mux_path = base + "_audio.mp4"
            final = mux_audio(video_path, wav_path, mux_path)
            if final == mux_path and os.path.exists(mux_path):
                os.replace(mux_path, video_path)
            if args.audio_save_path is None and os.path.exists(wav_path):
                os.remove(wav_path)
        saved.append(video_path)
        print(f"Video saved to: {video_path}", flush=True)

    for p in saved:
        logger.info(f"saved: {p}")
    return saved


# ---------------------------------------------------------------------------
# Decode-only mode
# ---------------------------------------------------------------------------


def decode_only(args, device):
    from minimax_video.model_loader import load_audio_vae, load_vae
    from minimax_video.pipeline import MiniMaxH3Pipeline

    with open(args.latent_path, "rb"):
        pass  # fail early with a clear error if unreadable
    sd = load_safetensors(args.latent_path, device="cpu")
    if "latent" not in sd or "audio_latent" not in sd:
        raise ValueError(f"{args.latent_path} must hold 'latent' and 'audio_latent' keys")

    vae = load_vae(args.ckpt_dir, device, vae_dtype=str_to_dtype(args.vae_dtype), vae_path=args.vae)
    audio_vae = load_audio_vae(args.ckpt_dir, device, audio_vae_path=args.audio_vae)
    if args.vae_tiling:
        vae.enable_tiling()
    pipe = MiniMaxH3Pipeline(vae=vae, audio_vae=audio_vae, device=device)

    frames = pipe.decode_video(sd["latent"].to(device))
    waveform, sample_rate = pipe.decode_audio(sd["audio_latent"].to(device))

    base = os.path.splitext(args.output_filename)[0] if args.output_filename else os.path.join(
        args.save_path, f"minimax_decode_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(args.save_path, exist_ok=True)
    video_path = base + ".mp4"
    save_video(frames, video_path, fps=24)
    wav_path = args.audio_save_path or (base + ".wav")
    save_audio_wav(waveform, wav_path, sample_rate)
    final = mux_audio(video_path, wav_path, base + "_audio.mp4")
    if final != video_path and os.path.exists(final):
        os.replace(final, video_path)
    if args.audio_save_path is None and os.path.exists(wav_path):
        os.remove(wav_path)
    print(f"Video saved to: {video_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one(args, task, device, seed, extra_references=None, motion_context=None):
    from minimax_video.model_loader import load_audio_vae, load_vae

    pipe = build_pipeline_shell(args, device)
    generator = torch.Generator().manual_seed(seed)  # CPU generator: device-independent draws

    # Setup (geometry + media preparation; no weights needed).
    prompt = read_text_or_path(args.prompt)
    height, width = resolve_canvas_args(args)
    if height is None and args.aspect_ratio and not (args.image_path or args.last_image_path):
        from minimax_video.packing import resolve_canvas_size

        aspect_w, aspect_h = (float(x) for x in args.aspect_ratio.split(":"))
        height, width = resolve_canvas_size(aspect_w, aspect_h)

    image = Image.open(args.image_path) if args.image_path else None
    last_image = Image.open(args.last_image_path) if args.last_image_path else None
    references = build_references(args) if task == "ref2va" else None
    if extra_references:
        references = (references or []) + list(extra_references)
    num_frames = args.video_length if args.video_length else None

    plan = pipe.setup(
        task,
        height=height,
        width=width,
        num_frames=num_frames,
        image=image,
        last_image=last_image,
        references=references,
        motion_context=motion_context,
        audio_motion_mode=getattr(args, "chain_audio_mode", "timeline"),
    )
    logger.info(
        f"task={plan.task} size={plan.width}x{plan.height} frames={plan.num_frames} "
        f"steps={args.infer_steps} seed={seed}"
    )

    # Stage A: conditioner -> prompt embeds (freed afterwards).
    prompt_embeds, text_token_tags = encode_prompt_stage(args, plan.task, plan, prompt, device)

    # Stage B: VAEs -> condition latents. Condition noise is the request generator's FIRST draw.
    logger.info("Loading VAE...")
    vae = load_vae(args.ckpt_dir, device, vae_dtype=str_to_dtype(args.vae_dtype), vae_path=args.vae)
    audio_vae = load_audio_vae(args.ckpt_dir, device, audio_vae_path=args.audio_vae)
    if args.vae_tiling:
        vae.enable_tiling()
    pipe.vae, pipe.audio_vae = vae, audio_vae
    pipe.vae_latent_channels = vae.config.latent_channels
    pipe.audio_latent_channels = audio_vae.config.latent_channels
    pipe.audio_sampling_rate = audio_vae.config.sampling_rate

    condition_latents, audio_condition_latents = pipe.encode_conditions(plan, generator=generator)
    vae.to("cpu")
    audio_vae.to("cpu")
    clean_memory_on_device(device)

    # Stage C: transformer -> denoise.
    transformer = load_transformer_stage(args, plan.task, device)
    pipe.transformer = transformer
    pipe.patch_size = tuple(transformer.config.patch_size)

    layout = pipe.build_layout(plan, text_token_tags)
    mm = None
    if args.offload_engine == "diffusers_mm":
        mm = _apply_diffusers_mm(transformer, args.mm_strategy, device, layout.sequence_length)
    latents, audio_latents = pipe.prepare_latents(
        plan,
        generator=generator,
        condition_latents=condition_latents,
        audio_condition_latents=audio_condition_latents,
        layout=layout,
    )
    timesteps, audio_timesteps, row_timestep_plan = pipe.set_timesteps(args.infer_steps, layout)

    previewer_holder = {}
    stopped_early = False
    progress_bar = tqdm(total=len(timesteps), desc="denoise")
    callback = make_step_callback(args, pipe, plan, layout, progress_bar, previewer_holder)
    try:
        latents, audio_latents = pipe.denoise(
            layout,
            latents,
            audio_latents,
            prompt_embeds.to(device),
            timesteps,
            audio_timesteps,
            row_timestep_plan,
            step_callback=callback,
        )
    except _StopGeneration:
        stopped_early = True
        logger.info("stop requested: decoding the current state")
    finally:
        progress_bar.close()

    if args.attn_mode == "sol":
        from minimax_video.sol_attn import SOL_CTX

        logger.info(f"sol-attn attention calls: {SOL_CTX.stats}")
        if (
            args.sol_strict
            and not stopped_early
            and len(timesteps) > args.sol_dense_steps  # a run that is all warmup is dense by design
            and SOL_CTX.stats["sparse"] == 0
        ):
            raise RuntimeError(
                f"--sol_strict: the denoise loop finished without a single sparse attention call "
                f"(stats: {SOL_CTX.stats})"
            )

    if mm is not None:
        mm.unregister_component("transformer")  # removes the offload hooks and the manager's own refs
        release_host_cache = mm.release_host_cache
        mm = None
    else:
        release_host_cache = None
    del transformer
    pipe.transformer = None
    gc.collect()  # int8/fp8-patched modules sit in reference cycles; free them before the VAEs return
    if release_host_cache is not None:
        release_host_cache()  # gc does not return the group-offload pinned host pool
    clean_memory_on_device(device)

    # Stage D: VAEs back -> decode + save.
    video_latents = pipe.unpack_video_latents(latents, plan, layout)
    audio_latent_tensor = pipe.unpack_audio_latents(audio_latents, plan, layout)

    frames = waveform = sample_rate = None
    if args.output_type in ("video", "both"):
        vae.to(device)
        audio_vae.to(device)
        frames = pipe.decode_video(video_latents.to(device))
        waveform, sample_rate = pipe.decode_audio(audio_latent_tensor.to(device))
        vae.to("cpu")
        audio_vae.to("cpu")
        clean_memory_on_device(device)

    return plan, video_latents, audio_latent_tensor, frames, waveform, sample_rate, stopped_early


def save_chain_segment(args, task, seed, plan, video_latents, audio_latents, frames, waveform, sample_rate,
                       base: str, clip_index: int):
    clip_base = f"{base}_clip{clip_index + 1}"
    metadata = None if args.no_metadata else build_metadata(args, task, seed, plan)

    if args.output_type in ("latent", "both") and video_latents is not None:
        mem_eff_save_file(
            {
                "latent": video_latents.detach().cpu().contiguous(),
                "audio_latent": audio_latents.detach().cpu().contiguous(),
            },
            clip_base + "_latent.safetensors",
            metadata=metadata,
        )
        logger.info(f"saved: {clip_base}_latent.safetensors")

    if args.chain_keep_segments and frames is not None:
        video_path = clip_base + ".mp4"
        save_video(frames, video_path, fps=24)
        if waveform is not None:
            wav_path = clip_base + ".wav"
            save_audio_wav(waveform, wav_path, sample_rate)
            final = mux_audio(video_path, wav_path, clip_base + "_audio.mp4")
            if final != video_path and os.path.exists(final):
                os.replace(final, video_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        logger.info(f"saved: {video_path}")


def _ffprobe_csv(path: str, *probe_args) -> str:
    result = subprocess.run(["ffprobe", "-v", "error", *probe_args, "-of", "csv=p=0", path],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def join_with_source(source_path: str, continuation_path: str, out_path: str,
                     width: int, height: int, sample_rate: int) -> bool:
    """Re-encode concat of the source video and the generated continuation, scaling the source
    onto the generated canvas and its audio onto the audio VAE's rate."""
    src_audio = _ffprobe_csv(source_path, "-select_streams", "a:0",
                             "-show_entries", "stream=codec_type") == "audio"
    cont_audio = _ffprobe_csv(continuation_path, "-select_streams", "a:0",
                              "-show_entries", "stream=codec_type") == "audio"

    v0 = f"[0:v]fps=24,scale={width}:{height}:flags=lanczos,setsar=1,format=yuv420p[v0]"
    aformat = f"aformat=sample_rates={sample_rate}:channel_layouts=stereo"
    if src_audio and cont_audio:
        graph = (f"{v0};[0:a]{aformat}[a0];[1:a]{aformat}[a1];"
                 f"[v0][a0][1:v][a1]concat=n=2:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]
    elif cont_audio:
        delay_ms = int(float(_ffprobe_csv(source_path, "-show_entries", "format=duration") or 0) * 1000)
        graph = f"{v0};[v0][1:v]concat=n=2:v=1:a=0[v];[1:a]{aformat},adelay={delay_ms}:all=1[a]"
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        graph = f"{v0};[v0][1:v]concat=n=2:v=1:a=0[v]"
        maps = ["-map", "[v]"]

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", source_path, "-i", continuation_path,
           "-filter_complex", graph, *maps, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    if "[a]" in maps[-1]:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(out_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg concat with source failed: {result.stderr.strip()}")
        return False
    return True


def trim_chain_segment(frames, waveform, sample_rate, num_context_frames):
    """Drop the regenerated pinned run off the front of a segment, picture and sound together.

    Trimming only the picture would slide the whole soundtrack `num_context_frames / 24` seconds early, and matching
    the tail keeps every segment exactly as long as its picture so the sample grid's rounding cannot accumulate."""
    from minimax_video.motion_context import trim_segment_audio

    frames = frames[num_context_frames:]
    if waveform is None:
        return frames, None
    cut, want = trim_segment_audio(waveform.shape[-1], num_context_frames, frames.shape[0], sample_rate)
    waveform = waveform[..., cut:]
    if waveform.shape[-1] > want:
        waveform = waveform[..., :want]
    elif waveform.shape[-1] < want:
        waveform = torch.nn.functional.pad(waveform, (0, want - waveform.shape[-1]))
    return frames, waveform


def run_chain(args, base_task, device):
    """Generate --chain_count segments and join them into one video.

    Under --chain_mode motion each segment after the first pins the previous segment's tail as conditioning rows on
    its own timeline, regenerates those frames and has them trimmed back off, so the join is a continuation rather
    than a cut. The older reference modes carry the previous segment as a reference instead. With --chain_extend an
    existing video seeds the chain the same way and is prepended to the joined output.

    --last_image_path rides the final segment and pins the end of the chain; --image_path only survives a segment
    that starts cold, because a motion context already anchors frame 0."""
    from minimax_video.motion_context import (
        MiniMaxH3MotionContext,
        chain_frame_budget,
        motion_context_num_frames,
    )
    from minimax_video.packing import align_num_frames, audio_latent_num_frames
    from minimax_video.packing_ref2va import MiniMaxH3Reference, resample_reference_frames

    if not args.video_length:
        raise ValueError("chaining needs an explicit --video_length per segment")
    if args.num_outputs != 1:
        logger.warning("chaining forces --num_outputs 1")
        args.num_outputs = 1
    if args.output_type == "latent":
        logger.warning("chaining needs decoded frames; using --output_type both")
        args.output_type = "both"

    motion_mode = args.chain_mode == "motion"
    if not motion_mode and args.prompt_cache:
        logger.warning("prompt cache disabled while chaining: per-segment references change")
        args.prompt_cache = None

    span = 0
    if motion_mode:
        # Everything the chain arithmetic depends on is checked before a single weight is loaded.
        raw_frames = align_num_frames(args.video_length)
        span = motion_context_num_frames(args.chain_context_length)
        if span != args.chain_context_length:
            logger.warning(
                f"--chain_context_length {args.chain_context_length} is off the video VAE grid, using {span} "
                "(the grid is 1 and every 17 * n + 5)"
            )
        budget = chain_frame_budget(raw_frames, span, args.chain_count, extend=bool(args.chain_extend))
        for index, delivered in enumerate(budget[:-1]):
            if delivered < span:
                raise ValueError(
                    f"segment {index + 1} delivers {delivered} frames, fewer than the {span} frame motion context "
                    f"the next one needs. Raise --video_length or lower --chain_context_length."
                )
        logger.info(
            f"motion context: {span} frames pinned per link, {sum(budget)} frames delivered "
            f"({sum(budget) / 24:.2f}s) from {args.chain_count} x {raw_frames} generated"
        )

    base_seed = args.seed
    base = output_base(args, base_task, base_seed, 0)
    all_frames, all_waveforms = [], []
    extra_ref = motion_context = sample_rate = None

    if args.chain_extend:
        source = MiniMaxH3Reference(video=args.chain_extend, match_canvas=True)
        if args.video_size is None:
            from minimax_video.packing import MINIMAX_H3_MAX_PIXELS

            height, width = source.video.shape[1], source.video.shape[2]
            scale = min(1.0, (MINIMAX_H3_MAX_PIXELS / (height * width)) ** 0.5)
            args.video_size = [max(32, round(height * scale / 32) * 32),
                               max(32, round(width * scale / 32) * 32)]
            logger.info(f"canvas matched to the source video: {args.video_size[1]}x{args.video_size[0]}")
        if motion_mode:
            # An mp4 carries no H3 latent, so the tail is re-encoded through the same VAEs the links use; only this
            # first seam pays for that.
            frames = resample_reference_frames(source.video, float(source.fps))
            motion_context = MiniMaxH3MotionContext(
                frames=frames[-span:],
                waveform=source.audio,
                sample_rate=source.sample_rate,
                previous_num_frames=frames.shape[0],
                previous_num_audio_latents=audio_latent_num_frames(frames.shape[0]),
                num_frames=span,
                audio_num_frames=args.chain_audio_context,
                encode_mode=args.chain_context_encode,
                crop=args.chain_context_crop,
            )
            del source
        elif args.chain_mode == "last_frame":
            extra_ref = MiniMaxH3Reference(image=Image.fromarray(source.video[-1]), match_canvas=True)
            del source
        else:
            extra_ref = source

    keyframe_paths = (args.image_path, args.last_image_path)
    for i in range(args.chain_count):
        print(f"=== Generating clip {i + 1}/{args.chain_count} ===", flush=True)
        task = "ref2va" if extra_ref is not None else base_task
        if any(keyframe_paths):
            # The first frame anchors frame 0, which a motion context already pins with the previous clip's picture,
            # so it only survives a cold start. The last frame anchors the final frame, which no context run reaches,
            # so it rides the clip that closes the chain.
            args.image_path = keyframe_paths[0] if i == 0 and not (motion_mode and args.chain_extend) else None
            args.last_image_path = keyframe_paths[1] if i == args.chain_count - 1 else None
            if i == 0:
                if keyframe_paths[0] and not args.image_path:
                    logger.info("first frame dropped: the motion context anchors the start of an extended clip")
                if keyframe_paths[1] and args.chain_count > 1:
                    logger.info(f"last frame held for clip {args.chain_count}: it pins the end of the chain")
        plan, video_latents, audio_latents, frames, waveform, sample_rate, stopped_early = run_one(
            args, task, device, base_seed + i,
            extra_references=[extra_ref] if extra_ref is not None else None,
            motion_context=motion_context,
        )
        if i == 0 and args.video_size is None:
            args.video_size = [plan.height, plan.width]  # pin the auto canvas for all later segments
        raw_num_frames, raw_num_audio_latents = plan.num_frames, plan.num_audio_latents
        if motion_context is not None and frames is not None:
            frames, waveform = trim_chain_segment(frames, waveform, sample_rate, plan.motion_context.num_frames)
        all_frames.append(frames)
        all_waveforms.append(waveform)
        save_chain_segment(args, task, base_seed + i, plan, video_latents, audio_latents,
                           frames, waveform, sample_rate, base, i)
        del video_latents
        if stopped_early:
            if keyframe_paths[1] and i < args.chain_count - 1:
                logger.warning("stopped before the final clip: the last frame never got pinned")
            logger.info(f"stop requested in clip {i + 1}: joining the segments generated so far")
            break
        if i + 1 < args.chain_count:
            if motion_mode:
                # The delivered tail carries the picture; the audio latents are the untrimmed ones, because the
                # overhang is a property of the clip the VAE encoded and the trim only removes its front.
                motion_context = MiniMaxH3MotionContext(
                    frames=frames[-span:],
                    audio_latents=audio_latents.detach().cpu(),
                    previous_num_frames=raw_num_frames,
                    previous_num_audio_latents=raw_num_audio_latents,
                    num_frames=span,
                    audio_num_frames=args.chain_audio_context,
                    encode_mode=args.chain_context_encode,
                    crop=args.chain_context_crop,
                )
            elif args.chain_mode == "last_frame":
                extra_ref = MiniMaxH3Reference(image=Image.fromarray(frames[-1]), match_canvas=True)
            else:
                extra_ref = MiniMaxH3Reference(video=frames, audio=waveform[0].detach().cpu(),
                                               sample_rate=sample_rate, match_canvas=True)
        del audio_latents
        clean_memory_on_device(device)

    # One encode + one mux: raw waveforms are concatenated so the seams carry no AAC padding.
    final_path = base + ".mp4"
    video_path = (base + "_continuation.mp4") if args.chain_extend else final_path
    stem = os.path.splitext(video_path)[0]
    joined = np.concatenate(all_frames, axis=0)
    save_video(joined, video_path, fps=24)
    if all(w is not None for w in all_waveforms):
        wav_path = args.audio_save_path or (stem + ".wav")
        save_audio_wav(torch.cat([w.detach().float().cpu() for w in all_waveforms], dim=-1),
                       wav_path, sample_rate)
        final = mux_audio(video_path, wav_path, stem + "_audio.mp4")
        if final != video_path and os.path.exists(final):
            os.replace(final, video_path)
        if args.audio_save_path is None and os.path.exists(wav_path):
            os.remove(wav_path)

    if args.chain_extend:
        height, width = joined.shape[1], joined.shape[2]
        if join_with_source(args.chain_extend, video_path, final_path, width, height,
                            int(sample_rate) if sample_rate else 32000):
            if not args.chain_keep_segments and os.path.exists(video_path):
                os.remove(video_path)
        else:
            logger.warning("keeping the continuation alone as the output")
            os.replace(video_path, final_path)
    print(f"Video saved to: {final_path}", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.latent_path:
        if args.ckpt_dir is None and (args.vae is None or args.audio_vae is None):
            raise ValueError("decode-only mode needs --ckpt_dir (or --vae and --audio_vae)")
        decode_only(args, device)
        return

    if args.ckpt_dir is None:
        raise ValueError("--ckpt_dir is required")
    if not args.prompt:
        raise ValueError("--prompt is required")

    task = detect_task(args)
    if args.seed is None or args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    start = time.time()
    if args.chain_count > 1 or args.chain_extend:
        run_chain(args, task, device)
        logger.info(f"done in {time.time() - start:.1f}s")
        return

    for i in range(args.num_outputs):
        seed = args.seed + i
        plan, video_latents, audio_latents, frames, waveform, sample_rate, _ = run_one(args, task, device, seed)
        save_output(args, plan.task, seed, plan, video_latents, audio_latents, frames, waveform, sample_rate, index=i)
        clean_memory_on_device(device)
    logger.info(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
