#!/usr/bin/env python3
"""Open-H3-Regenerate-2K upscale/regenerate driver (subprocess entry for h3.py).

Takes a 768P/576P draft video (or a normalized H3 latent) and regenerates it at
2K/4K with the Open-H3-Regenerate-2K pipeline from NVlabs/Sana (sol-engine branch):
latent upscaler -> 2/3-step refine with the merged LoRA -> H3 VAE decode.

Setup is automatic and cached under --weights-dir (default weights/open_h3):
  repo/            sparse clone of NVlabs/Sana sol-engine (models/Open-H3-Regenerate-2K)
  lora/ latent_upsampler/ vae_fast/   fetched by the repo's download.py --skip-base
  base_ref/        symlink farm over the local MiniMax-H3 Diffusers dir with localized
                   model indexes, so nothing is re-fetched from the Hub

Prints machine-readable lines for the GUI:
  PROGRESS: <frac 0..1> <text>
  OUTPUT: <video path>
  ERROR: <text>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

SANA_URL = "https://github.com/NVlabs/Sana"
SANA_BRANCH = "sol-engine"
MODEL_SUBPATH = Path("models") / "Open-H3-Regenerate-2K"
HUB_REPO = "MiniMaxAI/MiniMax-H3"
INDEX_FILES = ("model_index.json", "modular_model_index.json")


def progress(frac, text):
    print(f"PROGRESS: {frac:.2f} {text}", flush=True)


def fail(text):
    print(f"ERROR: {text}", flush=True)
    sys.exit(1)


def stream_child(cmd, cwd, frac, label):
    """Run a setup subprocess, forwarding its output as PROGRESS lines."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.strip().lstrip("\r")
        if not line:
            continue
        tail.append(line)
        tail = tail[-40:]
        print(f"PROGRESS: {frac:.2f} {label}: {line}", flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{label} failed (exit {code}): " + " | ".join(tail[-6:]))


def ensure_code(weights: Path) -> Path:
    """Sparse-clone NVlabs/Sana (sol-engine) once; return the model directory."""
    repo = weights / "repo"
    model_dir = repo / MODEL_SUBPATH
    if (model_dir / "infer.py").is_file() and (model_dir / "open_h3").is_dir():
        return model_dir
    if repo.exists():
        shutil.rmtree(repo)
    progress(0.02, "Fetching Open-H3-Regenerate-2K code from NVlabs/Sana (sol-engine)...")
    stream_child(["git", "clone", "--filter=blob:none", "--depth", "1", "--branch", SANA_BRANCH,
                  "--no-checkout", SANA_URL, str(repo)], weights, 0.02, "clone")
    stream_child(["git", "sparse-checkout", "set", str(MODEL_SUBPATH), "techniques/sparse_backends"],
                 repo, 0.04, "checkout")
    stream_child(["git", "checkout", SANA_BRANCH], repo, 0.04, "checkout")
    if not (model_dir / "infer.py").is_file():
        raise RuntimeError(f"sparse clone did not produce {model_dir}")
    return model_dir


def ensure_weights(weights: Path, model_dir: Path, force: bool):
    """Fetch LoRA / latent upscaler / fast VAE / adaln table via the repo's download.py."""
    marker = weights / ".weights_ready"
    required = [
        weights / "lora" / "open_h3_regenerate_2k_lora.safetensors",
        weights / "lora" / "sharp_curve_branches.safetensors",
        weights / "lora" / "adaln_t_table.bin",
        weights / "latent_upsampler" / "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    ]
    if not force and marker.is_file() and all(p.is_file() for p in required):
        return
    progress(0.06, "Downloading Open-H3-Regenerate-2K weights (LoRA + latent upscaler + VAE)...")
    stream_child([sys.executable, "download.py", "--weights", str(weights), "--skip-base"],
                 model_dir, 0.06, "download")
    marker.write_text("ok")


def _localize_index(obj, base_dir: str):
    """Point every MiniMaxAI/MiniMax-H3 component reference at the local snapshot."""
    if isinstance(obj, dict):
        if obj.get("pretrained_model_name_or_path") == HUB_REPO:
            obj["pretrained_model_name_or_path"] = base_dir
            obj["revision"] = None
        for value in obj.values():
            _localize_index(value, base_dir)
    elif isinstance(obj, list):
        for value in obj:
            _localize_index(value, base_dir)


def prepare_base_ref(ckpt_dir: Path, base_ref: Path):
    """Symlink the local MiniMax-H3 Diffusers dir and rewrite its indexes to resolve locally."""
    src = ckpt_dir.resolve()
    for index_file in INDEX_FILES:
        if not (src / index_file).is_file():
            raise RuntimeError(f"{src} is not a MiniMax-H3 Diffusers snapshot (missing {index_file})")
    if base_ref.is_symlink() or base_ref.is_file():
        base_ref.unlink()
    elif base_ref.exists():
        shutil.rmtree(base_ref)  # only symlinks + rewritten json live here; never the sources
    base_ref.mkdir(parents=True)
    target = str(base_ref.resolve())
    for item in src.iterdir():
        if item.name in INDEX_FILES:
            continue
        try:
            os.symlink(item.resolve(), base_ref / item.name)
        except (OSError, NotImplementedError):  # e.g. filesystem without symlink support
            shutil.copytree(item, base_ref / item.name) if item.is_dir() else shutil.copy2(item, base_ref / item.name)
    for index_file in INDEX_FILES:
        index = json.loads((src / index_file).read_text())
        _localize_index(index, target)
        (base_ref / index_file).write_text(json.dumps(index, indent=2))
    return base_ref


def check_environment():
    try:
        from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel  # noqa: F401
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "the Open-H3 pipeline needs diffusers with MiniMax-H3 support (diffusers>=0.40): "
            f"`pip install -r <weights>/repo/{MODEL_SUBPATH}/requirements.txt` or pick a different "
            f"Python in the tab. Import error: {exc}")


def run(args):
    weights = Path(args.weights_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    weights.mkdir(parents=True, exist_ok=True)

    model_dir = ensure_code(weights)
    ensure_weights(weights, model_dir, force=args.force_download)

    progress(0.08, "Preparing local MiniMax-H3 base model reference...")
    base_ref = prepare_base_ref(ckpt_dir, weights / "base_ref")

    sys.path.insert(0, str(model_dir))
    check_environment()
    import torch
    from open_h3.pipeline import Pipeline, encode_mp4, encode_prompt, encode_video

    torch.cuda.set_device(0)
    native = None
    if args.input_latents:
        loaded = torch.load(args.input_latents, map_location="cpu", weights_only=True)["video_latents"]
        native = loaded[0] if loaded.ndim == 5 else loaded
        progress(0.10, "Loaded input latents.")
    else:
        progress(0.10, "Encoding draft video with the H3 VAE...")
        native = encode_video(base_ref, args.input_video, args.input_size, args.vae_seed)
        progress(0.25, "Draft video encoded.")

    progress(0.30, "Encoding prompt (loads the 32B text encoder, then frees it)...")
    text = encode_prompt(base_ref, args.prompt)

    vae = args.vae if args.vae != "auto" else ("fast" if args.fwd == "sol_engine" else "standard")
    progress(0.40, f"Loading refine pipeline (fwd={args.fwd}, vae={vae}, resolution={args.resolution})...")
    pipe = Pipeline(weights, base_ref, fwd=args.fwd, steps=args.steps,
                    start_sigma=args.start_sigma if args.start_sigma and args.start_sigma > 0 else None,
                    resolution=args.resolution, vae=vae)
    progress(0.45, f"Pipeline ready (sigmas={[round(x, 4) for x in pipe.sigmas]}, "
                   f"gpu-mem={torch.cuda.memory_allocated() / 2**30:.1f}GB). Refining...")

    frames, _, times = pipe(text, native, 42, args.seed)
    progress(0.90, f"Refine done ({' '.join(f'{k}={v:.1f}s' for k, v in times.items())}). Writing MP4...")
    encode_mp4(frames, args.output)
    progress(1.0, f"Saved {args.output}")
    print(f"OUTPUT: {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-video", help="draft video (encoded at --input-size, first 121 frames)")
    parser.add_argument("--input-latents", help="torch.save dict with normalized 'video_latents'")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt-dir", default="MiniMax-H3", help="local MiniMax-H3 Diffusers snapshot")
    parser.add_argument("--weights-dir", default=os.path.join("weights", "open_h3"))
    parser.add_argument("--resolution", choices=["768p", "1080p", "2k", "4k"], default="2k")
    parser.add_argument("--steps", type=int, choices=[2, 3], default=3)
    parser.add_argument("--start-sigma", type=float, default=0.0, help="0 = use the --steps default grid")
    parser.add_argument("--input-size", choices=["768p", "576p"], default="768p")
    parser.add_argument("--fwd", choices=["standard", "sol_engine"], default="standard")
    parser.add_argument("--vae", choices=["auto", "standard", "fast"], default="auto")
    parser.add_argument("--vae-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1766580932)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="run the bootstrap and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.setup_only:
            weights = Path(args.weights_dir).resolve()
            weights.mkdir(parents=True, exist_ok=True)
            model_dir = ensure_code(weights)
            ensure_weights(weights, model_dir, force=args.force_download)
            prepare_base_ref(Path(args.ckpt_dir).resolve(), weights / "base_ref")
            progress(1.0, "Setup complete.")
            return
        if not args.input_video and not args.input_latents:
            fail("give --input-video or --input-latents")
        run(args)
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        fail(str(exc))


if __name__ == "__main__":
    main()
