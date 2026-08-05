#!/usr/bin/env python3
"""Fit MiniMax-H3 latent -> RGB preview factors from real encoded pairs. One-time calibration.

The UI's latent preview maps normalized video-VAE latents (24ch) to RGB with a per-channel
linear transform (blissful_tuner/latent_preview.py), which currently holds a grayscale
placeholder for MiniMax-H3. This script derives the real factors and prints them as a
paste-ready block to hard-code there — after that, no one ever runs this again.

Only the video VAE is loaded (no DiT, no text encoder, no generation): sample frames are
encoded through the exact keyframe path of the pipeline (stretch onto the model canvas,
ImageNet-normalize, `_encode_clip`, normalize with the VAE's latents_mean/std) and the
normalized latents are least-squares fitted against the source RGB downsampled to the
latent grid (16x).

Run on the machine that has the MiniMax-H3 weights:

    python minimax_engine/fit_minimax_preview_factors.py \
        --ckpt_dir MiniMax-H3 --media outputs photos/

Point --media at any mix of image files, video files, or directories of them — previously
generated outputs/*.mp4 work well. Use diverse, colorful material; 50+ frames give a
stable fit.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# Runnable from anywhere: the H1111 repo root provides utils/ and blissful_tuner/;
# this directory (minimax_engine/) provides minimax_video/.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from minimax_video.packing import (  # noqa: E402
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    resolve_canvas_size,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", type=str, default="MiniMax-H3",
                        help="path to the MiniMax-H3 HF snapshot dir (or use --vae)")
    parser.add_argument("--vae", type=str, default=None, help="explicit video VAE path override")
    parser.add_argument("--media", type=str, nargs="+", required=True,
                        help="image/video files or directories to fit on")
    parser.add_argument("--frames_per_video", type=int, default=8,
                        help="frames sampled evenly from each video (default 8)")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="total frame cap across all media (default 200)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_here, "minimax_preview_factors.json"),
                        help="JSON record of the fit (the values themselves get hard-coded)")
    return parser.parse_args()


def collect_media(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(os.path.join(p, f) for f in sorted(os.listdir(p)))
        else:
            files.append(p)
    images = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    videos = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
    return images, videos


def video_frames(path, count):
    """Sample `count` frames evenly from a video as PIL images."""
    import av

    with av.open(path) as container:
        frames = [f.to_image() for f in container.decode(container.streams.video[0])]
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def iter_frames(args):
    images, videos = collect_media(args.media)
    if not images and not videos:
        raise SystemExit(f"No image/video files found under: {args.media}")
    print(f"Found {len(images)} images and {len(videos)} videos")
    n = 0
    for path in images:
        if n >= args.max_frames:
            return
        yield path, Image.open(path).convert("RGB")
        n += 1
    for path in videos:
        for frame in video_frames(path, args.frames_per_video):
            if n >= args.max_frames:
                return
            yield path, frame.convert("RGB")
            n += 1


@torch.no_grad()
def encode_frame(vae, image, device, pixel_mean, pixel_std, latents_mean, latents_std):
    """The pipeline's keyframe encode, minus the posterior sampling: returns
    (normalized_latent [24, h, w], target_rgb [3, h, w] in [0, 1])."""
    try:
        height, width = resolve_canvas_size(*image.size)
    except ValueError as e:  # aspect outside 1:4..4:1
        print(f"  skipping frame: {e}")
        return None
    image = image.resize((width, height), Image.LANCZOS)
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0)  # [3, H, W]

    pixels = (rgb.to(device)[None, :, None] - pixel_mean) / pixel_std  # [1, 3, 1, H, W]
    moments = vae._encode_clip(pixels)  # [1, 2C, 1, h, w]
    mu = moments[:, : moments.shape[1] // 2].float()  # posterior mean, noise-free
    latent = ((mu - latents_mean) / latents_std)[0, :, 0]  # [24, h, w]

    target = torch.nn.functional.adaptive_avg_pool2d(rgb[None], latent.shape[-2:])[0]  # [3, h, w]
    return latent.cpu(), target.cpu()


def main():
    args = parse_args()
    device = torch.device(args.device)

    from minimax_video.model_loader import load_vae

    print("Loading video VAE (float32)...")
    vae = load_vae(args.ckpt_dir, device=device, vae_dtype=torch.float32, vae_path=args.vae)
    vae.enable_tiling()  # release default; keeps encode activations small

    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    xs, ys = [], []
    for path, frame in iter_frames(args):
        pair = encode_frame(vae, frame, device, pixel_mean, pixel_std, latents_mean, latents_std)
        if pair is None:
            continue
        latent, target = pair
        xs.append(latent.reshape(latent.shape[0], -1).T)  # [h*w, 24]
        ys.append(target.reshape(target.shape[0], -1).T)  # [h*w, 3]
        print(f"  encoded {os.path.basename(path)} -> latent {tuple(latent.shape)}")

    if not xs:
        raise SystemExit("No frames survived encoding; nothing to fit.")

    x = torch.cat(xs).double()  # [N, 24]
    y = torch.cat(ys).double()  # [N, 3]
    print(f"Fitting on {x.shape[0]} latent pixels from {len(xs)} frames...")

    a = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], dim=1)  # [N, 25]
    solution = torch.linalg.lstsq(a, y).solution  # [25, 3]
    factors, bias = solution[:-1], solution[-1]

    residual = a @ solution - y
    r2 = (1 - (residual**2).sum(dim=0) / ((y - y.mean(dim=0)) ** 2).sum(dim=0)).tolist()
    print(f"R^2 per channel (R, G, B): {[f'{v:.4f}' for v in r2]}")

    payload = {
        "rgb_factors": [[round(v, 4) for v in row] for row in factors.tolist()],
        "bias": [round(v, 4) for v in bias.tolist()],
        "fit_r2": [round(v, 4) for v in r2],
        "fit_pixels": int(x.shape[0]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}\n")

    rows = payload["rgb_factors"]
    print("Paste-ready block for blissful_tuner/latent_preview.py ('minimax' entry):\n")
    print('            "minimax": {')
    print('                "rgb_factors": [')
    for i in range(0, 24, 2):
        pair = ", ".join(
            "[" + ", ".join(f"{v: .4f}".strip() for v in rows[j]) + "]" for j in (i, i + 1)
        )
        print(f"                    {pair},")
    print("                ],")
    print(f'                "bias": {payload["bias"]},')
    print("            },")


if __name__ == "__main__":
    main()
