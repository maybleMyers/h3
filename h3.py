# h3.py — standalone MiniMax-H3 GUI (MiniMax + Frame Interpolation + Video Info tabs)
# Assembled from H1111/h1111.py (branch h3 @ ece03a2); backend lives in minimax_engine/.
import gradio as gr
import subprocess
import threading
import time
import os
import random
import tiktoken
import html
from urllib.parse import quote as url_quote
import sys
from typing import List, Tuple, Optional, Generator, Dict, Any
import json
from gradio import themes
from gradio.themes.utils import colors
from PIL import Image
import math
import cv2
import shutil
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "minimax_engine")
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

logger = logging.getLogger(__name__)

# Job queue system for background generation (survives browser disconnects)
from wan_job_queue import get_queue, JobQueue, JobStatus, Job
from wan_worker import Worker

# Global tracking for the queue-based generation
wan22_current_output_filename = None
wan22_worker_thread = None
wan22_worker_instance = None

UI_CONFIGS_DIR = "ui_configs"
MINIMAX_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "minimax_defaults.json")
INTERP_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "interp_defaults.json")


# ===== Queue infrastructure (shared with the background worker) =====
def format_elapsed_time(seconds: float) -> str:
    """47s / 4m 32s / 1h 02m — for job timing shown in the queue tabs."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def wan22_poll_active_job(current_job_id: str, current_batch_id: str):
    """Poll the queue for job status updates.

    Called by Timer to get live updates. Returns updated state.
    """
    queue = get_queue()

    # Get all jobs in batch
    jobs = []
    if current_batch_id:
        jobs = queue.get_batch_jobs(current_batch_id)
    elif current_job_id:
        job = queue.get_job(current_job_id)
        if job:
            jobs = [job]

    # If no jobs found, try to find running job
    if not jobs:
        running_jobs = queue.get_running_jobs()
        if running_jobs:
            jobs = running_jobs
            current_batch_id = jobs[0].batch_id or ""

    all_videos = []
    running_job = None
    status_parts = []
    completed_count = 0
    failed_count = 0
    preview_path = None

    for job in jobs:
        if job.status == JobStatus.COMPLETED.value:
            completed_count += 1
            if job.output_filename and os.path.exists(job.output_filename):
                seed = job.parameters.get('seed', 'unknown')
                caption = f"Seed: {seed}"
                if job.elapsed_time:
                    caption += f" | {format_elapsed_time(job.elapsed_time)}"
                all_videos.append((job.output_filename, caption))
        elif job.status == JobStatus.RUNNING.value:
            running_job = job
            if job.preview_path and os.path.exists(job.preview_path):
                preview_path = job.preview_path
        elif job.status == JobStatus.FAILED.value:
            failed_count += 1
            status_parts.append(f"Job {job.id} failed: {job.error_message[:50] if job.error_message else 'Unknown error'}")
        elif job.status == JobStatus.CANCELLED.value:
            status_parts.append(f"Job {job.id} cancelled")

    total_jobs = len(jobs)
    timer_active = False
    progress_text = ""

    if running_job:
        timer_active = True
        progress_text = running_job.progress_text or f"Progress: {running_job.progress:.0f}%"
        status_parts.insert(0, f"Processing {completed_count + 1}/{total_jobs}")
    elif completed_count == total_jobs and total_jobs > 0:
        total_elapsed = sum(j.elapsed_time for j in jobs if j.status == JobStatus.COMPLETED.value)
        done_msg = f"All {total_jobs} generation(s) complete!"
        if total_elapsed:
            done_msg += f" (total {format_elapsed_time(total_elapsed)})"
        status_parts.insert(0, done_msg)
        progress_text = "Done"
        timer_active = False
    elif completed_count + failed_count == total_jobs and total_jobs > 0:
        total_elapsed = sum(j.elapsed_time for j in jobs if j.status == JobStatus.COMPLETED.value)
        done_msg = f"Batch complete: {completed_count} succeeded, {failed_count} failed"
        if total_elapsed:
            done_msg += f" (total {format_elapsed_time(total_elapsed)})"
        status_parts.insert(0, done_msg)
        progress_text = "Done"
        timer_active = False
    elif total_jobs == 0:
        status_parts.append("No jobs found - may still be initializing")
        timer_active = True
        progress_text = "Waiting..."
    else:
        # Still pending
        pending = total_jobs - completed_count - failed_count
        status_parts.insert(0, f"Pending: {pending} jobs")
        timer_active = True
        progress_text = "Waiting for worker..."

    status_text = " | ".join(status_parts) if status_parts else "Processing..."

    # Return: (videos, preview_list, status, progress, job_id, batch_id, timer)
    return (
        all_videos,
        [preview_path] if preview_path else [],
        status_text,
        progress_text,
        current_job_id,
        current_batch_id,
        gr.Timer(value=2.0, active=timer_active)
    )


def wan22_stop_queue_generation(current_batch_id: str):
    """Cancel all jobs in the current batch and stop processing."""
    global wan22_current_output_filename

    queue = get_queue()

    if current_batch_id:
        cancelled_jobs = queue.cancel_batch(current_batch_id)
        print(f"[Queue] Cancelled {len(cancelled_jobs)} jobs in batch {current_batch_id}")

    # Return reset state
    return (
        gr.update(),  # videos - keep existing gallery
        [],  # preview
        "Generation cancelled",  # status
        "Stopped",  # progress
        "",  # job_id
        "",  # batch_id
        gr.Timer(value=2.0, active=False)  # stop timer
    )


def wan22_stop_and_decode():
    """Create signal file to stop generation and decode current latents."""
    queue = get_queue()
    running_jobs = queue.get_running_jobs()

    if not running_jobs:
        return "No active generation to stop"

    signals_sent = 0
    for job in running_jobs:
        if job.output_filename:
            signal_file = job.output_filename + ".stop_decode"
            try:
                with open(signal_file, 'w') as f:
                    f.write('decode')
                signals_sent += 1
            except Exception as e:
                print(f"Error creating signal file: {e}")

    if signals_sent > 0:
        return f"Stop & Decode signal sent to {signals_sent} job(s)..."
    return "No active generation to stop"

# ===== MiniMax module-level helpers =====
MINIMAX_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MINIMAX_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


def minimax_align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next 17n+5 the MiniMax-H3 video VAE can encode."""
    num_frames = max(1, int(num_frames))
    while num_frames % 17 != 5:
        num_frames += 1
    return num_frames


# The encodable 17n+5 frame counts inside the released 5-15 s window (124..345 @ 24 fps).
MINIMAX_VIDEO_LENGTH_CHOICES = [("0 — derive from audio reference", 0)] + [
    (f"{n} frames ({n / 24:.2f} s)", n) for n in range(124, 346, 17)
]


def minimax_reference_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ("image" if ext in MINIMAX_IMAGE_EXTENSIONS
            else "audio" if ext in MINIMAX_AUDIO_EXTENSIONS else "video")


def minimax_reference_preview_html(files) -> str:
    """Render the accumulated reference list as a card grid (index badge, kind badge,
    inline media preview, per-card remove button wired through the hidden textbox+button)."""
    files = files or []
    if not files:
        return (
            '<div class="minimax-ref-empty">No references yet — drop images, videos or '
            'audio into the box above.</div>'
        )
    counts = {"image": 0, "video": 0, "audio": 0}
    cards = []
    for i, path in enumerate(files):
        kind = minimax_reference_kind(path)
        counts[kind] += 1
        url = "/gradio_api/file=" + url_quote(path)
        name = html.escape(os.path.basename(path))
        if kind == "image":
            media = f'<img src="{url}" loading="lazy" alt="{name}">'
        elif kind == "video":
            media = f'<video src="{url}" controls preload="metadata"></video>'
        else:
            media = f'<audio src="{url}" controls preload="metadata"></audio>'
        cards.append(
            f'<div class="minimax-ref-card">'
            f'<div class="minimax-ref-head">'
            f'<span class="minimax-ref-index">{i + 1}</span>'
            f'<span class="minimax-ref-kind minimax-ref-kind-{kind}">{kind}</span>'
            f'<button type="button" class="minimax-ref-remove" title="Remove this reference" '
            f'onclick="minimaxRemoveRef({i + 1})">&#10005;</button>'
            f'</div>'
            f'<div class="minimax-ref-media">{media}</div>'
            f'<div class="minimax-ref-name" title="{name}">{name}</div>'
            f'</div>'
        )
    summary = (
        f'<div class="minimax-ref-summary">{counts["image"]}/9 images &middot; '
        f'{counts["video"]}/3 videos &middot; {counts["audio"]}/3 audio &middot; '
        f'{len(files)}/12 total</div>'
    )
    return summary + '<div class="minimax-ref-grid">' + "".join(cards) + "</div>"


def minimax_order_references(files, order_text: str):
    """Order the multi-upload reference list by the 1-based indexes in `order_text`
    (blank = upload order). Returns the ordered path list."""
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
    order = [t for t in str(order_text or "").replace(",", " ").split() if t]
    if not order:
        return paths
    try:
        indexes = [int(t) - 1 for t in order]
    except ValueError:
        raise gr.Error(f"Reference order must be 1-based indexes like '2 1 3', got: {order_text!r}")
    if sorted(indexes) != list(range(len(paths))):
        raise gr.Error(
            f"Reference order must use each of 1..{len(paths)} exactly once, got: {order_text!r}"
        )
    return [paths[i] for i in indexes]


def minimax_submit_to_queue(
    prompt: str,
    task_override: str,
    input_image: str,
    last_image: str,
    reference_files,
    reference_order: str,
    reference_strip_audio: bool,
    aspect_ratio: str,
    width,
    height,
    video_length,
    infer_steps: int,
    flow_shift,
    audio_flow_shift,
    base_seed: int,
    batch_size: int,
    save_path: str,
    enable_preview: bool,
    preview_steps: int,
    preview_vae: str,
    # Model Paths
    ckpt_dir: str,
    dit_path: str,
    vae_path: str,
    audio_vae_path: str,
    text_encoder_path: str,
    # Advanced
    num_outputs: int,
    prompt_cache: bool,
    # Performance
    attn_mode: str,
    blocks_to_swap: int,
    classic_block_swap: bool,
    act_chunk_rows,
    fp8: bool,
    fp8_scaled: bool,
    fp8_fast: bool,
    fp8_exclude_adaln: bool,
    int8_fast: bool,
    text_encoder_gpu_layers,
    text_encoder_stream: bool,
    vae_tiling: bool,
    dit_dtype: str,
    vae_dtype: str,
    compile_enabled: bool,
    # LoRAs
    lora_folder: str,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
) -> Tuple[str, List[str]]:
    """Submit MiniMax-H3 generation job(s) to the shared queue.

    Drives minimax_engine/minimax_generate_video.py. Tasks: t2va (text), fl2va (first/last
    keyframe), ref2va (ordered image/video/audio references — separate transformer_ref
    partition). Guidance-distilled: no negative prompt, no CFG.
    """
    queue = get_queue()
    batch_count = int(batch_size)
    batch_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    job_ids = []

    os.makedirs(save_path, exist_ok=True)

    def opt_number(v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if float(v) == 0:
            return None
        return v

    flow_shift = opt_number(flow_shift)
    audio_flow_shift = opt_number(audio_flow_shift)

    references = minimax_order_references(reference_files, reference_order)

    if task_override and task_override != "auto":
        task_name = task_override
    elif references:
        task_name = "ref2va"
    elif input_image or last_image:
        task_name = "fl2va"
    else:
        task_name = "t2va"

    if task_name == "ref2va" and not references:
        raise gr.Error("ref2va needs at least one reference file")
    ref_kinds = []
    for path in references:
        ext = os.path.splitext(path)[1].lower()
        ref_kinds.append("image" if ext in MINIMAX_IMAGE_EXTENSIONS
                         else "audio" if ext in MINIMAX_AUDIO_EXTENSIONS else "video")
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        if ref_kinds.count(kind) > limit:
            raise gr.Error(f"MiniMax-H3 accepts at most {limit} {kind} references, got {ref_kinds.count(kind)}")
    if len(ref_kinds) > 12:
        raise gr.Error(f"MiniMax-H3 accepts at most 12 references, got {len(ref_kinds)}")

    # Frame count: 17n+5 at 24 fps, 5-15 s. 0/blank on ref2va = derive from the audio reference.
    video_length = opt_number(video_length)
    if video_length is not None:
        video_length = minimax_align_num_frames(video_length)
        if not 124 <= video_length <= 345:
            raise gr.Error(
                f"MiniMax-H3 generates 5-15 seconds at 24 fps: video length (snapped to 17n+5) must be "
                f"124-345 frames, got {video_length}"
            )
    elif task_name != "ref2va":
        video_length = 124

    # Explicit pixel dimensions (both set) override the auto canvas; multiples of 32.
    width = opt_number(width)
    height = opt_number(height)
    if width is not None and height is not None:
        width = max(32, (int(width) // 32) * 32)
        height = max(32, (int(height) // 32) * 32)
    else:
        width = height = None

    for i in range(batch_count):
        current_seed = base_seed
        if base_seed == -1:
            current_seed = random.randint(0, 2**31 - 1)
        elif batch_count > 1:
            current_seed = base_seed + i

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(save_path, f"minimax_{task_name}_{timestamp}_{current_seed}.mp4")
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"minimax_{run_id}"

        command = [
            sys.executable, os.path.join(ENGINE_DIR, "minimax_generate_video.py"),
            "--prompt", str(prompt),
            "--ckpt_dir", str(ckpt_dir),
            "--task", str(task_name),
            "--infer_steps", str(int(infer_steps)),
            "--num_outputs", str(int(num_outputs)),
            "--attn_mode", str(attn_mode),
            "--blocks_to_swap", str(int(blocks_to_swap)),
            "--act_chunk_rows", str(int(act_chunk_rows) if act_chunk_rows is not None else 0),
            "--dit_dtype", str(dit_dtype),
            "--vae_dtype", str(vae_dtype),
            "--save_path", str(save_path),
            "--output_type", "video",
            "--output_filename", output_filename,
            "--video_length", str(int(video_length) if video_length is not None else 0),
        ]

        if width is not None and height is not None:
            command.extend(["--video_size", str(height), str(width)])
        elif aspect_ratio and task_name != "fl2va":
            command.extend(["--aspect_ratio", str(aspect_ratio)])

        if current_seed >= 0:
            command.extend(["--seed", str(current_seed)])

        if input_image:
            command.extend(["--image_path", str(input_image)])
        if last_image:
            command.extend(["--last_image_path", str(last_image)])
        for path in references:
            command.extend(["--reference", str(path)])
        if reference_strip_audio:
            command.append("--reference_strip_audio")

        if flow_shift is not None:
            command.extend(["--flow_shift", str(flow_shift)])
        if audio_flow_shift is not None:
            command.extend(["--audio_flow_shift", str(audio_flow_shift)])

        if dit_path and str(dit_path).strip():
            command.extend(["--dit", str(dit_path).strip()])
        if vae_path and str(vae_path).strip():
            command.extend(["--vae", str(vae_path).strip()])
        if audio_vae_path and str(audio_vae_path).strip():
            command.extend(["--audio_vae", str(audio_vae_path).strip()])
        if text_encoder_path and str(text_encoder_path).strip():
            command.extend(["--text_encoder", str(text_encoder_path).strip()])

        if fp8:
            command.append("--fp8")
        if fp8_scaled:
            command.append("--fp8_scaled")
        if fp8_fast:
            command.append("--fp8_fast")
        if fp8_exclude_adaln:
            command.append("--fp8_exclude_adaln")
        if int8_fast:
            command.append("--int8_fast")
        if classic_block_swap:
            command.append("--classic_block_swap")
        if vae_tiling:
            command.append("--vae_tiling")
        if compile_enabled:
            command.append("--compile")
        te_layers = text_encoder_gpu_layers
        if te_layers is not None and str(te_layers).strip() != "":
            command.extend(["--text_encoder_gpu_layers", str(int(te_layers))])
        if text_encoder_stream:
            command.append("--text_encoder_stream")
        if prompt_cache:
            command.extend(["--prompt_cache", os.path.join(save_path, "minimax_prompt_cache.safetensors")])

        if enable_preview:
            command.extend(["--preview", str(max(1, int(preview_steps)))])
            command.extend(["--preview_suffix", unique_preview_suffix])
            if preview_vae and str(preview_vae).strip():
                command.extend(["--preview_vae", str(preview_vae).strip()])

        # LoRA handling (shared lora folder listing, same as the Cosmos tab)
        lora_weights_paths = []
        lora_multipliers_values = []
        lora_inputs = [
            (lora1_str, lora1_mult),
            (lora2_str, lora2_mult),
            (lora3_str, lora3_mult),
            (lora4_str, lora4_mult),
        ]
        if lora_folder and os.path.exists(lora_folder):
            for name, mult in lora_inputs:
                if name and name != "None":
                    path = os.path.join(lora_folder, name)
                    if os.path.exists(path):
                        lora_weights_paths.append(path)
                        lora_multipliers_values.append(str(mult))
        if lora_weights_paths:
            command.extend(["--lora_weight"] + lora_weights_paths)
            command.extend(["--lora_multiplier"] + lora_multipliers_values)

        parameters = {
            "model_type": "MiniMax-H3",
            "prompt": prompt,
            "task": task_name,
            "aspect_ratio": aspect_ratio,
            "video_length": video_length,
            "fps": 24,
            "infer_steps": infer_steps,
            "flow_shift": flow_shift,
            "audio_flow_shift": audio_flow_shift,
            "seed": current_seed,
            "num_outputs": num_outputs,
            "ckpt_dir": ckpt_dir,
            "attn_mode": attn_mode,
            "blocks_to_swap": blocks_to_swap,
            "compile": compile_enabled,
            "save_path": save_path,
        }
        if input_image:
            parameters["image_path"] = input_image
        if last_image:
            parameters["last_image_path"] = last_image
        if references:
            parameters["references"] = references

        job = queue.add_job(
            command=command,
            parameters=parameters,
            output_filename=output_filename,
            batch_id=batch_id,
            batch_index=i,
            batch_total=batch_count,
        )
        job_ids.append(job.id)
        print(f"[Queue] MiniMax job {job.id} queued (batch {batch_id}, item {i+1}/{batch_count})")

    return batch_id, job_ids


def minimax_generate_via_queue(*args):
    """Queue-based MiniMax-H3 generation; returns immediately and polls via Timer."""
    batch_id, job_ids = minimax_submit_to_queue(*args)
    first_job_id = job_ids[0] if job_ids else ""
    status_msg = f"Queued batch {batch_id} ({len(job_ids)} job(s): {', '.join(job_ids)})"
    return (
        [],
        [],
        status_msg,
        "Waiting for worker to start...",
        first_job_id,
        batch_id,
        gr.Timer(value=2.0, active=True),
    )

def start_wan22_worker():
    """Start the background worker thread for processing Wan2.2 queue jobs."""
    global wan22_worker_thread, wan22_worker_instance

    if wan22_worker_thread is not None and wan22_worker_thread.is_alive():
        print("[Worker] Wan2.2 worker already running")
        return

    queue = get_queue()
    wan22_worker_instance = Worker(queue, poll_interval=2.0, use_signals=False)
    wan22_worker_thread = threading.Thread(target=wan22_worker_instance.run, daemon=True)
    wan22_worker_thread.start()
    print("[Worker] Wan2.2 background worker started")

def set_random_seed():
    """Returns -1 to set the seed input to random."""
    return -1

def extract_video_metadata(video_path: str) -> Dict:
    """Extract metadata from video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = json.loads(result.stdout.decode('utf-8'))
        if 'format' in metadata and 'tags' in metadata['format']:
            comment = metadata['format']['tags'].get('comment', '{}')
            return json.loads(comment)
        return {}
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}

def count_prompt_tokens(prompt: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt)
    return len(tokens)


def get_lora_options(lora_folder: str = "lora") -> List[str]:
    if not os.path.exists(lora_folder):
        return ["None"]
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors') or f.endswith('.pt')]
    lora_files.sort(key=str.lower)
    return ["None"] + lora_files

# =============================================================================
# Frame Interpolation / Upscaling (GIMM-VFI / BiM-VFI / ESRGAN / SwinIR / BasicVSR++)
# Checkpoints auto-download from HuggingFace into weights/ on first use.
# =============================================================================
INTERP_WEIGHTS_REPO = "maybleMyers/interpolate"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
GIMM_VFI_DIR = os.path.join(ENGINE_DIR, "GIMM-VFI")

GIMM_MODELS = {
    "GIMM-VFI-R (RAFT)": {
        "type": "gimm",
        "checkpoint": "gimmvfi_r_arb.pt",
        "aux": ["raft-things.pth"],
    },
    "GIMM-VFI-R-P (RAFT+Perceptual)": {
        "type": "gimm",
        "checkpoint": "gimmvfi_r_arb_lpips.pt",
        "aux": ["raft-things.pth"],
    },
    "GIMM-VFI-F (FlowFormer)": {
        "type": "gimm",
        "checkpoint": "gimmvfi_f_arb.pt",
        "aux": ["flowformer_sintel.pth"],
    },
    "GIMM-VFI-F-P (FlowFormer+Perceptual)": {
        "type": "gimm",
        "checkpoint": "gimmvfi_f_arb_lpips.pt",
        "aux": ["flowformer_sintel.pth"],
    },
    "BiM-VFI (Bidirectional Motion)": {
        "type": "bim",
        "checkpoint": "bim_vfi.pth",
    },
}

# Upscaler model configurations
UPSCALER_MODELS = {
    "Real-ESRGAN x2": {
        "type": "esrgan",
        "scale": 2,
        "checkpoint": "RealESRGAN_x2plus.pth",
    },
    "Real-ESRGAN x4": {
        "type": "esrgan",
        "scale": 4,
        "checkpoint": "RealESRGAN_x4plus.pth",
    },
    "SwinIR x4": {
        "type": "swinir",
        "scale": 4,
        "checkpoint": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    },
    "BasicVSR++ x4 (Temporal)": {
        "type": "basicvsr",
        "scale": 4,
        "checkpoint": "basicvsr_plusplus_reds4.pth",
    },
}


def ensure_interp_weight(filename: str) -> str:
    """Return the path to a checkpoint in weights/, downloading it from HuggingFace on first use."""
    dest = os.path.join(WEIGHTS_DIR, filename)
    if os.path.exists(dest):
        return dest
    from huggingface_hub import hf_hub_download
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Downloading {filename} from {INTERP_WEIGHTS_REPO} to {WEIGHTS_DIR}...")
    hf_hub_download(repo_id=INTERP_WEIGHTS_REPO, filename=filename, local_dir=WEIGHTS_DIR)
    return dest


def link_gimm_aux_weight(weight_path: str):
    """GIMM-VFI resolves RAFT/FlowFormer weights via 'pretrained_ckpt/...' relative to its own
    directory, so link the downloaded file there (falling back to a copy if symlinks fail)."""
    ckpt_dir = os.path.join(GIMM_VFI_DIR, "pretrained_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    dest = os.path.join(ckpt_dir, os.path.basename(weight_path))
    if os.path.exists(dest):
        return
    try:
        os.symlink(os.path.abspath(weight_path), dest)
    except OSError:
        shutil.copy2(weight_path, dest)


def interpolate_video(
    input_video: str,
    model_variant: str,
    checkpoint_path: str,
    config_path: str,
    interp_factor: int,
    ds_scale: float,
    output_fps_override: float,
    raft_iters: int,
    pyr_level: int,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Unified dispatcher for video frame interpolation.
    Runs interpolation in a subprocess for complete VRAM cleanup.
    """
    if not input_video:
        yield None, "Error: No input video provided", 0.0
        return

    model_info = GIMM_MODELS.get(model_variant, {})
    model_type = model_info.get("type", "gimm")

    # Resolve checkpoint: user override, else auto-download into weights/ on first use
    if not checkpoint_path:
        try:
            yield None, f"Checking weights for {model_variant}...", 0.02
            checkpoint_path = ensure_interp_weight(model_info["checkpoint"])
            for aux_name in model_info.get("aux", []):
                link_gimm_aux_weight(ensure_interp_weight(aux_name))
        except Exception as e:
            yield None, f"Error downloading weights: {e}", 0.0
            return

    # Create output path
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"interpolated_{model_type}_{int(time.time())}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Build subprocess command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interp_script = os.path.join(ENGINE_DIR, "interpolate_video.py")

    command = [
        sys.executable, interp_script,
        "--input", input_video,
        "--output", output_path,
        "--model-type", model_type,
        "--variant", model_variant,
        "--factor", str(int(interp_factor)),
        "--pyr-level", str(int(pyr_level)),
        "--ds-scale", str(float(ds_scale)),
        "--output-fps", str(float(output_fps_override)),
        "--seed", str(int(seed)),
    ]

    if checkpoint_path:
        command.extend(["--checkpoint", checkpoint_path])
    if config_path:
        command.extend(["--config", config_path])

    print("\n" + "=" * 80)
    print("LAUNCHING INTERPOLATION SUBPROCESS:")
    print(" ".join(command))
    print("=" * 80 + "\n")

    yield None, "Starting interpolation subprocess...", 0.05

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        output_file = None
        last_status = "Processing..."

        while True:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    print(line)
                    if line.startswith("PROGRESS:"):
                        last_status = line[9:].strip()
                        # Try to extract percentage
                        progress = 0.1
                        if "%" in last_status:
                            try:
                                pct = int(last_status.split("(")[1].split("%")[0])
                                progress = 0.1 + (pct / 100) * 0.8
                            except:
                                pass
                        yield None, last_status, progress
                    elif line.startswith("OUTPUT:"):
                        output_file = line[7:].strip()
                    elif line.startswith("ERROR:"):
                        yield None, line[6:].strip(), 0.0
                        return

            if process.poll() is not None:
                # Read remaining output
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        print(line)
                        if line.startswith("OUTPUT:"):
                            output_file = line[7:].strip()
                        elif line.startswith("ERROR:"):
                            yield None, line[6:].strip(), 0.0
                            return
                break

        return_code = process.returncode

        if return_code == 0 and output_file and os.path.exists(output_file):
            yield output_file, f"Done! Output saved to {output_file}", 1.0
        else:
            yield None, f"Interpolation failed (exit code {return_code})", 0.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()


def upscale_video(
    input_video: str,
    model_variant: str,
    model_path_override: str,
    tile_size: int,
    half_precision: bool,
    motion_blur: bool,
    blur_strength: float,
    blur_samples: int,
    crf: int,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Unified video upscaling dispatcher.
    Launches upscale_video.py as subprocess for VRAM cleanup.
    """
    # Validate input
    if not input_video or not os.path.exists(input_video):
        yield None, "Error: No input video provided", 0.0
        return

    # Get model config
    model_config = UPSCALER_MODELS.get(model_variant)
    if not model_config:
        yield None, f"Error: Unknown model variant: {model_variant}", 0.0
        return

    model_type = model_config["type"]
    scale = model_config["scale"]

    # Determine model path: user override, else auto-download into weights/ on first use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path_override and model_path_override.strip():
        model_path = model_path_override.strip()
        if not os.path.exists(model_path):
            yield None, f"Error: Model not found at {model_path}", 0.0
            return
    else:
        try:
            yield None, f"Checking weights for {model_variant}...", 0.03
            model_path = ensure_interp_weight(model_config["checkpoint"])
        except Exception as e:
            yield None, f"Error downloading {model_variant}: {e}", 0.0
            return

    # Create output path
    output_dir = os.path.join("outputs", "upscaled")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    input_name = os.path.splitext(os.path.basename(input_video))[0]
    output_path = os.path.join(output_dir, f"{input_name}_upscaled_{scale}x_{timestamp}.mp4")

    yield None, f"Starting {model_variant} upscaling...", 0.05

    # Build command
    cmd = [
        sys.executable, os.path.join(ENGINE_DIR, "upscale_video.py"),
        "--input", input_video,
        "--output", output_path,
        "--model-type", model_type,
        "--model-path", model_path,
        "--scale", str(scale),
        "--tile-size", str(tile_size),
        "--crf", str(crf),
        "--seed", str(seed),
    ]

    if half_precision:
        cmd.append("--half")

    if motion_blur:
        cmd.extend([
            "--motion-blur",
            "--blur-strength", str(blur_strength),
            "--blur-samples", str(blur_samples),
        ])

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_file = None
        last_status = "Processing..."

        while True:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    print(line)
                    if line.startswith("PROGRESS:"):
                        last_status = line[9:].strip()
                        # Try to extract percentage
                        progress = 0.1
                        if "%" in last_status:
                            try:
                                pct = int(last_status.split("(")[1].split("%")[0])
                                progress = 0.1 + (pct / 100) * 0.8
                            except:
                                pass
                        yield None, last_status, progress
                    elif line.startswith("OUTPUT:"):
                        output_file = line[7:].strip()
                    elif line.startswith("ERROR:"):
                        yield None, line[6:].strip(), 0.0
                        return

            if process.poll() is not None:
                # Read remaining output
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        print(line)
                        if line.startswith("OUTPUT:"):
                            output_file = line[7:].strip()
                        elif line.startswith("ERROR:"):
                            yield None, line[6:].strip(), 0.0
                            return
                break

        return_code = process.returncode

        if return_code == 0 and output_file and os.path.exists(output_file):
            yield output_file, f"Done! Output saved to {output_file}", 1.0
        else:
            yield None, f"Upscaling failed (exit code {return_code})", 0.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()


# UI setup
with gr.Blocks(
    theme=themes.Default(
        primary_hue=colors.Color(
            name="custom",
            c50="#E6F0FF",
            c100="#CCE0FF",
            c200="#99C1FF",
            c300="#66A3FF",
            c400="#3384FF",
            c500="#0060df",  # This is your main color
            c600="#0052C2",
            c700="#003D91",
            c800="#002961",
            c900="#001430",
            c950="#000A18"
        )
    ),
    css="""
    .gallery-item:first-child { border: 2px solid #4CAF50 !important; }
    .gallery-item:first-child:hover { border-color: #45a049 !important; }
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    .refresh-btn {
        max-width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .light-blue-btn {
        background: linear-gradient(to bottom right, #AEC6CF, #9AB8C4) !important; /* Light blue gradient */
        color: #333 !important; /* Darker text for readability */
        border: 1px solid #9AB8C4 !important; /* Subtle border */
    }
    .light-blue-btn:hover {
        background: linear-gradient(to bottom right, #9AB8C4, #8AA9B5) !important; /* Slightly darker on hover */
        border-color: #8AA9B5 !important;
    }
    .minimax-hidden { display: none !important; }
    .minimax-ref-summary {
        font-size: 0.85em;
        opacity: 0.8;
        margin: 4px 0 6px 2px;
    }
    .minimax-ref-empty {
        font-size: 0.85em;
        opacity: 0.6;
        margin: 4px 0 6px 2px;
    }
    .minimax-ref-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 10px;
    }
    .minimax-ref-card {
        border: 1px solid var(--border-color-primary, #444);
        border-radius: 8px;
        padding: 6px;
        background: var(--background-fill-secondary, rgba(128,128,128,0.05));
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
    }
    .minimax-ref-head {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .minimax-ref-index {
        font-weight: bold;
        background: #0060df;
        color: white;
        border-radius: 4px;
        padding: 0 6px;
        font-size: 0.85em;
    }
    .minimax-ref-kind {
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-radius: 4px;
        padding: 0 5px;
        color: white;
    }
    .minimax-ref-kind-image { background: #27ae60; }
    .minimax-ref-kind-video { background: #8e44ad; }
    .minimax-ref-kind-audio { background: #e67e22; }
    .minimax-ref-remove {
        margin-left: auto;
        border: none;
        background: transparent;
        color: var(--body-text-color, inherit);
        opacity: 0.6;
        cursor: pointer;
        font-size: 0.95em;
        line-height: 1;
        padding: 2px 4px;
    }
    .minimax-ref-remove:hover { opacity: 1; color: #e74c3c; }
    .minimax-ref-media img, .minimax-ref-media video {
        width: 100%;
        max-height: 140px;
        object-fit: contain;
        border-radius: 4px;
        display: block;
    }
    .minimax-ref-media audio { width: 100%; }
    .minimax-ref-name {
        font-size: 0.75em;
        opacity: 0.75;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    """,

) as demo:
    params_state = gr.State()

    demo.load(None, None, None, js=r"""
        () => {
            document.title = 'H1111';

            // A file dropped outside an upload zone would otherwise navigate the
            // browser to the file, wiping out the page. Swallow stray drops at the
            // window level; Gradio's own dropzones handle their drops first.
            window.addEventListener('dragover', (e) => { e.preventDefault(); }, false);
            window.addEventListener('drop', (e) => { e.preventDefault(); }, false);

            // Per-card remove buttons in the MiniMax reference preview: write the
            // 1-based index into the hidden textbox, then click the hidden button.
            window.minimaxRemoveRef = (idx) => {
                const box = document.querySelector('#minimax_ref_remove_idx textarea, #minimax_ref_remove_idx input');
                if (!box) return;
                box.value = String(idx);
                box.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => {
                    const btn = document.querySelector('#minimax_ref_remove_btn button, button#minimax_ref_remove_btn, #minimax_ref_remove_btn');
                    if (btn) btn.click();
                }, 60);
            };

            let lastProgressText = '';

            function updateTitle(text) {
                if (text && text.trim() && text !== lastProgressText) {
                    lastProgressText = text;
                    // Match formats like:
                    // "Generating: 95% (38/40 steps) - ETA: 03:01" (queue format)
                    // "45%|████     | 45/100 [01:23<01:45" (raw TQDM format)

                    // Try queue format first: "XX% ... ETA: HH:MM:SS"
                    let match = text.match(/(\d+)%.*?ETA:\s*([\d:]+)/);
                    if (match) {
                        document.title = `[${match[1]}% ETA: ${match[2]}] - H1111`;
                        return;
                    }

                    // Try raw TQDM format: "XX%|...[...<HH:MM:SS"
                    match = text.match(/(\d+)%\|.*\[.*<([\d:?]+)/);
                    if (match) {
                        document.title = `[${match[1]}% ETA: ${match[2]}] - H1111`;
                        return;
                    }
                }
                // Reset title if no progress info found and we had progress before
                if (!text || !text.trim()) {
                    if (document.title !== 'H1111') {
                        document.title = 'H1111';
                    }
                }
            }

            // Poll all progress textareas every 500ms for value changes
            setInterval(() => {
                // Select progress elements from all tabs
                const selectors = [
                    'textarea.scroll-hide',  // Direct generation tabs
                    '#wan22_progress_text input',
                    '#wan22_progress_text textarea',
                    '#svi_progress_text input',
                    '#svi_progress_text textarea',
                    '[id$="_progress_text"] input',  // Any element ending with _progress_text
                    '[id$="_progress_text"] textarea'
                ];
                const progressElements = document.querySelectorAll(selectors.join(', '));
                progressElements.forEach(element => {
                    if (element && element.value) {
                        updateTitle(element.value);
                    }
                });
            }, 500);
        }
        """)

    with gr.Tabs() as tabs:
        with gr.Tab(id=21, label="MiniMax") as minimax_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    minimax_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt",
                        value="A red fox trotting through a snowy pine forest, snow crunching underfoot.",
                        lines=5,
                    )
                with gr.Column(scale=1):
                    minimax_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    minimax_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    minimax_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    minimax_progress_text = gr.Textbox(label="Progress", interactive=False, value="",
                                                       elem_id="minimax_progress_text")

            with gr.Row():
                minimax_generate_btn = gr.Button("Generate", elem_classes="green-btn")
                minimax_stop_btn = gr.Button("Stop Generation", variant="stop")
                minimax_stop_decode_btn = gr.Button("Stop & Decode", variant="secondary")

            # Queue system state components
            minimax_job_id_state = gr.State(value="")
            minimax_batch_id_state = gr.State(value="")
            minimax_poll_timer = gr.Timer(value=2.0, active=False)

            with gr.Row():
                with gr.Column():
                    minimax_task_override = gr.Dropdown(
                        label="Task Override",
                        choices=["auto", "t2va", "fl2va", "ref2va"],
                        value="auto",
                        info="auto: any reference → ref2va, any keyframe → fl2va, else t2va",
                    )
                    gr.Markdown("### Keyframes (fl2va)")
                    with gr.Row():
                        minimax_input_image = gr.Image(
                            label="First Frame (stretched onto the canvas; sets the aspect ratio)",
                            type="filepath",
                        )
                        minimax_last_image = gr.Image(
                            label="Last Frame (cover-cropped; can be used on its own)",
                            type="filepath",
                        )

                    with gr.Accordion("References (ref2va)", open=False):
                        gr.Markdown(
                            "Up to **9 images / 3 videos / 3 audio clips**, 12 total. New drops are "
                            "**added** to the set. The order is semantic (it labels the references and "
                            "advances the shared rotary clock) — reorder with 1-based indexes below. "
                            "A video reference conditions on its soundtrack too unless stripped. Leave "
                            "Video Length at 0 to derive the duration from a single audio-bearing "
                            "reference."
                        )
                        minimax_reference_state = gr.State(value=[])
                        minimax_reference_files = gr.File(
                            label="Drop images / videos / audio here (or click to browse)",
                            file_count="multiple",
                            type="filepath",
                            height=110,
                            elem_id="minimax_ref_dropzone",
                        )
                        minimax_reference_preview = gr.HTML(value=minimax_reference_preview_html([]))
                        with gr.Row():
                            minimax_reference_order = gr.Textbox(
                                label="Reference order (1-based indexes, blank = upload order)", value=""
                            )
                            minimax_reference_strip_audio = gr.Checkbox(
                                label="Strip soundtrack from video references", value=False
                            )
                            minimax_reference_clear_btn = gr.Button("Clear all references", size="sm")
                        # Hidden plumbing for the per-card ✕ buttons in the HTML preview
                        # (kept visible=True so they exist in the DOM; hidden via CSS).
                        minimax_reference_remove_idx = gr.Textbox(
                            value="", elem_id="minimax_ref_remove_idx",
                            elem_classes=["minimax-hidden"],
                        )
                        minimax_reference_remove_btn = gr.Button(
                            "remove", elem_id="minimax_ref_remove_btn",
                            elem_classes=["minimax-hidden"],
                        )

                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        minimax_aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio (auto canvas: 768px short edge, ×32; ignored for fl2va)",
                            choices=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2:1", "1:2"],
                            value="16:9",
                        )
                    minimax_original_dims = gr.Textbox(visible=False, value="")
                    with gr.Row():
                        minimax_width = gr.Number(label="Width (blank = auto; ×32)", value=None, step=32)
                        minimax_calc_height_btn = gr.Button("→")
                        minimax_calc_width_btn = gr.Button("←")
                        minimax_height = gr.Number(label="Height (blank = auto; ×32)", value=None, step=32)
                    minimax_video_length = gr.Dropdown(
                        label="Video Length (frames @ 24 fps; the VAE encodes 17n+5 frames, 5–15 s)",
                        choices=MINIMAX_VIDEO_LENGTH_CHOICES,
                        value=124,
                    )
                    minimax_infer_steps = gr.Slider(
                        minimum=2, maximum=100, step=1, label="Sampling Steps (model evals = steps − 1)",
                        value=50,
                    )
                    with gr.Row():
                        minimax_flow_shift = gr.Number(label="Flow Shift (blank = checkpoint 12.0)", value=None)
                        minimax_audio_flow_shift = gr.Number(label="Audio Flow Shift (blank = checkpoint 3.0)",
                                                             value=None)
                    with gr.Row():
                        minimax_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        minimax_random_seed_btn = gr.Button("🎲")

                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            minimax_num_outputs = gr.Number(label="Num Outputs (per job)", value=1, minimum=1, step=1)
                            minimax_prompt_cache = gr.Checkbox(
                                label="Prompt Cache", value=False,
                                info="reuse cached prompt embeddings when inputs match — skips the ~30B "
                                     "conditioner load on repeat runs",
                            )

                with gr.Column():
                    minimax_output = gr.Gallery(
                        label="Generated Output (Click to select)",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_minimax", allow_preview=True, preview=True,
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        minimax_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        minimax_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                          label="Preview Every N Steps")
                        minimax_preview_vae = gr.Textbox(
                            label="Preview TAE Checkpoint (optional)", value="",
                            info="blank = fast latent2rgb preview; path to taeh3.pth (madebyollin/taehv) "
                                 "for full-resolution TAE previews",
                        )
                        minimax_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True,
                            elem_id="minimax_preview_gallery",
                        )
                    with gr.Accordion("LoRA", open=False):
                        with gr.Row():
                            minimax_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            minimax_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        minimax_lora_weights = []
                        minimax_lora_multipliers = []
                        for i in range(4):
                            with gr.Row():
                                minimax_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2,
                                ))
                                minimax_lora_multipliers.append(gr.Number(
                                    label="Multiplier", value=1.0, scale=1, interactive=True,
                                ))

            with gr.Accordion("Model Paths", open=True):
                minimax_ckpt_dir = gr.Textbox(
                    label="Checkpoint Dir",
                    value="MiniMax-H3",
                    info="path to the cloned MiniMaxAI/MiniMax-H3 HF snapshot dir",
                )
                with gr.Row():
                    minimax_dit_path = gr.Textbox(label="DiT Override (blank = per-task transformer[_ref])", value="",
                                                  info="dir, merged file, or an int8 convrot export (auto-detected)")
                    minimax_vae_path = gr.Textbox(label="VAE Override (blank = ckpt_dir vae)", value="")
                    minimax_audio_vae_path = gr.Textbox(label="Audio VAE Override (blank = ckpt_dir audio_vae)",
                                                        value="")
                minimax_text_encoder_path = gr.Textbox(
                    label="Text Encoder Override (blank = ckpt_dir text_encoder)", value="",
                    info="single-file override, e.g. an int8 convrot export; the 'ultra_p' file also carries "
                         "the vision tower",
                )

            with gr.Accordion("Performance", open=True):
                with gr.Row():
                    minimax_attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers"],
                        value="sdpa",
                    )
                    minimax_blocks_to_swap = gr.Slider(
                        minimum=0, maximum=49, step=1,
                        label="Block Swap to Save VRAM (50 transformer blocks — max 49)", value=25,
                    )
                with gr.Row():
                    minimax_classic_block_swap = gr.Checkbox(
                        label="Classic Block Swap", value=False,
                        info="legacy rolling swap instead of pinned sub-block weight streaming "
                             "(streaming pins ~1.2 GB/block bf16 or ~0.6 GB/block fp8 of host RAM)",
                    )
                    minimax_act_chunk_rows = gr.Number(
                        label="Activation Chunk Rows (0 = off)", value=32768, step=1, minimum=0,
                        info="process row-wise ops (AdaLN, rotary, FF, output heads) in slices of this many "
                             "rows to bound activation peaks on long/large runs",
                    )
                with gr.Row():
                    minimax_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    minimax_fp8_scaled = gr.Checkbox(
                        label="Use Scaled FP8 (DiT)", value=False,
                        info="off = full-quality bf16 (61.7 GB, needs block swap on 48 GB cards); "
                             "on = ~31 GB resident, lossy runtime quantization",
                    )
                    minimax_fp8_fast = gr.Checkbox(label="FP8 Fast", value=False, info="scaled_mm fp8 matmul")
                    minimax_fp8_exclude_adaln = gr.Checkbox(
                        label="FP8: exclude AdaLN", value=False,
                        info="keep the AdaLN projections in bf16 (+~13 GB, higher fidelity)",
                    )
                    minimax_int8_fast = gr.Checkbox(
                        label="INT8 Fast", value=False,
                        info="int8 convrot checkpoints only: torch._int_mm with dynamic activation "
                             "quantization instead of dequantize-per-forward",
                    )
                    minimax_vae_tiling = gr.Checkbox(label="VAE Tiling", value=True,
                                                     info="the release ships with spatial tiling on")
                with gr.Row():
                    minimax_text_encoder_gpu_layers = gr.Number(
                        label="Text Encoder GPU Layers (-1 = all, 0 = CPU/streamed)", value=0, step=1,
                        info="the Qwen3-VL conditioner is ~30B; it runs once per job",
                    )
                    minimax_text_encoder_stream = gr.Checkbox(
                        label="Stream Text Encoder Layers", value=True,
                        info="move CPU-resident conditioner layers through the GPU one at a time",
                    )
                with gr.Row():
                    minimax_dit_dtype = gr.Dropdown(label="DiT Dtype", choices=["bfloat16", "float16"],
                                                    value="bfloat16")
                    minimax_vae_dtype = gr.Dropdown(
                        label="VAE Dtype", choices=["float32"], value="float32",
                        info="decode runs fp16-autocast over fp32 weights (checkpoint contract)",
                    )
                    minimax_compile = gr.Checkbox(
                        label="Enable torch.compile",
                        value=False,
                        info="Function-level JIT compile. Compatible with all dtypes and block swap. First run slower."
                    )
                minimax_save_path = gr.Textbox(label="Save Path", value="outputs")
                with gr.Row():
                    minimax_save_defaults_btn = gr.Button("Save Defaults")
                    minimax_load_defaults_btn = gr.Button("Load Defaults")
                    minimax_defaults_status = gr.Textbox(label="Defaults Status", interactive=False, visible=False)

        with gr.Tab("Video Info") as video_info_tab:
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                send_to_minimax_btn = gr.Button("Send to MiniMax", variant="primary")

            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Frame Interpolation") as frame_interp_tab:
            gr.Markdown("### Increase Video FPS using GIMM-VFI\nState-of-the-art frame interpolation for smooth slow motion and higher frame rates.")

            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    interp_input_video = gr.Video(label="Input Video", sources=["upload"])

                    # Model Settings
                    with gr.Accordion("Model Settings", open=True):
                        interp_model_variant = gr.Dropdown(
                            label="Model Variant",
                            choices=list(GIMM_MODELS.keys()),
                            value="GIMM-VFI-R-P (RAFT+Perceptual)",
                            info="R=RAFT (faster), F=FlowFormer (better quality), P=Perceptual loss (recommended)"
                        )
                        interp_checkpoint_path = gr.Textbox(
                            label="Checkpoint Path (optional)",
                            placeholder="Leave empty to use default for selected variant",
                            info="Override the default checkpoint path"
                        )
                        interp_config_path = gr.Textbox(
                            label="Config Path (optional)",
                            placeholder="Leave empty to use default for selected variant",
                            info="Override the default config path"
                        )

                    # Interpolation Settings
                    with gr.Accordion("Interpolation Settings", open=True):
                        interp_factor = gr.Slider(
                            label="Interpolation Factor",
                            minimum=2,
                            maximum=16,
                            value=2,
                            step=1,
                            info="2=2x FPS (1 new frame), 4=4x FPS (3 new frames), 8=8x FPS (7 new frames)"
                        )
                        interp_ds_scale = gr.Slider(
                            label="DS Scale (for high-res)",
                            minimum=0.25,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            info="Downscale factor: 1.0=SD/HD, 0.5=2K (~8GB VRAM), 0.25=4K (~11GB VRAM)"
                        )
                        interp_output_fps = gr.Number(
                            label="Output FPS Override",
                            value=0,
                            minimum=0,
                            info="0 = auto (input FPS × factor). Set manually for custom output FPS."
                        )

                    # Advanced Settings
                    with gr.Accordion("Advanced", open=False):
                        interp_raft_iters = gr.Slider(
                            label="RAFT Iterations (GIMM-VFI only)",
                            minimum=12,
                            maximum=32,
                            value=20,
                            step=1,
                            info="More iterations = better quality, slower (GIMM-VFI only)"
                        )
                        interp_pyr_level = gr.Slider(
                            label="Pyramid Level (BiM-VFI only)",
                            minimum=0,
                            maximum=8,
                            value=0,
                            step=1,
                            info="0=auto (based on resolution), 5=<1080p, 6=1080p, 7=4K+"
                        )
                        interp_seed = gr.Number(
                            label="Seed",
                            value=0,
                            info="Random seed for reproducibility"
                        )

                    # Upscaling Settings
                    with gr.Accordion("Upscaling", open=False):
                        upscale_enable = gr.Checkbox(
                            label="Enable Upscaling",
                            value=False,
                            info="Apply spatial upscaling (standalone or after interpolation)"
                        )
                        upscale_model = gr.Dropdown(
                            label="Upscaler Model",
                            choices=list(UPSCALER_MODELS.keys()),
                            value="Real-ESRGAN x2",
                            info="ESRGAN/SwinIR: frame-by-frame, BasicVSR++: temporal-aware"
                        )
                        upscale_tile_size = gr.Slider(
                            label="Tile Size",
                            minimum=0,
                            maximum=1024,
                            value=512,
                            step=64,
                            info="0=no tiling (more VRAM), 512=balanced, lower=less VRAM"
                        )
                        upscale_half = gr.Checkbox(
                            label="Half Precision (FP16)",
                            value=True,
                            info="Faster, less VRAM, slight quality loss"
                        )
                        upscale_model_path = gr.Textbox(
                            label="Custom Model Path (optional)",
                            placeholder="Leave empty for default model",
                            info="Override the default checkpoint path"
                        )
                        upscale_crf = gr.Slider(
                            label="Output CRF",
                            minimum=10,
                            maximum=30,
                            value=18,
                            step=1,
                            info="Video quality: lower=better quality, larger file (18=good default)"
                        )

                    # Motion Blur Settings (for masking deformation artifacts)
                    with gr.Accordion("Motion Blur (Artifact Masking)", open=False):
                        motion_blur_enable = gr.Checkbox(
                            label="Enable Motion Blur",
                            value=False,
                            info="Add blur along motion vectors to mask deformation artifacts"
                        )
                        motion_blur_strength = gr.Slider(
                            label="Blur Strength",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            info="Higher = more blur along motion direction"
                        )
                        motion_blur_samples = gr.Slider(
                            label="Blur Samples",
                            minimum=3,
                            maximum=15,
                            value=7,
                            step=2,
                            info="More samples = smoother blur (use odd numbers)"
                        )

                    # Action Buttons
                    with gr.Row():
                        interp_generate_btn = gr.Button("🎬 Interpolate", variant="primary", elem_classes="green-btn")
                        upscale_btn = gr.Button("🔍 Upscale", variant="secondary")
                    with gr.Row():
                        interp_save_defaults_btn = gr.Button("💾 Save Defaults")
                        interp_load_defaults_btn = gr.Button("📂 Load Defaults")

                # Output Column
                with gr.Column(scale=1):
                    interp_output_video = gr.Video(label="Interpolated Video")
                    interp_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    interp_progress = gr.Slider(
                        label="Progress",
                        minimum=0,
                        maximum=1,
                        value=0,
                        interactive=False,
                        visible=True
                    )

                    gr.Markdown("""
                    **Notes:**
                    - Checkpoints download automatically from [maybleMyers/interpolate](https://huggingface.co/maybleMyers/interpolate) into `weights/` on first use
                    - **GIMM-VFI**: R=RAFT (faster), F=FlowFormer (better quality), P=Perceptual loss
                    - **BiM-VFI**: Bidirectional motion field interpolation
                    - For 2K/4K video with GIMM-VFI, reduce DS Scale to fit in VRAM
                    - BiM-VFI auto-detects pyramid level based on resolution (or set manually)

                    **Upscaling:**
                    - **Real-ESRGAN / SwinIR**: frame-by-frame, **BasicVSR++**: temporal-aware
                    - Motion blur uses RAFT flow to mask deformation artifacts
                    """)

            # Frame Interpolation event handlers
            interp_generate_btn.click(
                fn=interpolate_video,
                inputs=[
                    interp_input_video,
                    interp_model_variant,
                    interp_checkpoint_path,
                    interp_config_path,
                    interp_factor,
                    interp_ds_scale,
                    interp_output_fps,
                    interp_raft_iters,
                    interp_pyr_level,
                    interp_seed,
                ],
                outputs=[interp_output_video, interp_status, interp_progress]
            )

            # Upscaling event handler
            upscale_btn.click(
                fn=upscale_video,
                inputs=[
                    interp_input_video,  # Use same input video
                    upscale_model,
                    upscale_model_path,
                    upscale_tile_size,
                    upscale_half,
                    motion_blur_enable,
                    motion_blur_strength,
                    motion_blur_samples,
                    upscale_crf,
                    interp_seed,  # Reuse same seed
                ],
                outputs=[interp_output_video, interp_status, interp_progress]
            )

            # Save/Load defaults
            interp_ui_default_components = [
                interp_model_variant, interp_checkpoint_path, interp_config_path,
                interp_factor, interp_ds_scale, interp_output_fps,
                interp_raft_iters, interp_pyr_level, interp_seed,
                upscale_enable, upscale_model, upscale_tile_size, upscale_half,
                upscale_model_path, upscale_crf,
                motion_blur_enable, motion_blur_strength, motion_blur_samples,
            ]
            interp_ui_default_keys = [
                "interp_model_variant", "interp_checkpoint_path", "interp_config_path",
                "interp_factor", "interp_ds_scale", "interp_output_fps",
                "interp_raft_iters", "interp_pyr_level", "interp_seed",
                "upscale_enable", "upscale_model", "upscale_tile_size", "upscale_half",
                "upscale_model_path", "upscale_crf",
                "motion_blur_enable", "motion_blur_strength", "motion_blur_samples",
            ]

            def save_interp_defaults(*values):
                os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
                settings_to_save = dict(zip(interp_ui_default_keys, values))
                try:
                    with open(INTERP_DEFAULTS_FILE, 'w') as f:
                        json.dump(settings_to_save, f, indent=2)
                    return "Interpolation defaults saved successfully."
                except Exception as e:
                    return f"Error saving interpolation defaults: {e}"

            def load_interp_defaults(request: gr.Request = None):
                if not os.path.exists(INTERP_DEFAULTS_FILE):
                    if request:  # Button click with no saved file
                        return [gr.update()] * len(interp_ui_default_keys) + ["No defaults file found."]
                    return [gr.update()] * len(interp_ui_default_keys) + [""]
                try:
                    with open(INTERP_DEFAULTS_FILE, 'r') as f:
                        loaded_settings = json.load(f)
                except Exception as e:
                    return [gr.update()] * len(interp_ui_default_keys) + [f"Error loading defaults: {e}"]
                updates = []
                for i, key in enumerate(interp_ui_default_keys):
                    component = interp_ui_default_components[i]
                    default_value = getattr(component, 'value', None)
                    updates.append(gr.update(value=loaded_settings.get(key, default_value)))
                return updates + ["Interpolation defaults loaded successfully."]

            interp_save_defaults_btn.click(
                fn=save_interp_defaults,
                inputs=interp_ui_default_components,
                outputs=[interp_status]
            )
            interp_load_defaults_btn.click(
                fn=load_interp_defaults,
                inputs=None,
                outputs=interp_ui_default_components + [interp_status]
            )

            def initial_load_interp_defaults():
                return load_interp_defaults(None)[:-1]
            demo.load(
                fn=initial_load_interp_defaults,
                inputs=None,
                outputs=interp_ui_default_components
            )

    # ===== MiniMax Event Handlers =====
    minimax_generate_btn.click(
        fn=minimax_generate_via_queue,
        inputs=[
            minimax_prompt,
            minimax_task_override,
            minimax_input_image,
            minimax_last_image,
            minimax_reference_state,
            minimax_reference_order,
            minimax_reference_strip_audio,
            minimax_aspect_ratio,
            minimax_width,
            minimax_height,
            minimax_video_length,
            minimax_infer_steps,
            minimax_flow_shift,
            minimax_audio_flow_shift,
            minimax_seed,
            minimax_batch_size,
            minimax_save_path,
            minimax_enable_preview,
            minimax_preview_steps,
            minimax_preview_vae,
            # Model Paths
            minimax_ckpt_dir,
            minimax_dit_path,
            minimax_vae_path,
            minimax_audio_vae_path,
            minimax_text_encoder_path,
            # Advanced
            minimax_num_outputs,
            minimax_prompt_cache,
            # Performance
            minimax_attn_mode,
            minimax_blocks_to_swap,
            minimax_classic_block_swap,
            minimax_act_chunk_rows,
            minimax_fp8,
            minimax_fp8_scaled,
            minimax_fp8_fast,
            minimax_fp8_exclude_adaln,
            minimax_int8_fast,
            minimax_text_encoder_gpu_layers,
            minimax_text_encoder_stream,
            minimax_vae_tiling,
            minimax_dit_dtype,
            minimax_vae_dtype,
            minimax_compile,
            # LoRAs
            minimax_lora_folder,
            *minimax_lora_weights,
            *minimax_lora_multipliers,
        ],
        outputs=[minimax_output, minimax_preview_output, minimax_batch_progress, minimax_progress_text,
                 minimax_job_id_state, minimax_batch_id_state, minimax_poll_timer],
        queue=True
    )

    minimax_poll_timer.tick(
        fn=wan22_poll_active_job,
        inputs=[minimax_job_id_state, minimax_batch_id_state],
        outputs=[minimax_output, minimax_preview_output, minimax_batch_progress, minimax_progress_text,
                 minimax_job_id_state, minimax_batch_id_state, minimax_poll_timer]
    )

    minimax_stop_btn.click(
        fn=wan22_stop_queue_generation,
        inputs=[minimax_batch_id_state],
        outputs=[minimax_output, minimax_preview_output, minimax_batch_progress, minimax_progress_text,
                 minimax_job_id_state, minimax_batch_id_state, minimax_poll_timer],
        queue=False
    )

    minimax_stop_decode_btn.click(
        fn=wan22_stop_and_decode,
        outputs=[minimax_batch_progress],
        queue=False
    )

    minimax_random_seed_btn.click(fn=set_random_seed, inputs=None, outputs=[minimax_seed])

    minimax_prompt.change(fn=count_prompt_tokens, inputs=minimax_prompt, outputs=minimax_token_counter)

    # Keyframe upload snaps the canvas to the image (first frame wins — it anchors the
    # geometry; the last frame is cover-cropped). Cleared images blank the fields back to auto.
    def update_minimax_dimensions(input_image, last_image):
        image = input_image or last_image
        if image is None:
            return "", gr.update(value=None), gr.update(value=None)
        img = Image.open(image)
        w, h = img.size
        w = max(32, (w // 32) * 32)
        h = max(32, (h // 32) * 32)
        return f"{w}x{h}", w, h

    minimax_input_image.change(
        fn=update_minimax_dimensions,
        inputs=[minimax_input_image, minimax_last_image],
        outputs=[minimax_original_dims, minimax_width, minimax_height],
    )
    minimax_last_image.change(
        fn=update_minimax_dimensions,
        inputs=[minimax_input_image, minimax_last_image],
        outputs=[minimax_original_dims, minimax_width, minimax_height],
    )

    def minimax_calc_width(height, original_dims):
        if not original_dims or not height:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split("x"))
        return gr.update(value=max(32, math.floor(height * orig_w / orig_h / 32) * 32))

    def minimax_calc_height(width, original_dims):
        if not original_dims or not width:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split("x"))
        return gr.update(value=max(32, math.floor(width * orig_h / orig_w / 32) * 32))

    minimax_calc_width_btn.click(
        fn=minimax_calc_width,
        inputs=[minimax_height, minimax_original_dims],
        outputs=[minimax_width],
    )
    minimax_calc_height_btn.click(
        fn=minimax_calc_height,
        inputs=[minimax_width, minimax_original_dims],
        outputs=[minimax_height],
    )

    minimax_lora_refresh_outputs_list = []
    for i in range(len(minimax_lora_weights)):
        minimax_lora_refresh_outputs_list.extend([minimax_lora_weights[i], minimax_lora_multipliers[i]])

    def refresh_minimax_loras(folder: str) -> List[gr.update]:
        choices = get_lora_options(folder)
        updates = []
        for _ in range(4):
            updates.extend([gr.update(choices=choices, value="None"), gr.update(value=1.0)])
        return updates

    minimax_lora_refresh_btn.click(
        fn=refresh_minimax_loras,
        inputs=[minimax_lora_folder],
        outputs=minimax_lora_refresh_outputs_list
    )

    def minimax_default_reference_order(files):
        count = len(files or [])
        return " ".join(str(i + 1) for i in range(count))

    def minimax_add_references(state_files, uploaded):
        """Append newly dropped files to the accumulated set, then clear the dropzone
        so it stays an always-available drop target."""
        files = list(state_files or [])
        for f in (uploaded or []):
            path = f if isinstance(f, str) else getattr(f, "name", str(f))
            if path and path not in files:
                files.append(path)
        return (files, None, minimax_default_reference_order(files),
                minimax_reference_preview_html(files))

    minimax_reference_files.upload(
        fn=minimax_add_references,
        inputs=[minimax_reference_state, minimax_reference_files],
        outputs=[minimax_reference_state, minimax_reference_files,
                 minimax_reference_order, minimax_reference_preview],
    )

    def minimax_remove_reference(state_files, idx_text):
        files = list(state_files or [])
        try:
            idx = int(str(idx_text).strip())
        except ValueError:
            return files, minimax_default_reference_order(files), minimax_reference_preview_html(files)
        if 1 <= idx <= len(files):
            files.pop(idx - 1)
        return (files, minimax_default_reference_order(files),
                minimax_reference_preview_html(files))

    minimax_reference_remove_btn.click(
        fn=minimax_remove_reference,
        inputs=[minimax_reference_state, minimax_reference_remove_idx],
        outputs=[minimax_reference_state, minimax_reference_order, minimax_reference_preview],
    )

    def minimax_clear_references():
        return [], None, "", minimax_reference_preview_html([])

    minimax_reference_clear_btn.click(
        fn=minimax_clear_references,
        inputs=None,
        outputs=[minimax_reference_state, minimax_reference_files,
                 minimax_reference_order, minimax_reference_preview],
    )

    minimax_ui_default_components_ORDERED_LIST = [
        minimax_ckpt_dir,
        minimax_dit_path,
        minimax_vae_path,
        minimax_audio_vae_path,
        minimax_text_encoder_path,
        minimax_attn_mode,
        minimax_blocks_to_swap,
        minimax_classic_block_swap,
        minimax_act_chunk_rows,
        minimax_fp8,
        minimax_fp8_scaled,
        minimax_fp8_fast,
        minimax_fp8_exclude_adaln,
        minimax_int8_fast,
        minimax_text_encoder_gpu_layers,
        minimax_text_encoder_stream,
        minimax_vae_tiling,
        minimax_dit_dtype,
        minimax_vae_dtype,
        minimax_compile,
        minimax_save_path,
        minimax_lora_folder,
        minimax_aspect_ratio,
        minimax_video_length,
        minimax_infer_steps,
        minimax_flow_shift,
        minimax_audio_flow_shift,
        minimax_task_override,
        minimax_num_outputs,
        minimax_prompt_cache,
        minimax_enable_preview,
        minimax_preview_steps,
        minimax_preview_vae,
    ] + minimax_lora_weights + minimax_lora_multipliers

    minimax_ui_default_keys = [
        "minimax_ckpt_dir",
        "minimax_dit_path",
        "minimax_vae_path",
        "minimax_audio_vae_path",
        "minimax_text_encoder_path",
        "minimax_attn_mode",
        "minimax_blocks_to_swap",
        "minimax_classic_block_swap",
        "minimax_act_chunk_rows",
        "minimax_fp8",
        "minimax_fp8_scaled",
        "minimax_fp8_fast",
        "minimax_fp8_exclude_adaln",
        "minimax_int8_fast",
        "minimax_text_encoder_gpu_layers",
        "minimax_text_encoder_stream",
        "minimax_vae_tiling",
        "minimax_dit_dtype",
        "minimax_vae_dtype",
        "minimax_compile",
        "minimax_save_path",
        "minimax_lora_folder",
        "minimax_aspect_ratio",
        "minimax_video_length",
        "minimax_infer_steps",
        "minimax_flow_shift",
        "minimax_audio_flow_shift",
        "minimax_task_override",
        "minimax_num_outputs",
        "minimax_prompt_cache",
        "minimax_enable_preview",
        "minimax_preview_steps",
        "minimax_preview_vae",
    ] + [f"minimax_lora_weight_{i+1}" for i in range(4)] + \
        [f"minimax_lora_multiplier_{i+1}" for i in range(4)]

    def save_minimax_defaults(*values):
        os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
        settings_to_save = {}
        for i, key in enumerate(minimax_ui_default_keys):
            settings_to_save[key] = values[i]
        try:
            with open(MINIMAX_DEFAULTS_FILE, 'w') as f:
                json.dump(settings_to_save, f, indent=2)
            return "MiniMax defaults saved successfully."
        except Exception as e:
            return f"Error saving MiniMax defaults: {e}"

    def load_minimax_defaults(request: gr.Request):
        if not os.path.exists(MINIMAX_DEFAULTS_FILE):
            if request:
                return [gr.update()] * len(minimax_ui_default_keys) + ["No defaults file found."]
            else:
                return [gr.update()] * len(minimax_ui_default_keys) + [""]

        try:
            with open(MINIMAX_DEFAULTS_FILE, 'r') as f:
                loaded_settings = json.load(f)
        except Exception as e:
            return [gr.update()] * len(minimax_ui_default_keys) + [f"Error loading defaults: {e}"]

        lora_folder = loaded_settings.get("minimax_lora_folder", "lora")
        lora_choices = get_lora_options(lora_folder)

        updates = []
        for i, key in enumerate(minimax_ui_default_keys):
            component = minimax_ui_default_components_ORDERED_LIST[i]
            default_value_from_component = None
            if hasattr(component, 'value'):
                default_value_from_component = component.value

            value_to_set = loaded_settings.get(key, default_value_from_component)

            if key == "minimax_video_length" and value_to_set is not None:
                # Defaults saved before the dropdown may hold floats or off-grid counts.
                try:
                    v = int(float(value_to_set))
                    value_to_set = 0 if v <= 0 else min(
                        max(124, minimax_align_num_frames(v)), 345
                    )
                except (TypeError, ValueError):
                    value_to_set = 124

            if "lora_weight" in key:
                if value_to_set not in lora_choices:
                    value_to_set = "None"
                updates.append(gr.update(choices=lora_choices, value=value_to_set))
            else:
                updates.append(gr.update(value=value_to_set))

        return updates + ["MiniMax defaults loaded successfully."]

    minimax_save_defaults_btn.click(
        fn=save_minimax_defaults,
        inputs=minimax_ui_default_components_ORDERED_LIST,
        outputs=[minimax_defaults_status]
    )
    minimax_load_defaults_btn.click(
        fn=load_minimax_defaults,
        inputs=None,
        outputs=minimax_ui_default_components_ORDERED_LIST + [minimax_defaults_status]
    )

    def initial_load_minimax_defaults():
        results_and_status = load_minimax_defaults(None)
        return results_and_status[:-1]

    demo.load(
        fn=initial_load_minimax_defaults,
        inputs=None,
        outputs=minimax_ui_default_components_ORDERED_LIST
    )


    # ===== Video Info handlers =====
    def get_video_info(video_path: str) -> dict:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return {}

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0

            # Ensure video length does not exceed 201 frames
            if total_frames > 201:
                total_frames = 201
                duration = total_frames / fps  # Adjust duration accordingly

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration  # Might be useful in some contexts
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return {}
        
    def extract_video_details(video_path: str) -> Tuple[dict, str]:
        metadata = extract_video_metadata(video_path)
        video_details = get_video_info(video_path)

        # Combine metadata with video details
        for key, value in video_details.items():
            if key not in metadata:
                metadata[key] = value

        # Ensure video length does not exceed 201 frames
        if 'video_length' in metadata:
            metadata['video_length'] = min(metadata['video_length'], 201)
        else:
            metadata['video_length'] = min(video_details.get('total_frames', 0), 201)

        # Return both the updated metadata and a status message
        return metadata, "Video details extracted successfully"

    video_input.upload(
        fn=extract_video_details,
        inputs=video_input,
        outputs=[metadata_output, status]
    )

    # ===== Video Info -> MiniMax send =====
    def change_to_minimax_tab():
        return gr.Tabs(selected=21)

    def handle_send_to_minimax_tab(metadata: dict, video_path: str) -> Tuple[str, Dict]:
        # Prefer the pristine embedded metadata over the merged Video Info JSON:
        # extract_video_details caps video_length at 201, which would corrupt
        # MiniMax lengths above 201 frames.
        params = dict(metadata or {})
        if video_path:
            fresh = extract_video_metadata(video_path)
            if fresh:
                params = fresh
        if not params:
            return "No generation metadata found in video", {}
        model_type = params.get("model_type")
        if model_type and model_type != "MiniMax-H3":
            return f"Metadata is from {model_type} - mapping best-effort", params
        return "Parameters ready for MiniMax", params

    MINIMAX_SEND_OUTPUT_COUNT = 22

    def apply_minimax_params(params: dict):
        if not params:
            return [gr.update()] * MINIMAX_SEND_OUTPUT_COUNT

        def opt(key, cast=None):
            value = params.get(key)
            if value is None:
                return gr.update()
            try:
                return gr.update(value=cast(value) if cast else value)
            except (TypeError, ValueError):
                return gr.update()

        # Keyframes / references: only restore files that still exist on disk
        image_path = params.get("image_path")
        if image_path and not os.path.exists(image_path):
            image_path = None
        last_image_path = params.get("last_image_path")
        if last_image_path and not os.path.exists(last_image_path):
            last_image_path = None
        references = [p for p in (params.get("references") or [])
                      if isinstance(p, str) and os.path.exists(p)]

        # Reproduce the recorded task only if its required inputs survived
        task = params.get("task", "auto")
        if task not in ("t2va", "fl2va", "ref2va"):
            task = "auto"
        elif task == "fl2va" and not (image_path or last_image_path):
            task = "auto"
        elif task == "ref2va" and not references:
            task = "auto"

        # 0/None = derive from audio reference; else snap to an encodable 17n+5 count
        raw_length = params.get("video_length")
        if raw_length in (None, "", 0):
            video_length = 0
        else:
            try:
                video_length = min(max(124, minimax_align_num_frames(int(float(raw_length)))), 345)
            except (TypeError, ValueError):
                video_length = 124

        aspect = params.get("aspect_ratio")
        aspect_update = (gr.update(value=aspect)
                         if aspect in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2:1", "1:2")
                         else gr.update())
        attn = params.get("attn_mode")
        attn_update = (gr.update(value=attn)
                       if attn in ("torch", "sdpa", "flash", "flashattn", "flash2",
                                   "flash3", "sageattn", "xformers")
                       else gr.update())

        reference_order = " ".join(str(i + 1) for i in range(len(references)))

        return [
            gr.update(value=params.get("prompt", "")),        # minimax_prompt
            gr.update(value=task),                            # minimax_task_override
            gr.update(value=image_path),                      # minimax_input_image
            gr.update(value=last_image_path),                 # minimax_last_image
            references,                                       # minimax_reference_state
            None,                                             # minimax_reference_files
            reference_order,                                  # minimax_reference_order
            minimax_reference_preview_html(references),       # minimax_reference_preview
            aspect_update,                                    # minimax_aspect_ratio
            gr.update(value=None),                            # minimax_width (auto)
            gr.update(value=None),                            # minimax_height (auto)
            gr.update(value=video_length),                    # minimax_video_length
            opt("infer_steps", int),                          # minimax_infer_steps
            gr.update(value=params.get("flow_shift")),        # minimax_flow_shift
            gr.update(value=params.get("audio_flow_shift")),  # minimax_audio_flow_shift
            opt("seed", int),                                 # minimax_seed
            opt("num_outputs", int),                          # minimax_num_outputs
            opt("ckpt_dir", str),                             # minimax_ckpt_dir
            attn_update,                                      # minimax_attn_mode
            opt("blocks_to_swap", int),                       # minimax_blocks_to_swap
            opt("compile", bool),                             # minimax_compile
            opt("save_path", str),                            # minimax_save_path
        ]

    send_to_minimax_btn.click(
        fn=handle_send_to_minimax_tab,
        inputs=[metadata_output, video_input],
        outputs=[status, params_state],
    ).then(
        fn=apply_minimax_params,
        inputs=[params_state],
        outputs=[
            minimax_prompt, minimax_task_override, minimax_input_image, minimax_last_image,
            minimax_reference_state, minimax_reference_files, minimax_reference_order,
            minimax_reference_preview,
            minimax_aspect_ratio, minimax_width, minimax_height, minimax_video_length,
            minimax_infer_steps, minimax_flow_shift, minimax_audio_flow_shift,
            minimax_seed, minimax_num_outputs, minimax_ckpt_dir, minimax_attn_mode,
            minimax_blocks_to_swap, minimax_compile, minimax_save_path,
        ],
    ).then(
        fn=change_to_minimax_tab,
        inputs=None,
        outputs=[tabs],
    )


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp_frames", exist_ok=True)
    start_wan22_worker()
    demo.queue().launch(server_name="0.0.0.0", share=False)
