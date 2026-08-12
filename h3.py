# h3.py — standalone MiniMax-H3 GUI (MiniMax + Frame Interpolation + Video Info tabs)
# Assembled from H1111/h1111.py (branch h3 @ ece03a2); backend lives in minimax_engine/.
import os

# Offline by default: no telemetry ping, no PyPI version check, no font fetch.
# Must precede the gradio/huggingface imports — both read these at import time.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "tiktoken"))

import gradio as gr
import subprocess
import threading
import time
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
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")


def setup_logging(port: int) -> None:
    """Tee the root logger to a per-instance rotating file.

    Runs are hours long across several instances, and until now everything went
    to stdout only — there was no way to reconstruct an incident afterwards.
    """
    from logging.handlers import RotatingFileHandler

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, f"h3_{port}.log"),
        maxBytes=16 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stdout))

# Job queue system for background generation (survives browser disconnects)
from wan_job_queue import (
    get_queue, set_queue_port, FileLock, JobQueue, JobStatus, Job, QueueUnavailable,
)
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


def _render_batch(jobs: List[Job]):
    """Build the gallery/status view for one batch.

    Returns (videos, preview_path, status_text, progress_text, still_active).
    """
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
    elif completed_count + failed_count == total_jobs and total_jobs > 0:
        total_elapsed = sum(j.elapsed_time for j in jobs if j.status == JobStatus.COMPLETED.value)
        done_msg = f"Batch complete: {completed_count} succeeded, {failed_count} failed"
        if total_elapsed:
            done_msg += f" (total {format_elapsed_time(total_elapsed)})"
        status_parts.insert(0, done_msg)
        progress_text = "Done"
    elif total_jobs == 0:
        status_parts.append("No jobs found for this batch")
    else:
        # Still pending
        pending = total_jobs - completed_count - failed_count
        status_parts.insert(0, f"Pending: {pending} jobs")
        timer_active = True
        progress_text = "Waiting for worker..."

    status_text = " | ".join(status_parts) if status_parts else "Processing..."
    return all_videos, preview_path, status_text, progress_text, timer_active


def wan22_poll_active_job(current_job_id: str, current_batch_id: str):
    """Poll the queue for status of the batch this browser window submitted.

    A window only ever tracks its own batch. Falling back to whatever job
    happened to be running would make a second window display — and its Stop
    button cancel — the first window's generation.
    """
    idle = (
        gr.update(), gr.update(), gr.update(), gr.update(),
        current_job_id, current_batch_id,
        gr.Timer(value=2.0, active=False), gr.update(),
    )

    if not current_batch_id and not current_job_id:
        return idle

    try:
        queue = get_queue()
        if current_batch_id:
            jobs = queue.get_batch_jobs(current_batch_id)
        else:
            job = queue.get_job(current_job_id)
            jobs = [job] if job else []
    except QueueUnavailable as e:
        # Hold the last-known view and keep polling — a transient read failure
        # must never look like "your generation disappeared".
        logger.warning("Queue read failed during poll: %s", e)
        return (
            gr.update(), gr.update(), gr.update(), "Queue busy, retrying...",
            current_job_id, current_batch_id,
            gr.Timer(value=2.0, active=True), gr.update(),
        )

    all_videos, preview_path, status_text, progress_text, timer_active = _render_batch(jobs)

    return (
        all_videos if all_videos or jobs else gr.update(),
        [preview_path] if preview_path else [],
        status_text,
        progress_text,
        current_job_id,
        current_batch_id,
        gr.Timer(value=2.0, active=timer_active),
        # Forget a finished batch so a later reload does not re-attach to it.
        current_batch_id if timer_active else "",
    )


def minimax_reattach(stored_batch_id: str):
    """Re-bind this window to the batch it submitted, after a reload or dropout.

    gr.State is keyed on the gradio session, which a page reload discards — so
    without this the server keeps generating for another 50 minutes into a GUI
    that shows nothing. The batch id comes back from the browser's localStorage.
    """
    stored = (stored_batch_id or "").strip()
    if not stored:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            "", "", gr.Timer(value=2.0, active=False), "",
        )

    try:
        jobs = get_queue().get_batch_jobs(stored)
    except QueueUnavailable as e:
        # Keep the binding and let the timer retry rather than dropping it.
        logger.warning("Queue unreadable during reattach: %s", e)
        return (
            gr.update(), gr.update(), "Reconnecting...", "",
            "", stored, gr.Timer(value=2.0, active=True), stored,
        )

    if not jobs:
        logger.info("Stored batch %s is unknown; detaching", stored)
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            "", "", gr.Timer(value=2.0, active=False), "",
        )

    all_videos, preview_path, status_text, progress_text, timer_active = _render_batch(jobs)
    logger.info("Reattached to batch %s (%d job(s), active=%s)",
                stored, len(jobs), timer_active)

    return (
        all_videos,
        [preview_path] if preview_path else [],
        status_text,
        progress_text,
        jobs[0].id,
        stored,
        gr.Timer(value=2.0, active=timer_active),
        stored if timer_active else "",
    )


def wan22_stop_queue_generation(current_batch_id: str):
    """Cancel all jobs in this window's batch and stop processing."""
    if not current_batch_id:
        return (
            gr.update(), gr.update(),
            "No attached generation to stop", "",
            "", "", gr.Timer(value=2.0, active=False), "",
        )

    try:
        cancelled_jobs = get_queue().cancel_batch(current_batch_id)
    except QueueUnavailable as e:
        logger.error("Could not cancel batch %s: %s", current_batch_id, e)
        return (
            gr.update(), gr.update(),
            f"Could not reach the queue to cancel: {e}", "",
            "", current_batch_id, gr.Timer(value=2.0, active=True), current_batch_id,
        )

    logger.info("Cancelled %d job(s) in batch %s", len(cancelled_jobs), current_batch_id)
    return (
        gr.update(),  # videos - keep existing gallery
        [],  # preview
        f"Cancelled {len(cancelled_jobs)} job(s)",  # status
        "Stopped",  # progress
        "",  # job_id
        "",  # batch_id
        gr.Timer(value=2.0, active=False),  # stop timer
        "",  # forget the batch
    )


def wan22_stop_and_decode(current_batch_id: str):
    """Signal this window's running jobs to stop and decode what they have.

    Scoped to the caller's batch — signalling every running job would stop
    another window's generation as well.
    """
    if not current_batch_id:
        return "No attached generation to stop"

    try:
        jobs = get_queue().get_batch_jobs(current_batch_id)
    except QueueUnavailable as e:
        return f"Could not reach the queue: {e}"

    running_jobs = [j for j in jobs if j.status == JobStatus.RUNNING.value]
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
                logger.error("Error creating signal file: %s", e)

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


# The encodable 17n+5 frame counts inside the 4-30 s window (107..719 @ 24 fps).
MINIMAX_VIDEO_LENGTH_CHOICES = [("0 — derive from audio reference", 0)] + [
    (f"{n} frames ({n / 24:.2f} s)", n) for n in range(107, 721, 17)
]


def minimax_assemble_template_prompt(
    task_name: str,
    imd: str,
    subjects: str,
    summary: str,
    retention: str,
    detailed: str,
    soundscape: str,
    music: str,
    has_first: bool,
    has_last: bool,
    video_length,
) -> str:
    """Assemble a structured H3 prompt from the template fields.

    Follows the official H3-Context-IR output format (MiniMax-H3
    skills/h3-prompt-writing/references/): blank-line-separated `field: value` sections,
    preceded for fl2va by the keyframe alignment instruction line.
    """
    def clean(text):
        return str(text or "").strip()

    soundscape = clean(soundscape) or "N/A"
    music = clean(music) or "N/A"
    sections = []

    if task_name == "ref2va":
        # Ref2VA sections put the content on the line after the label
        # (base-mode sections keep it on the same line).
        for field, value in (
            ("subject_definitions", clean(subjects)),
            ("summary", clean(summary)),
            ("retention_analysis", clean(retention)),
            ("detailed_description", clean(detailed)),
        ):
            if not value:
                raise gr.Error(f"Prompt Template: `{field}` is required for ref2va")
            sections.append(f"{field}:\n{value}")
        sections.append(f"overall_soundscape:\n{soundscape}")
        sections.append(f"non_diegetic_music:\n{music}")
        return "\n\n".join(sections)
    else:
        if not clean(imd):
            raise gr.Error("Prompt Template: `integrated_multimodal_description` is required")
        if task_name == "fl2va" and (has_first or has_last):
            duration = f"{int(video_length) / 24:.2f}"
            if has_first and has_last:
                sections.append(
                    "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
                    "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) "
                    f"aligns with the {duration}-second mark of the target video."
                )
            elif has_first:
                sections.append(
                    "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                    "(from [Shot 1]) is fully referenced."
                )
            else:
                sections.append(
                    "How the reference pictures align with the target video — <Picture 1> "
                    f"(from [Shot N]) aligns with the {duration}-second mark of the target video."
                )
        sections.append(f"integrated_multimodal_description: {clean(imd)}")

    sections.append(f"overall_soundscape: {soundscape}")
    sections.append(f"non_diegetic_music: {music}")
    return "\n\n".join(sections)


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


MINIMAX_TAEHV_DEFAULT = os.path.join("weights", "taeh3.safetensors")
_TAEHV_RAW_BASE = "https://raw.githubusercontent.com/madebyollin/taehv/main"


def minimax_ensure_taehv_checkpoint(path: str) -> str:
    """Return `path`, downloading it from madebyollin/taehv first when it does not exist.

    Mirrors blissful_tuner.latent_preview.ensure_taehv_checkpoint (not imported here: that module
    pulls in torch, which the UI process avoids). Only `tae*.safetensors` names are attempted;
    the download goes to a temp file and is renamed into place so an interrupted fetch never
    leaves a truncated checkpoint behind.
    """
    if os.path.exists(path):
        return path
    name = os.path.basename(path)
    if not (name.startswith("tae") and name.endswith(".safetensors")):
        raise gr.Error(f"TAEHV preview checkpoint {path} was not found (only tae*.safetensors "
                       f"names can be auto-downloaded)")
    url = f"{_TAEHV_RAW_BASE}/safetensors/{name}"
    print(f"{path} not found — downloading {name} from {url}")
    import urllib.request
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".download"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise gr.Error(f"TAEHV auto-download from {url} failed: {e}")
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def minimax_submit_to_queue(
    prompt: str,
    task_override: str,
    input_image: str,
    last_image: str,
    reference_files,
    reference_order: str,
    reference_strip_audio: bool,
    prompt_template: bool,
    tpl_imd: str,
    tpl_subjects: str,
    tpl_summary: str,
    tpl_retention: str,
    tpl_detailed: str,
    tpl_soundscape: str,
    tpl_music: str,
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
    use_taehv: bool,
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
    sol_tau: float,
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
    # Chaining
    chain_enable: bool = False,
    chain_count: int = 2,
    chain_mode: str = "motion",
    chain_keep_segments: bool = False,
    chain_extend_video: str = None,
    chain_context_length: int = 22,
    chain_context_encode: str = "video",
    chain_audio_context: int = 0,
    chain_audio_mode: str = "timeline",
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

    chain_extend = str(chain_extend_video).strip() if chain_extend_video else ""
    chaining = bool(chain_enable) and (int(chain_count) >= 2 or bool(chain_extend))
    # Motion context anchors the start of an extension, so a First Frame cannot be used there.
    if chaining and chain_mode == "motion" and chain_extend and input_image:
        input_image = None

    if task_override and task_override != "auto":
        task_name = task_override
    elif references:
        task_name = "ref2va"
    elif input_image or last_image:
        task_name = "fl2va"
    else:
        task_name = "t2va"

    if bool(chain_enable) and not chaining:
        raise gr.Error("Chaining with 1 segment needs an Extend video")
    if chaining and chain_extend and not os.path.exists(chain_extend):
        raise gr.Error(f"Extend video not found: {chain_extend}")

    if task_name == "ref2va" and not references and not (chaining and chain_extend):
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

    # Chaining: one job runs chain_count segments and joins them. The reference carry-over modes need headroom under
    # the per-kind limits; motion context is conditioning rows, not a reference, and consumes no slot.
    if chaining:
        if int(num_outputs) > 1:
            raise gr.Error("Chaining produces one joined video per job; set Num Outputs to 1 "
                           "(use Batch Count for multiple chains)")
        if chain_mode == "last_frame" and ref_kinds.count("image") > 8:
            raise gr.Error("Chaining (last frame) adds an image reference per segment: at most 8 "
                           "image references")
        if chain_mode == "video" and ref_kinds.count("video") > 2:
            raise gr.Error("Chaining (previous video) adds a video reference per segment: at most 2 "
                           "video references")
        if chain_mode != "motion" and len(ref_kinds) > 11:
            raise gr.Error("Chaining adds a reference per segment: at most 11 references")

    # Frame count: 17n+5 at 24 fps, 4-30 s. 0/blank on ref2va = derive from the audio reference.
    video_length = opt_number(video_length)
    if video_length is not None:
        video_length = minimax_align_num_frames(video_length)
        if not 107 <= video_length <= 719:
            raise gr.Error(
                f"MiniMax-H3 generates 4-30 seconds at 24 fps: video length (snapped to 17n+5) must be "
                f"107-719 frames, got {video_length}"
            )
    elif task_name != "ref2va":
        video_length = 124
    if chaining and video_length is None:
        raise gr.Error("Chaining needs an explicit per-segment Video Length")

    # Motion context: the pinned run costs delivered frames, and every segment has to leave enough for the next.
    chain_budget = ()
    if chaining and chain_mode == "motion":
        from minimax_video.motion_context import chain_frame_budget, motion_context_num_frames

        try:
            chain_span = motion_context_num_frames(int(chain_context_length))
            chain_budget = chain_frame_budget(int(video_length), chain_span, int(chain_count),
                                              extend=bool(chain_extend))
        except ValueError as error:
            raise gr.Error(str(error))
        for index, delivered in enumerate(chain_budget[:-1]):
            if delivered < chain_span:
                raise gr.Error(
                    f"Segment {index + 1} delivers {delivered} frames, fewer than the {chain_span} frame motion "
                    f"context the next one needs. Raise Video Length or lower Context Length."
                )
        if chain_audio_context and int(chain_audio_context) > chain_span:
            raise gr.Error(
                f"Audio Context ({int(chain_audio_context)} frames) is wider than Context Length ({chain_span}); "
                "the soundtrack window would land on the text rows. Use 0 to follow the video context."
            )

    # Explicit pixel dimensions (both set) override the auto canvas; multiples of 32.
    width = opt_number(width)
    height = opt_number(height)
    if width is not None and height is not None:
        width = max(32, (int(width) // 32) * 32)
        height = max(32, (int(height) // 32) * 32)
    else:
        width = height = None

    if prompt_template:
        # Chaining puts the First Frame on segment 1 and the Last Frame on the last one, and each segment sees a
        # single <Picture 1>, so one alignment sentence cannot describe both.
        if chaining and input_image and last_image:
            raise gr.Error(
                "Chaining pins the First Frame on the first segment and the Last Frame on the last one, so the "
                "Prompt Template cannot align both. Clear one keyframe, or write the prompt yourself."
            )
        prompt = minimax_assemble_template_prompt(
            task_name, tpl_imd, tpl_subjects, tpl_summary, tpl_retention, tpl_detailed,
            tpl_soundscape, tpl_music, bool(input_image), bool(last_image), video_length,
        )

    # TAEHV previews: resolve + download the checkpoint up front so generation never waits on
    # (or fails over) a network fetch mid-run.
    taehv_path = None
    if enable_preview and use_taehv:
        taehv_path = str(preview_vae or "").strip() or MINIMAX_TAEHV_DEFAULT
        minimax_ensure_taehv_checkpoint(taehv_path)

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

        if chaining:
            command.extend(["--chain_count", str(int(chain_count)), "--chain_mode", str(chain_mode)])
            if chain_mode == "motion":
                command.extend([
                    "--chain_context_length", str(int(chain_context_length)),
                    "--chain_context_encode", str(chain_context_encode),
                    "--chain_audio_context", str(int(chain_audio_context or 0)),
                    "--chain_audio_mode", str(chain_audio_mode),
                ])
            if chain_keep_segments:
                command.append("--chain_keep_segments")
            if chain_extend:
                command.extend(["--chain_extend", chain_extend])

        if current_seed >= 0:
            command.extend(["--seed", str(current_seed)])

        if attn_mode == "sol":
            command.extend(["--sol_tau", str(float(sol_tau) if sol_tau is not None else 1.0)])

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
        # A reference-mode chain changes its references per segment and the cache key cannot see that. Motion context
        # never reaches the conditioner, so a motion chain's prompt embeds are the same every segment.
        if prompt_cache and (not chaining or chain_mode == "motion"):
            command.extend(["--prompt_cache", os.path.join(save_path, "minimax_prompt_cache.safetensors")])

        if enable_preview:
            command.extend(["--preview", str(max(1, int(preview_steps)))])
            command.extend(["--preview_suffix", unique_preview_suffix])
            if taehv_path:
                command.extend(["--preview_vae", taehv_path])

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
            "video_length": video_length,
            "fps": 24,
            "infer_steps": infer_steps,
            "flow_shift": flow_shift,
            "audio_flow_shift": audio_flow_shift,
            "seed": current_seed,
            "num_outputs": num_outputs,
            "ckpt_dir": ckpt_dir,
            "attn_mode": attn_mode,
            "sol_tau": sol_tau,
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
        if chaining:
            parameters["chain_count"] = int(chain_count)
            parameters["chain_mode"] = chain_mode
            parameters["video_length"] = int(video_length) * int(chain_count)
            if chain_mode == "motion":
                parameters["chain_context_length"] = int(chain_context_length)
                parameters["chain_context_encode"] = chain_context_encode
                parameters["chain_audio_mode"] = chain_audio_mode
                parameters["video_length"] = sum(chain_budget)
            if chain_extend:
                parameters["chain_extend"] = chain_extend

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
    logger.info("Queued batch %s with %d job(s)", batch_id, len(job_ids))
    return (
        [],
        [],
        status_msg,
        "Waiting for worker to start...",
        first_job_id,
        batch_id,
        gr.Timer(value=2.0, active=bool(job_ids)),
        batch_id if job_ids else "",
    )

# ===== Instance isolation (one h3.py per GPU on the same box) =====
INSTANCE_PORT_BASE = 7860
INSTANCE_PORT_SPAN = 32
_instance_lock = None  # Held for the process lifetime; never let it be collected.


def _port_available(port: int) -> bool:
    """Whether we could bind the port we are about to serve on."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def claim_instance_port(requested: Optional[int] = None) -> int:
    """Reserve a port for this instance and pin the queue file to it.

    Gradio picks a free port itself, but only inside launch() — long after the
    worker has already opened a queue file. Two instances would therefore share
    wan_job_queue.json, where each new worker's startup reconciliation and each
    poll would see the other's jobs. Claiming the port up front gives every
    instance its own wan_job_queue_<port>.json.

    The lock file is what actually serializes instances against each other; the
    bind test only rules out unrelated processes already on the port.
    """
    global _instance_lock

    candidates = (
        [requested] if requested
        else range(INSTANCE_PORT_BASE, INSTANCE_PORT_BASE + INSTANCE_PORT_SPAN)
    )

    for port in candidates:
        lock = FileLock(os.path.join(BASE_DIR, f".instance_{port}.lock"), timeout=0.05)
        if not lock.acquire():
            continue  # Another h3.py owns this slot.
        if not _port_available(port):
            lock.release()
            continue
        _instance_lock = lock
        return port

    if requested:
        raise RuntimeError(f"Port {requested} is already in use")
    raise RuntimeError(
        f"No free port in {INSTANCE_PORT_BASE}-{INSTANCE_PORT_BASE + INSTANCE_PORT_SPAN - 1}"
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

_TOKEN_ENCODER = None
_TOKEN_ENCODER_LOADED = False


def _get_token_encoder():
    """cl100k_base out of TIKTOKEN_CACHE_DIR, never off the network.

    tiktoken fetches the BPE ranks over HTTP on a cache miss, and read_file is the
    only path that does so — stubbing it keeps the lookup local. The ranks ship in
    weights/tiktoken/; without them the counter degrades to an estimate.
    """
    global _TOKEN_ENCODER, _TOKEN_ENCODER_LOADED
    if _TOKEN_ENCODER_LOADED:
        return _TOKEN_ENCODER
    _TOKEN_ENCODER_LOADED = True

    from tiktoken import load as tiktoken_load

    def _offline(*args, **kwargs):
        raise OSError("cl100k_base BPE ranks are not cached locally")

    original = tiktoken_load.read_file
    tiktoken_load.read_file = _offline
    try:
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("cl100k_base not cached in %s — prompt token count is an estimate",
                       os.environ["TIKTOKEN_CACHE_DIR"])
    finally:
        tiktoken_load.read_file = original
    return _TOKEN_ENCODER


def count_prompt_tokens(prompt: str) -> int:
    enc = _get_token_encoder()
    if enc is None:
        return len(prompt) // 4
    return len(enc.encode(prompt))


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
    analytics_enabled=False,
    # System font stacks only — gradio's default sans is a GoogleFont, which makes
    # the browser hit fonts.googleapis.com on every page load.
    theme=themes.Default(
        font=("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI",
              "Roboto", "Helvetica Neue", "Arial", "sans-serif"),
        font_mono=("ui-monospace", "Consolas", "Menlo", "monospace"),
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
                        value="A giant green and blue kitty cat eats a deliscious blueberry given to her by a handsomely dressed frog.",
                        lines=5,
                    )
                with gr.Column(scale=1):
                    minimax_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    with gr.Row():
                        minimax_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                        minimax_prompt_template = gr.Checkbox(
                            label="Prompt Template", value=False,
                            info="structured Context-IR prompt fields for the selected task; "
                                 "the freeform prompt box is ignored while enabled",
                        )
                with gr.Column(scale=2):
                    minimax_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    minimax_progress_text = gr.Textbox(label="Progress", interactive=False, value="",
                                                       elem_id="minimax_progress_text")

            with gr.Group(visible=False) as minimax_template_group:
                gr.Markdown(
                    "Structured prompt per the official H3 format "
                    "(`MiniMax-H3/skills/h3-prompt-writing/references/`). The keyframe alignment "
                    "line for fl2va is generated automatically from the keyframes and video length."
                )
                minimax_tpl_imd = gr.Textbox(
                    label="integrated_multimodal_description",
                    info="[Shot 1] style, subjects, action, camera grammar; (S1) speakers; "
                         "<d>[Language] dialogue</d>; later shots open with 'At MM:SS.mmm, the camera cuts to'",
                    lines=5,
                )
                minimax_tpl_subjects = gr.Textbox(
                    label="subject_definitions", visible=False,
                    info="<Subject 1>: ... one line per subject, tied to <Picture/Video N> references",
                    lines=3,
                )
                minimax_tpl_summary = gr.Textbox(
                    label="summary", visible=False,
                    info="starts with the bracketed task type, e.g. [reference generation] ...",
                    lines=2,
                )
                minimax_tpl_retention = gr.Textbox(
                    label="retention_analysis", visible=False,
                    info="<Subject/Picture/Video N> (...): fully_preserved / partially_preserved / "
                         "attribute_transfer / weak_reference; audio: fully_copy / partially_copy / "
                         "reference / weak_reference",
                    lines=3,
                )
                minimax_tpl_detailed = gr.Textbox(
                    label="detailed_description", visible=False,
                    info="shot-by-shot description, same grammar as integrated_multimodal_description",
                    lines=5,
                )
                with gr.Row():
                    minimax_tpl_soundscape = gr.Textbox(
                        label="overall_soundscape",
                        info="ambience only, no dialogue or music; N/A for silence", lines=2,
                    )
                    minimax_tpl_music = gr.Textbox(
                        label="non_diegetic_music",
                        info="score only: instrumentation, tempo, dynamics; N/A if none", lines=2,
                    )

            with gr.Row():
                minimax_generate_btn = gr.Button("Generate", elem_classes="green-btn")
                minimax_stop_btn = gr.Button("Stop Generation", variant="stop")
                minimax_stop_decode_btn = gr.Button("Stop & Decode", variant="secondary")

            # Queue system state components
            minimax_job_id_state = gr.State(value="")
            minimax_batch_id_state = gr.State(value="")
            minimax_poll_timer = gr.Timer(value=2.0, active=False)

            # gr.State dies with the gradio session, so the batch id is mirrored
            # into browser localStorage through these two hidden boxes: _persist
            # is written out, _restore is read back in on page load.
            minimax_batch_persist = gr.Textbox(value="", visible=False,
                                               elem_id="minimax_batch_persist")
            minimax_batch_restore = gr.Textbox(value="", visible=False,
                                               elem_id="minimax_batch_restore")

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
                            label="First Frame",
                            type="filepath",
                        )
                        minimax_last_image = gr.Image(
                            label="Last Frame",
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
                    minimax_original_dims = gr.Textbox(visible=False, value="")
                    with gr.Row():
                        minimax_width = gr.Number(label="Width (blank = auto; ×32)", value=None, step=32)
                        minimax_calc_height_btn = gr.Button("→")
                        minimax_calc_width_btn = gr.Button("←")
                        minimax_height = gr.Number(label="Height (blank = auto; ×32)", value=None, step=32)
                    minimax_video_length = gr.Dropdown(
                        label="Video Length (frames @ 24 fps; the VAE encodes 17n+5 frames, 4–30 s)",
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

                    minimax_chain_enable = gr.Checkbox(
                        label="Chaining", value=False,
                        info="generate several segments of the length above and join them into one video. "
                             "Every segment keeps your references; segments after the first also condition "
                             "on the previous one. Motion context continues the previous segment's motion; "
                             "the reference modes lock identity/style only, so their seams read as cuts.",
                    )
                    with gr.Accordion("Chaining Options", open=True, visible=False) as minimax_chain_accordion:
                        minimax_chain_count = gr.Slider(minimum=1, maximum=10, step=1, value=2,
                                                        label="Segments (1 only when extending)")
                        minimax_chain_extend_video = gr.Video(
                            label="Extend existing video (optional)", sources=["upload"], height=200,
                        )
                        gr.Markdown(
                            "*Extension mode: the video above seeds the chain (same carry-over logic) and is "
                            "prepended to the joined output — handy for testing which prompt/mode continues "
                            "best. 1 segment = a single continuation. With Width/Height blank the canvas "
                            "matches the source video's resolution (×32, capped at the model's max area).*"
                        )
                        minimax_chain_mode = gr.Radio(
                            choices=[("Motion context (true continuation — pins the previous segment's tail)",
                                      "motion"),
                                     ("Last frame (image reference — matched cut)", "last_frame"),
                                     ("Previous video (video reference — matched cut, carries audio, "
                                      "~2x sequence length/VRAM)", "video")],
                            value="motion", label="Carry-over conditioning",
                        )
                        with gr.Group(visible=True) as minimax_chain_motion_group:
                            with gr.Row():
                                minimax_chain_context_length = gr.Dropdown(
                                    choices=[1, 5, 22, 39], value=22, label="Context Length (frames)",
                                    info="the previous segment's tail pinned into the next one. 22 is the tested "
                                         "balance; each segment delivers this many frames fewer.",
                                )
                                minimax_chain_audio_context = gr.Number(
                                    label="Audio Context (frames, 0 = follow)", value=0, minimum=0, step=1,
                                    info="a window wider than the video context risks landing on the text rows",
                                )
                            with gr.Row():
                                minimax_chain_context_encode = gr.Radio(
                                    choices=[("Video (one VAE call — motion lives in the latent)", "video"),
                                             ("Frames (one call per frame — diagnostic)", "frames")],
                                    value="video", label="Context Encode",
                                )
                                minimax_chain_audio_mode = gr.Radio(
                                    choices=[("Timeline (continue the soundtrack)", "timeline"),
                                             ("Off (audio restarts each segment)", "off")],
                                    value="timeline", label="Audio Context",
                                )
                            minimax_chain_budget = gr.Markdown("")
                            gr.Markdown(
                                "*Keyframes: the Last Frame rides the final segment and pins the end of the "
                                "chain. The First Frame is ignored when extending — the context already anchors "
                                "the start — and otherwise opens segment 1 only.*"
                            )
                        minimax_chain_keep_segments = gr.Checkbox(
                            label="Keep per-segment files", value=False,
                            info="also write each segment as its own mp4 next to the joined video",
                        )

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
                        with gr.Row():
                            minimax_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                            minimax_use_taehv = gr.Checkbox(
                                label="Use TAEHV Previews", value=False,
                                info="full-resolution TAE previews (taeh3.safetensors, auto-downloaded "
                                     "before generation starts); off = fast latent2rgb",
                            )
                        minimax_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                          label="Preview Every N Steps")
                        minimax_preview_vae = gr.Textbox(
                            label="Preview TAE Checkpoint (optional)", value="",
                            info="custom checkpoint path used when 'Use TAEHV Previews' is checked; "
                                 "blank = weights/taeh3.safetensors (known madebyollin/taehv checkpoints "
                                 "are downloaded automatically if the file is missing)",
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
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers",
                                 "sol"],
                        value="sdpa",
                        info="'sol' = NVIDIA Sol-Attn block-sparse attention (training-free, DiT only; "
                             "needs triton + SM80+)",
                    )
                    minimax_sol_tau = gr.Slider(
                        minimum=0.8, maximum=1.25, step=0.01, value=1.0,
                        label="Sol-Attn tau",
                        info="routing threshold scale, only used when Attention Mode is 'sol'; "
                             "higher = sparser/faster, lower = denser/safer",
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
                send_first_frame_btn = gr.Button("Send First Frame + Prompt")

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
                outputs=[interp_output_video, interp_status, interp_progress],
                # These generators hold their slot for the whole run. Pinning
                # them to one shared id keeps them off the poller's slots while
                # still letting only one heavy job run at a time.
                concurrency_id="heavy",
                concurrency_limit=1,
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
                outputs=[interp_output_video, interp_status, interp_progress],
                concurrency_id="heavy",
                concurrency_limit=1,
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
    # Every handler that changes what this window is tracking writes the same
    # shape, so submit / poll / stop / reattach stay in step with each other.
    MINIMAX_TRACKING_OUTPUTS = [
        minimax_output, minimax_preview_output, minimax_batch_progress,
        minimax_progress_text, minimax_job_id_state, minimax_batch_id_state,
        minimax_poll_timer, minimax_batch_persist,
    ]

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
            minimax_prompt_template,
            minimax_tpl_imd,
            minimax_tpl_subjects,
            minimax_tpl_summary,
            minimax_tpl_retention,
            minimax_tpl_detailed,
            minimax_tpl_soundscape,
            minimax_tpl_music,
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
            minimax_use_taehv,
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
            minimax_sol_tau,
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
            # Chaining
            minimax_chain_enable,
            minimax_chain_count,
            minimax_chain_mode,
            minimax_chain_keep_segments,
            minimax_chain_extend_video,
            minimax_chain_context_length,
            minimax_chain_context_encode,
            minimax_chain_audio_context,
            minimax_chain_audio_mode,
        ],
        outputs=MINIMAX_TRACKING_OUTPUTS,
        queue=True
    )

    # The poller must never queue behind a long interpolation/upscale job, or
    # every open window's progress display freezes for the duration.
    minimax_poll_timer.tick(
        fn=wan22_poll_active_job,
        inputs=[minimax_job_id_state, minimax_batch_id_state],
        outputs=MINIMAX_TRACKING_OUTPUTS,
        concurrency_id="poll",
        concurrency_limit=None,
    )

    minimax_stop_btn.click(
        fn=wan22_stop_queue_generation,
        inputs=[minimax_batch_id_state],
        outputs=MINIMAX_TRACKING_OUTPUTS,
        queue=False
    )

    minimax_stop_decode_btn.click(
        fn=wan22_stop_and_decode,
        inputs=[minimax_batch_id_state],
        outputs=[minimax_batch_progress],
        queue=False
    )

    # Mirror the attached batch into localStorage whenever it changes, and read
    # it back on page load so a reload re-binds to the running generation.
    minimax_batch_persist.change(
        fn=None,
        inputs=[minimax_batch_persist],
        js="""(v) => {
            try {
                if (v) localStorage.setItem('h3_minimax_batch', v);
                else localStorage.removeItem('h3_minimax_batch');
            } catch (e) {}
        }""",
    )

    demo.load(
        fn=None,
        inputs=None,
        outputs=[minimax_batch_restore],
        js="""() => {
            try { return localStorage.getItem('h3_minimax_batch') || ''; }
            catch (e) { return ''; }
        }""",
    ).then(
        fn=minimax_reattach,
        inputs=[minimax_batch_restore],
        outputs=MINIMAX_TRACKING_OUTPUTS,
    )

    minimax_random_seed_btn.click(fn=set_random_seed, inputs=None, outputs=[minimax_seed])

    minimax_prompt.change(fn=count_prompt_tokens, inputs=minimax_prompt, outputs=minimax_token_counter)

    # Keyframe upload snaps the canvas to the image (first frame wins — it anchors the
    # geometry; the last frame is cover-cropped). Cleared images blank the fields back to auto.
    def update_minimax_dimensions(input_image, last_image, chain_enable, chain_extend_video):
        # An extension takes its canvas from the source video, so a keyframe must not snap the fields there.
        if chain_enable and chain_extend_video:
            return "", gr.update(value=None), gr.update(value=None)
        image = input_image or last_image
        if image is None:
            return "", gr.update(value=None), gr.update(value=None)
        img = Image.open(image)
        w, h = img.size
        w = max(32, (w // 32) * 32)
        h = max(32, (h // 32) * 32)
        return f"{w}x{h}", w, h

    minimax_canvas_inputs = [
        minimax_input_image, minimax_last_image, minimax_chain_enable, minimax_chain_extend_video
    ]
    for trigger in (minimax_input_image.change, minimax_last_image.change,
                    minimax_chain_enable.change, minimax_chain_extend_video.change):
        trigger(
            fn=update_minimax_dimensions,
            inputs=minimax_canvas_inputs,
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

    minimax_reference_upload_dep = minimax_reference_files.upload(
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

    minimax_reference_remove_dep = minimax_reference_remove_btn.click(
        fn=minimax_remove_reference,
        inputs=[minimax_reference_state, minimax_reference_remove_idx],
        outputs=[minimax_reference_state, minimax_reference_order, minimax_reference_preview],
    )

    def minimax_clear_references():
        return [], None, "", minimax_reference_preview_html([])

    minimax_reference_clear_dep = minimax_reference_clear_btn.click(
        fn=minimax_clear_references,
        inputs=None,
        outputs=[minimax_reference_state, minimax_reference_files,
                 minimax_reference_order, minimax_reference_preview],
    )

    # Prompt Template visibility: the group follows the checkbox; the fields follow the
    # effective task (same resolution as submit: override wins, else references → ref2va,
    # keyframes → fl2va, else t2va).
    def minimax_update_template_fields(enabled, task_override, references, input_image, last_image):
        if task_override and task_override != "auto":
            task = task_override
        elif references:
            task = "ref2va"
        elif input_image or last_image:
            task = "fl2va"
        else:
            task = "t2va"
        is_ref = task == "ref2va"
        return (
            gr.update(visible=bool(enabled)),
            gr.update(visible=not is_ref),   # integrated_multimodal_description
            gr.update(visible=is_ref),       # subject_definitions
            gr.update(visible=is_ref),       # summary
            gr.update(visible=is_ref),       # retention_analysis
            gr.update(visible=is_ref),       # detailed_description
        )

    minimax_template_vis_inputs = [minimax_prompt_template, minimax_task_override,
                                   minimax_reference_state, minimax_input_image, minimax_last_image]
    minimax_template_vis_outputs = [minimax_template_group, minimax_tpl_imd, minimax_tpl_subjects,
                                    minimax_tpl_summary, minimax_tpl_retention, minimax_tpl_detailed]
    for _event in (
        minimax_prompt_template.change,
        minimax_task_override.change,
        minimax_input_image.change,
        minimax_last_image.change,
        # Reference-set mutations chain after their handler so the state is current.
        minimax_reference_upload_dep.then,
        minimax_reference_remove_dep.then,
        minimax_reference_clear_dep.then,
    ):
        _event(
            fn=minimax_update_template_fields,
            inputs=minimax_template_vis_inputs,
            outputs=minimax_template_vis_outputs,
        )

    minimax_chain_enable.change(
        fn=lambda enabled: gr.update(visible=bool(enabled)),
        inputs=[minimax_chain_enable],
        outputs=[minimax_chain_accordion],
    )

    def minimax_chain_budget_text(mode, count, length, context_length, extend):
        if mode != "motion":
            return ""
        try:
            from minimax_video.motion_context import chain_frame_budget, motion_context_num_frames
            from minimax_video.packing import align_num_frames

            raw = align_num_frames(int(length))
            span = motion_context_num_frames(int(context_length))
            budget = chain_frame_budget(raw, span, max(1, int(count)), extend=bool(extend))
        except (ValueError, TypeError) as error:
            return f"*⚠️ {error}*"
        total = sum(budget)
        return (f"*{max(1, int(count))} × {raw} frames generated → **{total} delivered** ({total / 24:.2f}s), "
                f"{span} regenerated and trimmed per link.*")

    minimax_chain_budget_inputs = [minimax_chain_mode, minimax_chain_count, minimax_video_length,
                                   minimax_chain_context_length, minimax_chain_extend_video]
    for _component in minimax_chain_budget_inputs:
        _component.change(fn=minimax_chain_budget_text, inputs=minimax_chain_budget_inputs,
                          outputs=[minimax_chain_budget])
    minimax_chain_mode.change(
        fn=lambda mode: gr.update(visible=mode == "motion"),
        inputs=[minimax_chain_mode],
        outputs=[minimax_chain_motion_group],
    )

    minimax_ui_default_components_ORDERED_LIST = [
        minimax_ckpt_dir,
        minimax_dit_path,
        minimax_vae_path,
        minimax_audio_vae_path,
        minimax_text_encoder_path,
        minimax_attn_mode,
        minimax_sol_tau,
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
        minimax_video_length,
        minimax_infer_steps,
        minimax_flow_shift,
        minimax_audio_flow_shift,
        minimax_task_override,
        minimax_num_outputs,
        minimax_prompt_cache,
        minimax_enable_preview,
        minimax_use_taehv,
        minimax_preview_steps,
        minimax_preview_vae,
        minimax_chain_enable,
        minimax_chain_count,
        minimax_chain_mode,
        minimax_chain_keep_segments,
        minimax_chain_context_length,
        minimax_chain_context_encode,
        minimax_chain_audio_context,
        minimax_chain_audio_mode,
    ] + minimax_lora_weights + minimax_lora_multipliers

    minimax_ui_default_keys = [
        "minimax_ckpt_dir",
        "minimax_dit_path",
        "minimax_vae_path",
        "minimax_audio_vae_path",
        "minimax_text_encoder_path",
        "minimax_attn_mode",
        "minimax_sol_tau",
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
        "minimax_video_length",
        "minimax_infer_steps",
        "minimax_flow_shift",
        "minimax_audio_flow_shift",
        "minimax_task_override",
        "minimax_num_outputs",
        "minimax_prompt_cache",
        "minimax_enable_preview",
        "minimax_use_taehv",
        "minimax_preview_steps",
        "minimax_preview_vae",
        "minimax_chain_enable",
        "minimax_chain_count",
        "minimax_chain_mode",
        "minimax_chain_keep_segments",
        "minimax_chain_context_length",
        "minimax_chain_context_encode",
        "minimax_chain_audio_context",
        "minimax_chain_audio_mode",
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
                        max(107, minimax_align_num_frames(v)), 719
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
                video_length = min(max(107, minimax_align_num_frames(int(float(raw_length)))), 719)
            except (TypeError, ValueError):
                video_length = 124

        attn = params.get("attn_mode")
        attn_update = (gr.update(value=attn)
                       if attn in ("torch", "sdpa", "flash", "flashattn", "flash2",
                                   "flash3", "sageattn", "xformers", "sol")
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
            opt("sol_tau", float),                            # minimax_sol_tau
            opt("blocks_to_swap", int),                       # minimax_blocks_to_swap
            opt("compile", bool),                             # minimax_compile
            opt("save_path", str),                            # minimax_save_path
        ]

    def handle_send_first_frame(video_path: str, metadata: dict):
        if not video_path:
            return "No video loaded", gr.update(), gr.update()
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Could not read first frame from video", gr.update(), gr.update()
        frame_path = os.path.join("temp_frames", f"first_frame_{int(time.time())}.png")
        os.makedirs("temp_frames", exist_ok=True)
        cv2.imwrite(frame_path, frame)
        prompt = (metadata or {}).get("prompt") or extract_video_metadata(video_path).get("prompt")
        prompt_update = gr.update(value=prompt) if prompt else gr.update()
        suffix = " and prompt" if prompt else " (no prompt in metadata)"
        return f"First frame{suffix} sent to MiniMax", gr.update(value=frame_path), prompt_update

    send_first_frame_btn.click(
        fn=handle_send_first_frame,
        inputs=[video_input, metadata_output],
        outputs=[status, minimax_input_image, minimax_prompt],
    ).then(
        fn=change_to_minimax_tab,
        inputs=None,
        outputs=[tabs],
    )

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
            minimax_width, minimax_height, minimax_video_length,
            minimax_infer_steps, minimax_flow_shift, minimax_audio_flow_shift,
            minimax_seed, minimax_num_outputs, minimax_ckpt_dir, minimax_attn_mode,
            minimax_sol_tau, minimax_blocks_to_swap, minimax_compile, minimax_save_path,
        ],
    ).then(
        fn=change_to_minimax_tab,
        inputs=None,
        outputs=[tabs],
    )


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="MiniMax-H3 GUI")
    p.add_argument("--server_port", type=int, default=None,
                   help=f"Port to serve on. Default: first free slot from "
                        f"{INSTANCE_PORT_BASE}. The queue file is named after it.")
    p.add_argument("--gpu", type=str, default=None,
                   help="GPU index for this instance (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Gradio default concurrency limit (default: 8).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp_frames", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Claim the port before anything opens the queue: set_queue_port decides the
    # queue filename, and get_queue() caches it on first call.
    port = claim_instance_port(args.server_port)
    set_queue_port(port)
    setup_logging(port)

    logger.info("Starting h3 on port %d (GPU %s, queue %s)",
                port, os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                get_queue().queue_file)

    start_wan22_worker()
    demo.queue(default_concurrency_limit=args.concurrency).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
