r"""
Motion context: continuing a MiniMax-H3 clip rather than imitating it.

A reference is packed *before* the target on the shared rotary clock, which is what makes the model read it as a
separate clip to match. A motion context is the same machinery pointed the other way: the previous clip's tail is
encoded and pinned as conditioning rows at *interior* coordinates of the new clip's own timeline, at
`origin + 5/3 * p` for pixel frame `p`. The model then reads those rows as this clip, so far, and generates forward
out of them. Nothing else changes — no mask, no timestep manipulation, no latent blending.

The pinned run occupies the first `num_frames` frames of the new clip, which the model regenerates, so the caller
trims that many frames off the front of the delivered picture *and* its soundtrack before joining segments.

Two grids have to agree for the join to land:

- the video VAE encodes `17 * n + 5` pixel frames into `5 * n + 2` latent frames, each covering `(1, 4, 4, 4, 4)`
  pixel frames in turn, so a context length off that grid encodes to steps that cover the *first* frames of the
  slice rather than the last and the join skips forward by the difference;
- audio runs on a 40 Hz grid rounded to the *nearest* latent, so an end-aligned soundtrack window sits up to a third
  of a rotary unit away from where the pinned picture ends. `motion_context_audio_overhang` is that signed error and
  the placement compensates for it.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .packing import (
    _ROPE_FRAME_RESCALE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    audio_latent_num_frames,
    latent_pixel_frame_offsets,
    prepare_keyframe_image,
    video_latent_num_frames,
)


logger = logging.getLogger(__name__)


# The context lengths offered by the UI. The grid the VAE actually distinguishes is `1` and every `17 * n + 5`, and
# the CLI accepts all of it; these are the four that are worth the rows.
MINIMAX_H3_MOTION_CONTEXT_LENGTHS = (1, 5, 22, 39)


def motion_context_num_frames(num_frames: int) -> int:
    r"""
    Snap a context length *down* onto the grid the video VAE distinguishes: `1`, or `17 * n + 5`.

    Down rather than to the nearest: an off-grid run encodes to the same number of latent steps as the grid point
    below it, and those steps then cover the *first* frames of the run. The pinned tail would end early and the
    continuation would start from the wrong instant.

    Args:
        num_frames (`int`): The requested context length.

    Returns:
        `int`: The largest grid value that is at most `num_frames`.
    """
    if num_frames < 1:
        raise ValueError(f"A motion context needs at least one frame, got {num_frames}.")
    if num_frames < MINIMAX_H3_LATENTS_PER_CHUNK:
        return 1
    steps = (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK
    return steps * MINIMAX_H3_FRAMES_PER_CHUNK + MINIMAX_H3_LATENTS_PER_CHUNK


def motion_context_num_latent_frames(num_frames: int) -> int:
    r"""
    The latent frames a motion-context run of `num_frames` encodes to.

    A single frame goes through the VAE's clip encoder and stays one latent frame. Anything longer goes through the
    chunked encoder, which pads to a multiple of 17 and drops the trailing latents — so a single frame sent *there*
    would come back as two latent frames covering five, four of them padding.

    Args:
        num_frames (`int`): A context length already on the grid.

    Returns:
        `int`: The number of latent frames.
    """
    return 1 if num_frames == 1 else video_latent_num_frames(num_frames)


def motion_context_frame_anchors(num_frames: int, encode_mode: str) -> tuple[float, ...]:
    r"""
    The pixel-frame index every conditioning block of a motion context is pinned at.

    In `"video"` mode the run is one VAE call and each latent step becomes a block, so the anchors are the steps'
    own pixel offsets — `(0, 1, 5, 9, 13, 17, 18)` for a 22 frame run. In `"frames"` mode every frame is encoded on
    its own and pinned at its own index.

    Args:
        num_frames (`int`): A context length already on the grid.
        encode_mode (`str`): `"video"` or `"frames"`.

    Returns:
        `tuple[float, ...]`: One anchor per conditioning block.
    """
    if encode_mode == "frames" or num_frames == 1:
        return tuple(float(index) for index in range(num_frames))
    if encode_mode != "video":
        raise ValueError(f"A motion context encode mode must be 'video' or 'frames', got {encode_mode!r}.")
    return tuple(float(offset) for offset in latent_pixel_frame_offsets(motion_context_num_latent_frames(num_frames)))


def motion_context_audio_overhang(num_frames: int, num_audio_latents: int) -> float:
    r"""
    How far the end of a clip's audio grid sits from the end of its picture, in rotary units.

    MiniMax-H3 rounds a clip's soundtrack to the *nearest* 40 Hz latent, so this is `±1/3` of a latent: 124 frames
    want 206.67 latents and get 207, 362 frames want 603.33 and get 603.

    Args:
        num_frames (`int`): The clip's frame count.
        num_audio_latents (`int`): The clip's audio latent count per channel.

    Returns:
        `float`: The signed overhang.
    """
    return float(num_audio_latents) - _ROPE_FRAME_RESCALE * float(num_frames)


def motion_context_audio_start_offset(num_frames: int, num_audio_latents: int, overhang: float) -> float:
    r"""
    Where a motion context's soundtrack window starts, in rotary units from the target's own origin.

    The window and the pinned picture are the same instant of the same clip, so they have to *end* together: the
    picture ends at `5/3 * num_frames` of the new timeline, the window is `num_audio_latents` latents wide and
    advances one unit per latent, and `overhang` carries the source clip's grid rounding across.

    Args:
        num_frames (`int`): The pinned run's length in frames.
        num_audio_latents (`int`): The window's width in audio latents.
        overhang (`float`): The source clip's [`motion_context_audio_overhang`].

    Returns:
        `float`: The offset of the window's first latent, slightly negative when the audio grid rounded up.
    """
    return _ROPE_FRAME_RESCALE * float(num_frames) + float(overhang) - float(num_audio_latents)


def trim_segment_audio(num_samples: int, num_frames: int, num_delivered_frames: int, sample_rate: int):
    r"""
    How to cut a segment's soundtrack so it keeps step with its trimmed picture.

    Trimming only the picture would slide the whole soundtrack `num_frames / 24` seconds early — inaudible on
    ambience, squarely offbeat on anything with a pulse. Matching the tail as well keeps every segment exactly as
    long as its picture, so the ~8 ms the sample grid rounds by cannot accumulate down a chain.

    Args:
        num_samples (`int`): The segment's soundtrack length in samples.
        num_frames (`int`): The pinned run trimmed off the front.
        num_delivered_frames (`int`): The frames left after the trim.
        sample_rate (`int`): The soundtrack's sample rate.

    Returns:
        `tuple[int, int]`: Samples to drop from the front, and the length the remainder is padded or cut to.
    """
    cut = int(round(num_frames / MINIMAX_H3_FPS * sample_rate))
    if cut >= num_samples:
        raise ValueError(
            f"Trimming {num_frames} frames wants {cut} samples of a {num_samples} sample soundtrack. The segment's "
            "audio is shorter than its picture."
        )
    return cut, int(round(num_delivered_frames / MINIMAX_H3_FPS * sample_rate))


def chain_frame_budget(num_frames: int, num_context_frames: int, count: int, extend: bool = False) -> tuple[int, ...]:
    r"""
    The frames every segment of a chain delivers.

    Every segment generates `num_frames` and every segment that continues another spends `num_context_frames` of
    them regenerating the pinned run. Only the first segment of a cold chain keeps all of them.

    Args:
        num_frames (`int`): Frames generated per segment.
        num_context_frames (`int`): The pinned run's length.
        count (`int`): Number of segments.
        extend (`bool`): Whether the first segment continues an imported video rather than starting cold.

    Returns:
        `tuple[int, ...]`: Delivered frames per segment.
    """
    if num_context_frames >= num_frames:
        raise ValueError(
            f"A {num_context_frames} frame motion context does not fit in a {num_frames} frame segment. The pinned "
            "run has to be a small fraction of the timeline."
        )
    delivered = num_frames - num_context_frames
    return tuple(num_frames if index == 0 and not extend else delivered for index in range(count))


def prepare_motion_context_frames(frames: np.ndarray, height: int, width: int, crop: str) -> np.ndarray:
    r"""
    Put a motion-context tail onto the target canvas.

    Frames that already are the canvas are returned untouched, which is the path a previous segment of the same
    chain always takes; only an imported video pays for a resampling pass.

    Args:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The tail, uint8 RGB.
        height (`int`): Canvas height.
        width (`int`): Canvas width.
        crop (`str`): `"stretch"` or `"center"`.

    Returns:
        `np.ndarray`: The tail on the canvas.
    """
    if crop not in ("stretch", "center"):
        raise ValueError(f"A motion context crop must be 'stretch' or 'center', got {crop!r}.")
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    stretch = crop == "stretch"
    return np.stack(
        [np.array(prepare_keyframe_image(Image.fromarray(frame), height, width, stretch)) for frame in frames]
    )


@dataclass
class MiniMaxH3MotionContext:
    r"""
    The tail of the previous clip, to be continued.

    The caller fills the request half and [`MiniMaxH3Pipeline.setup`] resolves the rest against the target it is
    building. `frames` may be a whole clip: only its last `num_frames` are kept.

    Attributes:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`):
            The previous clip's frames, uint8 RGB.
        audio_latents (`torch.Tensor` of shape `(2, channels, num_latents)`):
            The previous clip's sampled audio latents, denormalized. Preferred over `waveform`: slicing the tail out
            of these skips a decode/encode round trip that dulls the sound at every link of a chain.
        waveform (`torch.Tensor` of shape `(2, num_samples)`):
            The previous clip's soundtrack, for a source that has no latents — an imported video.
        sample_rate (`int`): The sample rate of `waveform`.
        previous_num_frames (`int`): The previous clip's *untrimmed* frame count, for the audio overhang.
        previous_num_audio_latents (`int`): The previous clip's audio latent count per channel.
        num_frames (`int`): The requested context length; snapped down onto the VAE grid by `setup`.
        audio_num_frames (`int`): The soundtrack window's length in frames; `0` follows `num_frames`.
        encode_mode (`str`): `"video"` to encode the run in one call, `"frames"` one call per frame.
        crop (`str`): How to fit the tail onto the canvas, `"stretch"` or `"center"`.
        num_latent_frames (`int`): Resolved: latent frames the run encodes to.
        frame_anchors (`tuple[float, ...]`): Resolved: the pixel frame every conditioning block is pinned at.
        num_audio_latents (`int`): Resolved: the soundtrack window's width in latents.
        audio_start_offset (`float`): Resolved: the window's offset from the target origin, in rotary units.
    """

    frames: np.ndarray | None = None
    audio_latents: torch.Tensor | None = None
    waveform: torch.Tensor | None = None
    sample_rate: int | None = None
    previous_num_frames: int = 0
    previous_num_audio_latents: int = 0
    num_frames: int = 22
    audio_num_frames: int = 0
    encode_mode: str = "video"
    crop: str = "stretch"

    num_latent_frames: int = 0
    frame_anchors: tuple[float, ...] = ()
    num_audio_latents: int = 0
    audio_start_offset: float = 0.0

    @property
    def has_audio(self) -> bool:
        r"""Whether a soundtrack window can be built from this context."""
        return self.audio_latents is not None or self.waveform is not None


def resolve_motion_context(
    motion_context: MiniMaxH3MotionContext, height: int, width: int, num_frames: int, audio_mode: str = "timeline"
) -> MiniMaxH3MotionContext:
    r"""
    Resolve a motion context against the target it will be pinned into.

    Snaps the context length onto the VAE grid, fits the tail to the canvas and lays out the soundtrack window.

    Args:
        motion_context ([`MiniMaxH3MotionContext`]): The context, mutated in place and returned.
        height (`int`): Target canvas height.
        width (`int`): Target canvas width.
        num_frames (`int`): The target's frame count.
        audio_mode (`str`): `"timeline"` to pin the soundtrack too, `"off"` to pin picture only.

    Returns:
        [`MiniMaxH3MotionContext`]
    """
    if motion_context.frames is None:
        raise ValueError("A motion context needs the previous clip's frames.")

    span = motion_context_num_frames(motion_context.num_frames)
    if span != motion_context.num_frames:
        logger.warning(
            f"motion context: {motion_context.num_frames} frames is off the video VAE grid, using {span}. The grid "
            "is 1 and every 17 * n + 5."
        )
    if span >= num_frames:
        raise ValueError(
            f"A {span} frame motion context does not fit in a {num_frames} frame segment. The pinned run has to be "
            "a small fraction of the timeline."
        )
    available = motion_context.frames.shape[0]
    if available < span:
        raise ValueError(f"A {span} frame motion context needs {span} frames of the previous clip, got {available}.")

    motion_context.num_frames = span
    motion_context.num_latent_frames = motion_context_num_latent_frames(span)
    motion_context.frame_anchors = motion_context_frame_anchors(span, motion_context.encode_mode)
    motion_context.frames = prepare_motion_context_frames(
        motion_context.frames[available - span :], height, width, motion_context.crop
    )

    if audio_mode == "off" or not motion_context.has_audio:
        motion_context.num_audio_latents = 0
        motion_context.audio_start_offset = 0.0
        return motion_context

    window = audio_latent_num_frames(motion_context.audio_num_frames or span)
    if motion_context.audio_latents is not None:
        total = motion_context.audio_latents.shape[-1]
        if window > total:
            logger.warning(f"motion context: audio window clipped from {window} to the {total} latents available.")
            window = total
    if window < 1:
        raise ValueError("A motion context's soundtrack window rounds to zero audio latents.")

    overhang = motion_context_audio_overhang(
        motion_context.previous_num_frames, motion_context.previous_num_audio_latents
    )
    if not -0.5 < overhang < 0.5:
        logger.warning(
            f"motion context: audio overhang {overhang:g} is off the ±1/3 latent the 40 Hz grid rounds by; the "
            "previous clip's frame and latent counts disagree. Ignoring it."
        )
        overhang = 0.0

    motion_context.num_audio_latents = window
    motion_context.audio_start_offset = motion_context_audio_start_offset(span, window, overhang)
    return motion_context
