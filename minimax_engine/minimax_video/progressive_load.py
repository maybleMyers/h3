# Progressive DiT loading: the block stack streams in on a background thread while the denoise
# loop consumes it block by block. Loading the 50-block stack costs ~2 s/block against ~2.4 s/block
# of compute, so denoise step 1 hides almost all of it; steps 2..N are unaffected.
import logging
import re
import threading
import time

import torch

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


def split_block_keys(keys, num_blocks: int):
    """Partition checkpoint keys into (non-block keys, per-block key lists)."""
    non_block = []
    per_block = [[] for _ in range(num_blocks)]
    for key in keys:
        match = _BLOCK_RE.match(key)
        if match is None:
            non_block.append(key)
        else:
            index = int(match.group(1))
            if index >= num_blocks:
                raise ValueError(f"checkpoint has {key} but the config declares {num_blocks} blocks")
            per_block[index].append(key)
    return non_block, per_block


class BlockLoadGate:
    """Per-block readiness barrier between the loader thread and the denoise loop.

    A set flag means the block's weights are fully in place on both host and device — the loader
    synchronizes its CUDA stream before marking — so waiters need no further ordering.
    """

    def __init__(self, num_blocks: int):
        self._ready = [threading.Event() for _ in range(num_blocks)]
        self._error = None
        self.finished = threading.Event()
        self.stall_seconds = 0.0
        self.stalled_blocks = 0

    def is_ready(self, block_idx: int) -> bool:
        return self._ready[block_idx].is_set()

    def mark_ready(self, block_idx: int):
        self._ready[block_idx].set()

    def fail(self, exc: BaseException):
        """Unblock every waiter; each re-raises `exc` instead of reading unloaded weights."""
        self._error = exc
        for event in self._ready:
            event.set()
        self.finished.set()

    def wait(self, block_idx: int):
        event = self._ready[block_idx]
        if not event.is_set():
            start = time.perf_counter()
            event.wait()
            self.stall_seconds += time.perf_counter() - start
            self.stalled_blocks += 1
        if self._error is not None:
            raise self._error


class ProgressiveTransformerLoader:
    """Loads `model.transformer_blocks` in index order on a background thread.

    Each block is read, LoRA-merged, cast/quantized, assigned and placed (GPU for resident blocks,
    pinned CPU masters for streamed ones) before its gate flag is set.
    """

    def __init__(
        self,
        model,
        reads,
        device: torch.device,
        dit_dtype: torch.dtype,
        fp8: bool = False,
        fp8_scaled: bool = False,
        fp8_fast: bool = False,
        curve: bool = False,
        exclude_keys=(),
        lora_hook=None,
        report_unused=None,
    ):
        self.model = model
        self.reads = reads  # [block_idx] -> [(shard path, [keys]), ...]
        self.device = torch.device(device)
        self.dit_dtype = dit_dtype
        self.fp8 = fp8
        self.fp8_scaled = fp8_scaled
        self.fp8_fast = fp8_fast
        self.curve = curve
        self.exclude_keys = list(exclude_keys)
        self.lora_hook = lora_hook
        self.report_unused = report_unused

        self.num_blocks = len(reads)
        self.gate = BlockLoadGate(self.num_blocks)
        self._thread = None
        self._abort = threading.Event()
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

    # ----- lifecycle -----

    def start(self):
        self._thread = threading.Thread(target=self._run, name="minimax-progressive-load", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 120.0):
        """Stop the loader and join it. Idempotent; safe on the abort and exception paths."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._abort.set()
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("progressive load: loader thread did not stop within %.0fs", timeout)

    # ----- loader thread -----

    def _run(self):
        torch.set_grad_enabled(False)  # grad mode is thread-local; the main thread's no_grad does not apply
        start = time.time()
        try:
            for i in range(self.num_blocks):
                if self._abort.is_set():
                    self.gate.fail(RuntimeError("progressive load aborted"))
                    return
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self._load_block(i)
                    # all device work for this block is done before the flag is set, so waiters
                    # need no cross-stream event and the async H2D sources stay alive long enough
                    self._stream.synchronize()
                else:
                    self._load_block(i)
                self.gate.mark_ready(i)
        except BaseException as exc:  # surfaced on the compute thread by gate.wait()
            logger.exception("progressive load failed")
            self.gate.fail(exc)
            return

        offloader = getattr(self.model, "offloader", None)
        if offloader is not None:
            offloader.finalize()
        if self.report_unused is not None:
            self.report_unused()
        self.gate.finished.set()
        logger.info(
            f"progressive load: {self.num_blocks} blocks loaded in {time.time() - start:.1f}s "
            f"(denoise stalled {self.gate.stall_seconds:.1f}s on {self.gate.stalled_blocks} blocks)"
        )

    def _load_block(self, i: int):
        from modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
        from utils.safetensors_utils import stream_safetensors

        from .model_loader import FP8_TARGET_KEYS, apply_dtype_policy

        block = self.model.transformer_blocks[i]
        prefix = f"transformer_blocks.{i}."

        sd = {}
        for path, keys in self.reads[i]:
            for key, value in stream_safetensors(path, keys=keys):
                if self.lora_hook is not None:
                    value = self.lora_hook(key, value)
                sd[key] = value

        if self.fp8_scaled:
            # cleanup_every is effectively disabled: a per-block working set is small, and calling
            # empty_cache() from here would synchronize against the live denoise loop
            optimize_state_dict_with_fp8(
                sd,
                self.device,
                target_layer_keys=FP8_TARGET_KEYS,
                exclude_layer_keys=self.exclude_keys,
                quiet=True,
                cleanup_every=1 << 30,
            )
        apply_dtype_policy(sd, self.dit_dtype, self.fp8, self.curve, self.exclude_keys, scaled=self.fp8_scaled)

        local = {key[len(prefix):]: value for key, value in sd.items()}
        if self.fp8_scaled:
            # registers the scale_weight buffers, so it must precede the assign
            apply_fp8_monkey_patch(block, local, use_scaled_mm=self.fp8_fast, quiet=True)
        block.load_state_dict(local, strict=True, assign=True)
        block.eval().requires_grad_(False)

        offloader = getattr(self.model, "offloader", None)
        if offloader is not None:
            offloader.prepare_block(block, i)
        else:
            block.to(self.device)
