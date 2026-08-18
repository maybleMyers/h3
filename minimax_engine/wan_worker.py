"""
Background Worker for Wan2.2 Video Generation

This worker process runs independently of Gradio and processes jobs from the queue.
It continues running even when browsers disconnect, ensuring all queued jobs complete.

Adapted from Kandinsky5 worker.py for Wan2.2 video generation pipeline.

Usage:
    python wan_worker.py [--poll-interval 2.0] [--queue-file wan_job_queue.json]

The worker will automatically:
1. Poll for pending jobs
2. Execute them via subprocess
3. Update progress in the queue file
4. Handle cancellation requests
5. Clean up old completed jobs periodically
"""

import argparse
import json
import os
import sys
import re
import time
import signal
import subprocess
import threading
from typing import Optional, Tuple
from datetime import datetime
import imageio_ffmpeg

from wan_job_queue import get_queue, JobQueue, JobStatus, Job, QueueUnavailable


def get_ffmpeg_path():
    """Get ffmpeg executable path from imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    params_json = json.dumps(parameters, indent=2)
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    cmd = [
        get_ffmpeg_path(), '-y',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"[Worker] Failed to add metadata: {e.stderr.decode() if e.stderr else str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"[Worker] Metadata error: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


class Worker:
    """
    Background worker that processes video generation jobs.

    Runs as a daemon process or thread, independent of the Gradio frontend.
    """

    def __init__(self, queue: JobQueue, poll_interval: float = 2.0, use_signals: bool = False):
        self.queue = queue
        self.poll_interval = poll_interval
        self.running = True
        self.current_process: Optional[subprocess.Popen] = None
        self.current_job_id: Optional[str] = None
        self.current_clip_info: Optional[str] = None  # Track current clip progress (e.g., "Clip 2/4")
        self._last_loading_step = -100  # Track last printed loading step for filtering
        self._cancellation_stop_event = threading.Event()  # Signal to stop cancellation monitor
        self._was_cancelled = False  # Flag to indicate job was cancelled

        # Only setup signal handlers when running as main process (not in thread)
        if use_signals:
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except ValueError:
                # Signal handlers can only be set in main thread
                pass

    def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
            except:
                try:
                    self.current_process.kill()
                except:
                    pass

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[Worker] Received signal {signum}, shutting down...")
        self.stop()

    def _should_print_line(self, line: str) -> bool:
        """
        Filter rapid progress updates for model loading.
        Only print every 100 steps to avoid console spam.
        """
        # Skip empty lines
        if not line:
            return False
        # Check for model loading progress bars (tqdm format)
        # e.g., "Loading wan22_i2v_14B_high_noise_fp32_and_fp16.safetensors with LoRA merge:  69%|██████▊   | 752/1095"
        # or, when the bar has no total, "... with LoRA merge: 541it [01:01, 7.51it/s]"
        if "Loading" in line and ("with LoRA merge" in line or "safetensors" in line):
            match = re.search(r'\|\s*(\d+)/(\d+)', line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
            else:
                match = re.search(r'(\d+)it\s*\[', line)
                if not match:
                    return True
                current_step = int(match.group(1))
                total_steps = 0
            # A counter that moved backwards means a new bar (next shard, next job)
            if current_step < self._last_loading_step:
                self._last_loading_step = -100
            # Only print at 0%, every 100 steps, or at 100%
            if current_step == 0 or (total_steps and current_step >= total_steps) or current_step - self._last_loading_step >= 100:
                self._last_loading_step = current_step
                return True
            return False
        return True

    def parse_progress_line(self, line: str) -> Tuple[Optional[float], Optional[str], int, int]:
        """
        Parse progress bar lines and extract useful information for Wan2.2.

        Returns:
            Tuple of (progress_percent, progress_text, current_step, total_steps)
        """
        line = line.strip()

        # Loading checkpoint/model shards
        if "Loading checkpoint shards:" in line or "Loading model" in line:
            match = re.search(r'(\d+)%.*?(\d+/\d+)', line)
            if match:
                percent = float(match.group(1))
                fraction = match.group(2)
                # Loading is ~10% of total progress
                return percent * 0.1, f"Loading model: {percent:.0f}% ({fraction})", 0, 0

        # Building DiT model
        if "Building DiT" in line or "building model" in line.lower():
            return 10.0, "Building DiT model...", 0, 0

        # Loading DiT weights
        if "Loading DiT weights" in line or "loading weights" in line.lower():
            return 12.0, "Loading DiT weights...", 0, 0

        # Loading VAE
        if "Loading VAE" in line or "loading vae" in line.lower():
            return 13.0, "Loading VAE...", 0, 0

        # Loading T5/CLIP
        if "Loading T5" in line or "Loading CLIP" in line:
            return 14.0, "Loading text encoder...", 0, 0

        # Context windows progress (for Wan2.2)
        context_match = re.search(r'Processing window\s*(\d+)\s*/\s*(\d+)', line)
        if context_match:
            current = int(context_match.group(1))
            total = int(context_match.group(2))
            percent = (current / total) * 100
            adjusted_percent = 15.0 + (percent * 0.75)
            return adjusted_percent, f"Processing window {current}/{total}", current, total

        # Main generation progress bar (TQDM format)
        # Matches: "45%|████     | 45/100 [01:23<01:45"
        tqdm_match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+)', line)
        if tqdm_match:
            percent = float(tqdm_match.group(1))
            current = int(tqdm_match.group(2))
            total = int(tqdm_match.group(3))
            elapsed = tqdm_match.group(4)
            eta = tqdm_match.group(5)
            # Main generation is 15% to 95% of total progress
            adjusted_percent = 15.0 + (percent * 0.8)
            clip_prefix = f"{self.current_clip_info} - " if self.current_clip_info else ""
            return adjusted_percent, f"{clip_prefix}Generating: {percent:.0f}% ({current}/{total} steps) - ETA: {eta}", current, total

        # Alternative TQDM format (simpler)
        simple_tqdm = re.search(r'(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+)', line)
        if simple_tqdm:
            current = int(simple_tqdm.group(1))
            total = int(simple_tqdm.group(2))
            elapsed = simple_tqdm.group(3)
            eta = simple_tqdm.group(4)
            percent = (current / total) * 100
            adjusted_percent = 15.0 + (percent * 0.8)
            clip_prefix = f"{self.current_clip_info} - " if self.current_clip_info else ""
            return adjusted_percent, f"{clip_prefix}Generating: {percent:.0f}% ({current}/{total} steps) - ETA: {eta}", current, total

        # Time elapsed (completion)
        if "TIME ELAPSED:" in line:
            match = re.search(r'TIME ELAPSED:\s*([\d.]+)', line)
            if match:
                elapsed = float(match.group(1))
                return 95.0, f"Generation completed in {elapsed:.1f}s", 0, 0

        # Video saved
        if "Video saved to" in line or "Generated video is saved to" in line:
            return 100.0, "Video saved successfully!", 0, 0

        # VAE decoding
        if "Decoding" in line or "VAE" in line.upper() or "decoding video" in line.lower():
            return 96.0, "Decoding video...", 0, 0

        # Stop signal received (graceful stop)
        if "Stop signal received" in line:
            if "decoding" in line.lower():
                return 95.0, "Stopping - decoding current latents...", 0, 0
            elif "checkpoint" in line.lower() or "saving" in line.lower():
                return 95.0, "Stopping - saving checkpoint...", 0, 0

        # Encoding text
        if "Encoding" in line and ("text" in line.lower() or "prompt" in line.lower()):
            return 14.5, "Encoding text prompt...", 0, 0

        # Preparing latents
        if "Preparing" in line and "latent" in line.lower():
            return 15.0, "Preparing latents...", 0, 0

        # SVI clip progress (e.g., "=== Generating clip 2/4 ===")
        clip_match = re.search(r'=== Generating clip (\d+)/(\d+) ===', line)
        if clip_match:
            clip_current = int(clip_match.group(1))
            clip_total = int(clip_match.group(2))
            self.current_clip_info = f"Clip {clip_current}/{clip_total}"
            return 15.0, f"Starting {self.current_clip_info}...", 0, 0

        return None, None, 0, 0

    # A vanished job only counts as a cancellation after this many consecutive
    # misses (~1 min at the 2 s poll). Below that it is treated as a transient
    # queue problem — killing an hour-long generation over one bad read is far
    # worse than letting a genuinely-deleted job run to completion.
    MISSING_JOB_TOLERANCE = 30

    def check_cancellation(self, job_id: str) -> Optional[bool]:
        """Whether the job has been cancelled.

        Returns True (cancelled), False (still live), or None when the queue
        could not be read or the job is absent — never conflate those with a
        cancellation.
        """
        try:
            job = self.queue.get_job(job_id)
        except QueueUnavailable as e:
            print(f"[Worker] Queue unreadable while checking job {job_id}: {e}")
            return None

        if job is None:
            return None
        return job.status == JobStatus.CANCELLED.value

    def _cancellation_monitor(self, job_id: str):
        """
        Background thread that monitors for job cancellation and immediately
        terminates the subprocess when cancellation is detected.

        This runs in parallel with the main output-reading loop, checking for
        cancellation every 2 seconds so stops are near-instant instead of
        waiting for the next step output.
        """
        missing = 0
        while not self._cancellation_stop_event.is_set():
            cancelled = self.check_cancellation(job_id)

            if cancelled is None:
                missing += 1
                if missing == 1 or missing % 10 == 0:
                    print(f"[Worker] Job {job_id} not readable from queue "
                          f"({missing}/{self.MISSING_JOB_TOLERANCE}) - not cancelling")
                if missing < self.MISSING_JOB_TOLERANCE:
                    self._cancellation_stop_event.wait(2.0)
                    continue
                print(f"[Worker] Job {job_id} missing from queue for "
                      f"{missing} consecutive checks, treating as cancelled")
            elif cancelled:
                missing = 0
                print(f"[Worker] Job {job_id} cancellation detected, terminating immediately...")
            else:
                missing = 0
                self._cancellation_stop_event.wait(2.0)
                continue

            self._was_cancelled = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        print(f"[Worker] Process didn't terminate, killing...")
                        self.current_process.kill()
                except Exception as e:
                    print(f"[Worker] Error terminating process: {e}")
            return

    def run_job(self, job: Job) -> bool:
        """
        Execute a single job.

        Returns:
            True if job completed successfully, False otherwise
        """
        self.current_job_id = job.id
        self.current_clip_info = None  # Reset clip tracking for new job
        self._last_loading_step = -100  # Reset loading progress tracking
        self._was_cancelled = False  # Reset cancellation flag
        self._cancellation_stop_event.clear()  # Reset stop event
        monitor_thread = None  # Will hold the cancellation monitor thread
        print(f"\n[Worker] Starting job {job.id}")
        print(f"[Worker] Command: {' '.join(job.command)}")

        # Get preview path from job parameters
        preview_suffix = None
        for i, arg in enumerate(job.command):
            if arg == "--preview_suffix" and i + 1 < len(job.command):
                preview_suffix = job.command[i + 1]
                break

        preview_path = ""
        if preview_suffix:
            save_path = job.parameters.get('save_path', 'outputs')
            preview_path = os.path.join(save_path, "previews", f"latent_preview_{preview_suffix}.mp4")

        job_log = self._open_job_log(job)

        try:
            # Start the subprocess
            self.current_process = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

            # The job was already set RUNNING when it was claimed; just record the pid.
            self.queue.attach_process(job.id, self.current_process.pid)

            # Start cancellation monitor thread for immediate stop response
            monitor_thread = threading.Thread(
                target=self._cancellation_monitor,
                args=(job.id,),
                daemon=True
            )
            monitor_thread.start()

            last_preview_mtime = 0
            output_lines = []
            last_write_time = 0.0
            last_write_progress = -100.0

            # Monitor the process
            while True:
                # Check if process has finished (either normally or via cancellation monitor)
                if self.current_process.poll() is not None:
                    break

                # Read output line
                line = self.current_process.stdout.readline()
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    self._write_job_log(job_log, line)
                    if self._should_print_line(line):
                        print(f"[Job {job.id}] {line}")

                    # Parse progress
                    progress, progress_text, current_step, total_steps = self.parse_progress_line(line)
                    if progress is not None:
                        # Check for updated preview
                        current_preview = ""
                        if preview_path and os.path.exists(preview_path):
                            try:
                                mtime = os.path.getmtime(preview_path)
                                if mtime > last_preview_mtime:
                                    current_preview = preview_path
                                    last_preview_mtime = mtime
                            except:
                                pass

                        # Each write is a whole-file rewrite under the lock, and
                        # tqdm emits many lines a second. Throttling keeps an
                        # hour-long run from starving every other reader.
                        now = time.time()
                        should_write = (
                            current_preview
                            or abs(progress - last_write_progress) >= 1.0
                            or now - last_write_time >= 1.0
                        )
                        if should_write:
                            try:
                                self.queue.update_progress(
                                    job.id,
                                    progress=progress,
                                    progress_text=progress_text,
                                    current_step=current_step,
                                    total_steps=total_steps,
                                    preview_path=current_preview
                                )
                                last_write_time = now
                                last_write_progress = progress
                            except QueueUnavailable as e:
                                # A dropped progress tick is cosmetic; the run continues.
                                print(f"[Worker] Could not write progress for {job.id}: {e}")
                else:
                    # No output, sleep briefly
                    time.sleep(0.1)

            # Stop the cancellation monitor thread
            self._cancellation_stop_event.set()
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)

            # Check if job was cancelled by the monitor thread
            if self._was_cancelled:
                self.current_process = None
                self.current_job_id = None
                print(f"[Worker] Job {job.id} was cancelled")
                return False

            # Process finished - read any remaining output
            remaining = self.current_process.stdout.read()
            if remaining:
                for line in remaining.strip().split('\n'):
                    output_lines.append(line)
                    self._write_job_log(job_log, line)
                    if self._should_print_line(line):
                        print(f"[Job {job.id}] {line}")

            return_code = self.current_process.returncode
            self.current_process = None
            self.current_job_id = None

            # Check if output file exists
            if return_code == 0 and os.path.exists(job.output_filename):
                # Add metadata to the generated video
                try:
                    add_metadata_to_video(job.output_filename, job.parameters)
                    print(f"[Worker] Added metadata to {job.output_filename}")
                except Exception as meta_err:
                    print(f"[Worker] Warning: Failed to add metadata: {meta_err}")

                self._finalize(self.queue.mark_completed, job.id, return_code)
                print(f"[Worker] Job {job.id} completed successfully")
                return True
            else:
                error_msg = f"Process exited with code {return_code}"
                if not os.path.exists(job.output_filename):
                    error_msg += f", output file not found: {job.output_filename}"

                # Check last few lines for error messages
                for line in output_lines[-10:]:
                    if "error" in line.lower() or "exception" in line.lower():
                        error_msg = line
                        break

                self._finalize(self.queue.mark_failed, job.id, error_msg, return_code)
                print(f"[Worker] Job {job.id} failed: {error_msg}")
                return False

        except Exception as e:
            # Stop the cancellation monitor thread if running
            self._cancellation_stop_event.set()
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)
            self.current_process = None
            self.current_job_id = None
            self._finalize(self.queue.mark_failed, job.id, str(e))
            print(f"[Worker] Job {job.id} exception: {e}")
            return False

        finally:
            if job_log:
                try:
                    job_log.close()
                except Exception:
                    pass

    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "logs")

    def _open_job_log(self, job: Job):
        """Open this job's own log file.

        Console output scrolls away and is interleaved across instances; an
        hour-long run that fails at minute 55 needs its own transcript.
        """
        try:
            os.makedirs(self.LOG_DIR, exist_ok=True)
            handle = open(os.path.join(self.LOG_DIR, f"job_{job.id}.log"),
                          "w", encoding="utf-8")
            handle.write(f"# job {job.id} batch {job.batch_id} "
                         f"started {datetime.now().isoformat()}\n")
            handle.write(f"# command: {' '.join(job.command)}\n\n")
            handle.flush()
            return handle
        except OSError as e:
            print(f"[Worker] Could not open job log for {job.id}: {e}")
            return None

    @staticmethod
    def _write_job_log(handle, line: str) -> None:
        if handle is None:
            return
        try:
            handle.write(line + "\n")
            handle.flush()
        except OSError:
            pass

    @staticmethod
    def _finalize(mark, job_id: str, *args) -> None:
        """Write a job's terminal status, retrying past transient queue trouble.

        A lost terminal write is what strands the GUI on "Processing" forever,
        so it is worth several attempts. Recording the wrong terminal status is
        worse than none, so a failure here is left for recover_stale_jobs.
        """
        for attempt in range(5):
            try:
                mark(job_id, *args)
                return
            except QueueUnavailable as e:
                print(f"[Worker] Terminal write for {job_id} failed "
                      f"(attempt {attempt + 1}/5): {e}")
                time.sleep(0.5 * (attempt + 1))
        print(f"[Worker] Gave up writing terminal status for {job_id}")

    # A job may sit RUNNING with no pid for this long between being claimed and
    # its subprocess starting. Recovering inside that window would fail a job
    # that is about to run perfectly well.
    CLAIM_GRACE_SECONDS = 60.0

    @staticmethod
    def _process_alive(pid: int) -> bool:
        try:
            if os.name == 'nt':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def recover_stale_jobs(self):
        """
        Recover jobs marked RUNNING whose process is no longer alive.

        Covers a worker or subprocess that died without marking the job failed,
        which would otherwise leave the GUI polling "Processing" forever. Only a
        confirmed-dead pid recovers a job — a live but silent process may just be
        loading checkpoint shards, and failing it would be worse than waiting.
        """
        try:
            running_jobs = self.queue.get_running_jobs()
        except QueueUnavailable as e:
            print(f"[Worker] Skipping stale-job recovery, queue unreadable: {e}")
            return

        now = time.time()
        for job in running_jobs:
            if job.id == self.current_job_id:
                continue  # Ours, and we are watching it directly.

            if job.process_id:
                if self._process_alive(job.process_id):
                    continue
                reason = f"process {job.process_id} is gone"
            else:
                # Claimed but never attached a pid.
                age = now - (job.started_at or job.created_at)
                if age < self.CLAIM_GRACE_SECONDS:
                    continue
                reason = f"no process started after {age:.0f}s"

            print(f"[Worker] Recovering stale job {job.id} ({reason})")
            try:
                self.queue.mark_failed(job.id, f"Job did not survive: {reason}")
            except QueueUnavailable as e:
                print(f"[Worker] Could not mark {job.id} failed: {e}")

    def _foreign_running_job(self) -> Optional[Job]:
        """A live RUNNING job this worker did not start, if any.

        Killing the h3 server leaves its generation subprocess alive — it still
        finishes and writes its output — so a restarted worker has to yield the
        GPU to it rather than launching alongside it.
        """
        try:
            running = self.queue.get_running_jobs()
        except QueueUnavailable:
            return None

        for job in running:
            if job.id == self.current_job_id or not job.process_id:
                continue
            if self._process_alive(job.process_id):
                return job
        return None

    def run(self):
        """Main worker loop."""
        print("=" * 60)
        print(f"[Wan Worker] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Wan Worker] Queue file: {self.queue.queue_file}")
        print(f"[Wan Worker] Poll interval: {self.poll_interval}s")
        print("=" * 60)

        # Start this instance from a clean queue, but leave a generation that
        # outlived the last server alone. Only this instance's queue file is
        # touched; other instances keep their own.
        try:
            stats = self.queue.clear_session(self._process_alive)
            print(f"[Worker] Queue reset: cleared {stats['removed']} job(s) from the "
                  f"previous session, kept {stats['kept']} still generating")
        except QueueUnavailable as e:
            print(f"[Worker] Could not reset queue at startup: {e}")

        last_cleanup = time.time()
        cleanup_interval = 3600  # Clean up old jobs every hour
        last_recovery = time.time()
        recovery_interval = 60  # Re-check for orphaned running jobs every minute
        last_orphan_notice = 0.0

        while self.running:
            try:
                # A generation orphaned by a previous instance keeps running and
                # keeps its VRAM. Starting a second job on top of it would OOM
                # both, so wait for it instead.
                orphan = self._foreign_running_job()
                if orphan:
                    if time.time() - last_orphan_notice > 60:
                        print(f"[Worker] Waiting on job {orphan.id} from an earlier "
                              f"instance (pid {orphan.process_id})")
                        last_orphan_notice = time.time()
                    time.sleep(self.poll_interval)
                    continue

                # Take the next pending job (claim marks it RUNNING atomically)
                job = self.queue.claim_next_pending()

                if job:
                    self.run_job(job)
                else:
                    time.sleep(self.poll_interval)

                    if time.time() - last_recovery > recovery_interval:
                        self.recover_stale_jobs()
                        last_recovery = time.time()

                # Periodic cleanup of old jobs
                if time.time() - last_cleanup > cleanup_interval:
                    removed = self.queue.cleanup_old_jobs(max_age_hours=24.0)
                    if removed > 0:
                        print(f"[Worker] Cleaned up {removed} old jobs")
                    last_cleanup = time.time()

            except KeyboardInterrupt:
                break
            except QueueUnavailable as e:
                print(f"[Worker] Queue unavailable, retrying: {e}")
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"[Worker] Error in main loop: {e}")
                time.sleep(self.poll_interval)

        print(f"\n[Wan Worker] Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Background Worker")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                       help="Interval in seconds between queue polls (default: 2.0)")
    parser.add_argument("--queue-file", type=str, default=None,
                       help="Path to the job queue file. Auto-detects GPU-specific file "
                            "from CUDA_VISIBLE_DEVICES if not specified.")
    args = parser.parse_args()

    queue = get_queue(args.queue_file)
    # use_signals=True when running as standalone process
    worker = Worker(queue, poll_interval=args.poll_interval, use_signals=True)
    worker.run()


if __name__ == "__main__":
    main()
