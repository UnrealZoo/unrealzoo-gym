"""Read-only process measurements for an explicitly identified local UE editor."""
from __future__ import annotations

import math
import os
import time


def host_cpu_capacity() -> dict:
    """Unknown physical core counts remain null, never an invented zero."""
    try:
        import psutil
        return {"logical_cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_source": "psutil", "cpu_count_unavailable_reason": None}
    except (ImportError, OSError) as error:
        return {"logical_cpu_count": os.cpu_count(), "physical_cpu_count": None,
                "cpu_count_source": "os.cpu_count (logical only)",
                "cpu_count_unavailable_reason": str(error)}


class ExternalUEProcessMonitor:
    """Never owns, signals or waits for the process; identity changes fail closed."""

    def __init__(self, worker, pid: int, *, psutil_module=None, clock=time.perf_counter):
        if type(pid) is not int or pid <= 0:
            raise ValueError("--ue-pid must be a positive integer")
        if worker.owns_process is not False or worker.proc is not None:
            raise ValueError("Explicit UE PID monitoring is only for an attached external process")
        context_pid = (worker.world_context or {}).get("process_id")
        if (isinstance(context_pid, bool) or not isinstance(context_pid, (int, float))
                or not math.isfinite(context_pid) or context_pid != pid):
            raise ValueError(f"--ue-pid {pid} differs from game/context process_id {context_pid!r}")
        if psutil_module is None:
            try:
                import psutil as psutil_module
            except ImportError as error:
                raise RuntimeError("--ue-pid requires psutil in the benchmark Python runtime") from error
        self._psutil = psutil_module
        self._clock = clock
        self.pid = pid
        self.create_time = self._psutil.Process(pid).create_time()
        if not math.isfinite(self.create_time) or self.create_time <= 0:
            raise RuntimeError("UE process has no valid creation time")
        self._before = None

    def _snapshot(self) -> dict:
        # A new Process object is essential: psutil caches create_time on each
        # object, so rereading the original object cannot detect PID reuse.
        process = self._psutil.Process(self.pid)
        if process.create_time() != self.create_time:
            raise RuntimeError("UE PID was reused; refusing to combine different process lifetimes")
        cpu = process.cpu_times()
        timestamp = self._clock()
        rss = process.memory_info().rss
        if self._psutil.Process(self.pid).create_time() != self.create_time:
            raise RuntimeError("UE PID was reused while reading its resource counters")
        cpu_seconds = float(cpu.user + cpu.system)
        if not math.isfinite(cpu_seconds) or cpu_seconds < 0 or rss < 0:
            raise RuntimeError("UE process returned invalid CPU or RSS counters")
        return {"monotonic_seconds": timestamp, "cpu_seconds": cpu_seconds,
                "rss_bytes": int(rss)}

    def begin(self) -> None:
        if self._before is not None:
            raise RuntimeError("Process measurement window is already active")
        self._before = self._snapshot()
        self._rss_peak = self._before["rss_bytes"]
        self._samples = 1

    def sample(self) -> dict:
        if self._before is None:
            raise RuntimeError("Start the process measurement window before sampling")
        current = self._snapshot()
        self._rss_peak = max(self._rss_peak, current["rss_bytes"])
        self._samples += 1
        return current

    def finish(self, sampling_started: float, sampling_finished: float) -> dict:
        after = self.sample()
        before = self._before
        elapsed = after["monotonic_seconds"] - before["monotonic_seconds"]
        if not (before["monotonic_seconds"] <= sampling_started < sampling_finished
                <= after["monotonic_seconds"]) or elapsed <= 0:
            raise RuntimeError("Process CPU measurement must enclose the complete sampling interval")
        cpu_delta = after["cpu_seconds"] - before["cpu_seconds"]
        if cpu_delta < 0:
            raise RuntimeError("UE process CPU counter decreased during sampling")
        cores = cpu_delta / elapsed
        result = {
            "source": "psutil", "pid": self.pid, "process_create_time": self.create_time,
            "owns_process": False, "identity_checked_each_sample": True,
            "cpu_scope": "This UE process user+system CPU; excludes child processes and the Python client",
            "cpu_window_seconds": elapsed,
            "step_window_seconds": sampling_finished - sampling_started,
            "cpu_window_encloses_step_window": True,
            "cpu_seconds_delta": cpu_delta,
            "mean_cpu_cores": cores,
            "mean_cpu_percent_one_core": 100.0 * cores,
            "rss_before_bytes": before["rss_bytes"], "rss_after_bytes": after["rss_bytes"],
            "rss_peak_sampled_bytes": self._rss_peak, "rss_sample_count": self._samples,
            "rss_sampling": "Before, after each vector step, and after the complete sampling window; peak is a sampled lower bound",
            "before": before, "after": after,
        }
        self._before = None
        return result
