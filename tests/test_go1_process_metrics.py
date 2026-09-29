"""Identity and measurement-window checks without accessing any real process."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
from benchmark_go1_ue import parse_args
from ue_process_metrics import ExternalUEProcessMonitor, host_cpu_capacity


def worker(pid=42):
    return SimpleNamespace(owns_process=False, proc=None, world_context={"process_id": pid})


class FakePsutil:
    def __init__(self):
        self.creation = 1000.0
        self.rows = iter([(10.0, 2.0, 100), (12.0, 3.0, 500), (16.0, 4.0, 200)])
        self.calls = []

    def Process(self, pid):
        self.calls.append(pid)
        owner = self

        class Process:
            def __init__(self):
                self.cached_creation = owner.creation

            def create_time(self):
                return self.cached_creation

            def cpu_times(self):
                self.user, self.system, self.rss = next(owner.rows)
                return SimpleNamespace(user=self.user, system=self.system)

            def memory_info(self):
                return SimpleNamespace(rss=self.rss)

        return Process()


class ExternalProcessMetricTests(unittest.TestCase):
    def test_pid_mismatch_is_rejected_before_process_access(self):
        psutil = Mock()
        for context in (999, None, True, float("nan")):
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "game/context"):
                ExternalUEProcessMonitor(worker(context), 42, psutil_module=psutil)
        psutil.Process.assert_not_called()

    def test_monitor_never_acquires_process_ownership(self):
        attached = worker()
        psutil = FakePsutil()
        monitor = ExternalUEProcessMonitor(attached, 42, psutil_module=psutil,
                                           clock=iter([10.0, 12.0, 14.0]).__next__)
        monitor.begin()
        monitor.sample()
        measured = monitor.finish(10.25, 13.75)
        self.assertEqual(measured["cpu_seconds_delta"], 8.0)
        self.assertEqual(measured["cpu_window_seconds"], 4.0)
        self.assertEqual(measured["step_window_seconds"], 3.5)
        self.assertEqual(measured["mean_cpu_cores"], 2.0)
        self.assertEqual(measured["mean_cpu_percent_one_core"], 200.0)
        self.assertEqual(measured["rss_before_bytes"], 100)
        self.assertEqual(measured["rss_after_bytes"], 200)
        self.assertEqual(measured["rss_peak_sampled_bytes"], 500)
        self.assertEqual(measured["rss_sample_count"], 3)
        self.assertFalse(measured["owns_process"])
        self.assertIsNone(attached.proc)
        self.assertFalse(attached.owns_process)
        self.assertEqual(set(psutil.calls), {42})

    def test_fresh_process_identity_detects_cached_create_time_pid_reuse(self):
        psutil = FakePsutil()
        monitor = ExternalUEProcessMonitor(worker(), 42, psutil_module=psutil, clock=lambda: 10.0)
        monitor.begin()
        psutil.creation = 2000.0
        with self.assertRaisesRegex(RuntimeError, "PID was reused"):
            monitor.sample()

    def test_process_read_failure_is_not_faked_as_zero_usage(self):
        psutil = FakePsutil()
        monitor = ExternalUEProcessMonitor(worker(), 42, psutil_module=psutil, clock=lambda: 10.0)
        monitor.begin()
        with patch.object(psutil, "Process", side_effect=OSError("process disappeared")):
            with self.assertRaisesRegex(OSError, "disappeared"):
                monitor.finish(10.1, 10.5)

    def test_cpu_window_must_cover_complete_step_interval(self):
        monitor = ExternalUEProcessMonitor(worker(), 42, psutil_module=FakePsutil(),
                                           clock=iter([10.0, 12.0]).__next__)
        monitor.begin()
        with self.assertRaisesRegex(RuntimeError, "complete sampling interval"):
            monitor.finish(10.1, 12.5)

    def test_explicit_pid_needs_positive_pid_and_external_worker(self):
        for pid in (0, -1, True, 42.0):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                ExternalUEProcessMonitor(worker(), pid, psutil_module=Mock())
        owned = worker()
        owned.owns_process = True
        with self.assertRaisesRegex(ValueError, "only for an attached"):
            ExternalUEProcessMonitor(owned, 42, psutil_module=Mock())

    def test_cli_pid_is_attach_only_and_linux_launch_stays_supported(self):
        args = parse_args(["--connect", "127.0.0.1:9000", "--ue-pid", "42", "--output", "/tmp/not-created"])
        self.assertEqual(args.ue_pid, 42)
        self.assertEqual(args.runtime_diagnostics, "disabled")
        enabled = parse_args(["--connect", "127.0.0.1:9000", "--runtime-diagnostics", "enabled", "--output", "/tmp/not-created"])
        self.assertEqual(enabled.runtime_diagnostics, "enabled")
        self.assertIsNone(parse_args(["--ue-binary", "/mock/Game", "--output", "/tmp/not-created"]).ue_pid)
        for invalid in (
            ["--ue-binary", "/mock/Game", "--ue-pid", "42"],
            ["--connect", "127.0.0.1:9000", "--ue-pid", "0"],
        ):
            with self.subTest(invalid=invalid), patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args(invalid + ["--output", "/tmp/not-created"])

    def test_physical_cpu_unavailable_is_null(self):
        with patch.dict(sys.modules, {"psutil": None}), patch("ue_process_metrics.os.cpu_count", return_value=10):
            info = host_cpu_capacity()
        self.assertEqual(info["logical_cpu_count"], 10)
        self.assertIsNone(info["physical_cpu_count"])
        self.assertIsNotNone(info["cpu_count_unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
