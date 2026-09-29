"""Failure-path tests for the UE pool; no UE process or network is used."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue
import json
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
import ue_go1_env as module
from train_go1_ue import build_parser


class FakeSocket:
    def __init__(self):
        self.timeout = 60.0

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value


class FakeClient:
    def __init__(self, responses=()):
        self.sock = FakeSocket()
        self.send_message_id = 7
        self.recv_num_q = SimpleQueue()
        self.recv_data_q = SimpleQueue()
        for response in responses:
            self.recv_data_q.put(response)
        self.sent = []
        self.disconnected = False

    def send(self, value):
        self.sent.append(value)
        return True

    def disconnect(self):
        self.disconnected = True


class Activity:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def run(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.015)
        with self.lock:
            self.active -= 1


class FakeWorker:
    def __init__(self, index, robots=1, activity=None):
        self.proc = SimpleNamespace(pid=99999900 + index, returncode=None)
        self.port = 19000 + index
        self.tmpdir = Path("/tmp") / f"fake-ue-{index}"
        self.robots = [self.robot() for _ in range(robots)]
        self.reset_count = 0
        self.reset_result_counts = dict.fromkeys(module.RESET_RESULTS, 0)
        self.reset_request_counts = dict.fromkeys(module.RESET_MODES, 0)
        self.reset_calls = []
        self.cleanup_errors = []
        self.activity = activity
        self.closed = False

    @staticmethod
    def robot():
        return SimpleNamespace(
            state={"mock": True}, asset_contract={"mock": True}, reset_api_supported=False
        )

    def add_robot(self):
        if self.activity:
            self.activity.run()
        self.robots.append(self.robot())

    def reset_indices(self, ids, **kwargs):
        if self.activity:
            self.activity.run()
        self.reset_calls.extend(ids)
        self.reset_arguments = kwargs
        self.reset_count += len(ids)
        self.reset_result_counts["rebuilt"] += len(ids)
        self.reset_request_counts[kwargs["reset_mode"]] += len(ids)

    def step(self, actions, commands):
        if self.activity:
            self.activity.run()
        return [robot.state for robot in self.robots]

    def close(self):
        self.closed = True
        self.proc.returncode = 0


def fake_pool(workers, per_process=1):
    pool = module.UEGo1Pool.__new__(module.UEGo1Pool)
    pool.workers = workers
    pool.agents_per_process = per_process
    pool.requested_processes = len(workers)
    pool.output_dir = Path("/tmp")
    pool.executor = ThreadPoolExecutor(max_workers=len(workers))
    pool.closed = False
    pool.initialized = True
    pool.reset_mode = "auto"
    pool.cleanup_errors = []
    pool.step_seconds = 0.0
    pool.reset_wall_seconds = 0.0
    pool.vector_steps = 0
    pool.created_at = time.perf_counter()
    pool.check_resources = lambda: None
    return pool


class UEPoolTests(unittest.TestCase):
    def test_training_cli_exposes_only_auto_and_rebuild(self):
        parser = build_parser()
        required = ["--ue-binary", "/mock/Game", "--log-dir", "/tmp/mock-run"]
        self.assertEqual(parser.parse_args(required).reset_mode, "auto")
        self.assertEqual(parser.parse_args(required + ["--reset-mode", "rebuild"]).reset_mode, "rebuild")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(required + ["--reset-mode", "cached"])

    def test_runtime_diagnostics_cli_defaults_disabled_and_pool_rejects_nonbool(self):
        parser = build_parser()
        required = ["--ue-binary", "/mock/Game", "--log-dir", "/tmp/mock-run"]
        self.assertEqual(parser.parse_args(required).runtime_diagnostics, "disabled")
        self.assertEqual(parser.parse_args(required + ["--runtime-diagnostics", "enabled"]).runtime_diagnostics, "enabled")
        with patch.object(module, "UEWorker") as worker:
            with self.assertRaisesRegex(ValueError, "must be Boolean"):
                module.UEGo1Pool(None, 1, 1, 9000, Path("/tmp/not-created"),
                                connect="127.0.0.1:9000", runtime_diagnostics="disabled")
            worker.assert_not_called()

    def test_diagnostics_metrics_never_claim_old_server_is_disabled(self):
        robots = [module.Robot("old", np.zeros(3), np.zeros(3), state={"old": True}, asset_contract={"valid": True})]
        result = module.diagnostics_summary(robots, False)
        self.assertEqual(result["requested"], "disabled")
        self.assertEqual(result["unavailable_replicas"], 1)
        self.assertEqual(result["confirmed_disabled_replicas"], 0)
        self.assertFalse(result["request_verified_for_all_initialized"])
        self.assertIn("cannot be controlled or verified", result["unavailable_reason"])
        robots.append(module.Robot("new", np.zeros(3), np.zeros(3), state={"new": True},
                                   asset_contract={"valid": True}, runtime_diagnostics_enabled=False))
        result = module.diagnostics_summary(robots, False)
        self.assertEqual(result["control_supported_replicas"], 1)
        self.assertEqual(result["confirmed_disabled_replicas"], 1)
        self.assertFalse(result["request_verified_for_all_initialized"])
        self.assertTrue(module.diagnostics_summary(robots[1:], False)["request_verified_for_all_initialized"])

    def test_training_cli_connect_is_mutually_exclusive_with_binary(self):
        parser = build_parser()
        parsed = parser.parse_args(["--connect", "localhost:9000", "--log-dir", "/tmp/run"])
        self.assertIsNone(parsed.ue_binary)
        self.assertEqual(parsed.connect, "localhost:9000")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--connect", "localhost:9000", "--ue-binary", "/mock/Game", "--log-dir", "/tmp/run"])

    def test_growth_reuses_saved_spawn_after_existing_robot_moves(self):
        worker = module.UEWorker.__new__(module.UEWorker)
        worker.camera_location = np.zeros(3)
        worker.camera_rotation = np.zeros(3)
        worker.robots = []
        poses = {}
        settles = []

        def request(command):
            parts = command.split()
            if parts[1] == "/objects/spawn_from_path":
                name = f"robot{len(poses)}"
                poses[name] = {"location": list(map(float, parts[-3:])), "rotation": [0., 0., 0.]}
                return name
            _, _, name, field = parts[1].split("/")
            if parts[0] == "vget":
                return " ".join(map(str, poses[name][field]))
            if field == "settle_to_ground":
                settles.append(name)
                # The first trace sees the road; later traces would see a peer.
                poses[name]["location"][2] = 35.0 if len(settles) == 1 else 68.905
            else:
                poses[name][field] = list(map(float, parts[2:]))
            return "ok"

        worker.request = request
        worker.reset_one = Mock(return_value={})
        worker.add_robot()
        poses["robot0"]["location"] = [321.0, 0.0, 25.0]
        worker.add_robot()
        self.assertEqual(settles, ["robot0"])
        np.testing.assert_array_equal(worker.robots[1].location, [300.0, 0.0, 35.0])
        np.testing.assert_array_equal(worker.robots[0].location, worker.robots[1].location)

    def test_batch_wire_ids_and_responses(self):
        client = FakeClient([b"ok", b"done"])
        result = module.bounded_batch_request(client, ["first", "second"], 1.0)
        self.assertEqual(result, [b"ok", b"done"])
        self.assertEqual(client.sent, [b"7:first", b"8:second"])
        self.assertEqual(client.send_message_id, 9)
        self.assertEqual(client.recv_num_q.get(), -2)
        self.assertEqual(client.sock.gettimeout(), 60.0)

    def test_partial_batch_has_real_timeout(self):
        client = FakeClient([b"first response only"])
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            module.bounded_batch_request(client, ["first", "missing"], 0.02)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(client.sock.gettimeout(), 60.0)

    def test_batch_deadline_is_not_restarted_for_each_response(self):
        client = FakeClient()
        clock = [0.0]
        waits = []

        def send(value):
            clock[0] += 0.01
            return True

        def receive(timeout):
            waits.append(timeout)
            if len(waits) == 1:
                clock[0] += 0.025
                return b"first"
            raise Empty

        client.send = send
        client.recv_data_q = SimpleNamespace(get=receive)
        with patch.object(module.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                module.bounded_batch_request(client, ["first", "second"], 0.05)
        self.assertAlmostEqual(waits[0], 0.03)
        self.assertAlmostEqual(waits[1], 0.005)

    def test_timeout_marks_worker_connection_unusable(self):
        worker = module.UEWorker.__new__(module.UEWorker)
        worker.client = FakeClient()
        worker.closed = worker.failed = False
        worker.timeout = 0.01
        worker.request_seconds = 0.0
        with self.assertRaises(TimeoutError):
            worker.request_many(["missing response"])
        self.assertTrue(worker.failed)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            worker.request("must not retry physics")

    @unittest.skipUnless(hasattr(module.os, "killpg"), "Owned UE launch requires POSIX process groups; Windows uses attach")
    def test_close_tolerates_process_exit_race_and_is_idempotent(self):
        worker = module.UEWorker.__new__(module.UEWorker)
        worker.closed = False
        worker.cleanup_errors = []
        worker.client = FakeClient()
        client = worker.client
        worker.proc = Mock(pid=123456)
        worker.proc.poll.return_value = None
        with patch.object(module.os, "killpg", side_effect=ProcessLookupError) as kill:
            worker.close()
            worker.close()
        self.assertEqual(kill.call_count, 1)
        self.assertTrue(client.disconnected)
        self.assertIsInstance(client.recv_data_q.get(), ConnectionError)
        self.assertTrue(worker.closed)

    def test_pool_cleanup_continues_after_one_worker_failure(self):
        workers = [FakeWorker(0), FakeWorker(1)]
        workers[0].close = Mock(side_effect=RuntimeError("mock close error"))
        pool = fake_pool(workers)
        pool.close()
        self.assertTrue(workers[1].closed)
        self.assertTrue(any("mock close error" in error for error in pool.cleanup_errors))
        with self.assertRaises(RuntimeError):
            pool.executor.submit(lambda: None)

    def test_reset_indices_are_validated_before_any_mutation(self):
        workers = [FakeWorker(0, 2), FakeWorker(1, 2)]
        pool = fake_pool(workers, 2)
        try:
            for invalid in ([-1], [4], [1.0], [True], [False, 1], [0, 0], [[0]]):
                with self.subTest(invalid=invalid):
                    with self.assertRaises((ValueError, IndexError)):
                        pool.reset_indices(invalid)
                    self.assertFalse(any(worker.reset_calls for worker in workers))
            pool.reset_indices([0, 3])
            self.assertEqual(workers[0].reset_calls, [0])
            self.assertEqual(workers[1].reset_calls, [1])
        finally:
            pool.close()

    def test_model_build_waves_bounded_but_steps_still_parallel(self):
        activity = Activity()
        workers = [FakeWorker(index, 0, activity) for index in range(10)]
        pool = fake_pool(workers)
        try:
            pool.grow(1)
            self.assertLessEqual(activity.peak, 4)
            activity.peak = 0
            pool.reset_indices(range(10))
            self.assertLessEqual(activity.peak, 4)
            activity.peak = 0
            pool.step(np.zeros((10, 12)), np.zeros((10, 3)))
            self.assertGreater(activity.peak, 4)
        finally:
            pool.close()

    def test_pool_validates_all_spawns_and_modes_before_dispatch(self):
        workers = [FakeWorker(0, 2), FakeWorker(1, 2)]
        pool = fake_pool(workers, 2)
        try:
            for arguments in (
                {"spawn_locations": [[1, 2, 3]]},
                {"spawn_locations": [[1, 2, 3], [4, float("nan"), 6]]},
                {"spawn_rotations": [[0, 0], [0, 90]]},
                {"reset_mode": "cached"},
            ):
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ValueError):
                        pool.reset_indices([0, 3], **arguments)
                    self.assertFalse(any(worker.reset_calls for worker in workers))
            pool.reset_indices(
                [3, 0], spawn_locations=[[300, 0, 35], [500, 20, 40]],
                spawn_rotations=[[0, 90, 0], [0, 180, 0]], reset_mode="rebuild",
            )
            np.testing.assert_array_equal(workers[0].reset_arguments["spawn_locations"], [[500, 20, 40]])
            np.testing.assert_array_equal(workers[1].reset_arguments["spawn_rotations"], [[0, 90, 0]])
            self.assertEqual(workers[0].reset_arguments["reset_mode"], "rebuild")
            self.assertEqual(pool.reset_mode, "auto")
        finally:
            pool.close()

    def test_partial_growth_metrics_distinguish_actual_and_configured(self):
        pool = fake_pool([FakeWorker(0, 3), FakeWorker(1, 2)], 2)
        try:
            with patch.object(module, "host_resources", return_value={"mock": True}):
                metrics = pool.metrics()
            self.assertEqual(metrics["num_envs"], 5)
            self.assertEqual(metrics["configured_num_envs"], 4)
            self.assertFalse(metrics["uniform_ready"])
            with self.assertRaisesRegex(RuntimeError, "incompletely initialized"):
                pool.step(np.zeros((4, 12)), np.zeros((4, 3)))
        finally:
            pool.close()

    def test_metrics_report_actual_reset_modes_and_capabilities(self):
        workers = [FakeWorker(0, 2), FakeWorker(1, 2)]
        workers[0].reset_result_counts = {
            "initial_model_build": 2, "rebuilt": 0, "reused": 3,
        }
        workers[1].reset_result_counts = {
            "initial_model_build": 2, "rebuilt": 1, "reused": 0,
        }
        for worker in workers:
            worker.reset_count = sum(worker.reset_result_counts.values())
        for robot in workers[0].robots:
            robot.reset_api_supported = True
        pool = fake_pool(workers, 2)
        try:
            with patch.object(module, "host_resources", return_value={}):
                metrics = pool.metrics()
            self.assertEqual(metrics["robot_resets"], 8)
            self.assertEqual(metrics["reset_result_counts"], {
                "initial_model_build": 4, "rebuilt": 1, "reused": 3,
            })
            self.assertEqual(metrics["reset_api_v2_capable_replicas"], 2)
            self.assertEqual(metrics["reset_mode"], "auto")
            self.assertEqual(metrics["reset_request_counts"], {"auto": 0, "rebuild": 0})
            self.assertEqual(metrics["processes"][0]["reset_result_counts"]["reused"], 3)
        finally:
            pool.close()

    def test_constructor_failure_preserves_evidence_and_cleans_workers(self):
        first = FakeWorker(0, 0)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(module, "UEWorker", side_effect=[first, RuntimeError("startup failure")]),
                patch.object(module.UEGo1Pool, "check_resources"),
                patch.object(module, "host_resources", return_value={"mock": True}),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup failure"):
                    module.UEGo1Pool(Path(sys.executable), 2, 1, 19000, Path(directory))
            report = json.loads((Path(directory) / "startup_failure.json").read_text())
            self.assertEqual(report["fully_initialized_workers"], 1)
            self.assertEqual(report["actual_robots"], 0)
            self.assertEqual(report["host_at_failure"], {"mock": True})
            self.assertTrue(first.closed)

    @unittest.skipUnless(hasattr(module.os, "killpg"), "Owned UE launch requires POSIX process groups; Windows uses attach")
    def test_worker_uses_private_tmpdir_without_changing_parent_environment(self):
        client = FakeClient()
        client.connect = Mock()
        client.isconnected = Mock(return_value=True)
        responses = {
            "vget /camera/0/location": "0 0 100",
            "vget /camera/0/rotation": "0 0 0",
            "vset /action/game/pause": "ok",
            "vget /action/game/is_paused": "true",
        }
        client.request = lambda command, timeout: responses[command]
        process = Mock(pid=99999901)
        process.poll.return_value = None
        original_tmpdir = module.os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(sys.modules, {"unrealcv": SimpleNamespace(Client=lambda endpoint: client)}),
                patch.object(module.socket, "socket"),
                patch.object(module.subprocess, "Popen", return_value=process) as launch,
                patch.object(module.subprocess, "run", return_value=SimpleNamespace(
                    stdout="LISTEN 127.0.0.1:19000 users:(pid=99999901,fd=1)"
                )),
                patch.object(module.os, "killpg"),
            ):
                worker = module.UEWorker(
                    Path("/mock/Project/Binaries/Linux/Game"), 19000,
                    Path(directory) / "worker", 1.0,
                )
                try:
                    child_env = launch.call_args.kwargs["env"]
                    self.assertEqual(child_env["TMPDIR"], str(worker.tmpdir))
                    self.assertTrue(worker.tmpdir.is_dir())
                    self.assertEqual(module.os.environ.get("TMPDIR"), original_tmpdir)
                finally:
                    worker.close()


class UEAttachTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.context = {
            "is_game_world": True, "has_player_controller": True,
            "world_name": "UEDPIE_0_Suburb", "world_id": 1234, "world_type": "PIE",
        }
        self.paused = False
        self.commands = []
        self.clients = []
        self.created_names = []
        self.fail_after_spawn = False
        self.fail_after_pause = False

        def client_factory(endpoint):
            client = FakeClient()
            client.connect = Mock()
            client.isconnected = Mock(return_value=True)
            client.request = Mock(side_effect=self.respond)
            self.clients.append(client)
            return client

        self.factory = Mock(side_effect=client_factory)
        self.unrealcv = patch.dict(sys.modules, {"unrealcv": SimpleNamespace(Client=self.factory)})
        self.unrealcv.start()
        self.addCleanup(self.unrealcv.stop)

    def respond(self, command, timeout):
        self.commands.append(command)
        if command == "vget /action/game/context":
            return json.dumps(self.context)
        if command == "vget /camera/0/location":
            return "0 0 100"
        if command == "vget /camera/0/rotation":
            return "0 0 0"
        if command == "vget /action/game/is_paused":
            if self.fail_after_pause and self.paused:
                self.fail_after_pause = False
                raise TimeoutError("pause acknowledgement lost")
            return str(self.paused).lower()
        if command == "vset /action/game/pause":
            self.paused = True
            return "ok"
        if command == "vset /action/game/resume":
            self.paused = False
            return "ok"
        if command.startswith("vset /objects/spawn_from_path"):
            self.created_names.append(command.split()[3])
            return self.created_names[-1]
        if self.fail_after_spawn and "/rotation " in command:
            return "error failed to initialize own actor"
        return "ok"

    def attach(self, suffix="worker"):
        return module.UEWorker.attach("127.0.0.1:9000", self.directory / suffix, 1.0)

    def test_external_editor_never_spawned_or_killed_and_pause_is_restored(self):
        for original in (False, True):
            with self.subTest(original=original), patch.object(module.subprocess, "Popen") as launch, patch.object(module.os, "killpg", create=True) as kill:
                self.paused = original
                worker = self.attach(str(original))
                self.assertIsNone(worker.proc)
                self.assertFalse(worker.owns_process)
                self.assertTrue(self.paused)
                worker.owned_actor_names = ["uzgym_owned_one", "uzgym_owned_two"]
                worker.close()
                worker.close()
                self.assertEqual(self.paused, original)
                self.assertFalse(worker.cleanup_errors)
                launch.assert_not_called()
                kill.assert_not_called()
        destroys = [command for command in self.commands if command.endswith("/destroy")]
        self.assertEqual(destroys, ["vset /object/uzgym_owned_one/destroy", "vset /object/uzgym_owned_two/destroy"] * 2)
        self.assertTrue(all(client.disconnected for client in self.clients))
        self.assertFalse(any("quit" in cmd or "stop_pie" in cmd for cmd in self.commands))

    def test_no_pie_is_rejected_before_pause_or_actor_mutation(self):
        self.context.update(is_game_world=False, has_player_controller=False, world_type="Editor")
        with self.assertRaisesRegex(RuntimeError, "No active playable PIE"):
            self.attach()
        self.assertEqual(self.commands, ["vget /action/game/context"])
        self.assertTrue(self.clients[0].disconnected)
        self.assertFalse(self.paused)

    def test_changed_world_is_not_mutated_during_cleanup(self):
        worker = self.attach()
        worker.owned_actor_names = ["uzgym_owned"]
        self.context["world_id"] = 555
        self.commands.clear()
        worker.close()
        self.assertEqual(self.commands, ["vget /action/game/context"])
        self.assertTrue(any("PIE world changed" in message for message in worker.cleanup_errors))

    def test_initialization_failure_after_pause_restores_state(self):
        self.fail_after_pause = True
        with self.assertRaisesRegex(TimeoutError, "acknowledgement lost"):
            self.attach()
        self.assertFalse(self.paused)
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(all(client.disconnected for client in self.clients))

    def test_actor_created_before_initialization_failure_is_still_cleaned(self):
        worker = self.attach()
        self.fail_after_spawn = True
        with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
            worker.add_robot()
        self.assertEqual(len(worker.owned_actor_names), 1)
        self.assertEqual(worker.owned_actor_names, self.created_names)
        self.assertFalse(worker.robots)
        worker.close()
        self.assertIn(f"vset /object/{self.created_names[0]}/destroy", self.commands)
        self.assertFalse(self.paused)

    def test_failed_worker_cleanup_uses_fresh_connection_not_stale_responses(self):
        worker = self.attach()
        worker.failed = True
        worker.client.request.side_effect = TimeoutError("connection failed")
        worker.close()
        self.assertEqual(len(self.clients), 2)
        self.assertFalse(self.paused)
        self.assertFalse(worker.cleanup_errors)

    def test_attach_metrics_leave_unmeasured_process_values_null(self):
        worker = self.attach()
        pool = fake_pool([worker])
        pool.connect = "127.0.0.1:9000"
        try:
            metrics = pool.metrics()
            self.assertEqual(metrics["connection_mode"], "attach")
            self.assertEqual(metrics["render_mode"], "existing_editor_settings")
            process = metrics["processes"][0]
            self.assertIsNone(process["pid"])
            self.assertIsNone(process["cpu_seconds"])
            self.assertIsNone(process["VmRSS"])
            self.assertIsNone(process["write_bytes"])
            self.assertFalse(process["owns_process"])
        finally:
            pool.close()

    def test_invalid_connect_and_multiple_connections_fail_before_opening_socket(self):
        for value in ("host:9000", "192.168.1.250:9000", "localhost", "localhost:0", "localhost:65536",
                      "localhost:-1", "localhost:NaN", "http://localhost:9000", " localhost:9000", "[::1]:9000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.UEGo1Pool(None, 1, 1, 9000, self.directory, connect=value)
        with self.assertRaisesRegex(ValueError, "exactly one connection"):
            module.UEGo1Pool(None, 2, 1, 9000, self.directory, connect="localhost:9000")
        self.factory.assert_not_called()

    def test_unavailable_memory_probe_is_null_not_zero(self):
        with patch.object(module.Path, "is_file", return_value=False), patch.dict(sys.modules, {"psutil": None}):
            resources = module.host_resources(self.directory)
            self.assertIsNone(resources["available_ram_bytes"])
            self.assertIsNone(resources["total_ram_bytes"])
            self.assertIn("unavailable", resources["memory_metrics_unavailable_reason"])
            self.assertGreater(resources["free_disk_bytes"], 0)


class UEKeyboardV2ResetTests(unittest.TestCase):
    def fixture(self, randomize=False, seed=42):
        from test_go1_fast_step import telemetry_metadata, train_capability

        fixture = UEWorkerResetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.enable_api()
        fixture.state.update({**telemetry_metadata(fixture.model), **train_capability()})
        fixture.state.update(foot_heights=[.023] * 4, foot_current_air_time=[0.] * 4,
                             foot_current_contact_time=[0.] * 4,
                             foot_contact_forces_world=[0.] * 12, root_quat_wxyz=[1., 0, 0, 0],
                             runtime_diagnostics_enabled=False)
        fixture.state["control_targets"] = fixture.state["training_default_joint_positions"].copy()
        fixture.worker.training_telemetry = "keyboard_v2"
        fixture.worker.randomize_reset_pose = randomize
        fixture.worker.reset_pose_rng = np.random.default_rng(seed)
        fixture.worker.step_mode = "fast"

        def respond(command, timeout):
            if "/mujoco_go1_training_telemetry " in command:
                return json.dumps({**fixture.state, **fixture.forced_changes})
            if "/mujoco_go1_policy_reset_pose " in command:
                # The server restores the cached episode itself, while keeping
                # its preceding last_reset_result enum.
                fixture.state["sim_time"] = 0.0
                fixture.state["obs"][33:48] = [0.] * 15
                fixture.state["foot_current_air_time"] = [0.] * 4
                fixture.state["foot_current_contact_time"] = [0.] * 4
                fixture.state["training_reset_pose_offset"] = list(map(float, command.split()[2:]))
                return json.dumps({**fixture.state, **fixture.forced_changes})
            return fixture.respond(command, timeout)

        fixture.worker.client.request.side_effect = respond
        return fixture

    def test_profile_selected_before_model_build_and_metadata_survives_reused_reset(self):
        fixture = self.fixture()
        state = fixture.worker.reset_one(0, initial=True)
        commands = fixture.commands()
        self.assertLess(commands.index("vset /object/robot/mujoco_go1_training_profile keyboard_v2"),
                        commands.index("vset /object/robot/mujoco_quadruped_pose_preview/start go1"))
        self.assertEqual(state["control_targets"], [.1, .9, -1.8, -.1, .9, -1.8] * 2)
        old_hash = fixture.worker.robots[0].asset_contract["runtime_mjcf_sha256"]
        fixture.model.unlink()  # A reusable reset must not reload terrain/XML.
        fixture.worker.client.request.reset_mock()
        fixture.worker.reset_one(0)
        self.assertEqual(len(fixture.commands()), 1)
        self.assertIn("policy_reset auto", fixture.commands()[0])
        robot = fixture.worker.robots[0]
        self.assertEqual(robot.reset_metadata["soft_joint_pos_limits"], state["soft_joint_pos_limits"])
        self.assertEqual(robot.asset_contract["runtime_mjcf_sha256"], old_hash)
        self.assertEqual(robot.asset_contract["training_physics"]["joint_frictionloss"], [0.] * 12)

    def test_missing_or_changed_v2_contract_never_falls_back(self):
        for key, value in (("training_telemetry_mode", "disabled"), ("foot_heights", [0.] * 3),
                           ("foot_current_air_time", [.005] * 4),
                           ("training_physics_profile", "legacy"),
                           ("joint_frictionloss", [.3] * 12),
                           ("actuator_ctrl_limited", [True] * 12),
                           ("training_default_root_height", .27),
                           ("training_model_path", "/wrong/runtime.xml")):
            fixture = self.fixture()
            fixture.forced_changes = {key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                fixture.worker.reset_one(0, initial=True)
            self.assertTrue(fixture.worker.failed)
            self.assertEqual(fixture.worker.robots[0].state, {})
            self.assertEqual(fixture.worker.reset_count, 0)

    def test_random_root_pose_is_seeded_relative_to_cached_spawn_without_terrain_rebuild(self):
        fixtures = [self.fixture(randomize=True, seed=123) for _ in range(2)]
        sequences = []
        for fixture in fixtures:
            offsets = []
            for initial in (True, False, False):
                state = fixture.worker.reset_one(0, initial=initial)
                offsets.append(state["training_reset_pose_offset"])
                self.assertEqual(state["reset_spawn_location"], [300., 0., 35.])
                self.assertEqual(state["last_reset_result"], "initial_model_build" if initial else "reused")
                np.testing.assert_array_equal(fixture.worker.robots[0].location, [300., 0., 35.])
            sequences.append(np.array(offsets))
            self.assertEqual(sum("preview/start" in command for command in fixture.commands()), 1)
            self.assertEqual(sum("policy_reset_pose " in command for command in fixture.commands()), 3)
            self.assertEqual(sum("policy_reset auto " in command for command in fixture.commands()), 1)
        np.testing.assert_array_equal(*sequences)
        self.assertTrue(np.all(sequences[0] >= [-.5, -.5, .01, -np.pi]))
        self.assertTrue(np.all(sequences[0] <= [.5, .5, .05, np.pi]))
        self.assertFalse(np.array_equal(sequences[0][0], sequences[0][1]))

    def test_verified_cached_random_reset_uses_one_rpc_and_validates_fresh_state(self):
        fixture = self.fixture(randomize=True)
        fixture.worker.reset_one(0, initial=True)
        fixture.worker.client.request.reset_mock()
        fixture.worker.reset_one(0)
        self.assertEqual(len(fixture.commands()), 2)  # Replace initial_model_build with reused.
        robot = fixture.worker.robots[0]
        old_hash = robot.asset_contract["runtime_mjcf_sha256"]
        fixture.model.unlink()  # A reused pose reset cannot reload terrain/XML.
        fixture.state["sim_time"] = 2.0
        fixture.state["obs"][33:48] = [1.] * 15
        fixture.state["foot_current_air_time"] = [2.] * 4
        robot.command = np.ones(3)
        fixture.worker.client.request.reset_mock()
        state = fixture.worker.reset_one(0)
        self.assertEqual(len(fixture.commands()), 1)
        self.assertIn("mujoco_go1_policy_reset_pose ", fixture.commands()[0])
        self.assertEqual(state["sim_time"], 0.)
        self.assertEqual(state["last_reset_result"], "reused")
        np.testing.assert_array_equal(state["obs"][33:48], np.zeros(15))
        np.testing.assert_array_equal(state["foot_current_air_time"], np.zeros(4))
        np.testing.assert_array_equal(robot.command, np.zeros(3))
        self.assertEqual(robot.asset_contract["runtime_mjcf_sha256"], old_hash)
        self.assertEqual(robot.reset_metadata["training_reset_pose_offset"], state["training_reset_pose_offset"])
        self.assertEqual(robot.reset_metadata["soft_joint_pos_limits"], state["soft_joint_pos_limits"])
        self.assertEqual(fixture.worker.reset_result_counts,
                         {"initial_model_build": 1, "reused": 2, "rebuilt": 0})
        self.assertEqual(fixture.worker.reset_request_counts, {"auto": 2, "rebuild": 0})

    def test_random_reset_changed_spawn_or_rebuild_keeps_explicit_reset_and_fresh_asset(self):
        for kwargs in ({"spawn_location": [400., 100., 45.]},
                       {"spawn_rotation": [0., 90., 0.]}, {"reset_mode": "rebuild"}):
            with self.subTest(kwargs=kwargs):
                fixture = self.fixture(randomize=True)
                fixture.worker.reset_one(0, initial=True)
                fixture.worker.reset_one(0)
                old_hash = fixture.worker.robots[0].asset_contract["runtime_mjcf_sha256"]
                fixture.model.write_text(fixture.model.read_text().replace(
                    "<worldbody>", "<worldbody><!-- changed terrain -->"))
                fixture.worker.client.request.reset_mock()
                state = fixture.worker.reset_one(0, **kwargs)
                self.assertEqual(len(fixture.commands()), 2)
                self.assertIn("mujoco_go1_policy_reset " + kwargs.get("reset_mode", "auto"),
                              fixture.commands()[0])
                self.assertIn("mujoco_go1_policy_reset_pose ", fixture.commands()[1])
                self.assertEqual(state["last_reset_result"], "rebuilt")
                self.assertNotEqual(fixture.worker.robots[0].asset_contract["runtime_mjcf_sha256"], old_hash)
                # A rebuilt result also needs one ordinary auto reset before
                # the pose-only path can report the preserved reused result.
                for count in (2, 1):
                    fixture.worker.client.request.reset_mock()
                    state = fixture.worker.reset_one(0)
                    self.assertEqual(len(fixture.commands()), count)
                    self.assertEqual(state["last_reset_result"], "reused")

    def test_pose_only_reset_rejects_bad_state_or_coverage_without_retry_or_commit(self):
        failures = [
            {"training_reset_pose_offset": [0.] * 4},
            {"sim_time": .02},
            {"foot_current_air_time": [.005] * 4},
            {"runtime_diagnostics_enabled": True},
            {"training_physics_profile": "legacy"},
            {"reset_spawn_location": [999., 0., 35.]},
            {"last_reset_result": "initial_model_build"},
            "error requested root pose exceeds cached terrain coverage",
            TimeoutError("late pose reset reply"),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                fixture = self.fixture(randomize=True)
                fixture.worker.reset_one(0, initial=True)
                fixture.worker.reset_one(0)
                robot = fixture.worker.robots[0]
                old_state, old_asset = robot.state, robot.asset_contract
                old_counts = fixture.worker.reset_result_counts.copy()
                fixture.worker.client.request.reset_mock()
                if isinstance(failure, dict):
                    fixture.forced_changes = failure
                elif isinstance(failure, BaseException):
                    fixture.worker.client.request.side_effect = failure
                else:
                    fixture.worker.client.request.side_effect = None
                    fixture.worker.client.request.return_value = failure
                with self.assertRaises((RuntimeError, TimeoutError)):
                    fixture.worker.reset_one(0)
                self.assertEqual(len(fixture.commands()), 1)
                self.assertIn("mujoco_go1_policy_reset_pose ", fixture.commands()[0])
                self.assertTrue(fixture.worker.failed)
                self.assertIs(robot.state, old_state)
                self.assertIs(robot.asset_contract, old_asset)
                self.assertEqual(fixture.worker.reset_result_counts, old_counts)
                self.assertEqual(fixture.worker.reset_count, 2)

    def test_wrong_pose_ack_and_coverage_error_do_not_retry_or_commit(self):
        for failure in ("ack", "coverage"):
            fixture = self.fixture(randomize=True)
            original = fixture.worker.client.request.side_effect
            if failure == "ack":
                fixture.forced_changes["training_reset_pose_offset"] = [0.] * 4
            else:
                def reject(command, timeout):
                    if "policy_reset_pose " in command:
                        return "error requested root pose exceeds cached terrain coverage"
                    return original(command, timeout)
                fixture.worker.client.request.side_effect = reject
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                fixture.worker.reset_one(0, initial=True)
            self.assertTrue(fixture.worker.failed)
            self.assertEqual(fixture.worker.reset_count, 0)
            self.assertEqual(sum("policy_reset_pose " in command for command in fixture.commands()), 1)
            self.assertEqual(fixture.worker.robots[0].state, {})

    def test_pool_rejects_invalid_telemetry_randomization_before_creating_workers(self):
        for kwargs in ({"training_telemetry": "auto"}, {"randomize_reset_pose": True},
                       {"training_telemetry": "keyboard_v2", "randomize_reset_pose": 1},
                       {"seed": -1}, {"seed": True}):
            with patch.object(module, "UEWorker") as worker, self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.UEGo1Pool(None, 1, 1, 9000, Path("/tmp/not-created"), connect="127.0.0.1:9000", **kwargs)
            worker.assert_not_called()


class UEWorkerResetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.model = directory / "robot.xml"
        self.model.write_text('<mujoco><worldbody><site name="imu" '
                              'pos="-0.01592 -0.06659 -0.00617"/></worldbody></mujoco>')
        self.state = {
            "obs": np.zeros(48).tolist(), "sim_time": 0.0,
            "policy_profile": "velocity", "synchronous": True,
            "control_targets": [0.0, 0.9, -1.8] * 4,
            "environment_collision_report_path": str(directory / "robot_collision.csv"),
            "environment_geom_count": 2049,
        }
        self.worker = module.UEWorker.__new__(module.UEWorker)
        self.worker.directory = directory
        self.worker.robots = [module.Robot("robot", np.array([300., 0., 35.]), np.zeros(3))]
        self.worker.closed = self.worker.failed = False
        self.worker.timeout = 1.0
        self.worker.reset_mode = "auto"
        self.worker.request_seconds = self.worker.reset_seconds = 0.0
        self.worker.reset_count = 0
        self.worker.reset_result_counts = dict.fromkeys(module.RESET_RESULTS, 0)
        self.worker.reset_request_counts = dict.fromkeys(module.RESET_MODES, 0)
        self.worker.client = SimpleNamespace(request=Mock(side_effect=self.respond))
        self.forced_changes = {}

    def respond(self, command, timeout):
        if "/mujoco_go1_diagnostics " in command:
            self.state["runtime_diagnostics_enabled"] = command.endswith(" enabled")
            return json.dumps({**self.state, **self.forced_changes})
        if "/mujoco_go1_policy_reset " in command:
            parts = command.split()
            location = list(map(float, parts[3:6]))
            rotation = list(map(float, parts[6:9]))
            reuse = (parts[2] == "auto" and location == self.state["reset_spawn_location"]
                     and rotation == self.state["reset_spawn_rotation"])
            self.state.update(
                last_reset_result="reused" if reuse else "rebuilt",
                reset_spawn_location=location, reset_spawn_rotation=rotation,
            )
            return json.dumps({**self.state, **self.forced_changes})
        if command.endswith("/mujoco_go1_policy_sync/start"):
            return json.dumps({**self.state, **self.forced_changes})
        return "ok"

    def enable_api(self):
        self.state.update(
            reset_api_version=2, reset_modes=["auto", "rebuild"],
            last_reset_result="initial_model_build",
            reset_spawn_location=[300., 0., 35.], reset_spawn_rotation=[0., 0., 0.],
        )

    def commands(self):
        return [call.args[0] for call in self.worker.client.request.call_args_list]

    def test_capable_server_explicitly_sets_diagnostics_for_each_requested_mode(self):
        self.enable_api()
        for requested in (False, True):
            with self.subTest(requested=requested):
                self.state["runtime_diagnostics_enabled"] = True
                self.worker.runtime_diagnostics = requested
                self.worker.client.request.reset_mock()
                state = self.worker.reset_one(0, initial=True)
                setting = "enabled" if requested else "disabled"
                self.assertIn(f"vset /object/robot/mujoco_go1_diagnostics {setting}", self.commands())
                self.assertIs(state["runtime_diagnostics_enabled"], requested)
                self.assertIs(self.worker.robots[0].runtime_diagnostics_enabled, requested)

    def test_legacy_diagnostics_does_not_send_unsupported_rpc_for_either_request(self):
        for requested in (False, True):
            self.worker.runtime_diagnostics = requested
            self.worker.client.request.reset_mock()
            self.worker.reset_one(0, initial=True)
            self.assertFalse(any("mujoco_go1_diagnostics" in command for command in self.commands()))
            self.assertIsNone(self.worker.robots[0].runtime_diagnostics_enabled)
            summary = module.diagnostics_summary(self.worker.robots, requested)
            self.assertEqual(summary["unavailable_replicas"], 1)
            self.assertFalse(summary["request_verified_for_all_initialized"])

    def test_rebuild_retains_diagnostics_or_explicitly_reapplies_setting(self):
        self.enable_api()
        self.state["runtime_diagnostics_enabled"] = True
        self.worker.reset_one(0, initial=True)
        self.worker.client.request.reset_mock()
        self.worker.reset_one(0, reset_mode="rebuild")
        self.assertFalse(any("mujoco_go1_diagnostics" in command for command in self.commands()))
        self.state["runtime_diagnostics_enabled"] = True
        self.worker.client.request.reset_mock()
        state = self.worker.reset_one(0, reset_mode="rebuild")
        self.assertIn("vset /object/robot/mujoco_go1_diagnostics disabled", self.commands())
        self.assertFalse(state["runtime_diagnostics_enabled"])

    def test_diagnostics_boolean_schema_wrong_ack_and_lost_capability_fail_closed(self):
        for invalid in ("false", 0, None):
            with self.subTest(invalid=invalid):
                self.worker.failed = False
                self.state["runtime_diagnostics_enabled"] = invalid
                with self.assertRaisesRegex(RuntimeError, "must be a Boolean"):
                    self.worker.reset_one(0, initial=True)
                self.assertTrue(self.worker.failed)
                self.assertIsNone(self.worker.robots[0].runtime_diagnostics_enabled)
        self.worker.failed = False
        self.state["runtime_diagnostics_enabled"] = True
        self.forced_changes = {"runtime_diagnostics_enabled": True}
        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            self.worker.reset_one(0, initial=True)
        self.assertTrue(self.worker.failed)
        self.worker.failed = False
        self.forced_changes = {}
        self.worker.reset_one(0, initial=True)
        self.state.pop("runtime_diagnostics_enabled")
        with self.assertRaisesRegex(RuntimeError, "lost its runtime diagnostics capability"):
            self.worker.reset_one(0)
        self.assertTrue(self.worker.failed)

    def test_sampling_diagnostics_drift_fails_without_committing_wrong_state(self):
        self.state["runtime_diagnostics_enabled"] = True
        self.worker.reset_one(0, initial=True)
        previous = self.worker.robots[0].state
        changed = {**previous, "sim_time": 0.02, "runtime_diagnostics_enabled": True}
        self.worker.request_many = Mock(side_effect=[[], [json.dumps(changed)]])
        with self.assertRaisesRegex(RuntimeError, "changed during sampling"):
            self.worker.step(np.zeros((1, 12)), np.zeros((1, 3)))
        self.assertTrue(self.worker.failed)
        self.assertIs(self.worker.robots[0].state, previous)

    def test_same_spawn_auto_reuses_without_observing_moved_actor_pose(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        robot = self.worker.robots[0]
        self.assertTrue(robot.reset_api_supported)
        robot.command = np.ones(3)
        robot.state = {**robot.state, "sim_time": 2.0, "location": [999., 900., 40.]}
        self.worker.client.request.reset_mock()
        state = self.worker.reset_one(0)
        self.assertEqual(self.commands(), [
            "vset /object/robot/mujoco_go1_policy_reset auto "
            "300.000000000 0.000000000 35.000000000 0.000000000 0.000000000 0.000000000"
        ])
        self.assertEqual(state["sim_time"], 0.0)
        np.testing.assert_array_equal(robot.command, np.zeros(3))
        self.assertEqual(self.worker.reset_result_counts, {
            "initial_model_build": 1, "rebuilt": 0, "reused": 1,
        })
        self.assertEqual(self.worker.reset_request_counts, {"auto": 1, "rebuild": 0})

    def test_new_spawn_auto_rebuilds_then_reuses_new_saved_spawn(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        state = self.worker.reset_one(0, spawn_location=[400., 100., 45.], spawn_rotation=[0., 90., 0.])
        self.assertEqual(state["last_reset_result"], "rebuilt")
        self.worker.client.request.reset_mock()
        state = self.worker.reset_one(0)
        self.assertEqual(state["last_reset_result"], "reused")
        self.assertIn("400.000000000 100.000000000 45.000000000 0.000000000 90.000000000", self.commands()[0])
        np.testing.assert_array_equal(self.worker.robots[0].location, [400, 100, 45])
        np.testing.assert_array_equal(self.worker.robots[0].rotation, [0, 90, 0])

    def test_force_rebuild_refreshes_same_path_runtime_asset_hash(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        previous = self.worker.robots[0].asset_contract
        self.model.write_text(self.model.read_text().replace("<worldbody>", "<worldbody><!-- rebuilt terrain -->"))
        state = self.worker.reset_one(0, reset_mode="rebuild")
        contract = self.worker.robots[0].asset_contract
        self.assertEqual(state["last_reset_result"], "rebuilt")
        self.assertNotEqual(contract["runtime_mjcf_sha256"], previous["runtime_mjcf_sha256"])
        self.assertEqual(contract["runtime_mjcf"], previous["runtime_mjcf"])
        self.assertEqual(self.worker.reset_request_counts, {"auto": 0, "rebuild": 1})

    def test_ue_equivalent_euler_rotation_and_numeric_api_version_are_accepted(self):
        self.enable_api()
        self.state["reset_api_version"] = 2.0
        self.worker.reset_one(0, initial=True)
        self.forced_changes = {"reset_spawn_rotation": [0, -90, 180]}
        self.worker.reset_one(0, spawn_rotation=[180, 90, 0])
        np.testing.assert_array_equal(self.worker.robots[0].rotation, [0, -90, 180])

    def test_reused_reset_does_not_reread_or_recompile_runtime_model(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        previous = self.worker.robots[0].asset_contract
        self.model.unlink()
        self.worker.reset_one(0)
        self.assertIs(previous, self.worker.robots[0].asset_contract)

    def test_old_server_and_v1_cache_capability_use_rebuild_for_both_modes(self):
        self.state.update(cached_reset_supported=True, cached_reset_scope="fixed_spawn_static_scene")
        self.worker.reset_one(0, initial=True)
        for mode in module.RESET_MODES:
            with self.subTest(mode=mode):
                self.worker.client.request.reset_mock()
                self.worker.reset_one(0, reset_mode=mode, spawn_location=[450., 0., 35.])
                self.assertEqual(self.commands(), [
                    "vset /object/robot/mujoco_quadruped_pose_preview/stop",
                    "vset /object/robot/location 450.000000000 0.000000000 35.000000000",
                    "vset /object/robot/rotation 0.000000000 0.000000000 0.000000000",
                    "vset /object/robot/mujoco_quadruped_pose_preview/start go1",
                    "vset /object/robot/mujoco_go1_policy_command 0 0 0",
                    "vset /object/robot/mujoco_go1_policy_sync/start",
                ])
        self.assertEqual(self.worker.reset_result_counts["rebuilt"], 2)
        self.assertEqual(self.worker.reset_result_counts["reused"], 0)
        np.testing.assert_array_equal(self.worker.robots[0].location, [450, 0, 35])

    def test_legacy_rebuild_also_refreshes_runtime_asset_contract(self):
        self.worker.reset_one(0, initial=True)
        previous = self.worker.robots[0].asset_contract
        self.model.write_text(self.model.read_text().replace("<worldbody>", "<worldbody><!-- new spawn -->"))
        self.worker.reset_one(0, spawn_location=[500, 0, 35])
        self.assertNotEqual(previous["runtime_mjcf_sha256"], self.worker.robots[0].asset_contract["runtime_mjcf_sha256"])

    def test_invalid_v2_modes_are_rejected_during_negotiation(self):
        self.enable_api()
        self.state["reset_modes"] = ["auto", "cached"]
        with self.assertRaisesRegex(RuntimeError, "Invalid reset API modes"):
            self.worker.reset_one(0, initial=True)
        self.assertTrue(self.worker.failed)
        self.assertEqual(self.worker.reset_count, 0)

    def test_explicit_batch_requires_capability_during_initialization(self):
        self.worker.step_mode = "batch_parallel"
        with self.assertRaisesRegex(RuntimeError, "requires UE batch API"):
            self.worker.reset_one(0, initial=True)
        self.assertTrue(self.worker.failed)
        self.assertFalse(self.worker.robots[0].state)

    def test_batch_capability_is_negotiated_and_cannot_disappear_at_reset(self):
        self.enable_api()
        self.state.update(policy_step_batch_api_version=1, policy_step_batch_modes=["parallel", "serial"])
        self.worker.reset_one(0, initial=True)
        self.assertEqual(self.worker.robots[0].batch_step_modes, ("serial", "parallel"))
        self.state.pop("policy_step_batch_api_version")
        self.state.pop("policy_step_batch_modes")
        with self.assertRaisesRegex(RuntimeError, "lost its batch step capability"):
            self.worker.reset_one(0)
        self.assertTrue(self.worker.failed)

    def test_error_or_timeout_never_falls_back_or_commits_new_spawn(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        robot = self.worker.robots[0]
        previous_state = robot.state
        previous_contract = robot.asset_contract
        robot.command = np.ones(3)
        for failure in ("error cache invalid", "unknown command", None, TimeoutError("late reply")):
            with self.subTest(failure=failure):
                self.worker.failed = False
                self.worker.client.request.reset_mock()
                if isinstance(failure, BaseException):
                    self.worker.client.request.side_effect = failure
                else:
                    self.worker.client.request.side_effect = None
                    self.worker.client.request.return_value = failure
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.worker.reset_one(0, spawn_location=[500., 0., 35.])
                self.assertEqual(len(self.commands()), 1)
                self.assertIn("mujoco_go1_policy_reset auto 500.", self.commands()[0])
                self.assertTrue(self.worker.failed)
                self.assertIs(robot.state, previous_state)
                self.assertIs(robot.asset_contract, previous_contract)
                np.testing.assert_array_equal(robot.command, np.ones(3))
                np.testing.assert_array_equal(robot.location, [300, 0, 35])
                self.assertEqual(self.worker.reset_count, 1)
                self.assertEqual(self.worker.reset_result_counts["reused"], 0)
                with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                    self.worker.request("must not consume a late reply")

    def test_validates_physics_reference_and_spawn_before_commit(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        previous_state = self.worker.robots[0].state
        cases = [
            ({"sim_time": 0.02}, "Reset advanced physics"),
            ({"synchronous": False}, "synchronous Go1 velocity"),
            ({"policy_profile": "parkour"}, "synchronous Go1 velocity"),
            ({"control_targets": [0.1, 0.9, -1.8] * 4}, "initial joint reference"),
            ({"control_targets": [float("nan")] * 12}, "Invalid reset joint reference"),
            ({"obs": [0.] * 33 + [1.] * 12 + [0.] * 3}, "previous action and command"),
            ({"obs": [0.] * 45 + [1.] * 3}, "previous action and command"),
            ({"reset_api_version": None}, "no longer advertises"),
            ({"reset_spawn_location": [999, 0, 35]}, "does not match"),
            ({"reset_spawn_rotation": [0, 90, 0]}, "does not match"),
            ({"reset_spawn_rotation": [0, float("nan"), 0]}, "Invalid reset spawn"),
            ({"last_reset_result": "initial_model_build"}, "Unexpected UE reset result"),
        ]
        for changes, message in cases:
            with self.subTest(message=message):
                self.worker.failed = False
                self.forced_changes = changes
                with self.assertRaisesRegex(RuntimeError, message):
                    self.worker.reset_one(0)
                self.assertIs(self.worker.robots[0].state, previous_state)
                self.assertEqual(self.worker.reset_count, 1)
                self.assertTrue(self.worker.failed)

    def test_forced_rebuild_cannot_be_silently_reused(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        self.forced_changes = {"last_reset_result": "reused"}
        with self.assertRaisesRegex(RuntimeError, "Unexpected UE reset result"):
            self.worker.reset_one(0, reset_mode="rebuild")
        self.assertTrue(self.worker.failed)

    def test_rebuild_revalidates_changed_runtime_imu_before_commit(self):
        self.enable_api()
        self.worker.reset_one(0, initial=True)
        previous = self.worker.robots[0].asset_contract
        self.model.write_text(self.model.read_text().replace("-0.01592", "0.10000"))
        with self.assertRaisesRegex(RuntimeError, "IMU offset does not match"):
            self.worker.reset_one(0, reset_mode="rebuild", spawn_location=[500, 0, 35])
        self.assertIs(previous, self.worker.robots[0].asset_contract)
        np.testing.assert_array_equal(self.worker.robots[0].location, [300, 0, 35])

    def test_attach_reads_its_actual_saved_model_outside_worker_output(self):
        self.worker.owns_process = False
        with tempfile.TemporaryDirectory() as editor:
            saved = Path(editor) / "Saved" / "UnrealCV_MuJoCo" / "MJCF"
            saved.mkdir(parents=True)
            model = saved / "go1_runtime_000_robot.xml"
            model.write_bytes(self.model.read_bytes())
            self.state["environment_collision_report_path"] = str(model.with_name("go1_runtime_000_robot_collision.csv"))
            self.worker.reset_one(0, initial=True)
            self.assertEqual(self.worker.robots[0].asset_contract["runtime_mjcf"], str(model))
            self.assertTrue(self.worker.robots[0].asset_contract["runtime_mjcf_sha256"])

    def test_attach_rejects_another_actors_runtime_model(self):
        self.worker.owns_process = False
        self.state["environment_collision_report_path"] = str(self.worker.directory / "go1_runtime_000_other_collision.csv")
        with self.assertRaisesRegex(RuntimeError, "this actor's local"):
            self.worker.reset_one(0, initial=True)
        self.assertTrue(self.worker.failed)

    def test_worker_argument_errors_do_not_send_rpc_or_fail_connection(self):
        for args in ({"index": 1}, {"index": True}, {"index": 0, "reset_mode": "cached"},
                     {"index": 0, "spawn_location": [1, 2]},
                     {"index": 0, "spawn_rotation": [0, float("inf"), 0]}):
            with self.subTest(args=args):
                with self.assertRaises((ValueError, IndexError)):
                    self.worker.reset_one(**args)
                self.assertFalse(self.commands())
                self.assertFalse(self.worker.failed)


if __name__ == "__main__":
    unittest.main()
