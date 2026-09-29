"""Binary training batches preserve task semantics and never replay bad steps."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
import ue_go1_env as module
from ue_go1_task import Go1Task
import test_go1_batch_step as batch_tests
import test_ue_go1_env as env_tests


def train_capability():
    return {"policy_step_train_api_version": 1, "policy_step_train_modes": ["serial", "parallel"],
            "policy_step_train_columns": 89, "policy_step_train_dtype": "<f8"}


def train_v2_capability():
    return {"policy_step_train_v2_api_version": 2,
            "policy_step_train_v2_modes": ["serial", "parallel"],
            "policy_step_train_v2_columns": 117, "policy_step_train_v2_dtype": "<f8"}


def telemetry_metadata(model_path="/tmp/runtime.xml"):
    hard = np.array([[-.863, .863], [-.686, 4.501], [-2.818, -.888]] * 4)
    mid, half = hard.mean(axis=1), np.diff(hard, axis=1)[:, 0] * .45
    return {**train_v2_capability(), "training_telemetry_mode": "keyboard_v2",
            "training_telemetry_version": 2, "training_physics_profile": "keyboard_v2",
            "training_model_path": str(model_path),
            "soft_joint_pos_limits": np.column_stack((mid - half, mid + half)).ravel().tolist(),
            "joint_hard_limits": hard.ravel().tolist(),
            "actuator_stiffness": [15.8952426532, 15.8952426532, 35.7642959698] * 4,
            "actuator_damping": [1.0119225760, 1.0119225760, 2.2768257960] * 4,
            "actuator_force_limits": [-23.7, 23.7, -23.7, 23.7, -35.55, 35.55] * 4,
            "actuator_ctrl_limited": [False] * 12,
            "actuator_control_ranges": hard.ravel().tolist(),
            "joint_damping": [0.] * 12, "joint_frictionloss": [0.] * 12,
            "joint_armature": [.004026312, .004026312, .009059202] * 4,
            "training_default_joint_positions": [.1, .9, -1.8, -.1, .9, -1.8] * 2,
            "training_default_root_height": .278}


def packet(header, rows):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    magic = module.TRAIN_V2_MAGIC if header.get("api_version") == 2 else module.TRAIN_MAGIC
    return module.TRAIN_PREFIX.pack(magic, len(encoded), *rows.shape) + encoded + rows.astype("<f8").tobytes()


class FastStepTests(unittest.TestCase):
    def make_worker(self):
        worker, actions, commands = batch_tests.BatchStepTests().make_worker("fast")
        for robot in worker.robots:
            robot.state.update(train_capability(), runtime_diagnostics_enabled=False, sim_time=0.0)
            robot.train_step_modes = ("serial", "parallel")
            robot.runtime_diagnostics_enabled = False
        worker.client = Mock()
        worker.timeout = 1
        worker.request_seconds = 0.0
        worker.request = module.UEWorker.request.__get__(worker)
        full = batch_tests.BatchStepTests().response(worker, actions, commands)
        rows = np.asarray([
            [entry["state"]["sim_time"], *entry["state"]["obs"], *entry["state"]["control_targets"],
             *entry["state"]["foot_positions"], *entry["state"]["foot_velocities"], *entry["state"]["foot_contacts"]]
            for entry in full["robots"]
        ], dtype="<f8")
        header = {"api_version": 1, "mode": "parallel", "pose_sync": False,
                  "runtime_diagnostics_enabled": False, "actors": [robot.name for robot in worker.robots],
                  "timings_ms": dict.fromkeys(module.BATCH_TIMING_FIELDS, 0.1)}
        return worker, actions, commands, header, rows, [entry["state"] for entry in full["robots"]]

    def test_binary_step_uses_array_views_and_preserves_task_rewards(self):
        worker, actions, commands, header, rows, full_states = self.make_worker()
        initial = copy.deepcopy([robot.state for robot in worker.robots])
        # Exercise a header whose binary row offset is not float64-aligned.
        while (module.TRAIN_PREFIX.size + len(json.dumps(header, separators=(",", ":")).encode())) % 8 == 0:
            header["padding_test"] = str(header.get("padding_test", "")) + "x"
        worker.client.request.return_value = packet(header, rows)
        with patch.object(worker, "_validate_step_state", side_effect=AssertionError("No per-row full-state validator")):
            states = worker.step(actions, commands)
        sent = worker.client.request.call_args.args[0]
        prefix = "vset /mujoco/go1/policy_step_train parallel "
        self.assertTrue(sent.startswith(prefix))
        payload = json.loads(sent[len(prefix):])
        self.assertIs(payload["pose_sync"], False)
        self.assertEqual([entry["actor"] for entry in payload["robots"]], header["actors"])
        self.assertEqual(payload["robots"][1]["actions"], actions[1].tolist())
        worker.client.request.assert_called_once()
        worker.request_many.assert_not_called()
        self.assertEqual(worker.step_mode_counts["fast"], 1)
        for compact, full in zip(states, full_states):
            self.assertFalse(compact["actor_pose_synchronized"])
            for key in ("obs", "control_targets", "foot_positions", "foot_velocities", "foot_contacts"):
                np.testing.assert_array_equal(compact[key], full[key])
                self.assertFalse(compact[key].flags.writeable)
            self.assertEqual(compact["sim_time"], full["sim_time"])
        expected, actual = Go1Task(2, seed=42), Go1Task(2, seed=42)
        for task in (expected, actual):
            task.reset([0, 1], initial)
            task.set_commands(commands)
        left = expected.step(full_states, actions)
        right = actual.step(states, actions)
        for key in ("actor", "critic"):
            np.testing.assert_array_equal(left[0][key], right[0][key])
        for index in (1, 2, 3):
            np.testing.assert_array_equal(left[index], right[index])
        for key in left[4]["reward_terms"]:
            np.testing.assert_array_equal(left[4]["reward_terms"][key], right[4]["reward_terms"][key])
        timing = module.batch_timing_summary([worker])["by_mode"]["fast"]
        self.assertEqual(timing["measured_responses"], 1)

    def test_invalid_packets_never_commit_or_replay(self):
        failures = ("none", "error", "magic", "prefix", "rows", "columns", "header_length", "truncated", "extra",
                    "version", "mode", "pose_sync", "diagnostics", "identity", "timings", "header_nan",
                    "time", "obs_nan", "target_inf", "position_nan", "velocity_inf", "contacts", "action", "command")
        for failure in failures:
            worker, actions, commands, header, rows, _ = self.make_worker()
            if failure == "version": header["api_version"] = True
            elif failure == "mode": header["mode"] = "serial"
            elif failure == "pose_sync": header["pose_sync"] = True
            elif failure == "diagnostics": header["runtime_diagnostics_enabled"] = True
            elif failure == "identity": header["actors"].reverse()
            elif failure == "timings": header["timings_ms"]["physics"] = -1
            elif failure == "header_nan": header["unused"] = float("nan")
            elif failure == "time": rows[-1, 0] = .04
            elif failure == "obs_nan": rows[-1, 2] = np.nan
            elif failure == "target_inf": rows[-1, 50] = np.inf
            elif failure == "position_nan": rows[-1, 65] = np.nan
            elif failure == "velocity_inf": rows[-1, 75] = -np.inf
            elif failure == "contacts": rows[-1, -1] = 2
            elif failure == "action": rows[-1, 34] += 1
            elif failure == "command": rows[-1, 46] += 1
            raw = packet(header, rows)
            if failure == "none": raw = None
            elif failure == "error": raw = "error unavailable"
            elif failure == "magic": raw = b"INVALID!" + raw[8:]
            elif failure == "prefix": raw = module.TRAIN_MAGIC
            elif failure in ("rows", "columns", "header_length"):
                _, length, count, columns = module.TRAIN_PREFIX.unpack_from(raw)
                raw = module.TRAIN_PREFIX.pack(module.TRAIN_MAGIC,
                    2 ** 32 - 1 if failure == "header_length" else length,
                    count + 1 if failure == "rows" else count,
                    columns - 1 if failure == "columns" else columns) + raw[module.TRAIN_PREFIX.size:]
            elif failure == "truncated": raw = raw[:-1]
            elif failure == "extra": raw += b"\0"
            worker.client.request.return_value = raw
            original_states = [robot.state for robot in worker.robots]
            original_commands = [robot.command.copy() for robot in worker.robots]
            with self.subTest(failure=failure), self.assertRaises((RuntimeError, ValueError)):
                worker.step(actions, commands)
            self.assertTrue(worker.failed)
            self.assertFalse(any(worker.step_mode_counts.values()))
            worker.client.request.assert_called_once()
            worker.request_many.assert_not_called()
            for robot, original, command in zip(worker.robots, original_states, original_commands):
                self.assertIs(robot.state, original)
                np.testing.assert_array_equal(robot.command, command)
            with self.assertRaisesRegex(RuntimeError, "never retry"):
                worker.step(actions, commands)
            worker.client.request.assert_called_once()

    def test_fast_requires_complete_capability_and_diagnostics_off(self):
        self.assertEqual(module.supported_train_modes({}), ())
        self.assertEqual(module.supported_train_modes(train_capability()), ("serial", "parallel"))
        for key in train_capability():
            invalid = train_capability()
            invalid.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "capability/schema"):
                module.supported_train_modes(invalid)
        for key, value in (("policy_step_train_columns", 88), ("policy_step_train_dtype", ">f8"),
                           ("policy_step_train_api_version", True), ("policy_step_train_modes", ["parallel", "parallel"])):
            invalid = {**train_capability(), key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                module.supported_train_modes(invalid)
        for field, value in (("train_step_modes", ()), ("runtime_diagnostics_enabled", None),
                             ("runtime_diagnostics_enabled", True)):
            worker, actions, commands, _, _, _ = self.make_worker()
            setattr(worker.robots[1], field, value)
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                worker.step(actions, commands)
            worker.client.request.assert_not_called()
            worker.request_many.assert_not_called()
        with patch.object(module, "UEWorker") as constructor, self.assertRaisesRegex(ValueError, "runtime_diagnostics=False"):
            module.UEGo1Pool(None, 1, 1, 9000, Path("/tmp/not-created"), connect="127.0.0.1:9000",
                             step_mode="fast", runtime_diagnostics=True)
        constructor.assert_not_called()

    def test_auto_stays_full_state_and_timeout_never_falls_back(self):
        worker, actions, commands, _, _, _ = self.make_worker()
        summary = module.step_mode_summary(worker.robots, "auto")
        self.assertEqual(summary["selected"], "batch_parallel")
        self.assertEqual(summary["state_format"], "full_json")
        self.assertTrue(summary["actor_pose_sync_during_step"])
        worker.client.request.side_effect = TimeoutError("uncertain physics completion")
        with self.assertRaises(TimeoutError):
            worker.step(actions, commands)
        self.assertTrue(worker.failed)
        worker.client.request.assert_called_once()
        worker.request_many.assert_not_called()

    def test_initial_reset_negotiates_fast_and_later_reset_rejects_schema_change(self):
        fixture = env_tests.UEWorkerResetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.enable_api()
        fixture.worker.step_mode = "fast"
        with self.assertRaisesRegex(RuntimeError, "training API v1"):
            fixture.worker.reset_one(0, initial=True)
        self.assertFalse(any("policy_step_train" in command for command in fixture.commands()))
        fixture.worker.failed = False
        fixture.state.update(train_capability(), runtime_diagnostics_enabled=True)
        fixture.worker.runtime_diagnostics = False
        fixture.worker.reset_one(0, initial=True)
        self.assertEqual(fixture.worker.robots[0].train_step_modes, ("serial", "parallel"))
        self.assertFalse(fixture.worker.robots[0].runtime_diagnostics_enabled)
        fixture.worker.reset_one(0)
        before = fixture.worker.robots[0].state
        fixture.forced_changes = {"policy_step_train_dtype": ">f8"}
        with self.assertRaisesRegex(RuntimeError, "capability/schema"):
            fixture.worker.reset_one(0)
        self.assertTrue(fixture.worker.failed)
        self.assertIs(fixture.worker.robots[0].state, before)

    def test_parser_recovers_unrealcv_utf8_decoded_binary_losslessly(self):
        header = {"api_version": 1, "mode": "parallel", "pose_sync": False,
                  "runtime_diagnostics_enabled": False, "actors": ["robot"],
                  "timings_ms": dict.fromkeys(module.BATCH_TIMING_FIELDS, 0.0)}
        rows = np.zeros((1, 89), dtype="<f8")
        rows[0, 0] = 3.0  # This double and the zero numeric fields are valid UTF8 bytes.
        while True:
            raw = packet(header, rows)
            try:
                decoded = raw.decode("utf-8")
                break
            except UnicodeDecodeError:
                header["padding_test"] = str(header.get("padding_test", "")) + "x"
        recovered, _ = module.training_batch_from(decoded, ["robot"], np.array([2.98]),
                                                  np.zeros((1, 12)), np.zeros((1, 3)))
        np.testing.assert_array_equal(recovered, rows)


class KeyboardV2TransportTests(unittest.TestCase):
    def make_worker(self, mode="fast"):
        worker, actions, commands, header, legacy, _ = FastStepTests().make_worker()
        worker.training_telemetry = "keyboard_v2"
        worker.step_mode = mode
        for robot in worker.robots:
            robot.reset_metadata = {**telemetry_metadata(), "reset_api_version": 2,
                                    "reset_modes": ["auto", "rebuild"],
                                    "reset_spawn_location": [10, 20, 35],
                                    "last_reset_result": "reused"}
            robot.state.update(robot.reset_metadata)
            robot.state.update(foot_heights=[.023] * 4, foot_current_air_time=[0.] * 4,
                               foot_current_contact_time=[0.] * 4,
                               foot_contact_forces_world=[0.] * 12, root_quat_wxyz=[1., 0, 0, 0])
            robot.train_v2_step_modes = ("serial", "parallel")
        rows = np.zeros((len(worker.robots), module.TRAIN_V2_COLUMNS), dtype="<f8")
        rows[:, :89] = legacy
        rows[:, 85:89] = [1, 0, 1, 0]
        rows[:, 89:93] = [.023, .08, .023, .1]
        rows[:, 93:97] = [0., .02, 0., .02]
        rows[:, 97:101] = [.02, 0., .005, 0.]
        rows[:, 101:113] = [1., -2., 30., 0., 0., 0., -2., 3., 40., 0., 0., 0.]
        rows[:, 113:117] = [1., 0., 0., 0.]
        header["api_version"] = 2
        full = batch_tests.BatchStepTests().response(worker, actions, commands)
        for entry, row in zip(full["robots"], rows):
            entry["state"]["foot_contacts"] = row[85:89].tolist()
            entry["state"].update({key: row[section].tolist() for key, section in module.TRAIN_V2_FIELDS.items()})
        return worker, actions, commands, header, rows, full

    def test_binary_and_full_json_have_identical_training_inputs_and_reset_metadata(self):
        worker, actions, commands, header, rows, full = self.make_worker()
        expected = [worker._validate_step_state(robot, entry["state"], action, command)
                    for robot, entry, action, command in zip(worker.robots, full["robots"], actions, commands)]
        worker.client.request.return_value = packet(header, rows)
        actual = worker.step(actions, commands)
        self.assertTrue(worker.client.request.call_args.args[0].startswith(
            "vset /mujoco/go1/policy_step_train_v2 parallel "))
        for left, right in zip(actual, expected):
            for key in ("obs", "control_targets", "foot_positions", "foot_velocities", "foot_contacts", *module.TRAIN_V2_FIELDS):
                np.testing.assert_array_equal(left[key], right[key])
                self.assertFalse(left[key].flags.writeable)
            for key in (*module.TRAIN_V2_PHYSICS_FIELDS, "reset_spawn_location", "last_reset_result"):
                self.assertEqual(left[key], right[key])
            self.assertEqual(left["state_format"], "train_binary_v2")
        self.assertEqual(module.step_mode_summary(worker.robots, "fast", "keyboard_v2")["state_format"], "train_binary_v2")

    def test_full_json_v2_is_selected_without_using_binary_endpoint(self):
        worker, actions, commands, _, _, full = self.make_worker("batch_parallel")
        worker.client.request.return_value = json.dumps(full)
        actual = worker.step(actions, commands)
        self.assertTrue(worker.client.request.call_args.args[0].startswith("vset /mujoco/go1/policy_step_batch parallel "))
        self.assertEqual(actual, [entry["state"] for entry in full["robots"]])

    def test_malformed_v2_rows_never_commit_or_retry(self):
        mutations = {
            "height_nonfinite": lambda r: r.__setitem__((-1, 89), np.nan),
            "negative_air": lambda r: r.__setitem__((-1, 94), -.01),
            "both_clocks": lambda r: r.__setitem__((-1, 93), .005),
            "clock_exceeds_sim_time": lambda r: r.__setitem__((-1, 94), .03),
            "missing_contact_clock": lambda r: r.__setitem__((-1, 97), 0.),
            "missing_air_clock": lambda r: r.__setitem__((-1, 94), 0.),
            "force_nonfinite": lambda r: r.__setitem__((-1, 101), np.inf),
            "quaternion_norm": lambda r: r.__setitem__((-1, 113), 2.),
        }
        for name, mutate in mutations.items():
            worker, actions, commands, header, rows, _ = self.make_worker()
            mutate(rows)
            worker.client.request.return_value = packet(header, rows)
            old = [robot.state for robot in worker.robots]
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                worker.step(actions, commands)
            self.assertTrue(worker.failed)
            self.assertFalse(any(worker.step_mode_counts.values()))
            self.assertTrue(all(robot.state is original for robot, original in zip(worker.robots, old)))
            worker.client.request.assert_called_once()
            worker.request_many.assert_not_called()

    def test_clock_growth_and_static_limits_checked_against_previous_step(self):
        worker, _, _, _, rows, full = self.make_worker()
        state = full["robots"][0]["state"]
        previous = worker.robots[0].state
        previous["sim_time"] = 1.
        state["sim_time"] = 1.02
        state["foot_current_air_time"][1] = .5
        with self.assertRaisesRegex(RuntimeError, "faster than physics"):
            module.validate_training_telemetry(state, previous)
        state["foot_current_air_time"][1] = .02
        state["soft_joint_pos_limits"][0] -= .01
        with self.assertRaisesRegex(RuntimeError, "limits changed"):
            module.validate_training_telemetry(state, previous)

    def test_v2_capability_is_mandatory_for_every_execution_mode(self):
        self.assertEqual(module.supported_train_modes({}, 2), ())
        self.assertEqual(module.supported_train_modes(train_v2_capability(), 2), ("serial", "parallel"))
        for key in train_v2_capability():
            invalid = train_v2_capability(); invalid.pop(key)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                module.supported_train_modes(invalid, 2)
        for mode in module.STEP_MODES:
            worker, actions, commands, _, _, _ = self.make_worker(mode)
            worker.robots[-1].train_v2_step_modes = ()
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "no legacy fallback"):
                worker.step(actions, commands)
            worker.client.request.assert_not_called()
        worker, actions, commands, header, rows, _ = self.make_worker()
        header["api_version"] = 1
        worker.client.request.return_value = packet(header, rows[:, :89])
        with self.assertRaisesRegex(RuntimeError, "magic"):
            worker.step(actions, commands)
        self.assertTrue(worker.failed)

    def test_full_json_missing_telemetry_or_physics_changes_fail_closed(self):
        for key in ("foot_heights", "training_telemetry_version", "actuator_damping", "soft_joint_pos_limits"):
            worker, actions, commands, _, _, full = self.make_worker("batch_parallel")
            full["robots"][-1]["state"].pop(key)
            worker.client.request.return_value = json.dumps(full)
            before = [robot.state for robot in worker.robots]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                worker.step(actions, commands)
            self.assertTrue(all(robot.state is old for robot, old in zip(worker.robots, before)))
            self.assertTrue(worker.failed)


if __name__ == "__main__":
    unittest.main()
