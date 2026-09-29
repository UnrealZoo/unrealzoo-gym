"""Batch stepping must negotiate capability and validate every result atomically."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
import ue_go1_env as module
from benchmark_go1_ue import parse_args
from train_go1_ue import build_parser


def state(time=0.0, capable=True):
    observation = np.zeros(48)
    observation[8] = -1
    value = {"obs": observation.tolist(), "sim_time": time,
             "policy_profile": "velocity", "synchronous": True,
             "control_targets": [0, .9, -1.8] * 4,
             "foot_positions": [0.] * 12, "foot_velocities": [0.] * 12,
             "foot_contacts": [False] * 4}
    if capable:
        value.update(policy_step_batch_api_version=1, policy_step_batch_modes=["serial", "parallel"])
    return value


class BatchStepTests(unittest.TestCase):
    def make_worker(self, mode="auto", capable=True):
        worker = module.UEWorker.__new__(module.UEWorker)
        worker.closed = worker.failed = False
        worker.step_mode = mode
        worker.step_mode_counts = dict.fromkeys(module.EXECUTION_MODES, 0)
        worker.robots = [
            module.Robot(f"go1_{index}", np.zeros(3), np.zeros(3), state=state(index * .1, capable),
                         asset_contract={"valid": True},
                         batch_step_modes=("serial", "parallel") if capable else ())
            for index in range(2)
        ]
        worker.request = Mock()
        worker.request_many = Mock()
        actions = np.arange(24, dtype=np.float32).reshape(2, 12) / 10
        commands = np.array([[.1, .2, .3], [-.1, -.2, -.3]], dtype=np.float32)
        return worker, actions, commands

    def response(self, worker, actions, commands, mode="parallel"):
        entries = []
        for robot, action, command in zip(worker.robots, actions, commands):
            value = copy.deepcopy(robot.state)
            value["sim_time"] += .02
            value["obs"][33:45] = action.tolist()
            value["obs"][45:48] = command.tolist()
            entries.append({"actor": robot.name, "state": value})
        return {"policy_step_batch_api_version": 1, "mode": mode, "robots": entries}

    def use_transport(self, worker, response):
        """Exercise the production request/decoder, mocking only network I/O."""
        worker.client = Mock()
        worker.client.request.return_value = response
        worker.timeout = 1
        worker.request_seconds = 0.0
        worker.request = module.UEWorker.request.__get__(worker)

    def test_batch_decodes_once_without_serializing_individual_states(self):
        worker, actions, commands = self.make_worker()
        response = self.response(worker, actions, commands)
        self.use_transport(worker, json.dumps(response).encode())
        with patch.object(module.json, "loads", wraps=json.loads) as loads, \
                patch.object(module.json, "dumps", wraps=json.dumps) as dumps:
            result = worker.step(actions, commands)
        self.assertEqual(loads.call_count, 1)
        self.assertEqual(dumps.call_count, 1)  # Only the outgoing action payload.
        self.assertEqual(result, [entry["state"] for entry in response["robots"]])
        worker.client.request.assert_called_once()
        worker.request_many.assert_not_called()

    def test_strict_decoder_rejects_nonfinite_numbers_in_unvalidated_fields(self):
        # parse_constant handles the nonstandard NaN/Infinity tokens; parse_float
        # must also reject standards-compliant float overflow such as 1e999.
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            for location in ("state", "envelope"):
                worker, actions, commands = self.make_worker()
                response = self.response(worker, actions, commands)
                target = response["robots"][-1]["state"] if location == "state" else response
                target["extra_diagnostics"] = {"nested": [0, {"value": "INVALID_NUMBER"}]}
                raw = json.dumps(response).replace('"INVALID_NUMBER"', token)
                self.use_transport(worker, raw)
                original_states = [robot.state for robot in worker.robots]
                original_commands = [robot.command.copy() for robot in worker.robots]
                with self.subTest(token=token, location=location), self.assertRaisesRegex(ValueError, "Non-finite"):
                    worker.step(actions, commands)
                self.assertTrue(worker.failed)
                worker.client.request.assert_called_once()
                worker.request_many.assert_not_called()
                self.assertFalse(any(worker.step_mode_counts.values()))
                for robot, original, command in zip(worker.robots, original_states, original_commands):
                    self.assertIs(robot.state, original)
                    np.testing.assert_array_equal(robot.command, command)
                with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                    worker.step(actions, commands)
                worker.client.request.assert_called_once()

    def test_json_transport_keeps_error_and_legacy_text_contracts(self):
        worker, _, _ = self.make_worker()
        for raw in (None, "error unavailable", '{"success":false}', '{"error":"bad world"}'):
            self.use_transport(worker, raw)
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                worker.request("example", decode_json=True)
        self.use_transport(worker, b' {"success":true,"value":1} ')
        self.assertEqual(worker.request("legacy"), '{"success":true,"value":1}')
        self.assertEqual(worker.request("batch", decode_json=True), {"success": True, "value": 1})

    def test_auto_uses_one_parallel_rpc_with_all_actions_and_commands(self):
        worker, actions, commands = self.make_worker()
        response = self.response(worker, actions, commands)
        worker.request.return_value = response
        result = worker.step(actions, commands)
        command = worker.request.call_args.args[0]
        prefix = "vset /mujoco/go1/policy_step_batch parallel "
        self.assertTrue(command.startswith(prefix))
        payload = json.loads(command[len(prefix):])
        self.assertEqual(payload, {"robots": [
            {"actor": robot.name, "actions": action.tolist(), "command": command.tolist()}
            for robot, action, command in zip(worker.robots, actions, commands)
        ]})
        self.assertNotIn(" ", command[len(prefix):])
        worker.request.assert_called_once()
        worker.request_many.assert_not_called()
        self.assertEqual(result, [entry["state"] for entry in response["robots"]])
        np.testing.assert_array_equal(worker.robots[1].command, commands[1])
        self.assertEqual(worker.step_mode_counts, {"single": 0, "batch_serial": 0, "batch_parallel": 1, "fast": 0})

    def test_explicit_serial_requires_serial_ack(self):
        worker, actions, commands = self.make_worker("batch_serial")
        worker.request.return_value = self.response(worker, actions, commands, "serial")
        worker.step(actions, commands)
        self.assertIn("policy_step_batch serial ", worker.request.call_args.args[0])
        self.assertEqual(worker.step_mode_counts["batch_serial"], 1)

    def test_auto_old_binary_uses_legacy_path_and_records_reason(self):
        worker, actions, commands = self.make_worker(capable=False)
        replies = self.response(worker, actions, commands)
        worker.request_many.side_effect = [["ok", "ok"], [json.dumps(entry["state"]) for entry in replies["robots"]]]
        worker.step(actions, commands)
        worker.request.assert_not_called()
        self.assertEqual(worker.request_many.call_count, 2)
        summary = module.step_mode_summary(worker.robots, "auto")
        self.assertEqual(summary["selected"], "single")
        self.assertEqual(summary["batch_api_capable_replicas"], 0)
        self.assertIn("legacy", summary["unavailable_reason"])
        self.assertEqual(worker.step_mode_counts["single"], 1)

    def test_explicit_single_keeps_legacy_path_even_on_new_binary(self):
        worker, actions, commands = self.make_worker("single")
        response = self.response(worker, actions, commands)
        worker.request_many.side_effect = [["ok", "ok"], [json.dumps(entry["state"]) for entry in response["robots"]]]
        worker.step(actions, commands)
        worker.request.assert_not_called()
        self.assertEqual(worker.step_mode_counts["single"], 1)

    def test_explicit_batch_on_old_server_rejected_before_rpc(self):
        for mode in ("batch_serial", "batch_parallel"):
            worker, actions, commands = self.make_worker(mode, capable=False)
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "unsupported"):
                worker.step(actions, commands)
            worker.request.assert_not_called()
            worker.request_many.assert_not_called()

    def test_all_input_rows_validated_before_any_rpc(self):
        worker, actions, commands = self.make_worker()
        invalid_actions, invalid_commands = actions.copy(), commands.copy()
        invalid_actions[-1, -1] = np.nan
        invalid_commands[-1, -1] = np.inf
        for a, c in ((invalid_actions, commands), (actions, invalid_commands),
                     (actions[:1], commands), (actions, commands[:, :2])):
            with self.subTest(shape=(a.shape, c.shape)), self.assertRaises(ValueError):
                worker.step(a, c)
            worker.request.assert_not_called()
            worker.request_many.assert_not_called()
            self.assertFalse(worker.failed)
        worker.robots[1].name = worker.robots[0].name
        with self.assertRaisesRegex(ValueError, "unique"):
            worker.step(actions, commands)
        worker.request.assert_not_called()

    def test_bad_reply_never_partially_commits_or_falls_back(self):
        for failure in ("missing", "reordered", "duplicate", "mode", "version", "no_state",
                        "time", "action", "command", "foot", "nan", "lost_capability"):
            worker, actions, commands = self.make_worker()
            response = self.response(worker, actions, commands)
            entries = response["robots"]
            bad = entries[1]["state"]
            if failure == "missing": entries.pop()
            elif failure == "reordered": entries.reverse()
            elif failure == "duplicate": entries[1]["actor"] = entries[0]["actor"]
            elif failure == "mode": response["mode"] = "serial"
            elif failure == "version": response["policy_step_batch_api_version"] = True
            elif failure == "no_state": entries[1].pop("state")
            elif failure == "time": bad["sim_time"] += .02
            elif failure == "action": bad["obs"][33] += 1
            elif failure == "command": bad["obs"][45] += 1
            elif failure == "foot": bad["foot_contacts"] = [True]
            elif failure == "nan": bad["obs"][0] = float("nan")
            elif failure == "lost_capability": bad.pop("policy_step_batch_api_version")
            worker.request.return_value = response
            original_states = [robot.state for robot in worker.robots]
            original_commands = [robot.command.copy() for robot in worker.robots]
            with self.subTest(failure=failure), self.assertRaises((RuntimeError, ValueError, TypeError, KeyError)):
                worker.step(actions, commands)
            self.assertTrue(worker.failed)
            worker.request_many.assert_not_called()
            worker.request.assert_called_once()
            for robot, original, command in zip(worker.robots, original_states, original_commands):
                self.assertIs(robot.state, original)
                np.testing.assert_array_equal(robot.command, command)
            self.assertFalse(any(worker.step_mode_counts.values()))
            with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                worker.step(actions, commands)
            worker.request.assert_called_once()

    def test_request_timeout_is_not_replayed_as_single_steps(self):
        worker, actions, commands = self.make_worker()
        worker.request.side_effect = TimeoutError("uncertain physics completion")
        with self.assertRaises(TimeoutError):
            worker.step(actions, commands)
        self.assertTrue(worker.failed)
        worker.request_many.assert_not_called()

    def test_optional_server_wall_timings_are_accumulated_separately_from_cpu(self):
        worker, actions, commands = self.make_worker()
        for scale in (1, 2):
            response = self.response(worker, actions, commands)
            response["timings_ms"] = {key: float(scale * (index + 1)) for index, key in enumerate(module.BATCH_TIMING_FIELDS)}
            worker.request.return_value = response
            worker.step(actions, commands)
        report = module.batch_timing_summary([worker])
        values = report["by_mode"]["batch_parallel"]
        self.assertEqual(values["measured_responses"], 2)
        self.assertEqual(values["mean_ms"]["physics"], 4.5)
        self.assertEqual(values["sum_ms"]["finalize_and_report"], 12.0)
        self.assertIsNone(report["by_mode"]["batch_serial"]["mean_ms"]["physics"])
        self.assertIn("not CPU", report["source"])

    def test_invalid_timings_and_batch_limit_rejected_without_fallback(self):
        worker, actions, commands = self.make_worker()
        with patch.object(module, "MAX_BATCH_ROBOTS", 1), self.assertRaisesRegex(ValueError, "at most"):
            worker.step(actions, commands)
        worker.request.assert_not_called()
        response = self.response(worker, actions, commands)
        response["timings_ms"] = dict.fromkeys(module.BATCH_TIMING_FIELDS, 1.0)
        response["timings_ms"]["physics"] = -1.0
        worker.request.return_value = response
        with self.assertRaisesRegex(RuntimeError, "wall-clock timings"):
            worker.step(actions, commands)
        self.assertTrue(worker.failed)
        worker.request_many.assert_not_called()
        self.assertEqual(module.batch_timing_summary([worker])["by_mode"]["batch_parallel"]["measured_responses"], 0)

    def test_partial_or_invalid_batch_capability_is_rejected(self):
        self.assertEqual(module.supported_batch_modes({}), ())
        for invalid in (
            {"policy_step_batch_api_version": True, "policy_step_batch_modes": ["serial", "parallel"]},
            {"policy_step_batch_api_version": 1},
            {"policy_step_batch_modes": ["serial", "parallel"]},
            {"policy_step_batch_api_version": 1, "policy_step_batch_modes": ["serial", "serial"]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                module.supported_batch_modes(invalid)

    def test_cli_and_constructor_validate_step_modes(self):
        required = ["--connect", "127.0.0.1:9000", "--log-dir", "/tmp/not-created"]
        self.assertEqual(build_parser().parse_args(required).step_mode, "auto")
        for mode in module.STEP_MODES:
            self.assertEqual(build_parser().parse_args(required + ["--step-mode", mode]).step_mode, mode)
            self.assertEqual(parse_args(["--connect", "127.0.0.1:9000", "--output", "/tmp/not-created", "--step-mode", mode]).step_mode, mode)
        with patch.object(module, "UEWorker") as worker:
            with self.assertRaisesRegex(ValueError, "step_mode"):
                module.UEGo1Pool(None, 1, 1, 9000, Path("/tmp/not-created"),
                                connect="127.0.0.1:9000", step_mode="threads")
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
