"""Fair evaluation contracts, exercised without a UE connection or native physics."""
import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[1] / "example/rl/locomotion"
with patch.object(sys, "path", [str(DIRECTORY), *sys.path]):
    evaluation = importlib.import_module("evaluate_go1_ue")
    playback = importlib.import_module("play_go1")
    refresh = importlib.import_module("refresh_go1_ue_metrics")


def metadata(ue=False):
    result = {
        "joint_names": ",".join(playback.JOINT_NAMES),
        "observation_names": ",".join(playback.OBSERVATION_NAMES),
        "command_names": "twist",
        **{key: ",".join(map(str, values)) for key, values in playback.PROFILE.items()},
    }
    if ue:
        result.update({
            "unrealzoo_backend": "ue_mujoco", "control_dt": "0.02",
            "ue_task_profile": "ue_v310_velocity",
            "ue_observation_reference": "reset_control_targets",
            "ue_bridge_default_joint_pos": ",".join(map(str, [0, 0.9, -1.8] * 4)),
        })
    return result


def fresh_state():
    observation = np.zeros(48)
    observation[8] = -1
    return {
        "obs": observation.tolist(), "sim_time": 0.0,
        "synchronous": True, "policy_profile": "velocity",
        "control_targets": [0, 0.9, -1.8] * 4,
        "foot_velocities": [0.0] * 12, "foot_contacts": [True] * 4,
    }


class FakeSession:
    def __init__(self):
        self.inputs = []

    def get_inputs(self):
        return [types.SimpleNamespace(name="obs")]

    def get_outputs(self):
        return [types.SimpleNamespace(name="actions")]

    def run(self, names, inputs):
        self.inputs.append(inputs["obs"].copy())
        return [np.full((1, 12), 2.5, dtype=np.float32)]


class FakePool:
    """Only fabricated telemetry; no sockets, simulator or subprocesses."""
    def __init__(self, fall_step=None, fail_step=None):
        self.state = fresh_state()
        self.offset = np.array([-0.01592, -0.06659, -0.00617])
        contract = {"imu_offset_m": self.offset.tolist(), "environment_geom_count": 1}
        robot = types.SimpleNamespace(asset_contract=contract)
        self.workers = [types.SimpleNamespace(robots=[robot])]
        self.commands = []
        self.fall_step, self.fail_step = fall_step, fail_step

    def step(self, actions, commands):
        number = len(self.commands) + 1
        if number == self.fail_step:
            raise RuntimeError("fabricated RPC failure")
        self.commands.append(commands[0].copy())
        state = copy.deepcopy(self.state)
        state["sim_time"] += evaluation.CONTROL_DT
        gyro = np.array([0, 0, commands[0, 2]], dtype=float)
        root = np.array([commands[0, 0] + 0.1, commands[0, 1] - 0.2, 0])
        state["obs"][:3] = (root + np.cross(gyro, self.offset)).tolist()
        state["obs"][3:6] = gyro.tolist()
        state["obs"][33:45] = actions[0].tolist()
        state["control_clip_count"] = 0
        if number == self.fall_step:
            angle = np.radians(75)
            state["obs"][6:9] = [np.sin(angle), 0, -np.cos(angle)]
        self.state = state
        return [copy.deepcopy(state)]


class FakeTensor(np.ndarray):
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class FakeTensorDict(dict):
    def __init__(self, values, batch_size):
        super().__init__(values)
        self.batch_size = batch_size


class EvaluationTests(unittest.TestCase):
    def test_sequence_changes_commands_without_reset_and_preserves_history(self):
        pool, session = FakePool(), FakeSession()
        policy = evaluation.EvaluationPolicy(metadata(), session=session)
        policy.reset = Mock()
        schedule = [[.5, 0, 0]] * 2 + [[0, -.3, -.5]] * 3
        case = evaluation.run_case(pool, policy, "sequence", [0, 0, 0],
                                   [fresh_state()], 5, 2, command_schedule=schedule)
        np.testing.assert_allclose(pool.commands, [[0, 0, 0]] * 2 + schedule)
        policy.reset.assert_called_once_with()
        self.assertEqual(len(session.inputs), 7)
        # The switch at sample 5 sees the previous step's actual root velocity.
        np.testing.assert_allclose(session.inputs[4][0, :2], [.6, -.2], atol=1e-6)
        self.assertTrue(case["metrics"]["survived_full_command"])
        np.testing.assert_allclose(case["metrics"]["root_tracking_after_warmup"]["mae"],
                                   [.1, .2, 0], atol=1e-6)
        self.assertEqual(case["command_schedule"], schedule)

    def test_invalid_or_incomplete_sequence_is_rejected_before_physics(self):
        for schedule in ([[0, 0, 0]], [[0, 0, 0], [float("nan"), 0, 0]]):
            pool = FakePool()
            with self.assertRaisesRegex(ValueError, "command schedule"):
                evaluation.run_case(pool, evaluation.EvaluationPolicy(metadata(), session=FakeSession()),
                                    "sequence", [0, 0, 0], [fresh_state()], 2, 0,
                                    command_schedule=schedule)
            self.assertEqual(pool.commands, [])

    def test_reference_onnx_uses_v2_action_boundary_before_physics(self):
        pool = FakePool()
        policy = evaluation.EvaluationPolicy(metadata(), session=FakeSession())
        policy.act = lambda observation: np.full((1, 12), 8.0, dtype=np.float32)
        original_task = evaluation.Go1Task

        def task_without_optional_telemetry(*args, **kwargs):
            # This fixture supplies old synthetic telemetry; the test concerns
            # evaluation's pre-RPC action contract, not the v2 reward formulas.
            kwargs.pop("profile")
            return original_task(*args, **kwargs)

        with patch.object(evaluation, "Go1Task", side_effect=task_without_optional_telemetry):
            result = evaluation.run_case(pool, policy, "forward", [.5, 0, 0],
                                         [fresh_state()], 2, 0,
                                         environment_profile=evaluation.V2_PROFILE)
        for sample in result["samples"]:
            self.assertEqual(sample["action"], [5.0] * 12)
            self.assertEqual(sample["state"]["obs"][33:45], [5.0] * 12)

    def test_ground_height_uses_only_valid_probes_and_contact_slip_is_conditional(self):
        first = {
            "root_position_ue_cm": [0, 0, 140], "ground_support_z_ue_cm": 100,
            "local_ground_patch_active": True,
            "foot_ground_probe_hits": [True, False, True, True],
            "foot_ground_z_ue_cm": [100] * 4,
            "foot_contacts": [True, False, True, False],
            "foot_velocities": [3, 4, 0, 100, 0, 0, 0, 2, 0, 100, 0, 0],
        }
        second = {
            **first, "root_position_ue_cm": [0, 0, 120],
            "foot_ground_probe_hits": [True, True, False, True],
            "foot_contacts": [True, True, False, False],
            "foot_velocities": [0, 3, 0, 0, 4, 0, 100, 0, 0, 100, 0, 0],
        }
        # A stale support height/root position does not validate a failed probe.
        missing = {"root_position_ue_cm": [0, 0, -999], "ground_support_z_ue_cm": 100,
                   "local_ground_patch_active": False}
        result = evaluation.ground_contact_metrics([{"state": s} for s in (first, second, missing)])
        height = result["body_height_above_support_cm"]
        self.assertEqual(height["mean"], 30)
        self.assertEqual(height["min"], 20)
        self.assertEqual(height["p05"], 21)
        self.assertEqual(height["valid_fraction"], 2 / 3)
        self.assertEqual(result["root_ground_probe_valid_fraction"], 2 / 3)
        self.assertEqual(result["foot_ground_probe_valid_fraction"], 0.5)
        contact = result["contact_duty_fraction"]
        self.assertEqual(contact["overall"], 0.5)
        self.assertEqual(contact["per_foot"], [1, 0.5, 0.5, 0])
        self.assertEqual(contact["valid_sample_fraction"], 2 / 3)
        slip = result["contact_foot_slip_m_s"]
        self.assertEqual(slip["mean"], 3.5)
        self.assertAlmostEqual(slip["rms"], np.sqrt(13.5))
        self.assertEqual(slip["contact_samples_with_velocity"], 4)
        self.assertIsNone(slip["per_foot"][3]["mean"])

    def test_missing_probes_airborne_feet_and_malformed_fields_do_not_imply_success(self):
        state = {
            "root_position_ue_cm": [0, 0, 30], "ground_support_z_ue_cm": 0,
            "foot_ground_probe_hits": [True] * 4,
            "foot_ground_z_ue_cm": [float("nan")] * 4,
            "foot_contacts": [False] * 4, "foot_velocities": [0] * 12,
        }
        result = evaluation.ground_contact_metrics([{"state": state}])
        self.assertIsNone(result["body_height_above_support_cm"]["mean"])
        self.assertEqual(result["body_height_above_support_cm"]["valid_fraction"], 0)
        self.assertEqual(result["foot_ground_probe_valid_fraction"], 0)
        self.assertEqual(result["contact_duty_fraction"]["overall"], 0)
        self.assertIsNone(result["contact_foot_slip_m_s"]["mean"])
        empty = evaluation.ground_contact_metrics([])
        self.assertIsNone(empty["body_height_above_support_cm"]["valid_fraction"])
        self.assertIsNone(empty["contact_duty_fraction"]["overall"])
        malformed = evaluation.ground_contact_metrics([{"state": {"foot_contacts": "unknown"}}])
        self.assertIsNone(malformed["contact_duty_fraction"]["overall"])

    def test_offline_refresh_excludes_warmup_preserves_evidence_and_backs_up_input(self):
        result = {
            "evaluation_profile": "ue_go1_fixed_commands_v1",
            "cases": [{"name": "stand", "metrics": {"survived_full_command": True}, "samples": [
                {"phase": "warmup", "state": {"root_position_ue_cm": [0, 0, -100],
                 "ground_support_z_ue_cm": 0, "local_ground_patch_active": True}},
                {"phase": "command", "state": {"root_position_ue_cm": [0, 0, 30],
                 "ground_support_z_ue_cm": 0, "local_ground_patch_active": True}},
            ]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluation.json"
            original = json.dumps(result).encode()
            source.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                evaluation, "UEGo1Pool", side_effect=AssertionError("must stay offline"),
            ):
                refresh.main([str(source), "--in-place"])
            updated = json.loads(source.read_bytes())
            case = updated["cases"][0]
            self.assertEqual(case["samples"], result["cases"][0]["samples"])
            self.assertTrue(case["metrics"]["survived_full_command"])
            self.assertEqual(case["metrics"]["ground_contact_after_warmup"]["body_height_above_support_cm"]["mean"], 30)
            self.assertEqual(source.with_name("evaluation.before-ground-contact.json").read_bytes(), original)
            self.assertEqual(updated["metrics_postprocessing"][0]["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(updated["metrics_postprocessing"][0]["physics_steps_executed"], 0)

    def test_onnx_gets_calibrated_observation_once_and_actions_are_not_clipped(self):
        task = evaluation.Go1Task(1, observation_noise=False)
        observation = task.reset([0], [fresh_state()])["actor"]
        session = FakeSession()
        policy = evaluation.EvaluationPolicy(metadata(), session=session)
        action = policy.act(observation)
        np.testing.assert_array_equal(session.inputs[0], observation)
        np.testing.assert_allclose(observation[0, 9:21], [-0.1, 0, 0, 0.1, 0, 0] * 2)
        np.testing.assert_array_equal(action, 2.5)

    def test_pt_loader_uses_direct_deterministic_actor_without_wrapper_correction(self):
        actor = Mock(return_value=np.full((1, 12), 2.5, dtype=np.float32).view(FakeTensor))
        wrapper = types.SimpleNamespace(actor=actor, act=Mock(side_effect=AssertionError("legacy adapter")))
        loader = Mock(return_value=(wrapper, metadata(ue=True)))
        torch = types.SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            as_tensor=lambda value, device: np.asarray(value),
        )
        modules = {
            "ue_go1_checkpoint": types.SimpleNamespace(load_ue_checkpoint=loader),
            "torch": torch, "tensordict": types.SimpleNamespace(TensorDict=FakeTensorDict),
        }
        task = evaluation.Go1Task(1, observation_noise=False)
        observation = task.reset([0], [fresh_state()])["actor"]
        with patch.dict(sys.modules, modules):
            policy = evaluation.load_policy(Path("model.pt"))
            policy.reset()
            policy.act(observation)
        actor.reset.assert_called_once_with()
        self.assertFalse(actor.call_args.kwargs["stochastic_output"])
        np.testing.assert_array_equal(actor.call_args.args[0]["actor"], observation)
        wrapper.act.assert_not_called()

    def test_fixed_commands_root_metrics_and_warmup_are_separated(self):
        pool, session = FakePool(), FakeSession()
        case = evaluation.run_case(
            pool, evaluation.EvaluationPolicy(metadata(), session=session),
            "yaw", evaluation.CASES["yaw"], [fresh_state()], 4, 2,
        )
        np.testing.assert_array_equal(pool.commands[:2], np.zeros((2, 3)))
        np.testing.assert_array_equal(pool.commands[2:], [[0, 0, 0.5]] * 4)
        metrics = case["metrics"]
        self.assertTrue(metrics["survived_full_command"])
        self.assertEqual(metrics["command_steps_completed"], 4)
        np.testing.assert_allclose(metrics["root_tracking_after_warmup"]["rmse"], [0.1, 0.2, 0])
        self.assertNotEqual(
            metrics["root_tracking_after_warmup"]["rmse"],
            metrics["imu_tracking_after_warmup"]["rmse"],
        )
        self.assertEqual(metrics["control_target_clip_count"], 0)
        self.assertEqual(metrics["raw_action_abs_ge_one_fraction"], 1)
        self.assertEqual(len(case["samples"]), 6)
        self.assertEqual(case["samples"][0]["state"]["sim_time"], 0.02)
        self.assertFalse(case["task_manifest"]["actor_observation_noise"])

    def test_fall_stops_case_at_training_threshold(self):
        pool = FakePool(fall_step=3)
        case = evaluation.run_case(
            pool, evaluation.EvaluationPolicy(metadata(), session=FakeSession()),
            "stand", evaluation.CASES["stand"], [fresh_state()], 5, 1,
        )
        self.assertTrue(case["metrics"]["fallen"])
        self.assertFalse(case["metrics"]["survived_full_command"])
        self.assertEqual(case["metrics"]["first_fall_step"], 3)
        self.assertEqual(case["metrics"]["survived_command_seconds"], 0.04)
        self.assertLess(case["samples"][-1]["upright"], evaluation.FALL_UPRIGHT)

    def test_interrupted_case_keeps_samples_and_partial_metrics(self):
        pool, case = FakePool(fail_step=3), {}
        with self.assertRaisesRegex(RuntimeError, "RPC failure"):
            evaluation.run_case(
                pool, evaluation.EvaluationPolicy(metadata(), session=FakeSession()),
                "stand", evaluation.CASES["stand"], [fresh_state()], 5, 1, case=case,
            )
        self.assertEqual(len(case["samples"]), 2)
        self.assertFalse(case["metrics"]["survived_full_command"])

    def test_mismatched_checkpoint_reference_rejected_before_physics(self):
        pool = FakePool()
        state = fresh_state()
        state["control_targets"][0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "checkpoint calibration"):
            evaluation.run_case(
                pool, evaluation.EvaluationPolicy(metadata(ue=True), session=FakeSession()),
                "stand", evaluation.CASES["stand"], [state], 5, 1,
            )
        self.assertEqual(pool.commands, [])
        wrong = metadata()
        wrong["command_names"] = "height"
        with self.assertRaises(ValueError):
            evaluation.EvaluationPolicy(wrong, session=FakeSession())

    def test_runtime_asset_snapshot_checks_hash_and_preserves_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runtime.xml"
            data = b'<mujoco><worldbody><geom type="plane" size="10 10 1"/></worldbody></mujoco>'
            source.write_bytes(data)
            contract = {"runtime_mjcf": str(source), "runtime_mjcf_sha256": hashlib.sha256(data).hexdigest()}
            result = evaluation.snapshot_runtime_asset(contract, directory)
            self.assertEqual(Path(result["path"]).read_bytes(), data)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                evaluation.snapshot_runtime_asset(contract, directory)

    def test_cli_requires_explicit_checkpoint_valid_duration_and_local_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            checkpoint.touch()
            arguments = ["--connect", "127.0.0.1:9000", "--checkpoint", str(checkpoint),
                         "--output", str(Path(directory) / "result.json")]
            args = evaluation.parse_args(arguments)
            self.assertEqual(args.command_steps, 1000)
            self.assertEqual(args.warmup_steps, 25)
            self.assertEqual(evaluation.CASES["yaw"], (0, 0, 0.5))
            for extra in (["--episode-seconds", "nan"], ["--episode-seconds", "0.03"],
                          ["--connect", "192.168.1.250:9000"], ["--cases", "stand", "stand"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        evaluation.parse_args(arguments + extra)


if __name__ == "__main__":
    unittest.main()
