"""Guard the training-export/UE boundary without importing Gym or ONNX Runtime."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ENTRY = Path(__file__).resolve().parents[1] / "example/rl/locomotion/play_go1.py"
SPEC = importlib.util.spec_from_file_location("go1_playback", ENTRY)
playback = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(playback)


class FakeSession:
    def __init__(self):
        self.metadata = {
            "joint_names": ",".join(playback.JOINT_NAMES),
            "observation_names": ",".join(playback.OBSERVATION_NAMES),
            "command_names": "twist",
            **{key: ",".join("{:.3f}".format(x) for x in values) for key, values in playback.PROFILE.items()},
        }
        self.inputs = [types.SimpleNamespace(shape=[1, 48], type="tensor(float)")]
        self.outputs = [types.SimpleNamespace(shape=[1, 12], type="tensor(float)")]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map=self.metadata)


class FakeResetEnv:
    def __init__(self, paused=False, reset_error=None, camera_replies=None):
        self.paused = paused
        self.reset_error = reset_error
        self.camera_replies = list(camera_replies or ["0 0 100"])
        self.requests = []
        self.actor_name = ""
        self.spawn_location = None
        self.spawn_camera_id = "7"
        self.request_timeout = 30.0
        self.state = {"sim_time": 0.0, "control_targets": [0, 0.9, -1.8] * 4}

    def request(self, command):
        self.requests.append(command)
        if command == "vget /camera/7/location":
            response = self.camera_replies[0]
            if len(self.camera_replies) > 1:
                self.camera_replies.pop(0)
            return response
        if command == "vget /action/game/is_paused":
            return str(self.paused).lower()
        if command == "vset /action/game/pause":
            self.paused = True
            return "ok"
        if command == "vset /action/game/resume":
            self.paused = False
            return "ok"
        raise AssertionError(command)

    def reset(self):
        self.requests.append("reset")
        if self.reset_error is not None:
            raise self.reset_error
        self.state["sim_time"] = 0.0 if self.paused else 0.4
        return [0.0] * 48


class PlaybackContractTests(unittest.TestCase):
    def ue_session(self):
        session = FakeSession()
        session.metadata.update({
            "unrealzoo_backend": "ue_mujoco",
            "ue_task_profile": "ue_v310_velocity",
            "task_id": "ue_v310_velocity",
            "control_dt": "0.02",
            "ue_observation_reference": "reset_control_targets",
            "ue_bridge_default_joint_pos": ",".join(map(str, [0, 0.9, -1.8] * 4)),
        })
        return session

    def test_accepts_original_export_rounding_and_dynamic_batch(self):
        session = FakeSession()
        session.inputs[0].shape[0] = "batch"
        playback.validate_policy_contract(session)

    def test_rejects_policy_with_same_shape_but_wrong_semantics(self):
        for key, value in (
            ("joint_names", ",".join(reversed(playback.JOINT_NAMES))),
            ("observation_names", ",".join(reversed(playback.OBSERVATION_NAMES))),
            ("default_joint_pos", ",".join(["0"] * 12)),
            ("action_scale", ",".join(["0.5"] * 12)),
            ("joint_stiffness", ",".join(["40"] * 12)),
            ("joint_damping", ",".join(["0.5"] * 12)),
        ):
            with self.subTest(key=key):
                session = FakeSession()
                session.metadata[key] = value
                with self.assertRaises(ValueError):
                    playback.validate_policy_contract(session)

    def test_missing_and_nonfinite_metadata_are_rejected(self):
        session = FakeSession()
        del session.metadata["action_scale"]
        with self.assertRaises(ValueError):
            playback.validate_policy_contract(session)
        session = FakeSession()
        session.metadata["joint_damping"] = ",".join(["nan"] * 12)
        with self.assertRaises(ValueError):
            playback.validate_policy_contract(session)

    def test_rejects_wrong_dimensions_or_multiple_tensors(self):
        for inputs, outputs in (([1, 45], [1, 12]), ([1, 48], [1, 29]), ([8, 48], [8, 12])):
            session = FakeSession()
            session.inputs[0].shape = inputs
            session.outputs[0].shape = outputs
            with self.assertRaises(ValueError):
                playback.validate_policy_contract(session)
        session.inputs.append(copy.copy(session.inputs[0]))
        with self.assertRaises(ValueError):
            playback.validate_policy_contract(session)

    def test_simulation_must_be_synchronous_velocity_and_20ms(self):
        state = {"obs": [0] * 48, "policy_profile": "velocity", "synchronous": True, "sim_time": 0.02}
        playback.validate_state(state, 0.0)
        for change in ({"sim_time": 0.04}, {"synchronous": False}, {"policy_profile": "parkour"}, {"sim_time": math.nan}):
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError):
                    playback.validate_state({**state, **change}, 0.0)

    def test_metrics_exclude_warmup_and_record_fall(self):
        samples = [
            {"step": 1, "sim_time": 0.02, "command": [0, 0, 0], "linvel": [9, 9, 0], "gyro": [0, 0, 9], "upright": 1, "control_clip_count": 0},
            {"step": 2, "sim_time": 0.04, "command": [0.5, 0, 0.2], "linvel": [0.4, 0.1, 0], "gyro": [0, 0, 0.1], "upright": 0.4, "control_clip_count": 2},
        ]
        metrics = playback.summarize(samples, 1, 1.0, 0.5)
        self.assertEqual(metrics["tracking_after_warmup"]["samples"], 1)
        for value in metrics["tracking_after_warmup"]["rmse"]:
            self.assertAlmostEqual(value, 0.1)
        self.assertEqual(metrics["first_fall_step"], 2)
        self.assertEqual(metrics["control_target_clip_count"], 2)

    def test_engine_error_payloads_cannot_be_treated_as_observations(self):
        for response in (None, "null", "Error unknown command", '{"success":false}', '{"error":"missing endpoint"}'):
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    playback.check_response(response, "example")
        self.assertEqual(playback.check_response("ok", "example"), "ok")

    def test_ue_backend_requires_complete_and_consistent_calibration_metadata(self):
        playback.validate_policy_contract(self.ue_session())
        for missing in (
            "ue_task_profile", "ue_observation_reference",
            "ue_bridge_default_joint_pos", "control_dt",
        ):
            with self.subTest(missing=missing):
                session = self.ue_session()
                del session.metadata[missing]
                with self.assertRaises(ValueError):
                    playback.validate_policy_contract(session)
        for key, invalid in (
            ("ue_task_profile", "unknown"),
            ("task_id", "different_task"),
            ("ue_observation_reference", "policy_default"),
            ("control_dt", "0.04"),
            ("control_dt", "nan"),
            ("ue_bridge_default_joint_pos", "0,0,0"),
            ("ue_bridge_default_joint_pos", ",".join(["nan"] * 12)),
        ):
            with self.subTest(key=key, invalid=invalid):
                session = self.ue_session()
                session.metadata[key] = invalid
                with self.assertRaises(ValueError):
                    playback.validate_policy_contract(session)

    def test_joint_calibration_composes_with_the_real_policy_adapter_exactly_once(self):
        import numpy as np

        # Load the actual adapter without initializing ONNX Runtime or a network.
        adapter_path = ENTRY.parents[2] / "mujoco/common/pretrained_policy.py"
        spec = importlib.util.spec_from_file_location("go1_adapter_test", adapter_path)
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"onnxruntime": types.ModuleType("onnxruntime")}):
            spec.loader.exec_module(adapter)
        captured = []

        def capture(observation):
            captured.append(observation.copy())
            return np.zeros(12, dtype=np.float32)

        policy = adapter.PretrainedPolicy.__new__(adapter.PretrainedPolicy)
        policy.robot = "go1"
        policy.last_action = np.zeros(12, dtype=np.float32)
        policy.go1_default = np.array(playback.PROFILE["default_joint_pos"], dtype=np.float32)
        policy.checkpoint = capture
        raw = np.linspace(-0.05, 0.05, 48, dtype=np.float32)
        original = raw.copy()
        command = [0.5, 0.1, -0.2]
        metadata = self.ue_session().metadata
        for _ in range(2):
            prepared = playback.prepare_policy_observation(
                raw, command, metadata, adapter.GO1_BRIDGE_DEFAULT
            )
            policy.act(prepared, command)
        expected = raw[9:21] + np.array([0, 0.9, -1.8] * 4) - policy.go1_default
        for observed in captured:
            np.testing.assert_allclose(observed[9:21], expected, atol=1e-7)
            np.testing.assert_allclose(observed[45:48], command, atol=1e-7)
        np.testing.assert_array_equal(raw, original)

        # Existing external checkpoints retain their previous adapter semantics.
        prepared = playback.prepare_policy_observation(
            raw, command, FakeSession().metadata, adapter.GO1_BRIDGE_DEFAULT
        )
        policy.act(prepared, command)
        old_expected = raw[9:21] + adapter.GO1_BRIDGE_DEFAULT - policy.go1_default
        np.testing.assert_array_equal(captured[-1][9:21], old_expected)

    def test_new_policy_checks_actual_reset_reference_and_legacy_path_is_unchanged(self):
        state = {
            "sim_time": 0.0,
            "control_targets": [0, 0.9, -1.8] * 4,
        }
        metadata = self.ue_session().metadata
        result = playback.validate_reset_observation_reference(state, metadata)
        self.assertEqual(result["mode"], "checkpoint_reset_reference")
        for invalid in (
            {**state, "sim_time": 0.02},
            {**state, "sim_time": math.nan},
            {**state, "control_targets": playback.PROFILE["default_joint_pos"]},
            {**state, "control_targets": [0] * 3},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, RuntimeError)):
                    playback.validate_reset_observation_reference(invalid, metadata)
        self.assertEqual(
            playback.validate_reset_observation_reference({}, FakeSession().metadata),
            {"mode": "legacy_bridge_default"},
        )

    def test_calibrated_reset_restores_both_original_world_pause_states(self):
        for originally_paused in (False, True):
            with self.subTest(originally_paused=originally_paused):
                env = FakeResetEnv(paused=originally_paused)
                obs, calibration = playback.reset_with_policy_calibration(
                    env, self.ue_session().metadata
                )
                self.assertEqual(len(obs), 48)
                self.assertEqual(env.state["sim_time"], 0.0)
                self.assertEqual(env.paused, originally_paused)
                self.assertTrue(calibration["world_paused_during_reset"])
                self.assertLess(
                    env.requests.index("vget /camera/7/location"),
                    env.requests.index("vset /action/game/pause"),
                )
        legacy = FakeResetEnv()
        _, calibration = playback.reset_with_policy_calibration(legacy, FakeSession().metadata)
        self.assertEqual(legacy.requests, ["reset"])
        self.assertEqual(legacy.state["sim_time"], 0.4)
        self.assertEqual(calibration["mode"], "legacy_bridge_default")

    def test_calibrated_reset_restores_world_after_reset_or_validation_failure(self):
        for fail_inside_reset in (True, False):
            with self.subTest(fail_inside_reset=fail_inside_reset):
                env = FakeResetEnv(
                    reset_error=RuntimeError("reset failed") if fail_inside_reset else None
                )
                if not fail_inside_reset:
                    env.state["control_targets"] = playback.PROFILE["default_joint_pos"]
                with self.assertRaises(RuntimeError):
                    playback.reset_with_policy_calibration(env, self.ue_session().metadata)
                self.assertFalse(env.paused)
                self.assertEqual(env.requests[-2:], [
                    "vset /action/game/resume", "vget /action/game/is_paused",
                ])

    def test_camera_startup_retries_only_missing_sensor_before_world_pause(self):
        env = FakeResetEnv(camera_replies=["error invalid sensor id", "0 0 100"])
        with patch.object(playback.time, "sleep"):
            _, calibration = playback.reset_with_policy_calibration(
                env, self.ue_session().metadata
            )
        self.assertEqual(calibration["spawn_camera_ready"]["attempts"], 2)
        self.assertEqual(env.requests[:2], ["vget /camera/7/location"] * 2)
        self.assertEqual(env.request_timeout, 30.0)

        failed = FakeResetEnv(camera_replies=["error transport failed"])
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            playback.reset_with_policy_calibration(failed, self.ue_session().metadata)
        self.assertEqual(failed.requests, ["vget /camera/7/location"])
        self.assertEqual(failed.request_timeout, 30.0)

    def test_camera_startup_deadline_expires_without_pausing_world(self):
        env = FakeResetEnv(camera_replies=["error invalid sensor id"])
        clock = [0.0]

        def monotonic():
            clock[0] += 0.1
            return clock[0]

        with patch.object(playback.time, "monotonic", side_effect=monotonic):
            with patch.object(playback.time, "sleep"):
                with self.assertRaises(TimeoutError):
                    playback.reset_with_policy_calibration(
                        env, self.ue_session().metadata, startup_timeout=0.3
                    )
        self.assertFalse(env.paused)
        self.assertNotIn("vset /action/game/pause", env.requests)
        self.assertEqual(env.request_timeout, 30.0)


if __name__ == "__main__":
    unittest.main()
