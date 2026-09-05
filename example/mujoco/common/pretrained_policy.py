"""Pretrained ONNX policies used by the MuJoCo Gym example."""
from pathlib import Path

import numpy as np
import onnxruntime as ort


POLICY_ROOT = Path(__file__).resolve().parents[1] / "policies"
DEFAULT_PATHS = {
    "go1": POLICY_ROOT / "go1" / "velocity" / "policy.onnx",
    "g1": POLICY_ROOT / "g1" / "velocity" / "policy.onnx",
    "microduck_walking": POLICY_ROOT / "microduck" / "alpha_walking.onnx",
    "microduck_standing": POLICY_ROOT / "microduck" / "alpha_stand.onnx",
}
GO1_BRIDGE_DEFAULT = np.asarray(
    [
        0.1, 0.9, -1.8,
        -0.1, 0.9, -1.8,
        0.1, 0.9, -1.8,
        -0.1, 0.9, -1.8,
    ],
    dtype=np.float32,
)


class OnnxCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return self.session.run(
            [self.output.name], {self.input.name: observation}
        )[0][0].astype(np.float32)


class PretrainedPolicy:
    """Robot-specific checkpoint selection with a common act method."""

    def __init__(
        self,
        robot,
        policy_path="",
        walking_path="",
        standing_path="",
    ):
        self.robot = robot
        self.last_action = np.zeros(
            {"go1": 12, "g1": 29, "microduck": 14}[robot],
            dtype=np.float32,
        )

        if robot == "microduck":
            self.walking = OnnxCheckpoint(
                walking_path or DEFAULT_PATHS["microduck_walking"]
            )
            self.standing = OnnxCheckpoint(
                standing_path or DEFAULT_PATHS["microduck_standing"]
            )
        else:
            self.checkpoint = OnnxCheckpoint(
                policy_path or DEFAULT_PATHS[robot]
            )

        if robot == "go1":
            metadata = self.checkpoint.session.get_modelmeta().custom_metadata_map
            self.go1_default = np.asarray(
                metadata["default_joint_pos"].split(","), dtype=np.float32
            )

    def reset(self):
        self.last_action.fill(0.0)

    def act(self, observation, command):
        policy_observation = np.asarray(observation, dtype=np.float32).copy()

        if self.robot == "go1":
            joint_position = (
                policy_observation[9:21] + GO1_BRIDGE_DEFAULT
            )
            policy_observation[9:21] = joint_position - self.go1_default
            policy_observation[33:45] = self.last_action
            action = self.checkpoint(policy_observation)
        elif self.robot == "g1":
            action = self.checkpoint(policy_observation)
        else:
            moving = np.linalg.norm(command) >= 0.05
            checkpoint = self.walking if moving else self.standing
            action = checkpoint(policy_observation)

        self.last_action = action.copy()
        return action
