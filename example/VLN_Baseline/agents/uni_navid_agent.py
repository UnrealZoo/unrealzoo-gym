import os
import re
import sys

import numpy as np

from agents.base_vln_agent import BaseVlnAgent


VALID_ACTIONS = {"forward", "left", "right", "stop"}


class UniNavidAgent(BaseVlnAgent):
    """Adapter around Uni-NaVid's offline inference agent."""

    method_name = "uni_navid"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--model-path",
            default=None,
            help="Path to the Uni-NaVid checkpoint directory.",
        )
        parser.add_argument(
            "--uninavid-root",
            default=None,
            help="Path to the Uni-NaVid repo. Defaults to example/VLN_Baseline/Uni-NaVid.",
        )

    def __init__(self, args):
        if not args.model_path:
            raise ValueError("--model-path is required when --method uni_navid is selected.")
        self.uninavid_root = os.path.abspath(args.uninavid_root or self._default_uninavid_root())
        self.model_path = os.path.abspath(args.model_path)
        self._ensure_uninavid_importable()

        try:
            from offline_eval_uninavid import UniNaVid_Agent
        except ImportError as exc:
            raise ImportError(
                "Failed to import Uni-NaVid. Install the missing dependency shown above "
                "and check --uninavid-root if the module path is wrong."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Uni-NaVid model path does not exist: {self.model_path}")

        old_cwd = os.getcwd()
        try:
            # Uni-NaVid checkpoints may contain relative paths to model_zoo assets.
            os.chdir(self.uninavid_root)
            self.agent = UniNaVid_Agent(self.model_path)
        finally:
            os.chdir(old_cwd)
        self.latest_raw_result = None

    def reset(self):
        self.agent.reset()
        self.latest_raw_result = None

    def predict(self, observation, instruction, info=None):
        rgb = self._prepare_observation(observation)
        result = self.agent.act({"instruction": instruction, "observations": rgb})
        self.latest_raw_result = result
        return self._extract_actions(result)

    def _ensure_uninavid_importable(self):
        if not os.path.isdir(self.uninavid_root):
            raise FileNotFoundError(f"Uni-NaVid root does not exist: {self.uninavid_root}")
        if self.uninavid_root not in sys.path:
            sys.path.insert(0, self.uninavid_root)

    @staticmethod
    def _default_uninavid_root():
        vln_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(vln_root, "Uni-NaVid")

    @staticmethod
    def _prepare_observation(observation):
        obs = np.asarray(observation)
        if obs.ndim != 3:
            raise ValueError(f"Expected image observation with 3 dims, got shape {obs.shape}")
        if obs.shape[2] > 3:
            obs = obs[:, :, :3]
        if obs.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image observation, got shape {obs.shape}")
        if obs.dtype != np.uint8:
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        return obs

    @staticmethod
    def _extract_actions(result):
        raw_actions = result.get("actions", [])
        if isinstance(raw_actions, str):
            raw_text = raw_actions
        else:
            raw_text = " ".join(str(action) for action in raw_actions)

        actions = re.findall(r"\b(forward|left|right|stop)\b", raw_text.lower())
        actions = [action for action in actions if action in VALID_ACTIONS]
        if not actions:
            raise ValueError(f"Uni-NaVid returned no valid navigation actions: {raw_text!r}")
        return actions
