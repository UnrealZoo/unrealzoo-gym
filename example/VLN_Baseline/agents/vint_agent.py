import os
import sys
from collections import deque

import numpy as np
from PIL import Image

from agents.base_vln_agent import BaseVlnAgent


class VintAgent(BaseVlnAgent):
    """Adapter around ViNT's image-goal navigation policy."""

    method_name = "vint"
    model_key = "vint"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--visualnav-root",
            default=None,
            help="Path to visualnav-transformer. Defaults to example/VLN_Baseline/visualnav-transformer.",
        )
        parser.add_argument(
            "--visualnav-model-path",
            default=None,
            help="Optional checkpoint path for ViNT/NoMaD. Defaults to deployment/config/models.yaml.",
        )
        parser.add_argument(
            "--visualnav-config-path",
            default=None,
            help="Optional model config path for ViNT/NoMaD. Defaults to deployment/config/models.yaml.",
        )
        parser.add_argument(
            "--goal-image",
            default=None,
            help="Goal image used by ViNT/NoMaD image-goal policies.",
        )
        parser.add_argument(
            "--visualnav-waypoint",
            type=int,
            default=2,
            help="Predicted waypoint index used to choose the next action.",
        )
        parser.add_argument(
            "--visualnav-forward-threshold",
            type=float,
            default=0.05,
            help="Minimum forward waypoint value to emit forward instead of stop.",
        )
        parser.add_argument(
            "--visualnav-turn-threshold",
            type=float,
            default=0.05,
            help="Minimum lateral waypoint value to emit left/right.",
        )
        parser.add_argument(
            "--visualnav-num-samples",
            type=int,
            default=8,
            help="Number of NoMaD diffusion samples.",
        )

    def __init__(self, args):
        if not args.goal_image:
            raise ValueError("--goal-image is required when --method vint is selected.")
        self.visualnav_root = os.path.abspath(args.visualnav_root or self._default_visualnav_root())
        self._ensure_visualnav_importable()

        self.device = self._make_device()
        self.model_params = self._load_model_params(args)
        self.model_path = os.path.abspath(args.visualnav_model_path or self._default_model_path())
        self.model = self._load_model()
        self.goal_image = Image.open(args.goal_image).convert("RGB")
        self.context = deque(maxlen=int(self.model_params["context_size"]) + 1)
        self.waypoint_index = int(args.visualnav_waypoint)
        self.forward_threshold = float(args.visualnav_forward_threshold)
        self.turn_threshold = float(args.visualnav_turn_threshold)
        self.latest_waypoint = None

    def reset(self):
        self.context.clear()
        self.latest_waypoint = None

    def predict(self, observation, instruction, info=None):
        image = self._to_pil(observation)
        self.context.append(image)
        while len(self.context) < self.context.maxlen:
            self.context.appendleft(image)

        waypoint = self._predict_waypoint()
        self.latest_waypoint = waypoint
        return [self._waypoint_to_text_action(waypoint)]

    def _predict_waypoint(self):
        import torch
        from utils import to_numpy, transform_images

        obs_img = transform_images(list(self.context), self.model_params["image_size"]).to(self.device)
        goal_img = transform_images(self.goal_image, self.model_params["image_size"]).to(self.device)
        with torch.no_grad():
            _, waypoints = self.model(obs_img, goal_img)
        waypoints = to_numpy(waypoints)[0]
        idx = min(self.waypoint_index, len(waypoints) - 1)
        return waypoints[idx]

    def _load_model(self):
        from utils import load_model

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ViNT checkpoint not found: {self.model_path}. "
                "Pass --visualnav-model-path to a valid .pth file."
            )
        return load_model(self.model_path, self.model_params, self.device).to(self.device).eval()

    def _load_model_params(self, args):
        import yaml

        config_path = os.path.abspath(args.visualnav_config_path or self._default_config_path())
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"ViNT config not found: {config_path}. "
                "Pass --visualnav-config-path to a valid YAML config."
            )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _default_model_path(self):
        return self._path_from_models_yaml("ckpt_path")

    def _default_config_path(self):
        return self._path_from_models_yaml("config_path")

    def _path_from_models_yaml(self, key):
        import yaml

        models_path = os.path.join(self.visualnav_root, "deployment", "config", "models.yaml")
        with open(models_path, "r") as f:
            models = yaml.safe_load(f)
        rel_path = models[self.model_key][key]
        return os.path.abspath(os.path.join(self.visualnav_root, "deployment", "src", rel_path))

    def _ensure_visualnav_importable(self):
        if not os.path.isdir(self.visualnav_root):
            raise FileNotFoundError(f"visualnav-transformer root does not exist: {self.visualnav_root}")
        paths = [
            os.path.join(self.visualnav_root, "train"),
            os.path.join(self.visualnav_root, "deployment", "src"),
            os.path.join(self.visualnav_root, "diffusion_policy"),
            self.visualnav_root,
        ]
        for path in paths:
            if path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _default_visualnav_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "visualnav-transformer"))

    @staticmethod
    def _make_device():
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _to_pil(observation):
        obs = np.asarray(observation)
        if obs.ndim != 3:
            raise ValueError(f"Expected image observation with 3 dims, got shape {obs.shape}")
        if obs.shape[2] > 3:
            obs = obs[:, :, :3]
        if obs.dtype != np.uint8:
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        return Image.fromarray(obs).convert("RGB")

    def _waypoint_to_text_action(self, waypoint):
        waypoint = np.asarray(waypoint)
        x = float(waypoint[0]) if waypoint.shape[0] > 0 else 0.0
        y = float(waypoint[1]) if waypoint.shape[0] > 1 else 0.0
        if abs(y) > self.turn_threshold and abs(y) >= abs(x):
            return "left" if y > 0 else "right"
        if x > self.forward_threshold:
            return "forward"
        return "stop"
