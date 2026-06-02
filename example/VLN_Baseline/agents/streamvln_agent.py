import os
import sys
import types
from argparse import Namespace

import numpy as np
import torch
import transformers

from agents.base_vln_agent import BaseVlnAgent


class StreamVlnAgent(BaseVlnAgent):
    """Adapter around StreamVLN's online evaluator."""

    method_name = "streamvln"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--streamvln-root",
            default=None,
            help="Path to StreamVLN. Defaults to example/VLN_Baseline/StreamVLN.",
        )
        parser.add_argument(
            "--streamvln-model-path",
            default=None,
            help="Path to StreamVLN checkpoint directory.",
        )
        parser.add_argument("--streamvln-num-future-steps", type=int, default=4)
        parser.add_argument("--streamvln-num-frames", type=int, default=32)
        parser.add_argument("--streamvln-num-history", type=int, default=None)
        parser.add_argument("--streamvln-model-max-length", type=int, default=4096)
        parser.add_argument("--streamvln-device", default="cuda:0")
        parser.add_argument(
            "--streamvln-attn-implementation",
            default="sdpa",
            help="Transformers attention implementation. Use flash_attention_2 only if installed.",
        )

    def __init__(self, args):
        self.streamvln_root = os.path.abspath(args.streamvln_root or self._default_streamvln_root())
        self.model_path = os.path.abspath(args.streamvln_model_path or self._default_model_path())
        self.vision_tower_path = self._default_vision_tower_path()
        print(f"[StreamVLN] root: {self.streamvln_root}", flush=True)
        print(f"[StreamVLN] checkpoint: {self.model_path}", flush=True)
        self._ensure_streamvln_importable()

        try:
            print("[StreamVLN] importing model classes...", flush=True)
            from model.stream_video_vln import StreamVLNForCausalLM
            from streamvln_agent import VLNEvaluator
        except ImportError as exc:
            raise ImportError(
                "Failed to import StreamVLN. Install the missing dependency shown above "
                "and check --streamvln-root if the module path is wrong."
            ) from exc

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"StreamVLN checkpoint directory not found: {self.model_path}. "
                "Pass --streamvln-model-path to a valid checkpoint."
            )

        self.device = torch.device(args.streamvln_device if torch.cuda.is_available() else "cpu")
        print(f"[StreamVLN] device: {self.device}", flush=True)
        print("[StreamVLN] loading tokenizer/config...", flush=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=args.streamvln_model_max_length,
            padding_side="right",
        )
        config = transformers.AutoConfig.from_pretrained(self.model_path)
        self._use_local_vision_tower(config)
        print("[StreamVLN] loading model weights, this can take several minutes...", flush=True)
        self.model = StreamVLNForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=args.streamvln_attn_implementation,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
        )
        print("[StreamVLN] moving model to device...", flush=True)
        self.model.model.num_history = args.streamvln_num_history
        self.model.reset(1)
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()

        print("[StreamVLN] initializing evaluator...", flush=True)
        self.agent = VLNEvaluator(
            sim_sensors_config={"camera_intrinsic": self._camera_intrinsic()},
            model=self.model,
            tokenizer=self.tokenizer,
            args=Namespace(
                num_frames=args.streamvln_num_frames,
                num_future_steps=args.streamvln_num_future_steps,
                num_history=args.streamvln_num_history,
            ),
        )
        self.agent.device = self.device
        self.latest_raw_result = None
        print("[StreamVLN] ready.", flush=True)

    def reset(self):
        self.agent.reset_memory()
        self.latest_raw_result = None

    def predict(self, observation, instruction, info=None):
        rgb = self._prepare_observation(observation)
        action_ids, _, raw_output = self.agent.step(0, rgb, instruction, run_model=True)
        self.latest_raw_result = raw_output
        actions = [self._action_id_to_text(action_id) for action_id in action_ids]
        actions = [action for action in actions if action is not None]
        if not actions:
            raise ValueError(f"StreamVLN returned no valid actions: {raw_output!r}")
        return actions

    def _ensure_streamvln_importable(self):
        if not os.path.isdir(self.streamvln_root):
            raise FileNotFoundError(f"StreamVLN root does not exist: {self.streamvln_root}")
        self._install_depth_filter_stub()
        paths = [
            os.path.join(self.streamvln_root, "streamvln"),
            self.streamvln_root,
        ]
        for path in paths:
            if path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _default_streamvln_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "StreamVLN"))

    def _default_model_path(self):
        return os.path.join(
            self.streamvln_root,
            "checkpoints",
            "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world",
        )

    def _default_vision_tower_path(self):
        return os.path.join(
            self.streamvln_root,
            "checkpoints",
            "siglip-so400m-patch14-384",
        )

    def _use_local_vision_tower(self, config):
        weights_path = os.path.join(self.vision_tower_path, "model.safetensors")
        if not os.path.exists(weights_path):
            print(
                "[StreamVLN] local SigLIP vision tower weights not found; "
                "transformers may try to download google/siglip-so400m-patch14-384.",
                flush=True,
            )
            return
        print(f"[StreamVLN] using local vision tower: {self.vision_tower_path}", flush=True)
        config.mm_vision_tower = self.vision_tower_path
        config.vision_tower = self.vision_tower_path

    @staticmethod
    def _prepare_observation(observation):
        obs = np.asarray(observation)
        if obs.ndim != 3:
            raise ValueError(f"Expected image observation with 3 dims, got shape {obs.shape}")
        if obs.shape[2] > 3:
            obs = obs[:, :, :3]
        if obs.dtype != np.uint8:
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        return obs

    @staticmethod
    def _action_id_to_text(action_id):
        return {
            0: "stop",
            1: "forward",
            2: "left",
            3: "right",
        }.get(int(action_id))

    @staticmethod
    def _camera_intrinsic():
        return np.array(
            [
                [192.0, 0.0, 191.42857143, 0.0],
                [0.0, 192.0, 191.42857143, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _install_depth_filter_stub():
        if "depth_camera_filtering" in sys.modules:
            return
        module = types.ModuleType("depth_camera_filtering")
        module.filter_depth = lambda depth, *args, **kwargs: depth
        sys.modules["depth_camera_filtering"] = module
