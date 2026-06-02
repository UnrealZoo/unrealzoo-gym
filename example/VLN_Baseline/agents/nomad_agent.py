import os

from agents.vint_agent import VintAgent


class NomadAgent(VintAgent):
    """Adapter around NoMaD's image-goal diffusion navigation policy."""

    method_name = "nomad"
    model_key = "nomad"

    @classmethod
    def add_cli_args(cls, parser):
        # ViNT registers the shared visualnav CLI flags once.
        return

    def __init__(self, args):
        if not args.goal_image:
            raise ValueError("--goal-image is required when --method nomad is selected.")
        super().__init__(args)
        self.num_samples = int(args.visualnav_num_samples)
        self.noise_scheduler = self._build_noise_scheduler()

    def _predict_waypoint(self):
        import torch
        from utils import to_numpy, transform_images
        from vint_train.training.train_utils import get_action

        obs_images = transform_images(
            list(self.context),
            self.model_params["image_size"],
            center_crop=False,
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        goal_image = transform_images(
            self.goal_image,
            self.model_params["image_size"],
            center_crop=False,
        ).to(self.device)
        mask = torch.zeros(1).long().to(self.device)

        with torch.no_grad():
            obs_cond = self.model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=goal_image,
                input_goal_mask=mask,
            )
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

            naction = torch.randn(
                (self.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device,
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for timestep in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=timestep,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=naction,
                ).prev_sample

        waypoints = to_numpy(get_action(naction))[0]
        idx = min(self.waypoint_index, len(waypoints) - 1)
        return waypoints[idx]

    def _load_model(self):
        from utils import load_model

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"NoMaD checkpoint not found: {self.model_path}. "
                "Pass --visualnav-model-path to a valid .pth file."
            )
        return load_model(self.model_path, self.model_params, self.device).to(self.device).eval()

    def _build_noise_scheduler(self):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        return DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
