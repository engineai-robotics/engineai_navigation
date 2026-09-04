import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from engineai_rl_lab.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    get_actor_obs_normalizer,
)


def _get_logger_type(runner: OnPolicyRunner) -> str | None:
    if hasattr(runner, "logger_type"):
        return runner.logger_type
    logger = getattr(runner, "logger", None)
    return getattr(logger, "logger_type", None)


def _get_policy(runner: OnPolicyRunner):
    if hasattr(runner.alg, "get_policy"):
        return runner.alg.get_policy()
    return runner.alg.policy


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _get_logger_type(self) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                _get_policy(self),
                normalizer=get_actor_obs_normalizer(self),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _get_logger_type(self) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                _get_policy(self),
                normalizer=get_actor_obs_normalizer(self),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
