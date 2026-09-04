"""Gym registrations for PM01 navigation tasks."""

import gymnasium as gym

from . import agents, navigation_env_cfg


def _register(task_id: str, env_cfg, runner_cfg) -> None:
    gym.register(
        id=task_id,
        entry_point="engineai_rl_lab.tasks.navigation:NavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register(
    "Isaac-Navigation-PPO-PM01-v0",
    navigation_env_cfg.PM01NavigationEnvCfg,
    agents.range_image_rsl_rl_cfg.PM01NavRangeImagePPORunnerCfg,
)
