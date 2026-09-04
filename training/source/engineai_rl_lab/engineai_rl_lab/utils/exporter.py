# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import onnx
import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def get_actor_obs_normalizer(policy_or_runner: object) -> object | None:
    """Resolve the actor observation normalizer across RSL-RL versions.

    Older versions exposed the normalizer on the runner as ``obs_normalizer``.
    Newer versions keep it on the policy as ``actor_obs_normalizer``.
    """

    legacy_normalizer = getattr(policy_or_runner, "obs_normalizer", None)
    if legacy_normalizer is not None:
        return legacy_normalizer

    alg = getattr(policy_or_runner, "alg", None)
    policy = alg.get_policy() if hasattr(alg, "get_policy") else getattr(alg, "policy", policy_or_runner)
    return getattr(policy, "actor_obs_normalizer", getattr(policy, "obs_normalizer", None))


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if hasattr(actor_critic, "as_onnx"):
        policy_exporter = actor_critic.as_onnx(verbose=verbose)
        policy_exporter.to("cpu")
        policy_exporter.eval()
        torch.onnx.export(
            policy_exporter,
            policy_exporter.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=policy_exporter.input_names,
            output_names=policy_exporter.output_names,
        )
        return

    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

    def forward(self, x):
        return (self.actor(self.normalizer(x)),)

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        torch.onnx.export(
            self,
            (obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=[
                "actions",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "default_joint_pos": (
            getattr(env.scene["robot"].data, "default_joint_pos_nominal", env.scene["robot"].data.default_joint_pos[0])
            .cpu()
            .tolist()
        ),
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.default_joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.default_joint_damping[0].cpu().tolist(),
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
    }

    class CustomListDumper(yaml.SafeDumper):
        def represent_sequence(self, tag, sequence, flow_style=None):
            node = yaml.SafeDumper.represent_sequence(self, tag, sequence, flow_style=True)
            if isinstance(sequence, list):
                for i, item_node in enumerate(node.value):
                    if isinstance(sequence[i], str):
                        item_node.style = '"'
            return node

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.path.join(path, "deploy_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(
            metadata,
            f,
            Dumper=CustomListDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )
    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
