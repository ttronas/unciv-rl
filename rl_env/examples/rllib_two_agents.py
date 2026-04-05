"""Two-agent Unciv training example using Ray RLlib with action masking.

This script trains two RL agents to play Unciv against each other using PPO
with an ``ActionMaskingTorchRLModule`` that prevents the policy from selecting
illegal macro actions.

Architecture
------------
``UncivEnv`` (AEC, 2 agents)
    → ``UncivMacroWrapper``  – flattens obs, exposes ``Discrete(10)`` macro action
    → ``aec_to_parallel``    – PettingZoo AEC → Parallel conversion
    → ``PettingZooEnv``      – RLlib ``MultiAgentEnv`` adapter

The ``ActionMaskingTorchRLModule`` (defined below) is adapted from the
Ray RLlib reference implementation:
https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/action_masking_rlm.py

It reads ``obs["action_mask"]`` (a ``float32`` vector of shape
``(N_MACRO_ACTIONS,)``), applies it as a log-probability mask to the PPO
actor's action logits, and therefore never samples an illegal macro action.

Usage
-----
Ensure an Unciv server is running with the ``--rl`` flag::

    ./gradlew server:run --args="--rl --port 8080"

Then run this script::

    python -m rl_env.examples.rllib_two_agents \\
        --base-url http://localhost:8080 \\
        --num-iterations 10 \\
        --num-env-runners 1

Dependencies
------------
    pip install "ray[rllib]" torch
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Ray / RLlib imports (guarded so the module can still be imported without ray)
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    import torch

    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.framework import try_import_torch
    from ray.rllib.utils.torch_utils import FLOAT_MIN
    from ray.rllib.utils.typing import TensorType

    _torch, _nn = try_import_torch()
    _HAS_RAY = True
except ImportError:
    _HAS_RAY = False

# ---------------------------------------------------------------------------
# Env / wrapper imports
# ---------------------------------------------------------------------------
from rl_env.constants import N_MACRO_ACTIONS

# ---------------------------------------------------------------------------
# Environment name used for tune.register_env
# ---------------------------------------------------------------------------
ENV_NAME = "unciv_macro_multi_v0"


# ---------------------------------------------------------------------------
# ActionMaskingTorchRLModule
# ---------------------------------------------------------------------------
# Adapted from the Ray RLlib reference:
# rllib/examples/rl_modules/classes/action_masking_rlm.py
#
# The module expects each agent's observation dict to have two keys:
#   "observations"  – flat float32 Box vector (the actual state)
#   "action_mask"   – float32 Box vector of shape (N_MACRO_ACTIONS,)
#                     where 1.0 = legal, 0.0 = illegal
# The action space must be Discrete(N_MACRO_ACTIONS).
# ---------------------------------------------------------------------------

if _HAS_RAY:

    class ActionMaskingRLModule(RLModule):
        """Base mixin that sets up the observation space stripping the mask."""

        @override(RLModule)
        def __init__(
            self,
            *,
            observation_space: Optional[gym.Space] = None,
            action_space: Optional[gym.Space] = None,
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            model_config: Optional[Union[dict, DefaultModelConfig]] = None,
            catalog_class=None,
            **kwargs: Any,
        ) -> None:
            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError(
                    "UncivActionMaskingRLModule requires a Dict observation space "
                    "with keys 'observations' and 'action_mask'."
                )
            self.observation_space_with_mask = observation_space
            # The PPO base module builds its networks for the inner obs space only.
            self.observation_space = observation_space["observations"]
            self._checked_observations = False
            super().__init__(
                observation_space=self.observation_space,
                action_space=action_space,
                inference_only=inference_only,
                learner_only=learner_only,
                model_config=model_config,
                catalog_class=catalog_class,
                **kwargs,
            )

    class UncivActionMaskingTorchRLModule(ActionMaskingRLModule, PPOTorchRLModule):
        """PPO RLModule that applies the Unciv macro action mask.

        Inherits from both ``ActionMaskingRLModule`` (which strips the mask from
        the observation) and ``PPOTorchRLModule`` (which provides the actor /
        critic network and the full PPO forward logic).
        """

        @override(PPOTorchRLModule)
        def setup(self) -> None:
            super().setup()
            # Restore the full (with-mask) observation space on ``self`` so that
            # RLlib's framework sees the correct space.
            self.observation_space = self.observation_space_with_mask

        @override(PPOTorchRLModule)
        def _forward_inference(
            self, batch: Dict[str, TensorType], **kwargs: Any
        ) -> Dict[str, TensorType]:
            action_mask, batch = self._preprocess_batch(batch)
            outs = super()._forward_inference(batch, **kwargs)
            return self._mask_action_logits(outs, action_mask)

        @override(PPOTorchRLModule)
        def _forward_exploration(
            self, batch: Dict[str, TensorType], **kwargs: Any
        ) -> Dict[str, TensorType]:
            action_mask, batch = self._preprocess_batch(batch)
            outs = super()._forward_exploration(batch, **kwargs)
            return self._mask_action_logits(outs, action_mask)

        @override(PPOTorchRLModule)
        def _forward_train(
            self, batch: Dict[str, TensorType], **kwargs: Any
        ) -> Dict[str, TensorType]:
            outs = super()._forward_train(batch, **kwargs)
            return self._mask_action_logits(outs, batch["action_mask"])

        @override(ValueFunctionAPI)
        def compute_values(
            self,
            batch: Dict[str, TensorType],
            embeddings: Any = None,
        ) -> TensorType:
            if isinstance(batch[Columns.OBS], dict):
                action_mask, batch = self._preprocess_batch(batch)
                batch["action_mask"] = action_mask
            return super().compute_values(batch, embeddings)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _preprocess_batch(
            self, batch: Dict[str, TensorType]
        ) -> Tuple[TensorType, Dict[str, TensorType]]:
            self._check_batch(batch)
            action_mask = batch[Columns.OBS].pop("action_mask")
            batch[Columns.OBS] = batch[Columns.OBS].pop("observations")
            return action_mask, batch

        def _mask_action_logits(
            self,
            batch: Dict[str, TensorType],
            action_mask: TensorType,
        ) -> Dict[str, TensorType]:
            inf_mask = _torch.clamp(_torch.log(action_mask), min=FLOAT_MIN)
            batch[Columns.ACTION_DIST_INPUTS] += inf_mask
            return batch

        def _check_batch(self, batch: Dict[str, TensorType]) -> None:
            if not self._checked_observations:
                obs = batch[Columns.OBS]
                if "action_mask" not in obs or "observations" not in obs:
                    raise ValueError(
                        "Batch obs must contain keys 'action_mask' and "
                        "'observations'.  Got: " + str(list(obs.keys()))
                    )
                self._checked_observations = True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_unciv_env(config: dict) -> "UncivMultiAgentEnv":
    """Factory function registered with ``tune.register_env``.

    Creates a :class:`~rl_env.wrappers.UncivMultiAgentEnv` that wraps the
    AEC environment directly, without any ``aec_to_parallel`` conversion.
    The environment is turn-based: only the active agent is queried per step.

    Parameters
    ----------
    config:
        Dict accepted by ``tune.register_env`` factory.  Keys:

        * ``base_url`` – Unciv server URL (default: ``"http://localhost:8080"``)
        * ``num_agents`` – number of RL agents (default: ``2``)
        * ``num_ai`` – number of AI agents (default: ``0``)
        * ``include_map_planes`` – whether to include map planes (default: ``False``)
    """
    from rl_env.wrappers import UncivMultiAgentEnv  # imported here to avoid circular deps
    return UncivMultiAgentEnv(config)


# ---------------------------------------------------------------------------
# Build the RLlib PPO config
# ---------------------------------------------------------------------------

def build_ppo_config(
    base_url: str = "http://localhost:8080",
    num_iterations: int = 10,
    num_env_runners: int = 1,
    train_batch_size: int = 2000,
    include_map_planes: bool = False,
) -> "PPOConfig":
    """Return a ``PPOConfig`` for the two-agent Unciv environment.

    Both agents share a single :class:`UncivActionMaskingTorchRLModule`.

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server.
    num_iterations:
        Number of PPO training iterations.
    num_env_runners:
        Number of parallel environment runners.
    train_batch_size:
        Total number of env steps per training iteration.
    include_map_planes:
        Pass ``True`` to include the spatial map channels in the observation.
    """
    if not _HAS_RAY:
        raise ImportError("ray[rllib] and torch are required. Install with: pip install 'ray[rllib]' torch")

    env_config = {
        "base_url": base_url,
        "num_agents": 2,
        "num_ai": 0,
        "include_map_planes": include_map_planes,
    }

    from rl_env.wrappers.rllib_macro_wrapper import flat_obs_dim
    from gymnasium.spaces import Box, Discrete

    obs_flat_dim = flat_obs_dim(include_map_planes)
    obs_space = gym.spaces.Dict(
        {
            "observations": Box(
                low=-np.inf, high=np.inf, shape=(obs_flat_dim,), dtype=np.float32
            ),
            "action_mask": Box(
                low=0.0, high=1.0, shape=(N_MACRO_ACTIONS,), dtype=np.float32
            ),
        }
    )
    act_space = Discrete(N_MACRO_ACTIONS)

    tune.register_env(ENV_NAME, make_unciv_env)

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .env_runners(num_env_runners=num_env_runners)
        .training(train_batch_size=train_batch_size)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=UncivActionMaskingTorchRLModule,
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config={
                            "head_fcnet_hiddens": [256, 128],
                            "head_fcnet_activation": "relu",
                        },
                    )
                }
            )
        )
        # Replace the deprecated UnifiedLogger (removed in Ray 2.7) with a
        # no-op logger.  Training results are returned directly from
        # algo.train(); file-based logging is not needed for correctness.
        .debugging(logger_config={"type": "ray.tune.logger.NoopLogger"})
    )
    return config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two Unciv agents with PPO + action masking via Ray RLlib."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Unciv RL server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of PPO training iterations (default: 10)",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=1,
        help="Number of parallel environment runners (default: 1)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2000,
        help="Training batch size in env steps (default: 2000)",
    )
    parser.add_argument(
        "--include-map-planes",
        action="store_true",
        default=False,
        help="Include spatial map planes in observations (large: +28K dims)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not _HAS_RAY:
        raise SystemExit("ray[rllib] and torch are required. Install with: pip install 'ray[rllib]' torch")

    import os
    args = _parse_args()

    # Opt in to the future Ray default: do not override accelerator env vars
    # when num_gpus=0 (silences FutureWarning from ray._private.worker).
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    ray.init(ignore_reinit_error=True)

    config = build_ppo_config(
        base_url=args.base_url,
        num_iterations=args.num_iterations,
        num_env_runners=args.num_env_runners,
        train_batch_size=args.train_batch_size,
        include_map_planes=args.include_map_planes,
    )

    algo = config.build()
    for i in range(args.num_iterations):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get(
            "episode_reward_mean", float("nan")
        )
        print(f"Iteration {i + 1}/{args.num_iterations}  mean_reward={mean_reward:.3f}")

    algo.stop()
    ray.shutdown()
