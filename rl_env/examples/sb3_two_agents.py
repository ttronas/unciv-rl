"""Two-agent Unciv training example using MaskablePPO from sb3-contrib.

This script trains a shared RL policy on Unciv using **MaskablePPO** from
``sb3-contrib``, following the canonical PettingZoo multi-agent SB3 tutorial
(https://pettingzoo.farama.org/tutorials/sb3/).

Architecture (parameter-sharing)
---------------------------------
::

    UncivEnv (AEC, 2 agents)
      → UncivMacroWrapper        – flat Dict obs, Discrete(10) macro actions
      → UncivSB3PZWrapper        – Box obs, action_masks() for active agent
      → aec_to_parallel          – both agents act each parallel step
      → pettingzoo_env_to_vec_env_v1  – VecEnv (num_envs = num_agents = 2)
      → concat_vec_envs_v1       – stacked SB3 VecEnv
      → _MaskableVecEnvWrapper   – wires action_masks() into MaskablePPO
      → MaskablePPO              – one shared policy for all agent slots

A **single policy** controls all agents (parameter sharing / self-play).
Every agent slot in the ``VecEnv`` is presented to SB3 as a separate
"environment", so the policy learns from the perspective of all players
simultaneously.  Action masking ensures only legal macro actions are sampled.

Usage
-----
Ensure an Unciv server is running with the ``--rl`` flag::

    java -jar server/build/libs/UncivServer.jar --rl -no-auth -p 8080

Then run::

    python -m rl_env.examples.sb3_two_agents \\
        --base-url http://localhost:8080 \\
        --total-timesteps 50000

Dependencies
------------
    pip install stable-baselines3 sb3-contrib supersuit
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    from sb3_contrib import MaskablePPO
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train(
    base_url: str = "http://localhost:8080",
    num_agents: int = 2,
    total_timesteps: int = 50_000,
    n_steps: int = 512,
    batch_size: int = 64,
    include_map_planes: bool = False,
    seed: int | None = None,
    verbose: int = 1,
) -> "MaskablePPO":
    """Train a shared MaskablePPO policy on the Unciv environment.

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server.
    num_agents:
        Number of civilisations; all share the same policy.
    total_timesteps:
        Total environment steps.
    n_steps:
        Rollout buffer size per VecEnv step before a PPO update.
    batch_size:
        Mini-batch size for SGD updates.
    include_map_planes:
        Whether to include spatial map channels in observations.
    seed:
        RNG seed for reproducibility.
    verbose:
        Verbosity level (0 = silent, 1 = progress, 2 = debug).

    Returns
    -------
    MaskablePPO
        The trained model.
    """
    if not _HAS_SB3:
        raise ImportError(
            "sb3-contrib is required. Install with: "
            "pip install stable-baselines3 sb3-contrib supersuit"
        )

    from rl_env.wrappers.sb3_wrapper import make_unciv_vec_env

    vec_env = make_unciv_vec_env(
        base_url=base_url,
        num_agents=num_agents,
        num_ai=0,
        include_map_planes=include_map_planes,
        num_copies=1,
        seed=seed,
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose,
        policy_kwargs={"net_arch": [256, 128]},
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a shared MaskablePPO policy to play Unciv "
            "(PettingZoo parameter-sharing pattern, 2 agents)."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Unciv RL server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50_000,
        help="Total environment steps (default: 50 000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Rollout buffer size per PPO update (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates (default: 64)",
    )
    parser.add_argument(
        "--include-map-planes",
        action="store_true",
        default=False,
        help="Include spatial map planes in observations (adds ~28K dims)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="If given, save the trained model to this path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not _HAS_SB3:
        raise SystemExit(
            "sb3-contrib is required. Install with: "
            "pip install stable-baselines3 sb3-contrib supersuit"
        )

    args = _parse_args()
    model = train(
        base_url=args.base_url,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        include_map_planes=args.include_map_planes,
        seed=args.seed,
        verbose=1,
    )
    if args.save_path:
        model.save(args.save_path)
        print(f"Model saved to {args.save_path}.zip")
