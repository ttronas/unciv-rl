"""Two-agent Unciv training example using MaskablePPO from sb3-contrib.

This script trains one RL agent to play Unciv using **MaskablePPO** from
``sb3-contrib``.  A second agent (the *opponent*) is present in the same game
but plays a **random valid-action policy**, creating a self-play training setup:

Architecture
------------
``UncivEnv`` (AEC, 2 agents)
    → ``UncivMacroWrapper``  – flattens obs, exposes ``Discrete(10)`` macro action
    → ``UncivSB3Wrapper``    – single-agent gymnasium.Env (player_0 trains,
                                player_1 plays random valid actions)

The ``UncivSB3Wrapper`` implements ``action_masks() → bool[N_MACRO_ACTIONS]``,
which ``MaskablePPO`` calls before every action sample to prevent the policy from
selecting illegal macro actions.

Usage
-----
Ensure an Unciv server is running with the ``--rl`` flag::

    java -jar server/build/libs/UncivServer.jar --rl -no-auth -p 8080

Then run this script::

    python -m rl_env.examples.sb3_two_agents \\
        --base-url http://localhost:8080 \\
        --total-timesteps 50000

Dependencies
------------
    pip install stable-baselines3 sb3-contrib
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# sb3 / sb3-contrib imports (guarded so the module can still be imported
# without the library installed, e.g. during unit tests)
# ---------------------------------------------------------------------------
try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_checker import check_env
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_unciv_env(
    base_url: str = "http://localhost:8080",
    num_agents: int = 2,
    num_ai: int = 0,
    include_map_planes: bool = False,
    seed: int | None = None,
) -> "UncivSB3Wrapper":
    """Create a :class:`~rl_env.wrappers.UncivSB3Wrapper` for MaskablePPO.

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server.
    num_agents:
        Total RL-controlled players (one trains, the rest play randomly).
    num_ai:
        Number of built-in AI players.
    include_map_planes:
        Whether to include spatial map channels in observations (~28 K extra dims).
    seed:
        RNG seed for the random opponent policy.
    """
    from rl_env.wrappers.sb3_wrapper import UncivSB3Wrapper
    return UncivSB3Wrapper(
        base_url=base_url,
        num_agents=num_agents,
        num_ai=num_ai,
        include_map_planes=include_map_planes,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train(
    base_url: str = "http://localhost:8080",
    total_timesteps: int = 50_000,
    n_steps: int = 512,
    batch_size: int = 64,
    include_map_planes: bool = False,
    seed: int | None = None,
    verbose: int = 1,
) -> "MaskablePPO":
    """Train a MaskablePPO agent on the Unciv environment.

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server.
    total_timesteps:
        Total environment steps for training.
    n_steps:
        Number of steps per rollout buffer before an update.
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
            "sb3-contrib is required. Install with: pip install stable-baselines3 sb3-contrib"
        )

    env = make_unciv_env(
        base_url=base_url,
        num_agents=2,
        num_ai=0,
        include_map_planes=include_map_planes,
        seed=seed,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose,
        policy_kwargs={"net_arch": [256, 128]},
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MaskablePPO agent to play Unciv (self-play: "
            "player_0 trains, others act randomly)."
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
        help="Total environment steps for training (default: 50 000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Steps per rollout buffer before a PPO update (default: 512)",
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
        help="Include spatial map planes in observations (large: +28K dims)",
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
        help="If given, save the trained model to this path (e.g. unciv_maskable_ppo)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not _HAS_SB3:
        raise SystemExit(
            "sb3-contrib is required. Install with: pip install stable-baselines3 sb3-contrib"
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
