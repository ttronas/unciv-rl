"""Stable-Baselines3-compatible gymnasium.Env wrapper with self-play.

Wraps :class:`~rl_env.UncivEnv` (via :class:`~rl_env.wrappers.UncivMacroWrapper`)
as a **single-agent** ``gymnasium.Env`` suitable for ``MaskablePPO`` from
`sb3-contrib <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>`_.

Self-play design
----------------
Unciv is a multi-agent turn-based game.  SB3 expects a single-agent interface,
so this wrapper adopts a **self-play** strategy:

* The *training* agent always occupies the ``player_0`` slot.
* Every other agent (``player_1``, ``player_2``, …) is controlled by a
  **random-valid-action** opponent: on its turn, the wrapper samples uniformly
  from the legal macro actions reported by its action mask.
* After the training agent submits its action the wrapper automatically
  advances through all opponent turns (and any dead steps) until ``player_0``
  is active again — or the episode terminates.

The training agent's **cumulative reward delta** across all those intermediate
steps is returned as the step reward.

Action masking
--------------
``MaskablePPO`` calls ``env.action_masks()`` before sampling.  This method
returns the training agent's current macro mask as a ``bool`` array of shape
``(N_MACRO_ACTIONS,)``.

Spaces
------
* ``observation_space`` – ``Box(-inf, inf, (flat_obs_dim,), float32)``
* ``action_space``      – ``Discrete(N_MACRO_ACTIONS)``

Usage::

    from rl_env.wrappers.sb3_wrapper import UncivSB3Wrapper
    from sb3_contrib import MaskablePPO

    env = UncivSB3Wrapper(base_url="http://localhost:8080", num_agents=2)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from rl_env.constants import N_MACRO_ACTIONS
from rl_env.wrappers.rllib_macro_wrapper import UncivMacroWrapper, flat_obs_dim


class UncivSB3Wrapper(gymnasium.Env):
    """Single-agent ``gymnasium.Env`` around ``UncivEnv`` for MaskablePPO.

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server (default: ``"http://localhost:8080"``).
    num_agents:
        Total number of RL-controlled civilisations (default: ``2``).  One of
        them (``player_0``) is the training agent; the rest are self-play
        opponents using a random valid policy.
    num_ai:
        Number of built-in AI civilisations (default: ``0``).
    include_map_planes:
        Whether to include spatial map channels in the flat observation vector
        (adds ~28 K dimensions).  Must match the value used when building the
        underlying server game (default: ``False``).
    seed:
        Optional RNG seed for the random opponent policy.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        num_agents: int = 2,
        num_ai: int = 0,
        include_map_planes: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        from rl_env import UncivEnv  # local import to avoid circular deps

        self._training_agent = "player_0"
        self._include_map_planes = include_map_planes
        self._flat_dim = flat_obs_dim(include_map_planes)
        self._rng = np.random.default_rng(seed)

        aec_env = UncivEnv(
            base_url=base_url,
            num_agents=num_agents,
            num_ai=num_ai,
            num_city_states=0,
            no_barbarians=True,
            include_map_planes=include_map_planes,
        )
        self._aec = UncivMacroWrapper(aec_env, include_map_planes=include_map_planes)

        # Spaces (no server call needed – UncivMacroWrapper builds them in __init__)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._flat_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_MACRO_ACTIONS)

        # Cache for the latest training-agent action mask (needed by action_masks())
        self._current_mask: np.ndarray = np.ones(N_MACRO_ACTIONS, dtype=bool)

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._aec.reset(seed=seed, options=options)
        # Advance through any dead / non-training-agent initial turns
        self._advance_to_training_agent()
        obs = self._get_training_agent_obs()
        return obs, {}

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one action for the training agent, then auto-advance opponents."""
        # Snapshot cumulative reward before the training agent's step
        unciv_env = self._aec.env
        reward_before = unciv_env._cumulative_rewards.get(self._training_agent, 0.0)

        # Training agent step
        if (
            self._aec.terminations.get(self._training_agent, False)
            or self._aec.truncations.get(self._training_agent, False)
        ):
            self._aec.step(None)
        else:
            self._aec.step(int(action))

        # Let opponents play until it's the training agent's turn again
        self._advance_to_training_agent()

        # Reward = cumulative reward delta for training agent across all steps
        reward_after = unciv_env._cumulative_rewards.get(self._training_agent, 0.0)
        reward = float(reward_after - reward_before)

        terminated = self._aec.terminations.get(self._training_agent, False)
        truncated = self._aec.truncations.get(self._training_agent, False)
        done = terminated or truncated or (self._training_agent not in self._aec.agents)

        if done:
            obs = np.zeros(self._flat_dim, dtype=np.float32)
            self._current_mask = np.ones(N_MACRO_ACTIONS, dtype=bool)
        else:
            obs = self._get_training_agent_obs()

        return obs, reward, bool(terminated), bool(truncated), {}

    def action_masks(self) -> np.ndarray:
        """Return the current legal macro action mask for MaskablePPO.

        Returns a ``bool`` array of shape ``(N_MACRO_ACTIONS,)`` where ``True``
        means the action is legal.  Called automatically by ``MaskablePPO``
        before each action sample.

        .. note::
            This method must only be called after at least one :meth:`reset`.
            Before the first reset, the mask defaults to all-True (all actions
            permitted), which is not a meaningful game state.
        """
        return self._current_mask.copy()

    def render(self) -> None:
        self._aec.render()

    def close(self) -> None:
        self._aec.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance_to_training_agent(self) -> None:
        """Step through opponents and dead turns until player_0 is active."""
        while (
            self._aec.agents
            and self._aec.agent_selection != self._training_agent
        ):
            agent = self._aec.agent_selection
            if (
                self._aec.terminations.get(agent, False)
                or self._aec.truncations.get(agent, False)
            ):
                self._aec.step(None)
            else:
                opp_obs = self._aec.observe(agent)
                opp_mask = opp_obs["action_mask"].astype(bool)
                legal = np.where(opp_mask)[0]
                opp_action = int(self._rng.choice(legal)) if len(legal) > 0 else 0
                self._aec.step(opp_action)

    def _get_training_agent_obs(self) -> np.ndarray:
        """Observe the training agent and update the cached action mask."""
        raw = self._aec.observe(self._training_agent)
        self._current_mask = raw["action_mask"].astype(bool)
        return raw["observations"].copy()
