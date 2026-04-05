"""RLlib-compatible PettingZoo AEC wrapper with macro action masking.

Presents a simplified interface to the Unciv environment that is compatible
with Ray RLlib's ``PPO`` algorithm and the ``ActionMaskingTorchRLModule``:

* **Observation space** (per agent)::

      Dict({
          "observations": Box(-inf, inf, shape=(flat_dim,), dtype=float32),
          "action_mask":  Box(0.0, 1.0, shape=(N_MACRO_ACTIONS,), dtype=float32),
      })

* **Action space** (per agent)::

      Discrete(N_MACRO_ACTIONS)

The macro-action integer output by the RLlib agent is forwarded to the
underlying :class:`~rl_env.UncivEnv`; all remaining sub-action fields
(``unit_target``, ``tile_target``, …) default to 0.

The wrapper preserves the PettingZoo AEC API so the underlying environment
can still be tested with ``pettingzoo.test.api_test``.

:class:`UncivMultiAgentEnv` bridges the AEC environment to RLlib's
``MultiAgentEnv`` interface *without* an ``aec_to_parallel`` conversion,
preserving the turn-based semantics: only the currently active agent is
asked for an action on each step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseWrapper

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_MACRO_ACTIONS,
    N_SCALAR_FEATURES,
)


def flat_obs_dim(include_map_planes: bool = False) -> int:
    """Return the total number of floats in the flattened observation vector.

    Parameters
    ----------
    include_map_planes:
        When ``True``, the spatial map channels are included (adds
        ``N_MAP_CHANNELS × MAX_TILES`` dimensions).
    """
    dim = N_SCALAR_FEATURES + MAX_ENTITIES * ENTITY_FEATURES + MAX_ENTITIES
    if include_map_planes:
        dim += N_MAP_CHANNELS * MAX_TILES
    return dim


class UncivMacroWrapper(BaseWrapper):
    """PettingZoo AEC wrapper that exposes a flat, action-masked observation.

    This wrapper is the recommended bridge between :class:`~rl_env.UncivEnv`
    and Ray RLlib.  It:

    1. Flattens the nested ``Dict`` observation from :class:`~rl_env.UncivEnv`
       into a single ``float32`` vector.
    2. Exposes only the *macro* action mask (shape ``(N_MACRO_ACTIONS,)``) as a
       ``float32`` array, which is the format expected by
       ``ActionMaskingTorchRLModule``.
    3. Accepts a plain integer macro action and forwards it to the underlying
       environment as a full action dict (sub-action fields default to 0).

    The wrapper keeps the ``AECEnv`` interface fully intact, so
    ``pettingzoo.test.api_test`` still works on a wrapped instance.

    Parameters
    ----------
    env:
        A :class:`~rl_env.UncivEnv` (or any AEC env with the same observation
        structure).
    include_map_planes:
        When ``True`` the spatial map channels are included in the flat
        ``"observations"`` vector (adds ~28 K dimensions per agent).  Set to
        the same value used when constructing the underlying ``UncivEnv``.
    """

    def __init__(self, env: Any, include_map_planes: bool = False) -> None:
        super().__init__(env)
        self._include_map_planes = include_map_planes
        self._obs_flat_dim = flat_obs_dim(include_map_planes)

        # Pre-build the space objects once; they are the same for every agent.
        self._obs_space = spaces.Dict(
            {
                "observations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._obs_flat_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(N_MACRO_ACTIONS,),
                    dtype=np.float32,
                ),
            }
        )
        self._act_space = spaces.Discrete(N_MACRO_ACTIONS)

    # ------------------------------------------------------------------
    # Space overrides
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self._act_space

    # ------------------------------------------------------------------
    # Core AEC API overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        observations, infos = self.env.reset(seed=seed, options=options)
        return {a: self._reshape_obs(o) for a, o in observations.items()}, infos

    def observe(self, agent: str) -> dict:
        return self._reshape_obs(self.env.observe(agent))

    def step(self, action: int | np.integer | None) -> None:
        """Convert an integer macro action to a full action dict and step."""
        if action is None or (
            self.terminations.get(self.agent_selection, False)
            or self.truncations.get(self.agent_selection, False)
        ):
            self.env.step(None)
            return
        self.env.step({"macro": int(action)})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reshape_obs(self, obs: dict) -> dict:
        """Convert a raw ``UncivEnv`` observation dict to the RLlib format."""
        parts: list[np.ndarray] = [obs["scalars"].flatten()]
        parts.append(obs["entities"].flatten())
        if self._include_map_planes:
            parts.append(obs["map_planes"].flatten())
        parts.append(obs["visibility_mask"].flatten().astype(np.float32))
        flat_obs = np.concatenate(parts).astype(np.float32)

        # Extract the macro mask; fall back to "all actions allowed" if missing.
        action_mask_dict = obs.get("action_mask", {})
        raw_macro = action_mask_dict.get(
            "macro", np.ones(N_MACRO_ACTIONS, dtype=np.float32)
        )
        macro_mask = np.asarray(raw_macro, dtype=np.float32)

        return {"observations": flat_obs, "action_mask": macro_mask}


# ---------------------------------------------------------------------------
# RLlib MultiAgentEnv adapter (no aec_to_parallel needed)
# ---------------------------------------------------------------------------

try:
    from ray.rllib.env import MultiAgentEnv as _MultiAgentEnv
    _HAS_RLLIB = True
except ImportError:  # pragma: no cover
    _MultiAgentEnv = object  # type: ignore[assignment,misc]
    _HAS_RLLIB = False


class UncivMultiAgentEnv(_MultiAgentEnv):
    """RLlib ``MultiAgentEnv`` wrapping :class:`UncivMacroWrapper` (PettingZoo AEC).

    Bridges the turn-based AEC environment to RLlib **without** an
    ``aec_to_parallel`` conversion.  On each :meth:`step` only the currently
    active agent is asked for an action; all other agents remain idle until
    it is their turn.

    Parameters (passed as a ``config`` dict)
    -----------------------------------------
    base_url : str
        URL of the running Unciv RL server (default ``"http://localhost:8080"``).
    num_agents : int
        Number of RL-controlled civilisations (default ``2``).
    num_ai : int
        Number of AI-controlled civilisations (default ``0``).
    include_map_planes : bool
        Whether to include spatial map channels in observations (default ``False``).
    """

    def __init__(self, config=None) -> None:
        super().__init__()
        config = config or {}

        from rl_env import UncivEnv  # local import to avoid circular deps

        include_map_planes: bool = config.get("include_map_planes", False)
        num_agents: int = config.get("num_agents", 2)

        aec_env = UncivEnv(
            base_url=config.get("base_url", "http://localhost:8080"),
            num_agents=num_agents,
            num_ai=config.get("num_ai", 0),
            num_city_states=0,
            no_barbarians=True,
            include_map_planes=include_map_planes,
        )
        self._aec = UncivMacroWrapper(aec_env, include_map_planes=include_map_planes)

        # RLlib requires _agent_ids to be set before reset().
        self._agent_ids = {f"player_{i}" for i in range(num_agents)}

        # Spaces are constant for all agents (built in UncivMacroWrapper.__init__).
        self.observation_space = self._aec.observation_space("player_0")
        self.action_space = self._aec.action_space("player_0")

        # Baseline for per-step reward computation (updated on reset/step).
        self._last_cum_rewards: dict = {}

    # ------------------------------------------------------------------
    # MultiAgentEnv API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs_all, infos_all = self._aec.reset(seed=seed, options=options)
        self._agent_ids = set(self._aec.possible_agents)

        # Initialise cumulative-reward baseline for the new episode.
        self._last_cum_rewards = {a: 0.0 for a in self._aec.possible_agents}

        agent = self._aec.agent_selection
        return {agent: obs_all[agent]}, {agent: infos_all.get(agent, {})}

    def step(self, action_dict):
        agent = self._aec.agent_selection
        action = action_dict.get(agent)

        self._aec.step(action)

        # UncivEnv accumulates rewards into _cumulative_rewards and then
        # zeroes rewards[agent] at the end of step().  Read the delta from
        # the underlying UncivEnv instance (self._aec.env) to get the actual
        # per-step reward without going through BaseWrapper's private-attr guard.
        unciv_env = self._aec.env
        new_cum: dict = unciv_env._cumulative_rewards
        step_reward = new_cum.get(agent, 0.0) - self._last_cum_rewards.get(agent, 0.0)
        self._last_cum_rewards = dict(new_cum)

        all_done = len(self._aec.agents) == 0
        terminateds = {
            agent: self._aec.terminations.get(agent, False),
            "__all__": all_done,
        }
        truncateds = {
            agent: self._aec.truncations.get(agent, False),
            "__all__": False,
        }
        rewards = {agent: step_reward}

        obs: dict = {}
        infos: dict = {}
        if self._aec.agents:
            next_agent = self._aec.agent_selection
            obs[next_agent] = self._aec.observe(next_agent)
            infos[next_agent] = self._aec.infos.get(next_agent, {})

        return obs, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        self._aec.close()
