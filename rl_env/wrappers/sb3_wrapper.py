"""Stable-Baselines3-compatible wrappers using the PettingZoo-recommended pattern.

The canonical way to train SB3 agents on a multi-agent PettingZoo environment
is described in the official PettingZoo tutorial
(https://pettingzoo.farama.org/tutorials/sb3/):

    1. Wrap the AEC env so all agents share a flat ``Box`` observation space.
    2. Convert the AEC env to a **Parallel** env with
       ``pettingzoo.utils.conversions.aec_to_parallel``.
    3. Convert to a vectorised gymnasium env with
       ``supersuit.pettingzoo_env_to_vec_env_v1``.
    4. Stack multiple copies with ``supersuit.concat_vec_envs_v1``.
    5. Train a **single shared policy** (parameter-sharing) with
       ``MaskablePPO`` from ``sb3-contrib``.

Each agent slot in the resulting ``VecEnv`` is presented as a separate
"environment" from SB3's perspective.  The same policy is used for every agent
(parameter sharing / self-play).

Public API
----------
:class:`UncivSB3PZWrapper`
    PettingZoo AEC wrapper that exposes a flat ``Box`` observation space and
    provides ``action_masks()`` for the currently active agent.

:func:`make_unciv_vec_env`
    Factory that assembles the full chain described above and returns a
    ``VecEnv`` ready for ``MaskablePPO``.

Usage::

    from rl_env.wrappers.sb3_wrapper import make_unciv_vec_env
    from sb3_contrib import MaskablePPO

    vec_env = make_unciv_vec_env(base_url="http://localhost:8080", num_agents=2)
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=50_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseWrapper

from rl_env.constants import N_MACRO_ACTIONS
from rl_env.wrappers.rllib_macro_wrapper import UncivMacroWrapper, flat_obs_dim


# ---------------------------------------------------------------------------
# Step 1 – PettingZoo AEC wrapper
# ---------------------------------------------------------------------------

class UncivSB3PZWrapper(BaseWrapper):
    """PettingZoo AEC wrapper for the supersuit + MaskablePPO training pattern.

    This wrapper sits on top of :class:`~rl_env.wrappers.UncivMacroWrapper` and:

    * Exposes a flat ``Box(flat_obs_dim,)`` observation space (the "observations"
      array from the macro wrapper, without the action-mask sub-dict).
    * Provides ``action_masks() → bool[N_MACRO_ACTIONS]`` for the currently
      active agent, which is forwarded by the ``_MaskableVecEnvWrapper`` to
      ``MaskablePPO``.
    * Sets ``metadata["is_parallelizable"] = True`` so that
      ``pettingzoo.utils.conversions.aec_to_parallel`` can convert it to a
      Parallel environment safely.

    Do **not** use this wrapper directly with SB3; use :func:`make_unciv_vec_env`
    to get the full chain.

    Parameters
    ----------
    env:
        A :class:`~rl_env.wrappers.UncivMacroWrapper` instance.
    include_map_planes:
        Whether spatial map channels are included in the flat observation (must
        match the ``UncivMacroWrapper`` setting).  Default: ``False``.
    """

    metadata: dict[str, Any] = {
        "name": "unciv_sb3_pz_v0",
        "is_parallelizable": True,
        "render_modes": [],
    }
    render_mode: str | None = None

    def __init__(self, env: UncivMacroWrapper, include_map_planes: bool = False) -> None:
        super().__init__(env)
        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_obs_dim(include_map_planes),),
            dtype=np.float32,
        )
        self._act_space = spaces.Discrete(N_MACRO_ACTIONS)

    # ------------------------------------------------------------------
    # PettingZoo AEC API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    def observe(self, agent: str) -> np.ndarray:
        """Return the flat observation array for *agent*."""
        return self.env.observe(agent)["observations"].copy()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        flat_obs = {agent: raw["observations"].copy() for agent, raw in obs_dict.items()}
        return flat_obs, infos

    # ------------------------------------------------------------------
    # Action masking (forwarded to MaskablePPO by _MaskableVecEnvWrapper)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return the legal macro action mask for the currently active agent.

        Returns a ``bool`` array of shape ``(N_MACRO_ACTIONS,)`` where ``True``
        marks a legal action.  Only the currently active agent (``agent_selection``)
        returns a game-valid mask; other agents return all-``True`` (their sampled
        action is not applied to the game this step in a turn-based game).

        .. note::
            Call this method only when at least one :meth:`reset` has been
            performed and ``agent_selection`` is set.
        """
        return self.env.observe(self.agent_selection)["action_mask"].astype(bool)

    # ------------------------------------------------------------------
    # Required by supersuit's pettingzoo_env_to_vec_env_v1
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        return self.env.unwrapped


# ---------------------------------------------------------------------------
# Step 5 – thin VecEnvWrapper that wires action_masks() into SB3
# ---------------------------------------------------------------------------

def _make_maskable_vec_env_wrapper(venv, par_env):
    """Factory: wraps a supersuit SB3VecEnv to add action_masks() support.

    ``supersuit``'s ``SB3VecEnvWrapper`` delegates ``has_attr`` / ``env_method``
    to ``ConcatVecEnv``, which does not implement these SB3 methods.  This
    wrapper intercepts calls for ``"action_masks"`` and reads the current mask
    directly from the underlying :class:`UncivSB3PZWrapper` (accessible via
    ``par_env.aec_env``).

    For each agent slot in the VecEnv:

    * **Active agent** (``agent_selection`` in the AEC env): returns the real
      game mask from ``UncivSB3PZWrapper.action_masks()``.
    * **Inactive agents**: returns all-``True`` (their action is not applied to
      the game this turn-based step; any action is masked-valid).
    """
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class _Inner(VecEnvWrapper):
        def reset(self):
            return self.venv.reset()

        def step_wait(self):
            return self.venv.step_wait()

        def has_attr(self, attr_name, indices=None):
            if attr_name == "action_masks":
                return True
            try:
                return self.venv.has_attr(attr_name)
            except AttributeError:
                return False

        def env_method(self, method_name, *args, indices=None, **kwargs):
            if method_name == "action_masks":
                return self._get_action_masks()
            try:
                return self.venv.env_method(
                    method_name, *args, indices=indices, **kwargs
                )
            except AttributeError:
                raise AttributeError(
                    f"Method {method_name!r} is not supported by the environment"
                )

        def _get_action_masks(self):
            """Return one mask per VecEnv slot.

            Active agent slot → real game mask.
            All other slots   → all-True (action is discarded this step).
            """
            aec = self._par_env.aec_env
            current = aec.agent_selection
            masks = []
            for agent in self._par_env.possible_agents:
                if agent == current and agent in aec.agents:
                    masks.append(aec.action_masks())
                else:
                    masks.append(np.ones(N_MACRO_ACTIONS, dtype=bool))
            return masks

    wrapper = _Inner(venv)
    wrapper._par_env = par_env
    return wrapper


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_unciv_vec_env(
    base_url: str = "http://localhost:8080",
    num_agents: int = 2,
    num_ai: int = 0,
    include_map_planes: bool = False,
    num_copies: int = 1,
    seed: int | None = None,
):
    """Build a MaskablePPO-compatible SB3 ``VecEnv`` from :class:`~rl_env.UncivEnv`.

    Assembles the full PettingZoo-recommended chain::

        UncivEnv (AEC)
          → UncivMacroWrapper        (AEC, flat Dict obs)
          → UncivSB3PZWrapper        (AEC, Box obs + action_masks())
          → aec_to_parallel          (Parallel, parameter-sharing)
          → pettingzoo_env_to_vec_env_v1  (VecEnv, num_envs = num_agents)
          → concat_vec_envs_v1       (stacked SB3 VecEnv)
          → _MaskableVecEnvWrapper   (adds has_attr/env_method for masks)

    Parameters
    ----------
    base_url:
        URL of the running Unciv RL server.
    num_agents:
        Number of RL-controlled civilisations.  All agents share the same
        policy (parameter sharing).  Default: ``2``.
    num_ai:
        Number of built-in AI civilisations.  Default: ``0``.
    include_map_planes:
        Whether to include spatial map channels in observations (~28 K extra
        dims).  Default: ``False``.
    num_copies:
        Number of independent game copies to run in parallel (each game has
        ``num_agents`` agent slots, so total ``num_envs = num_agents *
        num_copies``).  Default: ``1``.
    seed:
        Optional seed passed to :meth:`~rl_env.UncivEnv.reset`.

    Returns
    -------
    VecEnv
        SB3-compatible vectorised environment with ``action_masks()`` support
        for ``MaskablePPO``.
    """
    import supersuit as ss
    from pettingzoo.utils.conversions import aec_to_parallel

    from rl_env import UncivEnv

    # Build one canonical AEC env that we will copy via cloudpickle
    aec_env = UncivEnv(
        base_url=base_url,
        num_agents=num_agents,
        num_ai=num_ai,
        num_city_states=0,
        no_barbarians=True,
        include_map_planes=include_map_planes,
    )
    aec_env = UncivMacroWrapper(aec_env, include_map_planes=include_map_planes)
    aec_env = UncivSB3PZWrapper(aec_env, include_map_planes=include_map_planes)

    # Convert AEC → Parallel (required by pettingzoo_env_to_vec_env_v1)
    par_env = aec_to_parallel(aec_env)

    # Parallel → VecEnv (num_envs = num_agents)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)

    # Stack num_copies of the VecEnv (each copy is a deep clone via cloudpickle)
    sb3_vec = ss.concat_vec_envs_v1(
        vec_env, num_copies, num_cpus=0, base_class="stable_baselines3"
    )

    # Add has_attr / env_method("action_masks") for MaskablePPO
    return _make_maskable_vec_env_wrapper(sb3_vec, par_env)

