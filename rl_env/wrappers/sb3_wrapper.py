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
    # AEC step override – enforce strict round-robin for aec_to_parallel
    # ------------------------------------------------------------------

    def step(self, action: int | np.integer | None) -> None:
        """Step the environment, auto-ending the turn when needed.

        ``pettingzoo.utils.conversions.aec_to_parallel`` requires that
        every call to its ``step()`` method cycles through **all** agents
        exactly once in order.  The underlying :class:`~rl_env.UncivEnv`
        only advances ``agent_selection`` when the action is
        ``MACRO_END_TURN`` (0); any other action keeps the same agent
        active, which breaks the parallel wrapper's cycle assertion.

        This override detects that case and immediately issues a follow-up
        ``MACRO_END_TURN`` (action ``0``) so that every single-action step
        from SB3's perspective also ends the agent's in-game turn.  The
        net effect is that each RL step = one game action + implicit turn end.
        """
        prev_agent = self.agent_selection
        super().step(action)
        # If the active agent did not change after the action (i.e. it was
        # not an end-turn action) and the agent is still alive, force an
        # automatic end-turn so aec_to_parallel can advance to the next agent.
        if (
            self.agent_selection == prev_agent
            and not self.terminations.get(prev_agent, False)
            and not self.truncations.get(prev_agent, False)
            and prev_agent in self.agents
        ):
            super().step(0)  # MACRO_END_TURN = 0

    # ------------------------------------------------------------------
    # Required by supersuit's pettingzoo_env_to_vec_env_v1
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        return self.env.unwrapped


# ---------------------------------------------------------------------------
# Step 5 – thin VecEnvWrapper that wires action_masks() into SB3
# ---------------------------------------------------------------------------

def _make_maskable_vec_env_wrapper(venv, live_par_envs):
    """Factory: wraps a supersuit SB3VecEnv to add action_masks() support.

    ``supersuit``'s ``SB3VecEnvWrapper`` delegates ``has_attr`` / ``env_method``
    to ``ConcatVecEnv``, which does not implement these SB3 methods.  This
    wrapper intercepts calls for ``"action_masks"`` and reads the current mask
    directly from the underlying :class:`UncivSB3PZWrapper` instances.

    ``concat_vec_envs_v1`` serialises the env tree via cloudpickle, so the
    original ``par_env`` passed *before* construction is disconnected from
    the live copies used during stepping.  Callers must therefore pass
    ``live_par_envs`` — a list of the ``aec_to_parallel_wrapper`` objects
    that are *actually* stepped, extracted via::

        live_par_envs = [m.par_env for m in sb3_vec.venv.vec_envs]

    For each agent slot in the VecEnv:

    * **Active agent** (``agent_selection`` in the AEC env): returns the real
      game mask from ``UncivSB3PZWrapper.action_masks()``.
    * **Inactive agents**: returns all-``True`` (their action is discarded this
      turn-based step; any choice is nominally legal from SB3's perspective).

    Parameters
    ----------
    venv:
        The ``SB3VecEnvWrapper(ConcatVecEnv(...))`` returned by
        ``concat_vec_envs_v1``.
    live_par_envs:
        One ``aec_to_parallel_wrapper`` per game copy, in the same order as
        ``sb3_vec.venv.vec_envs``.  These are the *live* objects that receive
        ``reset()`` / ``step()`` calls.
    """
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class _Inner(VecEnvWrapper):
        def reset(self):
            return self.venv.reset()

        def step_wait(self):
            return self.venv.step_wait()

        def seed(self, seed=None):
            # supersuit's ConcatVecEnv does not implement seed(); the Unciv
            # server's randomness is controlled via reset(seed=…) instead.
            return [None] * self.num_envs

        def get_attr(self, attr_name, indices=None):
            # supersuit's ConcatVecEnv does not implement get_attr(); SB3's
            # VecEnv.__init__ calls get_attr("render_mode") to detect it.
            if attr_name == "render_mode":
                return [None] * self.num_envs
            try:
                return self.venv.get_attr(attr_name, indices)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {attr_name!r} is not supported by the environment"
                )

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

            Active agent slot → real game mask from the live AEC env.
            All other slots   → all-True (action is discarded this step).
            """
            masks = []
            for par in _live_par_envs:
                aec = par.aec_env
                current = aec.agent_selection
                for agent in par.possible_agents:
                    if agent == current and agent in aec.agents:
                        masks.append(aec.action_masks())
                    else:
                        masks.append(np.ones(N_MACRO_ACTIONS, dtype=bool))
            return masks

    _live_par_envs = live_par_envs  # captured in the class closure
    return _Inner(venv)


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
          → MarkovVectorEnv          (VecEnv, num_envs = num_agents)
          → concat_vec_envs_v1       (stacked SB3 VecEnv)
          → _MaskableVecEnvWrapper   (adds has_attr/env_method for masks)

    .. note::
        ``concat_vec_envs_v1`` serialises the env via cloudpickle.  The live
        env objects that actually receive ``step()`` calls live inside
        ``ConcatVecEnv.vec_envs``.  We extract direct references to them so
        that :func:`_make_maskable_vec_env_wrapper` reads *current* action
        masks rather than the stale initial-state snapshot.

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
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

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

    # Parallel → VecEnv (num_envs = num_agents).
    # Use black_death=True to suppress the agents == possible_agents assertion:
    # aec_to_parallel_wrapper freezes possible_agents at __init__ time, so after
    # reset() rotates aec_env.agents (to align agents[0] with agent_selection),
    # par_env.agents and par_env.possible_agents have different orderings.
    # black_death=True skips the order-sensitive equality check and is safe here
    # because all agents in our game are always active until the episode ends.
    vec_env = MarkovVectorEnv(par_env, black_death=True)

    # Stack num_copies of the VecEnv (each copy is a deep clone via cloudpickle)
    sb3_vec = ss.concat_vec_envs_v1(
        vec_env, num_copies, num_cpus=0, base_class="stable_baselines3"
    )

    # Extract live par_env references from the deep-copied envs inside
    # ConcatVecEnv.  concat_vec_envs_v1 serialises the template via cloudpickle
    # so the original par_env is disconnected; the live copies live in
    # sb3_vec.venv (SB3VecEnvWrapper) → .venv (ConcatVecEnv) → .vec_envs[i].
    live_par_envs = [markov_env.par_env for markov_env in sb3_vec.venv.vec_envs]

    # Add has_attr / env_method("action_masks") for MaskablePPO
    return _make_maskable_vec_env_wrapper(sb3_vec, live_par_envs)

