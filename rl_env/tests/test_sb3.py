"""
Tests for the Stable-Baselines3 / sb3-contrib (MaskablePPO) integration.

Follows the PettingZoo-recommended pattern for MARL with SB3:
  AEC env → aec_to_parallel → pettingzoo_env_to_vec_env_v1
          → concat_vec_envs_v1 → _MaskableVecEnvWrapper → MaskablePPO

Unit tests (no server, no sb3 required)
    - :class:`TestUncivSB3PZWrapper` – PZ wrapper spaces and action_masks()

SB3 unit tests (sb3-contrib + supersuit required, no server)
    - :class:`TestMaskablePPOSetup` – full chain construction and prediction

Integration tests (sb3-contrib + supersuit + live Unciv server required)
    - :class:`TestMaskablePPOTraining` – short training run against live server

Set ``UNCIV_RL_URL`` to override the default server address.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_MACRO_ACTIONS,
    N_SCALAR_FEATURES,
)
from rl_env.spaces import UncivSpaces
from rl_env.wrappers.rllib_macro_wrapper import UncivMacroWrapper, flat_obs_dim
from rl_env.wrappers.sb3_wrapper import UncivSB3PZWrapper, _make_maskable_vec_env_wrapper

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import is_masking_supported
    import supersuit as ss
    from pettingzoo.utils.conversions import aec_to_parallel
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

_SERVER_URL = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")


def _server_available() -> bool:
    try:
        import requests
        resp = requests.get(f"{_SERVER_URL}/isalive", timeout=3.0)
        return resp.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Minimal mock AEC env (no server required)
# ---------------------------------------------------------------------------

def _make_mask_dict() -> dict:
    return {
        "macro": np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "unit_target": np.zeros(MAX_ENTITIES, dtype=np.int8),
        "unit_subaction": np.ones(7, dtype=np.int8),
        "city_target": np.zeros(MAX_ENTITIES, dtype=np.int8),
        "city_subaction": np.ones(4, dtype=np.int8),
        "production_target": np.ones(2, dtype=np.int8),
        "tech_target": np.ones(3, dtype=np.int8),
        "policy_target": np.ones(2, dtype=np.int8),
        "diplomacy_target": np.zeros(2, dtype=np.int8),
        "diplomacy_subaction": np.zeros(7, dtype=np.int8),
        "tile_target": np.zeros(MAX_TILES, dtype=np.int8),
        "improvement_target": np.ones(3, dtype=np.int8),
    }


def _make_obs_dict() -> dict:
    return {
        "scalars": np.ones(N_SCALAR_FEATURES, dtype=np.float32) * 2.0,
        "entities": np.ones((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32),
        "map_planes": np.ones((N_MAP_CHANNELS, MAX_TILES), dtype=np.float32),
        "visibility_mask": np.ones(MAX_ENTITIES, dtype=np.int8),
        "action_mask": _make_mask_dict(),
    }


class _MockAECEnv:
    """Minimal AEC-like stub compatible with PettingZoo's BaseWrapper.

    Adds the ``render_mode`` attribute and ``unwrapped`` property required by
    supersuit's ``pettingzoo_env_to_vec_env_v1``.
    """

    metadata = {
        "name": "mock_unciv_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }
    render_mode = None

    def __init__(self, num_agents: int = 2) -> None:
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos: dict = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._step_count = 0

    @property
    def unwrapped(self):
        return self

    def observation_space(self, agent: str):
        return UncivSpaces().observation_space()

    def action_space(self, agent: str):
        return UncivSpaces().action_space()

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._step_count = 0
        obs = {a: _make_obs_dict() for a in self.agents}
        infos = {a: {"action_mask": _make_mask_dict()} for a in self.agents}
        return obs, infos

    def observe(self, agent: str) -> dict:
        return _make_obs_dict()

    def step(self, action) -> None:
        self._step_count += 1
        idx = (
            self.agents.index(self.agent_selection)
            if self.agent_selection in self.agents
            else 0
        )
        next_idx = (idx + 1) % len(self.agents)
        self.agent_selection = self.agents[next_idx]

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


def _make_pz_wrapper(num_agents: int = 2) -> UncivSB3PZWrapper:
    """Build a UncivSB3PZWrapper around a mock AEC env (no server)."""
    mock = _MockAECEnv(num_agents=num_agents)
    macro = UncivMacroWrapper(mock)
    return UncivSB3PZWrapper(macro)


def _make_vec_env_from_mock(num_agents: int = 2):
    """Build the full supersuit chain + _make_maskable_vec_env_wrapper from mock (no server).

    Mirrors make_unciv_vec_env() exactly so that the unit tests exercise the
    same code path as the production stack.
    """
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

    pz = _make_pz_wrapper(num_agents)
    par = aec_to_parallel(pz)
    vec_env = MarkovVectorEnv(par, black_death=True)
    sb3_vec = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class="stable_baselines3")
    live_par_envs = [markov_env.par_env for markov_env in sb3_vec.venv.vec_envs]
    return _make_maskable_vec_env_wrapper(sb3_vec, live_par_envs)


# ---------------------------------------------------------------------------
# TestUncivSB3PZWrapper – no server, no sb3
# ---------------------------------------------------------------------------

class TestUncivSB3PZWrapper(unittest.TestCase):
    """Unit tests for UncivSB3PZWrapper in isolation (no server, no sb3)."""

    def setUp(self) -> None:
        self.wrapper = _make_pz_wrapper(num_agents=2)

    # --- spaces ---

    def test_observation_space_shape(self):
        expected = (flat_obs_dim(False),)
        self.assertEqual(self.wrapper.observation_space("player_0").shape, expected)

    def test_observation_space_dtype(self):
        self.assertEqual(
            self.wrapper.observation_space("player_0").dtype, np.float32
        )

    def test_action_space_size(self):
        self.assertEqual(self.wrapper.action_space("player_0").n, N_MACRO_ACTIONS)

    def test_all_agents_same_obs_space(self):
        sp0 = self.wrapper.observation_space("player_0")
        sp1 = self.wrapper.observation_space("player_1")
        self.assertEqual(sp0, sp1)

    def test_all_agents_same_act_space(self):
        self.assertEqual(
            self.wrapper.action_space("player_0"),
            self.wrapper.action_space("player_1"),
        )

    # --- observe ---

    def test_observe_returns_flat_array(self):
        obs = self.wrapper.observe("player_0")
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (flat_obs_dim(False),))
        self.assertEqual(obs.dtype, np.float32)

    def test_observe_is_copy(self):
        obs1 = self.wrapper.observe("player_0")
        obs2 = self.wrapper.observe("player_0")
        self.assertIsNot(obs1, obs2)

    # --- reset ---

    def test_reset_returns_flat_obs_per_agent(self):
        obs_dict, infos = self.wrapper.reset()
        for agent in self.wrapper.possible_agents:
            self.assertIn(agent, obs_dict)
            obs = obs_dict[agent]
            self.assertEqual(obs.shape, (flat_obs_dim(False),))
            self.assertEqual(obs.dtype, np.float32)

    def test_reset_returns_infos_dict(self):
        _, infos = self.wrapper.reset()
        self.assertIsInstance(infos, dict)

    # --- action_masks ---

    def test_action_masks_shape(self):
        self.wrapper.reset()
        mask = self.wrapper.action_masks()
        self.assertEqual(mask.shape, (N_MACRO_ACTIONS,))

    def test_action_masks_dtype(self):
        self.wrapper.reset()
        mask = self.wrapper.action_masks()
        self.assertEqual(mask.dtype, bool)

    def test_action_masks_reflects_underlying_mask(self):
        # Mock always returns macro mask [1,1,0,1,0,0,0,0,0,0]
        self.wrapper.reset()
        mask = self.wrapper.action_masks()
        expected = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_action_masks_returns_copy(self):
        self.wrapper.reset()
        mask1 = self.wrapper.action_masks()
        mask2 = self.wrapper.action_masks()
        self.assertIsNot(mask1, mask2)

    # --- metadata ---

    def test_metadata_is_parallelizable(self):
        self.assertTrue(
            self.wrapper.metadata.get("is_parallelizable", False),
            "metadata['is_parallelizable'] must be True for aec_to_parallel",
        )

    def test_has_render_mode(self):
        self.assertTrue(hasattr(self.wrapper, "render_mode"))

    def test_has_unwrapped(self):
        self.assertIsNotNone(self.wrapper.unwrapped)

    def test_include_map_planes_changes_obs_shape(self):
        wrapper_with_maps = UncivSB3PZWrapper(
            UncivMacroWrapper(_MockAECEnv(num_agents=2), include_map_planes=True),
            include_map_planes=True,
        )
        self.assertEqual(
            wrapper_with_maps.observation_space("player_0").shape,
            (flat_obs_dim(True),),
        )
        self.assertGreater(flat_obs_dim(True), flat_obs_dim(False))


# ---------------------------------------------------------------------------
# TestMaskablePPOSetup – sb3-contrib + supersuit required, no server
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_SB3, "sb3-contrib or supersuit not installed")
class TestMaskablePPOSetup(unittest.TestCase):
    """Verify the full supersuit chain + MaskablePPO can be built without a server."""

    def setUp(self) -> None:
        self.vec_env = _make_vec_env_from_mock(num_agents=2)

    def tearDown(self) -> None:
        self.vec_env.close()

    # --- VecEnv properties ---

    def test_num_envs_equals_num_agents(self):
        self.assertEqual(self.vec_env.num_envs, 2)

    def test_obs_space_is_box(self):
        from gymnasium import spaces
        self.assertIsInstance(self.vec_env.observation_space, spaces.Box)

    def test_obs_space_shape(self):
        self.assertEqual(
            self.vec_env.observation_space.shape, (flat_obs_dim(False),)
        )

    def test_action_space_is_discrete(self):
        from gymnasium import spaces
        self.assertIsInstance(self.vec_env.action_space, spaces.Discrete)

    # --- action masking support ---

    def test_is_masking_supported(self):
        self.assertTrue(
            is_masking_supported(self.vec_env),
            "VecEnv must report masking support for MaskablePPO",
        )

    def test_env_method_action_masks_returns_list(self):
        masks = self.vec_env.env_method("action_masks")
        self.assertIsInstance(masks, list)
        self.assertEqual(len(masks), self.vec_env.num_envs)

    def test_env_method_action_masks_shape(self):
        masks = self.vec_env.env_method("action_masks")
        for mask in masks:
            self.assertEqual(mask.shape, (N_MACRO_ACTIONS,))
            self.assertEqual(mask.dtype, bool)

    def test_has_attr_action_masks(self):
        self.assertTrue(self.vec_env.has_attr("action_masks"))

    # --- MaskablePPO construction ---

    def test_maskable_ppo_can_be_created(self):
        model = MaskablePPO("MlpPolicy", self.vec_env, verbose=0)
        self.assertIsNotNone(model)

    def test_maskable_ppo_policy_kwargs(self):
        model = MaskablePPO(
            "MlpPolicy",
            self.vec_env,
            verbose=0,
            policy_kwargs={"net_arch": [64, 64]},
        )
        self.assertIsNotNone(model)

    def test_maskable_ppo_short_learn(self):
        """MaskablePPO.learn() must complete without raising."""
        model = MaskablePPO(
            "MlpPolicy",
            self.vec_env,
            verbose=0,
            n_steps=4,
            batch_size=4,
        )
        model.learn(total_timesteps=8)
        self.assertGreaterEqual(model.num_timesteps, 8)

    def test_predict_action_in_action_space(self):
        obs = self.vec_env.reset()
        masks = np.stack(self.vec_env.env_method("action_masks"))
        model = MaskablePPO("MlpPolicy", self.vec_env, verbose=0)
        actions, _ = model.predict(obs, action_masks=masks)
        for action in actions:
            self.assertIn(int(action), range(N_MACRO_ACTIONS))

    def test_predict_respects_action_mask(self):
        """Predicted actions must all be legal according to the current masks."""
        obs = self.vec_env.reset()
        masks = np.stack(self.vec_env.env_method("action_masks"))
        model = MaskablePPO("MlpPolicy", self.vec_env, verbose=0)
        actions, _ = model.predict(obs, action_masks=masks)
        for i, action in enumerate(actions):
            self.assertTrue(
                masks[i, int(action)],
                f"Slot {i}: MaskablePPO chose illegal action {action}; mask={masks[i]}",
            )


# ---------------------------------------------------------------------------
# TestMaskablePPOTraining – sb3-contrib + supersuit + live server required
# ---------------------------------------------------------------------------

_SKIP_TRAINING = not (_HAS_SB3 and _server_available())
_SKIP_TRAINING_REASON = (
    "sb3-contrib/supersuit not installed or Unciv RL server not available; "
    "set UNCIV_RL_URL and ensure the server is running with --rl"
)


@unittest.skipIf(_SKIP_TRAINING, _SKIP_TRAINING_REASON)
class TestMaskablePPOTraining(unittest.TestCase):
    """End-to-end tests: MaskablePPO training against a live Unciv server.

    ``setUpClass`` creates the full supersuit chain and trains for a small
    number of timesteps so individual test methods can assert on the results
    without repeating the expensive training step.
    """

    TOTAL_TIMESTEPS: int = 200
    N_STEPS: int = 128
    BATCH_SIZE: int = 64

    @classmethod
    def setUpClass(cls) -> None:
        from rl_env.wrappers.sb3_wrapper import make_unciv_vec_env

        cls.vec_env = make_unciv_vec_env(
            base_url=_SERVER_URL,
            num_agents=2,
            num_ai=0,
            include_map_planes=False,
            num_copies=1,
        )
        cls.model = MaskablePPO(
            "MlpPolicy",
            cls.vec_env,
            n_steps=cls.N_STEPS,
            batch_size=cls.BATCH_SIZE,
            verbose=0,
            seed=42,
            policy_kwargs={"net_arch": [64, 64]},
        )
        cls.model.learn(total_timesteps=cls.TOTAL_TIMESTEPS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.vec_env.close()

    # --- post-training assertions ---

    def test_model_has_policy(self):
        self.assertIsNotNone(self.model.policy)

    def test_model_num_timesteps(self):
        self.assertGreaterEqual(self.model.num_timesteps, self.TOTAL_TIMESTEPS)

    def test_is_masking_supported_on_live_env(self):
        self.assertTrue(is_masking_supported(self.vec_env))

    def test_predict_legal_action_after_training(self):
        """After training, the policy must only pick legal macro actions."""
        obs = self.vec_env.reset()
        n_checks = 10
        for i in range(n_checks):
            masks = np.stack(self.vec_env.env_method("action_masks"))
            actions, _ = self.model.predict(
                obs, action_masks=masks, deterministic=True
            )
            for slot, action in enumerate(actions):
                self.assertTrue(
                    masks[slot, int(action)],
                    f"Check {i}, slot {slot}: model chose illegal action "
                    f"{action}; mask={masks[slot]}",
                )
            obs, _, dones, _ = self.vec_env.step(actions)
            if dones.any():
                obs = self.vec_env.reset()

    def test_action_mask_enforced_during_stepping(self):
        """Manual env stepping must always surface at least one legal action."""
        obs = self.vec_env.reset()
        for i in range(10):
            masks = np.stack(self.vec_env.env_method("action_masks"))
            for slot in range(self.vec_env.num_envs):
                self.assertTrue(
                    masks[slot].any(),
                    f"Step {i}, slot {slot}: no legal actions in mask={masks[slot]}",
                )
            actions = np.array(
                [int(np.random.choice(np.where(masks[s])[0]))
                 for s in range(self.vec_env.num_envs)]
            )
            obs, _, dones, _ = self.vec_env.step(actions)
            if dones.any():
                obs = self.vec_env.reset()

    def test_continued_learning_is_stable(self):
        """A second short .learn() call must not raise."""
        self.model.learn(total_timesteps=self.N_STEPS, reset_num_timesteps=False)


if __name__ == "__main__":
    unittest.main()
