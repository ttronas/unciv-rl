"""
Tests for the Stable-Baselines3 / sb3-contrib (MaskablePPO) integration.

Unit tests (no server, no sb3 required)
    - :class:`TestUncivSB3Wrapper` – wrapper spaces, action_masks, self-play loop

SB3 unit tests (sb3-contrib required, no server)
    - :class:`TestMaskablePPOSetup` – model construction, predict API

Integration tests (sb3-contrib + live Unciv server required)
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
from rl_env.wrappers.rllib_macro_wrapper import flat_obs_dim

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from sb3_contrib import MaskablePPO
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
# Minimal mock AEC env (mirrors the one in test_rllib.py; no server required)
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
    """Minimal AEC-like stub compatible with PettingZoo's BaseWrapper."""

    metadata = {
        "name": "mock_unciv_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }

    def __init__(self, num_agents: int = 2) -> None:
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos: dict = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.render_mode = None
        self._step_count = 0

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
        # Rotate through agents
        idx = self.agents.index(self.agent_selection) if self.agent_selection in self.agents else 0
        next_idx = (idx + 1) % len(self.agents)
        self.agent_selection = self.agents[next_idx]

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


def _make_sb3_wrapper(num_agents: int = 2) -> "UncivSB3Wrapper":
    """Build a UncivSB3Wrapper around a mock AEC env (no server)."""
    from rl_env.wrappers.rllib_macro_wrapper import UncivMacroWrapper
    from rl_env.wrappers.sb3_wrapper import UncivSB3Wrapper

    mock_aec = _MockAECEnv(num_agents=num_agents)
    macro_wrapper = UncivMacroWrapper(mock_aec)

    # Bypass the real UncivEnv constructor and inject the mock directly
    wrapper = UncivSB3Wrapper.__new__(UncivSB3Wrapper)
    # Minimal gymnasium.Env init
    gymnasium = __import__("gymnasium")
    gymnasium.Env.__init__(wrapper)

    wrapper._training_agent = "player_0"
    wrapper._include_map_planes = False
    wrapper._flat_dim = flat_obs_dim(False)
    wrapper._rng = np.random.default_rng(42)
    wrapper._aec = macro_wrapper
    wrapper._current_mask = np.ones(N_MACRO_ACTIONS, dtype=bool)

    import gymnasium as gym
    from gymnasium import spaces
    wrapper.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(wrapper._flat_dim,), dtype=np.float32
    )
    wrapper.action_space = spaces.Discrete(N_MACRO_ACTIONS)
    return wrapper


# ---------------------------------------------------------------------------
# TestUncivSB3Wrapper – no server, no sb3
# ---------------------------------------------------------------------------

class TestUncivSB3Wrapper(unittest.TestCase):
    """Unit tests for UncivSB3Wrapper in isolation (no server, no sb3)."""

    def setUp(self) -> None:
        self.wrapper = _make_sb3_wrapper(num_agents=2)

    # --- spaces ---

    def test_observation_space_shape(self):
        expected = (flat_obs_dim(False),)
        self.assertEqual(self.wrapper.observation_space.shape, expected)

    def test_observation_space_dtype(self):
        self.assertEqual(self.wrapper.observation_space.dtype, np.float32)

    def test_action_space_size(self):
        self.assertEqual(self.wrapper.action_space.n, N_MACRO_ACTIONS)

    # --- action_masks ---

    def test_action_masks_returns_bool_array(self):
        self.wrapper._aec.reset()
        mask = self.wrapper.action_masks()
        self.assertEqual(mask.dtype, bool)

    def test_action_masks_shape(self):
        self.wrapper._aec.reset()
        mask = self.wrapper.action_masks()
        self.assertEqual(mask.shape, (N_MACRO_ACTIONS,))

    def test_action_masks_reflects_underlying_mask(self):
        # The mock always returns macro mask [1,1,0,1,0,0,0,0,0,0]
        self.wrapper._aec.reset()
        # Manually trigger _get_training_agent_obs to populate _current_mask
        obs = self.wrapper._get_training_agent_obs()
        mask = self.wrapper.action_masks()
        expected = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_action_masks_returns_copy(self):
        self.wrapper._aec.reset()
        mask1 = self.wrapper.action_masks()
        mask2 = self.wrapper.action_masks()
        self.assertIsNot(mask1, mask2)

    # --- reset ---

    def test_reset_returns_flat_obs(self):
        obs, info = self.wrapper.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (flat_obs_dim(False),))
        self.assertEqual(obs.dtype, np.float32)

    def test_reset_returns_dict_info(self):
        _, info = self.wrapper.reset()
        self.assertIsInstance(info, dict)

    def test_reset_obs_in_observation_space(self):
        obs, _ = self.wrapper.reset()
        self.assertTrue(self.wrapper.observation_space.contains(obs))

    # --- step ---

    def test_step_returns_five_tuple(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        result = self.wrapper.step(0)
        self.assertEqual(len(result), 5)

    def test_step_obs_shape(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        obs, *_ = self.wrapper.step(0)
        self.assertEqual(obs.shape, (flat_obs_dim(False),))

    def test_step_obs_dtype(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        obs, *_ = self.wrapper.step(0)
        self.assertEqual(obs.dtype, np.float32)

    def test_step_reward_is_float(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        _, reward, *_ = self.wrapper.step(0)
        self.assertIsInstance(reward, float)

    def test_step_terminated_and_truncated_are_bool(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        _, _, terminated, truncated, _ = self.wrapper.step(0)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_step_info_is_dict(self):
        self.wrapper._aec.reset()
        self.wrapper._get_training_agent_obs()
        *_, info = self.wrapper.step(0)
        self.assertIsInstance(info, dict)

    # --- full reset-step cycle ---

    def test_full_reset_step_cycle(self):
        obs, _ = self.wrapper.reset()
        self.assertTrue(self.wrapper.observation_space.contains(obs))
        obs2, reward, terminated, truncated, _ = self.wrapper.step(0)
        self.assertTrue(self.wrapper.observation_space.contains(obs2))

    # --- advance_to_training_agent helper ---

    def test_advance_to_training_agent_restores_player0(self):
        self.wrapper._aec.reset()
        # Simulate: force the underlying env's agent_selection to player_1.
        # We must NOT set the attribute on the wrapper itself (that would create
        # a stale instance attribute that shadows BaseWrapper.__getattr__).
        self.wrapper._aec.env.agent_selection = "player_1"  # type: ignore[attr-defined]
        # _advance_to_training_agent should step the opponent once (mock rotates
        # player_1 → player_0 in a single step) and then exit the while-loop.
        self.wrapper._advance_to_training_agent()
        self.assertEqual(self.wrapper._aec.agent_selection, "player_0")


# ---------------------------------------------------------------------------
# TestMaskablePPOSetup – sb3-contrib required, no server
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_SB3, "sb3-contrib not installed")
class TestMaskablePPOSetup(unittest.TestCase):
    """Verify MaskablePPO can be constructed and calls action_masks()."""

    def setUp(self) -> None:
        self.wrapper = _make_sb3_wrapper(num_agents=2)

    def test_maskable_ppo_can_be_created(self):
        model = MaskablePPO("MlpPolicy", self.wrapper, verbose=0)
        self.assertIsNotNone(model)

    def test_maskable_ppo_policy_kwargs(self):
        model = MaskablePPO(
            "MlpPolicy",
            self.wrapper,
            verbose=0,
            policy_kwargs={"net_arch": [64, 64]},
        )
        self.assertIsNotNone(model)

    def test_predict_returns_action_in_action_space(self):
        model = MaskablePPO("MlpPolicy", self.wrapper, verbose=0)
        # Manually trigger a reset so _current_mask is populated
        obs, _ = self.wrapper.reset()
        action, _state = model.predict(obs, action_masks=self.wrapper.action_masks())
        self.assertIn(int(action), range(N_MACRO_ACTIONS))

    def test_predict_respects_action_mask(self):
        """Predicted action must be legal according to the current mask."""
        model = MaskablePPO("MlpPolicy", self.wrapper, verbose=0)
        obs, _ = self.wrapper.reset()
        mask = self.wrapper.action_masks()
        action, _ = model.predict(obs, action_masks=mask)
        self.assertTrue(
            mask[int(action)],
            f"MaskablePPO chose illegal action {action}; mask={mask}",
        )

    def test_action_masks_called_from_wrapper(self):
        """action_masks() must return a bool array compatible with MaskablePPO."""
        obs, _ = self.wrapper.reset()
        mask = self.wrapper.action_masks()
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(len(mask), N_MACRO_ACTIONS)


# ---------------------------------------------------------------------------
# TestMaskablePPOTraining – sb3-contrib + live server required
# ---------------------------------------------------------------------------

_SKIP_TRAINING = not (_HAS_SB3 and _server_available())
_SKIP_TRAINING_REASON = (
    "sb3-contrib not installed or Unciv RL server not available; "
    "set UNCIV_RL_URL and ensure the server is running with --rl"
)


@unittest.skipIf(_SKIP_TRAINING, _SKIP_TRAINING_REASON)
class TestMaskablePPOTraining(unittest.TestCase):
    """End-to-end tests: MaskablePPO training against live Unciv server.

    ``setUpClass`` creates the environment and trains for a small number of
    timesteps so individual test methods can assert on the results without
    repeating the expensive training step.
    """

    TOTAL_TIMESTEPS: int = 200  # enough for ~1 rollout buffer
    N_STEPS: int = 128          # rollout buffer size
    BATCH_SIZE: int = 64

    @classmethod
    def setUpClass(cls) -> None:
        from rl_env.examples.sb3_two_agents import make_unciv_env

        cls.env = make_unciv_env(
            base_url=_SERVER_URL,
            num_agents=2,
            num_ai=0,
            include_map_planes=False,
            seed=42,
        )
        cls.model = MaskablePPO(
            "MlpPolicy",
            cls.env,
            n_steps=cls.N_STEPS,
            batch_size=cls.BATCH_SIZE,
            verbose=0,
            seed=42,
            policy_kwargs={"net_arch": [64, 64]},
        )
        cls.model.learn(total_timesteps=cls.TOTAL_TIMESTEPS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    # ------------------------------------------------------------------
    # Post-training assertions
    # ------------------------------------------------------------------

    def test_model_has_policy(self):
        self.assertIsNotNone(self.model.policy)

    def test_model_num_timesteps(self):
        self.assertGreaterEqual(
            self.model.num_timesteps,
            self.TOTAL_TIMESTEPS,
            "Model should have trained for at least TOTAL_TIMESTEPS steps",
        )

    def test_predict_legal_action_after_training(self):
        """After training, the policy must always pick a legal macro action."""
        obs, _ = self.env.reset()
        n_checks = 10
        for i in range(n_checks):
            mask = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            self.assertTrue(
                mask[int(action)],
                f"Step {i}: model chose illegal action {action}; mask={mask}",
            )
            obs, _, terminated, truncated, _ = self.env.step(int(action))
            if terminated or truncated:
                obs, _ = self.env.reset()

    def test_action_mask_enforced_during_stepping(self):
        """Manual env stepping must always surface at least one legal action."""
        obs, _ = self.env.reset()
        for i in range(10):
            mask = self.env.action_masks()
            self.assertTrue(
                mask.any(),
                f"Step {i}: no legal actions in mask={mask}",
            )
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal))
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()

    def test_continued_learning_is_stable(self):
        """A second short .learn() call must not raise."""
        self.model.learn(total_timesteps=self.N_STEPS, reset_num_timesteps=False)


if __name__ == "__main__":
    unittest.main()
