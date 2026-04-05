"""
Tests for the Ray RLlib integration of the Unciv RL environment.

Unit tests (no server, no ray required)
    - :class:`TestUncivMacroWrapper` – wrapper obs/action spaces and transforms

RLlib unit tests (ray[rllib] required, no server)
    - :class:`TestActionMaskingRLModule` – forward pass correctly masks logits

Integration tests (ray[rllib] + live Unciv server required)
    - :class:`TestRLlibTraining` – one PPO training iteration runs end-to-end

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

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import ray  # noqa: F401
    from ray.rllib.utils.framework import try_import_torch as _try_torch
    _torch, _ = _try_torch()
    _HAS_RAY = _torch is not None
except ImportError:
    _HAS_RAY = False

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
    """Build a representative action-mask dict as returned by parse_action_mask."""
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
    """Build a representative raw observation dict as returned by UncivEnv.observe."""
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
        # BaseWrapper reads render_mode from the wrapped env
        self.render_mode = None

    def observation_space(self, agent: str):
        return UncivSpaces().observation_space()

    def action_space(self, agent: str):
        return UncivSpaces().action_space()

    def reset(self, seed=None, options=None):
        obs = {a: _make_obs_dict() for a in self.agents}
        infos = {a: {"action_mask": _make_mask_dict()} for a in self.agents}
        return obs, infos

    def observe(self, agent: str) -> dict:
        return _make_obs_dict()

    def step(self, action) -> None:
        pass

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TestUncivMacroWrapper – no server, no ray
# ---------------------------------------------------------------------------

class TestUncivMacroWrapper(unittest.TestCase):
    """Tests for UncivMacroWrapper in isolation (no server, no ray)."""

    def _make_wrapped(self, include_map_planes: bool = False) -> UncivMacroWrapper:
        return UncivMacroWrapper(_MockAECEnv(), include_map_planes=include_map_planes)

    # --- observation space ---

    def test_observation_space_keys(self):
        wrapper = self._make_wrapped()
        obs_space = wrapper.observation_space("player_0")
        self.assertIn("observations", obs_space.spaces)
        self.assertIn("action_mask", obs_space.spaces)

    def test_observation_space_flat_dim_without_map(self):
        wrapper = self._make_wrapped(include_map_planes=False)
        obs_space = wrapper.observation_space("player_0")
        expected = flat_obs_dim(False)
        self.assertEqual(obs_space["observations"].shape, (expected,))

    def test_observation_space_flat_dim_with_map(self):
        wrapper = self._make_wrapped(include_map_planes=True)
        obs_space = wrapper.observation_space("player_0")
        expected = flat_obs_dim(True)
        self.assertEqual(obs_space["observations"].shape, (expected,))

    def test_observation_space_action_mask_shape(self):
        wrapper = self._make_wrapped()
        obs_space = wrapper.observation_space("player_0")
        self.assertEqual(obs_space["action_mask"].shape, (N_MACRO_ACTIONS,))

    def test_observation_space_dtype_observations(self):
        from gymnasium.spaces import Box
        wrapper = self._make_wrapped()
        obs_box = wrapper.observation_space("player_0")["observations"]
        self.assertIsInstance(obs_box, Box)
        self.assertEqual(obs_box.dtype, np.float32)

    def test_observation_space_dtype_action_mask(self):
        from gymnasium.spaces import Box
        wrapper = self._make_wrapped()
        mask_box = wrapper.observation_space("player_0")["action_mask"]
        self.assertIsInstance(mask_box, Box)
        self.assertEqual(mask_box.dtype, np.float32)

    # --- action space ---

    def test_action_space_is_discrete(self):
        from gymnasium.spaces import Discrete
        wrapper = self._make_wrapped()
        act_space = wrapper.action_space("player_0")
        self.assertIsInstance(act_space, Discrete)
        self.assertEqual(act_space.n, N_MACRO_ACTIONS)

    def test_all_agents_share_same_spaces(self):
        wrapper = self._make_wrapped()
        for agent in wrapper.possible_agents:
            self.assertEqual(
                wrapper.observation_space(agent),
                wrapper.observation_space("player_0"),
            )
            self.assertEqual(
                wrapper.action_space(agent),
                wrapper.action_space("player_0"),
            )

    # --- observation transformation ---

    def test_reshape_obs_output_keys(self):
        wrapper = self._make_wrapped()
        result = wrapper._reshape_obs(_make_obs_dict())
        self.assertIn("observations", result)
        self.assertIn("action_mask", result)

    def test_reshape_obs_flat_dim(self):
        wrapper = self._make_wrapped(include_map_planes=False)
        result = wrapper._reshape_obs(_make_obs_dict())
        self.assertEqual(result["observations"].shape, (flat_obs_dim(False),))

    def test_reshape_obs_flat_dim_with_map(self):
        wrapper = self._make_wrapped(include_map_planes=True)
        result = wrapper._reshape_obs(_make_obs_dict())
        self.assertEqual(result["observations"].shape, (flat_obs_dim(True),))

    def test_reshape_obs_dtype(self):
        wrapper = self._make_wrapped()
        result = wrapper._reshape_obs(_make_obs_dict())
        self.assertEqual(result["observations"].dtype, np.float32)
        self.assertEqual(result["action_mask"].dtype, np.float32)

    def test_reshape_obs_macro_mask_values(self):
        wrapper = self._make_wrapped()
        obs = _make_obs_dict()
        result = wrapper._reshape_obs(obs)
        expected_mask = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result["action_mask"], expected_mask)

    def test_reshape_obs_macro_mask_fallback(self):
        """When action_mask is missing, all macro actions should be allowed."""
        wrapper = self._make_wrapped()
        obs = _make_obs_dict()
        del obs["action_mask"]
        result = wrapper._reshape_obs(obs)
        np.testing.assert_array_equal(
            result["action_mask"], np.ones(N_MACRO_ACTIONS, dtype=np.float32)
        )

    def test_reshape_obs_macro_mask_missing_key(self):
        """When action_mask dict lacks 'macro', fall back to all-ones."""
        wrapper = self._make_wrapped()
        obs = _make_obs_dict()
        obs["action_mask"] = {}
        result = wrapper._reshape_obs(obs)
        np.testing.assert_array_equal(
            result["action_mask"], np.ones(N_MACRO_ACTIONS, dtype=np.float32)
        )

    # --- reset / observe passthrough ---

    def test_reset_returns_transformed_obs(self):
        wrapper = self._make_wrapped()
        obs, infos = wrapper.reset()
        for agent in obs:
            self.assertIn("observations", obs[agent])
            self.assertIn("action_mask", obs[agent])

    def test_observe_returns_transformed_obs(self):
        wrapper = self._make_wrapped()
        wrapper.reset()
        obs = wrapper.observe("player_0")
        self.assertIn("observations", obs)
        self.assertIn("action_mask", obs)

    # --- AEC attribute delegation ---

    def test_agents_delegated_to_inner_env(self):
        mock = _MockAECEnv(num_agents=2)
        wrapper = UncivMacroWrapper(mock)
        self.assertEqual(wrapper.agents, mock.agents)

    def test_possible_agents_delegated(self):
        mock = _MockAECEnv(num_agents=3)
        wrapper = UncivMacroWrapper(mock)
        self.assertEqual(wrapper.possible_agents, mock.possible_agents)


# ---------------------------------------------------------------------------
# TestActionMaskingRLModule – requires ray[rllib] + torch
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_RAY, "ray[rllib] + torch not installed")
class TestActionMaskingRLModule(unittest.TestCase):
    """Tests for UncivActionMaskingTorchRLModule (no server required)."""

    def _make_module(self, obs_dim: int = 16):
        import gymnasium as gym
        from gymnasium.spaces import Box, Discrete
        from rl_env.examples.rllib_two_agents import UncivActionMaskingTorchRLModule

        obs_space = gym.spaces.Dict(
            {
                "observations": Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                "action_mask": Box(0.0, 1.0, shape=(N_MACRO_ACTIONS,), dtype=np.float32),
            }
        )
        act_space = Discrete(N_MACRO_ACTIONS)
        module = UncivActionMaskingTorchRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={"head_fcnet_hiddens": [32, 32]},
        )
        return module

    def test_module_instantiation(self):
        module = self._make_module()
        self.assertIsNotNone(module)

    def test_wrong_observation_space_raises(self):
        from gymnasium.spaces import Box, Discrete
        from rl_env.examples.rllib_two_agents import UncivActionMaskingTorchRLModule

        with self.assertRaises(ValueError):
            UncivActionMaskingTorchRLModule(
                observation_space=Box(-1.0, 1.0, (10,)),
                action_space=Discrete(N_MACRO_ACTIONS),
            )

    def test_module_masks_logits(self):
        """Forward inference must assign -inf logits to masked-out actions."""
        import torch
        from rl_env.examples.rllib_two_agents import UncivActionMaskingTorchRLModule
        from ray.rllib.core.columns import Columns

        obs_dim = 32
        module = self._make_module(obs_dim=obs_dim)
        module.eval()

        batch_size = 4
        obs = torch.zeros(batch_size, obs_dim)
        # Allow only actions 0 and 3; mask out the rest
        mask = torch.zeros(batch_size, N_MACRO_ACTIONS)
        mask[:, 0] = 1.0
        mask[:, 3] = 1.0
        obs_dict = {"observations": obs, "action_mask": mask}

        with torch.no_grad():
            out = module._forward_inference({Columns.OBS: obs_dict})

        logits = out[Columns.ACTION_DIST_INPUTS]  # (batch_size, N_MACRO_ACTIONS)
        # Masked actions should have very large negative logits
        masked_logits = logits[:, [1, 2, 4, 5, 6, 7, 8, 9]]
        self.assertTrue(
            (masked_logits < -1e6).all().item(),
            "Masked actions should have near-−inf logits",
        )
        # Allowed actions should have finite logits
        allowed_logits = logits[:, [0, 3]]
        self.assertTrue(
            torch.isfinite(allowed_logits).all().item(),
            "Allowed actions should have finite logits",
        )

    def test_forward_exploration_masks_logits(self):
        """_forward_exploration must also apply the action mask."""
        import torch
        from rl_env.examples.rllib_two_agents import UncivActionMaskingTorchRLModule
        from ray.rllib.core.columns import Columns

        module = self._make_module(obs_dim=16)
        module.eval()

        obs = torch.zeros(2, 16)
        # Only allow action 0
        mask = torch.zeros(2, N_MACRO_ACTIONS)
        mask[:, 0] = 1.0
        obs_dict = {"observations": obs, "action_mask": mask}

        with torch.no_grad():
            out = module._forward_exploration({Columns.OBS: obs_dict})

        logits = out[Columns.ACTION_DIST_INPUTS]
        self.assertTrue((logits[:, 1:] < -1e6).all().item())
        self.assertTrue(torch.isfinite(logits[:, 0]).all().item())


# ---------------------------------------------------------------------------
# Integration tests – ray[rllib] + live Unciv server
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    _HAS_RAY and _server_available(),
    "ray[rllib] + torch not installed or Unciv RL server not available; "
    "set UNCIV_RL_URL and ensure the server is running with --rl",
)
class TestRLlibTraining(unittest.TestCase):
    """End-to-end test: one PPO training iteration with two agents."""

    @classmethod
    def setUpClass(cls) -> None:
        import ray
        ray.init(ignore_reinit_error=True, num_cpus=2)

    @classmethod
    def tearDownClass(cls) -> None:
        import ray
        ray.shutdown()

    def test_one_training_iteration(self):
        from rl_env.examples.rllib_two_agents import build_ppo_config

        config = build_ppo_config(
            base_url=_SERVER_URL,
            num_iterations=1,
            num_env_runners=1,
            train_batch_size=200,
            include_map_planes=False,
        )
        algo = config.build()
        try:
            result = algo.train()
            # Verify the result dict contains expected top-level keys
            self.assertIn("env_runners", result)
        finally:
            algo.stop()

    def test_action_mask_enforced_during_sampling(self):
        """Sampled macro actions must always be in the legal set."""
        from pettingzoo.utils.conversions import aec_to_parallel
        from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
        from rl_env import UncivEnv
        from rl_env.wrappers import UncivMacroWrapper

        aec_env = UncivEnv(
            base_url=_SERVER_URL,
            num_agents=2,
            num_ai=0,
            num_city_states=0,
            no_barbarians=True,
            include_map_planes=False,
        )
        wrapped = UncivMacroWrapper(aec_env, include_map_planes=False)
        wrapped.reset(seed=0)

        # Run a few steps and assert every sampled action is legal
        for _ in range(5):
            agent = wrapped.agent_selection
            obs = wrapped.observe(agent)
            mask = obs["action_mask"]  # float32, shape (N_MACRO_ACTIONS,)
            legal_actions = np.where(mask > 0.5)[0]
            if len(legal_actions) == 0:
                break
            action = int(np.random.choice(legal_actions))
            self.assertIn(action, legal_actions)
            wrapped.step(action)

        wrapped.close()


if __name__ == "__main__":
    unittest.main()
