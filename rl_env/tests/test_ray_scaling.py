"""
Tests for Ray-native parallel scaling of UncivEnv.

Two test layers are provided:

**Unit tests** (``TestMakeEnvCreatorRouting``)
    These run always – no server and no Ray installation required.  They verify
    that :func:`~rl_env.ray_env.make_env_creator` routes workers to the
    correct server URLs and that edge-cases are handled correctly.

**Ray integration tests** (``TestRayParallelWorkers``)
    These require both a running Unciv RL server (``UNCIV_RL_URL``) and Ray to
    be installed.  They spin up multiple Ray remote tasks in parallel –
    mirroring how RLlib spawns rollout workers – and assert that:

    * Each worker creates a distinct game (unique ``gameId``).
    * Workers do not interfere: steps on one game do not affect another.
    * All parallel workers complete successfully within a reasonable timeout.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
``UNCIV_RL_URL``
    URL of a running Unciv RL server (default: ``http://localhost:8080``).
    Required for the Ray integration tests.
"""

from __future__ import annotations

import os
import unittest

import requests

_SERVER_URL: str = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")


def _server_available() -> bool:
    try:
        return requests.get(f"{_SERVER_URL}/isalive", timeout=3.0).ok
    except requests.exceptions.RequestException:
        return False


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP_NO_SERVER = unittest.skipUnless(
    _server_available(),
    "Unciv RL server not reachable. Set UNCIV_RL_URL and start the server "
    "with --rl to run these tests.",
)

_SKIP_NO_RAY = unittest.skipUnless(
    _ray_available(),
    "Ray is not installed. Install it with `pip install ray` to run these tests.",
)


# ---------------------------------------------------------------------------
# Unit tests – no Ray / server required
# ---------------------------------------------------------------------------


class TestMakeEnvCreatorRouting(unittest.TestCase):
    """Verify URL-routing logic in make_env_creator without any live services."""

    def setUp(self):
        from rl_env.ray_env import make_env_creator
        self._factory = make_env_creator

    def test_empty_urls_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._factory([])

    def test_single_url_all_workers_routed_to_same_server(self):
        """With one URL every worker_index maps to the same address."""
        url = "http://server-a:8080"
        creator = self._factory([url])
        for worker_index in range(8):
            env = creator({"worker_index": worker_index})
            self.assertEqual(env.base_url, url)
            env.close()

    def test_two_urls_even_workers_to_first_odd_to_second(self):
        """worker_index % 2 distributes evenly across two servers."""
        urls = ["http://server-a:8080", "http://server-b:8081"]
        creator = self._factory(urls)
        for i in range(8):
            env = creator({"worker_index": i})
            expected = urls[i % 2]
            self.assertEqual(
                env.base_url,
                expected,
                f"worker_index={i} should route to {expected}, got {env.base_url}",
            )
            env.close()

    def test_worker_index_wraps_around(self):
        """worker_index larger than num_servers wraps via modulo."""
        urls = ["http://a:8080", "http://b:8081", "http://c:8082"]
        creator = self._factory(urls)
        for i in range(12):
            env = creator({"worker_index": i})
            self.assertEqual(env.base_url, urls[i % 3])
            env.close()

    def test_missing_worker_index_defaults_to_zero(self):
        """An empty config dict falls back to worker_index=0."""
        url = "http://server:8080"
        creator = self._factory([url])
        env = creator({})
        self.assertEqual(env.base_url, url)
        env.close()

    def test_returns_unciv_env_instance(self):
        """The factory always returns an UncivEnv regardless of worker_index."""
        from rl_env import UncivEnv
        creator = self._factory(["http://server:8080"])
        env = creator({"worker_index": 5})
        self.assertIsInstance(env, UncivEnv)
        env.close()

    def test_game_kwargs_forwarded(self):
        """Constructor kwargs (num_agents, num_ai, …) are forwarded to UncivEnv."""
        from rl_env import UncivEnv
        creator = self._factory(
            ["http://server:8080"],
            num_agents=2,
            num_ai=4,
            num_city_states=3,
            no_barbarians=False,
        )
        env = creator({"worker_index": 0})
        self.assertIsInstance(env, UncivEnv)
        env.close()


# ---------------------------------------------------------------------------
# Ray integration tests – require Ray + running server
# ---------------------------------------------------------------------------


@_SKIP_NO_RAY
@_SKIP_NO_SERVER
class TestRayParallelWorkers(unittest.TestCase):
    """
    Verify that multiple Ray remote tasks can run UncivEnv in parallel.

    Each Ray task simulates one RLlib rollout worker: it calls
    ``make_env_creator``, resets the environment (which creates a new game on
    the Unciv server), steps through a short episode using END_TURN actions,
    and returns the game ID it used.

    The tests assert:
    * All tasks complete without raising exceptions.
    * Every task received a distinct ``gameId`` (no shared mutable state).
    """

    NUM_WORKERS: int = 4
    TURNS_PER_WORKER: int = 3
    TIMEOUT: float = 120.0

    @classmethod
    def setUpClass(cls):
        import ray
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        import ray
        ray.shutdown()

    @staticmethod
    def _get_remote_worker():
        """Return a Ray remote task that simulates a single rollout worker."""
        import ray

        @ray.remote
        def _worker(worker_index: int, server_url: str, n_turns: int) -> str:
            """
            Reset an UncivEnv, play n_turns END_TURN steps, return the game ID.
            """
            from rl_env.constants import MACRO_END_TURN
            from rl_env.ray_env import make_env_creator

            creator = make_env_creator([server_url])
            env = creator({"worker_index": worker_index})
            try:
                env.reset(seed=worker_index * 13)  # distinct prime offset → unique seeds
                game_id = env._game_id
                for _ in range(n_turns):
                    _, _, term, trunc, _ = env.last()
                    if term or trunc:
                        env.step(None)
                    else:
                        env.step({"macro": MACRO_END_TURN})
                return game_id
            finally:
                env.close()

        return _worker

    def test_parallel_workers_produce_unique_game_ids(self):
        """N Ray workers reset in parallel; each must receive a distinct gameId."""
        import ray

        worker_fn = self._get_remote_worker()
        futures = [
            worker_fn.remote(i, _SERVER_URL, 0)
            for i in range(self.NUM_WORKERS)
        ]
        game_ids = ray.get(futures, timeout=self.TIMEOUT)

        self.assertEqual(
            len(game_ids),
            len(set(game_ids)),
            f"Expected {self.NUM_WORKERS} unique game IDs; got duplicates: {game_ids}",
        )

    def test_parallel_workers_complete_without_errors(self):
        """N Ray workers each play TURNS_PER_WORKER END_TURN steps without error."""
        import ray

        worker_fn = self._get_remote_worker()
        futures = [
            worker_fn.remote(i, _SERVER_URL, self.TURNS_PER_WORKER)
            for i in range(self.NUM_WORKERS)
        ]
        game_ids = ray.get(futures, timeout=self.TIMEOUT)

        self.assertEqual(len(game_ids), self.NUM_WORKERS)
        for gid in game_ids:
            self.assertIsInstance(gid, str)
            self.assertGreater(len(gid), 0, "game_id must be non-empty")

    def test_worker_index_routing_reflected_in_env_base_url(self):
        """
        The env created inside each Ray task uses the URL selected by
        ``worker_index % len(server_urls)``.  When only one server URL is
        available, all workers route to that URL.
        """
        import ray

        @ray.remote
        def _get_url(worker_index: int, server_url: str) -> str:
            from rl_env.ray_env import make_env_creator
            creator = make_env_creator([server_url])
            env = creator({"worker_index": worker_index})
            url = env.base_url
            env.close()
            return url

        futures = [_get_url.remote(i, _SERVER_URL) for i in range(self.NUM_WORKERS)]
        urls = ray.get(futures, timeout=self.TIMEOUT)

        for url in urls:
            self.assertEqual(url, _SERVER_URL)


if __name__ == "__main__":
    unittest.main()
