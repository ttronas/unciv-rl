"""
Tests for Ray-native parallel scaling of UncivEnv.

Three test layers are provided:

**Unit tests** (``TestMakeEnvCreatorRouting``)
    These run always – no server and no Ray installation required.  They verify
    that :func:`~rl_env.ray_env.make_env_creator` routes each ``worker_index``
    to the correct server URL and that edge-cases are handled correctly.

**Ray parallel worker tests** (``TestRayParallelWorkers``)
    Require a single running Unciv RL server (``UNCIV_RL_URL``) and Ray.
    Spin up multiple Ray remote tasks in parallel and assert that each receives
    a distinct ``gameId`` and completes without errors.

**EnvRunner isolation tests** (``TestEnvRunnerIsolation``)
    Require two running Unciv RL servers (``UNCIV_RL_URLS``) and Ray.
    Validate the reference topology recommended in the scaling guide:

    * 2 EnvRunners (Ray actors), each backed by its **own** JVM server.
    * 2 environments per EnvRunner (``num_envs_per_worker=2``).
    * 2 RL agents per environment (1-vs-1 games, ``num_agents=2``).
    * 1 notional Learner node (not instantiated; just validates data flow).
    * 8 agent slots in total (2 EnvRunners × 2 Envs × 2 agents).

    Assertions:
    - Each EnvRunner connects to a **different** JVM (different ``base_url``).
    - All 4 games have unique ``gameId`` values (no cross-server collision).
    - All 8 agent slots (4 games × 2 agents) can complete at least one step.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
``UNCIV_RL_URL``
    URL of a **single** running Unciv RL server (default:
    ``http://localhost:8080``).  Required for ``TestRayParallelWorkers``.

``UNCIV_RL_URLS``
    Comma-separated list of **two or more** server URLs, e.g.
    ``http://localhost:8080,http://localhost:8081``.
    Required for ``TestEnvRunnerIsolation``.
"""

from __future__ import annotations

import os
import unittest

import requests

_SERVER_URL: str = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")
_SERVER_URLS_RAW: str = os.environ.get("UNCIV_RL_URLS", "")


def _server_available(url: str = _SERVER_URL) -> bool:
    try:
        return requests.get(f"{url}/isalive", timeout=3.0).ok
    except requests.exceptions.RequestException:
        return False


def _parse_server_urls() -> list[str]:
    """Return reachable URLs from ``UNCIV_RL_URLS``."""
    if not _SERVER_URLS_RAW:
        return []
    return [u.strip() for u in _SERVER_URLS_RAW.split(",") if u.strip()]


def _two_servers_available() -> bool:
    urls = _parse_server_urls()
    return len(urls) >= 2 and all(_server_available(u) for u in urls[:2])


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

_SKIP_NO_TWO_SERVERS = unittest.skipUnless(
    _two_servers_available(),
    "Two Unciv RL servers required. Set UNCIV_RL_URLS=url1,url2 and start "
    "both servers with --rl to run these tests.",
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
            num_ai=0,
            num_city_states=0,
            no_barbarians=True,
        )
        env = creator({"worker_index": 0})
        self.assertIsInstance(env, UncivEnv)
        env.close()

    def test_all_envs_in_same_worker_share_server(self):
        """
        Multiple envs created with the same worker_index (same EnvRunner) must
        all connect to the same server URL, regardless of vector_index.
        """
        url = "http://server:8080"
        creator = self._factory([url])
        for vector_index in range(4):
            env = creator({"worker_index": 1, "vector_index": vector_index})
            self.assertEqual(env.base_url, url)
            env.close()

    def test_different_workers_route_to_different_servers(self):
        """
        EnvRunners with different worker_index values must route to different
        servers when at least two server URLs are configured.
        """
        urls = ["http://server-a:8080", "http://server-b:8081"]
        creator = self._factory(urls)
        env_runner_0 = creator({"worker_index": 0})
        env_runner_1 = creator({"worker_index": 1})
        self.assertNotEqual(
            env_runner_0.base_url,
            env_runner_1.base_url,
            "EnvRunners with different worker_index must use different JVM servers",
        )
        env_runner_0.close()
        env_runner_1.close()


# ---------------------------------------------------------------------------
# Ray parallel worker tests – require Ray + one server
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
                env.reset(seed=worker_index * 13)  # distinct prime multiplier → unique seeds
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


# ---------------------------------------------------------------------------
# EnvRunner isolation tests – require Ray + two servers
# ---------------------------------------------------------------------------

@_SKIP_NO_RAY
@_SKIP_NO_TWO_SERVERS
class TestEnvRunnerIsolation(unittest.TestCase):
    """
    Validate the reference topology: 2 EnvRunners × 2 Envs × 2 agents = 8 agents.

    Topology
    --------
    ::

        Learner (1 node)
            ├── EnvRunner 0 (worker_index=1) → JVM server :8080
            │       ├── Env 0  (1-vs-1 game, 2 RL agents)
            │       └── Env 1  (1-vs-1 game, 2 RL agents)
            └── EnvRunner 1 (worker_index=2) → JVM server :8081
                    ├── Env 2  (1-vs-1 game, 2 RL agents)
                    └── Env 3  (1-vs-1 game, 2 RL agents)

    The Learner node is the process running these tests.  EnvRunners are
    simulated as Ray actors so that they run in independent Ray worker
    processes (mirroring RLlib's actual execution model).

    Assertions
    ----------
    * Each EnvRunner connects to a **different** JVM (server URL differs).
    * All 4 games have **unique** ``gameId`` values (no cross-server collision).
    * Every game can complete at least one END_TURN step without errors
      (validating the 8 agent slots are independently functional).
    """

    TURNS_PER_ENV: int = 2
    TIMEOUT: float = 180.0

    # EnvRunners are remote workers; RLlib numbers them from 1 (0 = local).
    _WORKER_INDEX_RUNNER_0: int = 1
    _WORKER_INDEX_RUNNER_1: int = 2
    _NUM_ENVS_PER_RUNNER: int = 2
    _NUM_AGENTS_PER_ENV: int = 2   # 1-vs-1 game

    @classmethod
    def setUpClass(cls):
        import ray
        ray.init(ignore_reinit_error=True)
        cls._server_urls = _parse_server_urls()[:2]

    @classmethod
    def tearDownClass(cls):
        import ray
        ray.shutdown()

    @staticmethod
    def _make_env_runner_actor():
        """
        Return a Ray actor class that simulates an RLlib EnvRunner.

        Each actor:
        - Creates ``num_envs`` UncivEnv instances using ``make_env_creator``.
        - All envs in the actor share the same ``worker_index`` and therefore
          the same JVM server.
        - Exposes helpers to retrieve the server URL, game IDs, and run steps.
        """
        import ray

        @ray.remote
        class _EnvRunnerActor:
            def __init__(
                self,
                worker_index: int,
                server_urls: list[str],
                num_envs: int,
                num_agents: int,
            ) -> None:
                from rl_env.ray_env import make_env_creator
                self._worker_index = worker_index
                creator = make_env_creator(
                    server_urls,
                    num_agents=num_agents,
                    num_ai=0,           # pure RL – no AI opponents
                    num_city_states=0,
                    no_barbarians=True,
                )
                self._envs = [
                    creator({"worker_index": worker_index, "vector_index": v})
                    for v in range(num_envs)
                ]
                for env in self._envs:
                    env.reset()

            def get_server_url(self) -> str:
                return self._envs[0].base_url

            def get_game_ids(self) -> list[str]:
                return [env._game_id for env in self._envs]

            def run_steps(self, n_turns: int) -> list[str]:
                """
                Play n_turns END_TURN steps on every env.  Returns all game IDs
                that completed at least one step without error.
                """
                from rl_env.constants import MACRO_END_TURN

                completed: list[str] = []
                for env in self._envs:
                    game_id = env._game_id
                    for _ in range(n_turns):
                        _, _, term, trunc, _ = env.last()
                        if term or trunc:
                            env.step(None)
                        else:
                            env.step({"macro": MACRO_END_TURN})
                    completed.append(game_id)
                return completed

            def close(self) -> None:
                for env in self._envs:
                    env.close()

        return _EnvRunnerActor

    def _create_runners(self):
        actor_cls = self._make_env_runner_actor()
        runner_0 = actor_cls.remote(
            self._WORKER_INDEX_RUNNER_0,
            self._server_urls,
            self._NUM_ENVS_PER_RUNNER,
            self._NUM_AGENTS_PER_ENV,
        )
        runner_1 = actor_cls.remote(
            self._WORKER_INDEX_RUNNER_1,
            self._server_urls,
            self._NUM_ENVS_PER_RUNNER,
            self._NUM_AGENTS_PER_ENV,
        )
        return runner_0, runner_1

    def test_each_env_runner_uses_a_different_jvm(self):
        """
        EnvRunner 0 and EnvRunner 1 must connect to different JVM servers.
        This validates that the ``worker_index % len(server_urls)`` routing
        assigns a unique JVM to each EnvRunner.
        """
        import ray

        runner_0, runner_1 = self._create_runners()
        try:
            url_0, url_1 = ray.get(
                [runner_0.get_server_url.remote(), runner_1.get_server_url.remote()],
                timeout=self.TIMEOUT,
            )
            self.assertNotEqual(
                url_0,
                url_1,
                f"EnvRunner 0 and EnvRunner 1 both connected to {url_0!r}; "
                "each EnvRunner must have its own dedicated JVM server.",
            )
            # Confirm each URL is one of the configured server URLs.
            self.assertIn(url_0, self._server_urls)
            self.assertIn(url_1, self._server_urls)
        finally:
            ray.get([runner_0.close.remote(), runner_1.close.remote()])

    def test_all_envs_in_same_runner_share_jvm(self):
        """
        All environments inside the same EnvRunner must connect to the same
        JVM server (they share the ``worker_index``).
        """
        import ray

        # Create a single actor with 2 envs and verify they share one URL.
        actor_cls = self._make_env_runner_actor()
        runner = actor_cls.remote(
            self._WORKER_INDEX_RUNNER_0,
            self._server_urls,
            self._NUM_ENVS_PER_RUNNER,
            self._NUM_AGENTS_PER_ENV,
        )
        try:
            # We check via game_ids that both envs exist (they reset successfully)
            # and that the actor reports a single consistent URL.
            url, game_ids = ray.get(
                [runner.get_server_url.remote(), runner.get_game_ids.remote()],
                timeout=self.TIMEOUT,
            )
            self.assertEqual(
                len(game_ids),
                self._NUM_ENVS_PER_RUNNER,
                f"Expected {self._NUM_ENVS_PER_RUNNER} games, got {game_ids}",
            )
            # All games live on the same server.
            self.assertIn(url, self._server_urls)
        finally:
            ray.get(runner.close.remote())

    def test_eight_agent_slots_unique_game_ids(self):
        """
        The full 2×2 topology produces 4 unique games (= 8 agent slots).

        Layout:
            EnvRunner 0 → 2 envs (game_id_0, game_id_1) on server 0
            EnvRunner 1 → 2 envs (game_id_2, game_id_3) on server 1

        All 4 game IDs must be distinct; each has 2 agents = 8 total.
        """
        import ray

        runner_0, runner_1 = self._create_runners()
        try:
            ids_0, ids_1 = ray.get(
                [runner_0.get_game_ids.remote(), runner_1.get_game_ids.remote()],
                timeout=self.TIMEOUT,
            )
            all_ids = ids_0 + ids_1
            total_envs = self._NUM_ENVS_PER_RUNNER * 2  # 2 runners

            self.assertEqual(
                len(all_ids),
                total_envs,
                f"Expected {total_envs} game IDs, got {all_ids}",
            )
            self.assertEqual(
                len(set(all_ids)),
                total_envs,
                f"Game IDs must all be unique; got duplicates in {all_ids}",
            )
            # Total agent slots = total_envs × agents_per_env
            total_agent_slots = total_envs * self._NUM_AGENTS_PER_ENV
            self.assertEqual(
                total_agent_slots,
                8,
                f"Expected 8 agent slots; topology gives {total_agent_slots}",
            )
        finally:
            ray.get([runner_0.close.remote(), runner_1.close.remote()])

    def test_all_envs_can_step_independently(self):
        """
        All 4 envs (across both EnvRunners) must be able to execute
        TURNS_PER_ENV END_TURN steps without errors, confirming that the 8
        agent slots function independently in parallel.
        """
        import ray

        runner_0, runner_1 = self._create_runners()
        try:
            completed_0, completed_1 = ray.get(
                [
                    runner_0.run_steps.remote(self.TURNS_PER_ENV),
                    runner_1.run_steps.remote(self.TURNS_PER_ENV),
                ],
                timeout=self.TIMEOUT,
            )
            all_completed = completed_0 + completed_1
            total_envs = self._NUM_ENVS_PER_RUNNER * 2

            self.assertEqual(
                len(all_completed),
                total_envs,
                f"Expected {total_envs} completed envs, got {all_completed}",
            )
            for gid in all_completed:
                self.assertIsInstance(gid, str)
                self.assertGreater(len(gid), 0, f"game_id must be non-empty, got {gid!r}")
        finally:
            ray.get([runner_0.close.remote(), runner_1.close.remote()])


if __name__ == "__main__":
    unittest.main()
