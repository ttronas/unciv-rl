"""
Tests for parallel-training options (Options 1–5).

These tests exercise the scaled-training infrastructure described in
``docs/Developers/RL-Environment.md``.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
``UNCIV_RL_URL``
    URL of a single running RL server (default: ``http://localhost:8080``).
    Required for all Option 1 / 4 / 5 single-server tests.

``UNCIV_RL_URLS``
    Comma-separated list of ≥2 server URLs, e.g.
    ``http://localhost:8080,http://localhost:8081``.
    Required for Options 2 / 3 tests when ``UNCIV_RL_JAR`` is not set.

``UNCIV_RL_JAR``
    Absolute path to the built ``server.jar``.  When set together with
    ``UNCIV_RL_ASSETS``, multi-server tests will automatically start and stop
    JAR processes on free ports so that no pre-running servers are needed.

``UNCIV_RL_ASSETS``
    Path to the directory that *contains* the ``jsons/`` folder (i.e. the
    directory you would ``cd`` into before running ``java -jar server.jar``).
    Typically ``<repo-root>/android/assets``.

Test classes
~~~~~~~~~~~~
:class:`TestOption1ConcurrentGames`
    Multiple concurrent games on a single server (Option 1).  Spawns N
    threads that each call ``new_game`` / ``step`` independently and verifies
    that game IDs are unique and no requests fail.

:class:`TestOption4MultipleEnvInstances`
    Multiple independent ``UncivEnv`` instances targeting the same server
    (Option 4, equivalent to ``num_envs_per_worker``).  Verifies that resets
    and steps across all instances are truly independent.

:class:`TestOption5PerGameMutex`
    Stress-tests the per-game mutex added in Option 5: N threads send
    concurrent requests to the *same* game and verify the server never
    returns a 5xx error and game state remains consistent.

:class:`TestOption2WorkerSharding`
    Worker-index → server-port sharding across ≥2 live servers (Option 2).
    Requires ``UNCIV_RL_URLS`` (≥2 URLs) or ``UNCIV_RL_JAR`` + ``UNCIV_RL_ASSETS``.

:class:`TestOption3MultiServerIndependence`
    Verifies that game state is strictly local to the server that created it
    (Option 3): a game created on server A returns 404 on server B.

:class:`TestMultiJarServerLifecycle`
    End-to-end: starts 2 real JAR server processes, runs a short episode on
    each, verifies independence, and shuts down cleanly.
    Requires ``UNCIV_RL_JAR`` + ``UNCIV_RL_ASSETS``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from rl_env.action_mapper import decode_action_result, encode_action
from rl_env.client import UncivRLClient
from rl_env.constants import (
    ENTITY_FEATURES,
    MACRO_END_TURN,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
)

# ---------------------------------------------------------------------------
# Configuration – read once at import time
# ---------------------------------------------------------------------------

_PRIMARY_URL: str = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")
_EXTRA_URLS_RAW: str = os.environ.get("UNCIV_RL_URLS", "")
_JAR_PATH: str = os.environ.get("UNCIV_RL_JAR", "")
_ASSETS_DIR: str = os.environ.get("UNCIV_RL_ASSETS", "")


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _url_available(url: str, timeout: float = 3.0) -> bool:
    """Return True if the server at *url* answers GET /isalive."""
    try:
        resp = requests.get(f"{url}/isalive", timeout=timeout)
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def _primary_available() -> bool:
    return _url_available(_PRIMARY_URL)


def _get_extra_urls() -> list[str]:
    """Return the list of reachable URLs from ``UNCIV_RL_URLS``."""
    if not _EXTRA_URLS_RAW:
        return []
    urls = [u.strip() for u in _EXTRA_URLS_RAW.split(",") if u.strip()]
    return [u for u in urls if _url_available(u)]


def _jar_available() -> bool:
    return (
        bool(_JAR_PATH)
        and os.path.isfile(_JAR_PATH)
        and bool(_ASSETS_DIR)
        and os.path.isdir(_ASSETS_DIR)
    )


def _multi_server_available() -> bool:
    """True if ≥2 servers are reachable *or* we can launch them from a JAR."""
    return len(_get_extra_urls()) >= 2 or _jar_available()


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

_SKIP_NO_PRIMARY = unittest.skipUnless(
    _primary_available(),
    "Primary RL server not available. "
    "Set UNCIV_RL_URL and start the server with --rl to run these tests.",
)

_SKIP_NO_MULTI = unittest.skipUnless(
    _multi_server_available(),
    "Multi-server tests require UNCIV_RL_URLS (≥2 reachable URLs) "
    "or UNCIV_RL_JAR + UNCIV_RL_ASSETS to auto-launch JAR servers.",
)

_SKIP_NO_JAR = unittest.skipUnless(
    _jar_available(),
    "JAR lifecycle tests require UNCIV_RL_JAR (path to server.jar) "
    "and UNCIV_RL_ASSETS (directory containing jsons/).",
)


# ---------------------------------------------------------------------------
# ManagedServer – start / stop a JAR server process on a free port
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find an available TCP port on localhost (127.0.0.1)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ManagedServer:
    """
    Manage the lifecycle of a single Unciv RL server JAR process.

    The server is started on a randomly chosen free port by :meth:`start`.
    Use as a context manager for automatic cleanup::

        with ManagedServer() as srv:
            client = UncivRLClient(base_url=srv.base_url)
            ...

    Parameters
    ----------
    jar_path:
        Path to ``server.jar``.  Defaults to the ``UNCIV_RL_JAR`` env var.
    assets_dir:
        Working directory for the server process (must contain ``jsons/``).
        Defaults to the ``UNCIV_RL_ASSETS`` env var.
    port:
        Port to listen on.  A free port is chosen automatically when *None*.
    startup_timeout:
        Seconds to wait for the server to become healthy before raising.
    """

    def __init__(
        self,
        jar_path: str = _JAR_PATH,
        assets_dir: str = _ASSETS_DIR,
        port: int | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        self.jar_path = jar_path
        self.assets_dir = assets_dir
        self.port: int = port if port is not None else _free_port()
        self.base_url: str = f"http://localhost:{self.port}"
        self._proc: subprocess.Popen[bytes] | None = None
        self._startup_timeout = startup_timeout

    def start(self) -> "ManagedServer":
        """Launch the server process and block until it is healthy."""
        self._proc = subprocess.Popen(
            [
                "java", "-jar", self.jar_path,
                "--rl", "-no-auth", "-no-chat",
                "-p", str(self.port),
            ],
            cwd=self.assets_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if _url_available(self.base_url, timeout=2.0):
                return self
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Server process on port {self.port} exited early "
                    f"(code={self._proc.returncode})"
                )
            time.sleep(1.0)
        self.stop()
        raise TimeoutError(
            f"Server on port {self.port} did not become healthy within "
            f"{self._startup_timeout}s"
        )

    def stop(self) -> None:
        """Terminate the server process if it is running."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    def __enter__(self) -> "ManagedServer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_client(url: str = _PRIMARY_URL) -> UncivRLClient:
    return UncivRLClient(base_url=url, timeout=60.0)


def _new_minimal_game(client: UncivRLClient, seed: int = 0) -> dict:
    """Create a 1-agent, 1-AI, no-barbarians game; return the response dict."""
    return client.new_game(
        num_agents=1,
        num_ai=1,
        num_city_states=0,
        no_barbarians=True,
        seed=seed,
    )


def _end_turn(client: UncivRLClient, game_id: str) -> dict:
    """Send an END_TURN action to *game_id* and return the ActionResult."""
    return client.step(game_id, encode_action({"macro": MACRO_END_TURN}))


def _worker_index_to_url(worker_index: int, server_urls: list[str]) -> str:
    """Option 2 sharding: ``worker_index % num_servers``."""
    return server_urls[worker_index % len(server_urls)]


# ---------------------------------------------------------------------------
# Helpers for building/accessing multi-server URL lists inside test classes
# ---------------------------------------------------------------------------

def _acquire_two_server_urls() -> tuple[list[str], list[ManagedServer]]:
    """
    Return ``(urls, managed_servers)``.

    Prefers ``UNCIV_RL_URLS``; falls back to starting 2 JAR processes when
    ``UNCIV_RL_JAR`` + ``UNCIV_RL_ASSETS`` are available.  Raises
    :class:`unittest.SkipTest` if neither can supply ≥2 servers.
    """
    managed: list[ManagedServer] = []
    urls = _get_extra_urls()

    if len(urls) >= 2:
        return urls[:], managed

    if _jar_available():
        for _ in range(2):
            srv = ManagedServer()
            srv.start()
            managed.append(srv)
            urls.append(srv.base_url)
        return urls, managed

    raise unittest.SkipTest(
        "Multi-server tests require UNCIV_RL_URLS (≥2 reachable URLs) "
        "or UNCIV_RL_JAR + UNCIV_RL_ASSETS."
    )


# ===========================================================================
# Option 1 – Many concurrent games on a single server
# ===========================================================================

@_SKIP_NO_PRIMARY
class TestOption1ConcurrentGames(unittest.TestCase):
    """
    Verify that many Ray workers can share one server without interfering with
    each other.  Each 'worker' creates its own game and runs a short episode
    independently.  This corresponds to Option 1 in the parallelisation plan:
    set ``num_rollout_workers`` high and let the ``ConcurrentHashMap`` handle
    the concurrent game storage.
    """

    NUM_WORKERS: int = 8
    TURNS_PER_WORKER: int = 3

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _make_client()

    def test_concurrent_game_creation_gives_unique_ids(self) -> None:
        """N simultaneous ``new_game`` calls must return N distinct game IDs."""
        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as pool:
            futures = [
                pool.submit(_new_minimal_game, self.client)
                for _ in range(self.NUM_WORKERS)
            ]
            game_ids = [f.result()["gameId"] for f in as_completed(futures)]

        self.assertEqual(
            len(game_ids),
            len(set(game_ids)),
            "Duplicate game IDs returned by concurrent new_game calls",
        )

    def test_concurrent_episodes_are_independent(self) -> None:
        """
        N workers each run TURNS_PER_WORKER END_TURN actions concurrently.
        All steps must succeed and game states must not cross-contaminate.
        """
        errors: list[str] = []

        def run_worker(seed: int) -> tuple[str, int]:
            resp = _new_minimal_game(self.client, seed=seed)
            gid = resp["gameId"]
            for _ in range(self.TURNS_PER_WORKER):
                result = _end_turn(self.client, gid)
                if not result.get("success"):
                    errors.append(f"game={gid}: END_TURN failed: {result}")
            state = self.client.get_state(gid)
            return gid, state["scalars"]["turn"]

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as pool:
            futures = [pool.submit(run_worker, i * 100) for i in range(self.NUM_WORKERS)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual([], errors, f"Worker errors: {errors}")
        game_ids = [r[0] for r in results]
        self.assertEqual(len(game_ids), len(set(game_ids)))
        for gid, turn in results:
            self.assertGreater(turn, 0, f"Game {gid} turn did not advance")

    def test_high_concurrency_no_server_errors(self) -> None:
        """
        16 simultaneous ``new_game`` calls must all succeed (2xx HTTP).
        Validates that the ``ConcurrentHashMap`` handles write bursts safely.
        """
        n = 16
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_new_minimal_game, self.client, i) for i in range(n)]
            for f in as_completed(futures):
                resp = f.result()  # raises if an HTTP error occurred
                self.assertIn("gameId", resp)

    def test_concurrent_observations_for_different_games(self) -> None:
        """
        After creating N games, reading their state concurrently must return
        the correct game ID for each game without cross-contamination.
        """
        game_ids: list[str] = [
            _new_minimal_game(self.client, seed=i)["gameId"]
            for i in range(self.NUM_WORKERS)
        ]

        def read(gid: str) -> tuple[str, str]:
            state = self.client.get_state(gid)
            return gid, state["gameId"]

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as pool:
            futures = [pool.submit(read, gid) for gid in game_ids]
            for f in as_completed(futures):
                requested, returned = f.result()
                self.assertEqual(
                    requested,
                    returned,
                    "Server returned state for a different game than requested",
                )


# ===========================================================================
# Option 4 – Multiple UncivEnv instances per worker
# ===========================================================================

@_SKIP_NO_PRIMARY
class TestOption4MultipleEnvInstances(unittest.TestCase):
    """
    Simulate ``num_envs_per_worker`` by creating multiple ``UncivEnv``
    instances in the same process, all pointing to the same server.  Each
    env creates its own game on ``reset()`` and its episode is independent of
    all others.
    """

    NUM_ENVS: int = 4
    STEPS: int = 3

    def setUp(self) -> None:
        from rl_env import UncivEnv

        self.envs = [
            UncivEnv(
                base_url=_PRIMARY_URL,
                num_agents=1,
                num_ai=1,
                num_city_states=0,
                no_barbarians=True,
            )
            for _ in range(self.NUM_ENVS)
        ]

    def tearDown(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except requests.exceptions.RequestException:
                pass

    def test_parallel_resets_return_unique_game_ids(self) -> None:
        """Each ``env.reset()`` must produce a distinct game ID."""

        def do_reset(env: Any, seed: int) -> str:
            env.reset(seed=seed)
            return env._game_id  # type: ignore[attr-defined]

        with ThreadPoolExecutor(max_workers=self.NUM_ENVS) as pool:
            futures = [
                pool.submit(do_reset, env, i * 7)
                for i, env in enumerate(self.envs)
            ]
            game_ids = [f.result() for f in as_completed(futures)]

        self.assertEqual(
            len(game_ids),
            len(set(game_ids)),
            "Two env instances received the same game ID from the server",
        )

    def test_parallel_resets_return_valid_observation_shapes(self) -> None:
        """All envs must return correctly-shaped observations from ``reset()``."""

        def do_reset(env: Any, seed: int) -> dict:
            obs, _ = env.reset(seed=seed)
            return obs

        with ThreadPoolExecutor(max_workers=self.NUM_ENVS) as pool:
            futures = [
                pool.submit(do_reset, env, i)
                for i, env in enumerate(self.envs)
            ]
            all_obs = [f.result() for f in as_completed(futures)]

        for obs in all_obs:
            for agent_obs in obs.values():
                self.assertEqual(agent_obs["scalars"].shape, (N_SCALAR_FEATURES,))
                self.assertEqual(
                    agent_obs["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES)
                )
                self.assertEqual(
                    agent_obs["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES)
                )

    def test_envs_step_independently(self) -> None:
        """Steps on one env must not affect another env's game state or game ID."""
        # Sequential resets to avoid threading noise on IDs
        for i, env in enumerate(self.envs):
            env.reset(seed=i * 13)

        game_ids_before = [env._game_id for env in self.envs]  # type: ignore[attr-defined]
        self.assertEqual(
            len(game_ids_before),
            len(set(game_ids_before)),
            "Envs share a game ID before stepping",
        )

        errors: list[str] = []

        def do_steps(env: Any) -> None:
            for _ in range(self.STEPS):
                env.step({"macro": MACRO_END_TURN})

        with ThreadPoolExecutor(max_workers=self.NUM_ENVS) as pool:
            futures = [pool.submit(do_steps, env) for env in self.envs]
            for f in as_completed(futures):
                try:
                    f.result()
                except requests.exceptions.RequestException as e:
                    errors.append(str(e))

        self.assertEqual([], errors, f"Env step errors: {errors}")
        # Game IDs must be unchanged (no accidental reset)
        game_ids_after = [env._game_id for env in self.envs]  # type: ignore[attr-defined]
        self.assertEqual(game_ids_before, game_ids_after)

    def test_sequential_resets_within_same_env_are_independent(self) -> None:
        """
        Calling ``reset()`` multiple times on the same env must always produce
        a valid new game (Option 4 robustness: each worker episode restarts
        cleanly without leaking state from the previous episode).
        """
        env = self.envs[0]
        previous_id: str | None = None
        for i in range(3):
            obs, _ = env.reset(seed=i)
            current_id: str = env._game_id  # type: ignore[attr-defined]
            self.assertIsNotNone(current_id)
            self.assertGreater(len(current_id), 0)
            # A new game should be created each time
            if previous_id is not None:
                self.assertNotEqual(
                    current_id,
                    previous_id,
                    "reset() returned the same game ID twice",
                )
            for agent_obs in obs.values():
                self.assertEqual(agent_obs["scalars"].shape, (N_SCALAR_FEATURES,))
            previous_id = current_id


# ===========================================================================
# Option 5 – Per-game mutex correctness under concurrency
# ===========================================================================

@_SKIP_NO_PRIMARY
class TestOption5PerGameMutex(unittest.TestCase):
    """
    Stress-test the per-game ``Mutex`` introduced in Option 5.

    All threads target the *same* ``gameId``; the mutex must serialise their
    access so that:
    * The server never returns a 5xx error.
    * ``GET /rl/state`` always returns internally-consistent data.
    * Concurrent ``reset`` calls are handled gracefully (one succeeds, the
      rest get 404 – no corruption or 500 errors).
    """

    NUM_CONCURRENT: int = 8

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _make_client()

    def test_concurrent_state_reads_never_error(self) -> None:
        """N threads simultaneously reading the same game must all succeed."""
        resp = _new_minimal_game(self.client)
        game_id = resp["gameId"]
        errors: list[Exception] = []

        def read_state() -> dict | None:
            try:
                return self.client.get_state(game_id)
            except requests.exceptions.RequestException as e:
                errors.append(e)
                return None

        with ThreadPoolExecutor(max_workers=self.NUM_CONCURRENT) as pool:
            futures = [pool.submit(read_state) for _ in range(self.NUM_CONCURRENT)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual([], errors, f"Concurrent state-read errors: {errors}")
        for r in results:
            if r is not None:
                self.assertIn("entities", r)
                self.assertEqual(r["gameId"], game_id)

    def test_concurrent_action_mask_reads_never_error(self) -> None:
        """N threads simultaneously reading the action mask must all succeed."""
        resp = _new_minimal_game(self.client)
        game_id = resp["gameId"]
        errors: list[Exception] = []

        def read_mask() -> dict | None:
            try:
                return self.client.get_action_mask(game_id)
            except requests.exceptions.RequestException as e:
                errors.append(e)
                return None

        with ThreadPoolExecutor(max_workers=self.NUM_CONCURRENT) as pool:
            futures = [pool.submit(read_mask) for _ in range(self.NUM_CONCURRENT)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual([], errors, f"Concurrent mask-read errors: {errors}")
        for r in results:
            if r is not None:
                self.assertIn("macroMask", r)

    def test_concurrent_end_turns_are_serialised_no_5xx(self) -> None:
        """
        N threads race to send END_TURN to the same game.

        The per-game mutex ensures requests are processed one at a time.
        We expect:
        * No 5xx HTTP errors.
        * The turn counter advances (at least one END_TURN succeeded).
        """
        resp = _new_minimal_game(self.client)
        game_id = resp["gameId"]
        initial_turn: int = self.client.get_state(game_id)["scalars"]["turn"]

        http_500_errors: list[Exception] = []

        def try_end_turn() -> None:
            try:
                _end_turn(self.client, game_id)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500:
                    http_500_errors.append(e)
            except requests.exceptions.ConnectionError:
                pass  # transient – not a 5xx server bug

        with ThreadPoolExecutor(max_workers=self.NUM_CONCURRENT) as pool:
            futures = [pool.submit(try_end_turn) for _ in range(self.NUM_CONCURRENT)]
            for f in as_completed(futures):
                f.result()

        self.assertEqual(
            [],
            http_500_errors,
            f"5xx errors from concurrent END_TURNs: {http_500_errors}",
        )
        final_turn: int = self.client.get_state(game_id)["scalars"]["turn"]
        self.assertGreater(final_turn, initial_turn, "Turn did not advance at all")

    def test_concurrent_resets_no_corrupt_state(self) -> None:
        """
        Multiple threads simultaneously reset the same game.

        Exactly one reset should succeed; others should receive 404 (game
        already removed by the winning thread).  No thread should receive a
        5xx error or corrupt response.
        """
        resp = _new_minimal_game(self.client)
        game_id = resp["gameId"]

        successes: list[dict] = []
        not_found: list[int] = []
        server_errors: list[Exception] = []
        lock = threading.Lock()

        def do_reset() -> None:
            try:
                r = self.client.reset(game_id)
                with lock:
                    successes.append(r)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    with lock:
                        not_found.append(404)
                elif e.response is not None and e.response.status_code >= 500:
                    with lock:
                        server_errors.append(e)

        threads = [threading.Thread(target=do_reset) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60.0)
            self.assertFalse(t.is_alive(), "Reset thread did not complete within 60 s")

        self.assertEqual([], server_errors, f"5xx errors from concurrent reset: {server_errors}")
        # At least one reset must have succeeded
        self.assertGreater(len(successes), 0, "All concurrent resets returned 404")
        # The successful response must contain a valid observation
        for success_resp in successes:
            self.assertIn("observation", success_resp)
            self.assertIn("gameId", success_resp)

    def test_mixed_read_write_concurrency(self) -> None:
        """
        Mix of state reads, mask reads, and END_TURN steps on the same game –
        no 5xx errors, and the final state is valid (entities field present).
        """
        resp = _new_minimal_game(self.client)
        game_id = resp["gameId"]
        errors: list[Exception] = []

        def read_state() -> None:
            try:
                self.client.get_state(game_id)
            except requests.exceptions.RequestException as e:
                errors.append(e)

        def read_mask() -> None:
            try:
                self.client.get_action_mask(game_id)
            except requests.exceptions.RequestException as e:
                errors.append(e)

        def do_step() -> None:
            try:
                _end_turn(self.client, game_id)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500:
                    errors.append(e)

        tasks = (
            [read_state] * 4
            + [read_mask] * 4
            + [do_step] * 4
        )

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(task) for task in tasks]
            for f in as_completed(futures):
                f.result()

        self.assertEqual([], errors, f"Mixed concurrency errors: {errors}")
        # Final state must be readable and valid
        final = self.client.get_state(game_id)
        self.assertIn("entities", final)


# ===========================================================================
# Option 2 – Worker-index sharding across multiple servers
# ===========================================================================

@_SKIP_NO_MULTI
class TestOption2WorkerSharding(unittest.TestCase):
    """
    Verify the ``worker_index → port`` sharding strategy documented in
    Option 2.  With N servers, worker ``i`` talks to
    ``server_urls[i % N]``; each server accumulates its own independent games.

    Servers are obtained from ``UNCIV_RL_URLS`` (preferred) or started
    automatically from ``UNCIV_RL_JAR`` + ``UNCIV_RL_ASSETS``.
    """

    NUM_WORKERS: int = 8

    @classmethod
    def setUpClass(cls) -> None:
        cls._managed: list[ManagedServer] = []
        cls.server_urls, cls._managed = _acquire_two_server_urls()

    @classmethod
    def tearDownClass(cls) -> None:
        for srv in cls._managed:
            srv.stop()

    def test_worker_routes_to_assigned_server(self) -> None:
        """
        Each worker creates a game on its assigned server.
        Verify that game IDs created on server A are *not* visible on server B.
        """
        num_servers = len(self.server_urls)
        game_per_server: dict[str, str] = {}  # creating_url → game_id

        for worker_idx in range(num_servers):
            url = _worker_index_to_url(worker_idx, self.server_urls)
            client = _make_client(url)
            resp = _new_minimal_game(client, seed=worker_idx * 99)
            game_per_server[url] = resp["gameId"]

        for creating_url, game_id in game_per_server.items():
            for other_url in self.server_urls:
                if other_url == creating_url:
                    continue
                other_client = _make_client(other_url)
                with self.assertRaises(requests.exceptions.HTTPError) as ctx:
                    other_client.get_state(game_id)
                self.assertEqual(
                    ctx.exception.response.status_code,
                    404,
                    f"Game {game_id} from {creating_url} was unexpectedly "
                    f"accessible on {other_url}",
                )

    def test_concurrent_sharded_workers_no_errors(self) -> None:
        """
        NUM_WORKERS workers run concurrently, each on its assigned server.
        All steps must succeed; game IDs must all be distinct.
        """
        errors: list[str] = []

        def run_sharded_worker(worker_idx: int) -> str:
            url = _worker_index_to_url(worker_idx, self.server_urls)
            client = _make_client(url)
            resp = _new_minimal_game(client, seed=worker_idx * 37)
            gid = resp["gameId"]
            for _ in range(3):
                result = _end_turn(client, gid)
                if not result.get("success"):
                    errors.append(
                        f"worker={worker_idx} url={url} game={gid}: {result}"
                    )
            return gid

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as pool:
            futures = [
                pool.submit(run_sharded_worker, i)
                for i in range(self.NUM_WORKERS)
            ]
            game_ids = [f.result() for f in as_completed(futures)]

        self.assertEqual([], errors, f"Worker errors: {errors}")
        self.assertEqual(
            len(game_ids),
            len(set(game_ids)),
            "Duplicate game IDs across sharded workers",
        )

    def test_sharding_distributes_load(self) -> None:
        """
        With NUM_WORKERS workers and ≥2 servers, every server should handle
        at least one worker (no server is left idle).
        """
        num_servers = len(self.server_urls)
        per_server: dict[str, int] = {url: 0 for url in self.server_urls}

        for worker_idx in range(self.NUM_WORKERS):
            url = _worker_index_to_url(worker_idx, self.server_urls)
            per_server[url] += 1

        for url, count in per_server.items():
            self.assertGreater(
                count,
                0,
                f"Server {url} received no workers "
                f"({self.NUM_WORKERS} workers across {num_servers} servers)",
            )

    def test_worker_stays_pinned_to_same_server(self) -> None:
        """
        A fixed worker_index always resolves to the same URL, ensuring sticky
        routing (the game lives on exactly one server for its entire lifetime).
        """
        for worker_idx in range(self.NUM_WORKERS * 3):
            url1 = _worker_index_to_url(worker_idx, self.server_urls)
            url2 = _worker_index_to_url(worker_idx, self.server_urls)
            self.assertEqual(
                url1,
                url2,
                f"worker_index={worker_idx} resolved to different URLs on repeat calls",
            )


# ===========================================================================
# Option 3 – Multi-server independence (in-memory state isolation)
# ===========================================================================

@_SKIP_NO_MULTI
class TestOption3MultiServerIndependence(unittest.TestCase):
    """
    Verify that game state is strictly local to the JVM process that created
    it.  This is the correctness property that makes worker pinning necessary:
    a request for game G must be sent to the *same* server that created G.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._managed: list[ManagedServer] = []
        cls.server_urls, cls._managed = _acquire_two_server_urls()
        cls.client_a = _make_client(cls.server_urls[0])
        cls.client_b = _make_client(cls.server_urls[1])

    @classmethod
    def tearDownClass(cls) -> None:
        for srv in cls._managed:
            srv.stop()

    def test_game_on_server_a_not_found_on_server_b(self) -> None:
        """A game created on server A must return 404 when queried on server B."""
        resp = _new_minimal_game(self.client_a)
        game_id = resp["gameId"]

        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self.client_b.get_state(game_id)
        self.assertEqual(ctx.exception.response.status_code, 404)

    def test_game_on_server_b_not_found_on_server_a(self) -> None:
        """Symmetrical: B's game is invisible to A."""
        resp = _new_minimal_game(self.client_b)
        game_id = resp["gameId"]

        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self.client_a.get_state(game_id)
        self.assertEqual(ctx.exception.response.status_code, 404)

    def test_steps_on_server_a_do_not_affect_server_b(self) -> None:
        """
        Advancing a game on server A must not change server B's game state.
        """
        resp_a = _new_minimal_game(self.client_a, seed=7)
        resp_b = _new_minimal_game(self.client_b, seed=7)
        initial_turn_b: int = self.client_b.get_state(resp_b["gameId"])["scalars"]["turn"]

        for _ in range(5):
            _end_turn(self.client_a, resp_a["gameId"])

        final_turn_b: int = self.client_b.get_state(resp_b["gameId"])["scalars"]["turn"]
        self.assertEqual(
            initial_turn_b,
            final_turn_b,
            "Steps on server A changed server B's game turn",
        )

    def test_both_servers_handle_independent_episodes(self) -> None:
        """
        Both servers can run full independent episodes simultaneously without
        interfering with each other.
        """
        errors: list[str] = []

        def run_episode(client: UncivRLClient, label: str, turns: int = 5) -> None:
            resp = _new_minimal_game(client, seed=42)
            gid = resp["gameId"]
            for i in range(turns):
                result = _end_turn(client, gid)
                if not result.get("success"):
                    errors.append(f"{label} turn={i} game={gid}: {result}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(run_episode, self.client_a, "server_a")
            fb = pool.submit(run_episode, self.client_b, "server_b")
            fa.result()
            fb.result()

        self.assertEqual([], errors, f"Episode errors: {errors}")

    def test_cross_server_reset_returns_404(self) -> None:
        """
        Attempting to reset a game on the wrong server must return 404, not a
        partial/corrupt state.
        """
        resp = _new_minimal_game(self.client_a)
        game_id = resp["gameId"]

        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self.client_b.reset(game_id)
        self.assertEqual(ctx.exception.response.status_code, 404)


# ===========================================================================
# JAR lifecycle – start N servers, run episodes, stop
# ===========================================================================

@_SKIP_NO_JAR
class TestMultiJarServerLifecycle(unittest.TestCase):
    """
    Full end-to-end lifecycle test: starts real server JAR processes, runs
    episodes, verifies independence, and shuts down cleanly.

    Requires ``UNCIV_RL_JAR`` (path to ``server.jar``) and
    ``UNCIV_RL_ASSETS`` (directory containing ``jsons/``).
    """

    NUM_SERVERS: int = 2
    TURNS_PER_EPISODE: int = 5

    def test_start_two_servers_run_episodes_stop(self) -> None:
        """
        Start NUM_SERVERS JARs, run episodes, assert independence, stop.
        This exercises the full Option 3 / docker-compose.rl.yml workflow.
        """
        servers: list[ManagedServer] = []
        try:
            for _ in range(self.NUM_SERVERS):
                srv = ManagedServer()
                srv.start()
                servers.append(srv)

            clients = [_make_client(srv.base_url) for srv in servers]

            # Create one game per server
            game_ids: list[str] = []
            for client in clients:
                resp = _new_minimal_game(client)
                game_ids.append(resp["gameId"])

            # Run episodes
            for client, game_id in zip(clients, game_ids):
                for _ in range(self.TURNS_PER_EPISODE):
                    result = _end_turn(client, game_id)
                    self.assertTrue(
                        result.get("success"),
                        f"END_TURN failed on {client.base_url}: {result}",
                    )

            # Verify isolation: each game exists only on its own server
            for i, (client, game_id) in enumerate(zip(clients, game_ids)):
                for j, other_client in enumerate(clients):
                    if i == j:
                        continue
                    with self.assertRaises(requests.exceptions.HTTPError) as ctx:
                        other_client.get_state(game_id)
                    self.assertEqual(
                        ctx.exception.response.status_code,
                        404,
                        f"Game from server {i} was visible on server {j}",
                    )

        finally:
            for srv in servers:
                srv.stop()

    def test_managed_server_context_manager(self) -> None:
        """``ManagedServer`` can be used as a context manager."""
        with ManagedServer() as srv:
            client = _make_client(srv.base_url)
            resp = _new_minimal_game(client)
            self.assertIn("gameId", resp)

        # Server must be unreachable after __exit__
        self.assertFalse(
            _url_available(srv.base_url, timeout=2.0),
            "Server still reachable after context manager exit",
        )

    def test_server_restarts_cleanly(self) -> None:
        """A server can be stopped and restarted on the same port."""
        port = _free_port()
        srv1 = ManagedServer(port=port)
        srv1.start()
        client = _make_client(srv1.base_url)
        resp1 = _new_minimal_game(client)
        old_game_id = resp1["gameId"]
        srv1.stop()

        # Restart on same port
        srv2 = ManagedServer(port=port)
        try:
            srv2.start()
            client2 = _make_client(srv2.base_url)

            # Old game must be gone (server restarted fresh)
            with self.assertRaises(requests.exceptions.HTTPError) as ctx:
                client2.get_state(old_game_id)
            self.assertEqual(ctx.exception.response.status_code, 404)

            # New game must work
            resp2 = _new_minimal_game(client2)
            self.assertIn("gameId", resp2)
        finally:
            srv2.stop()

    def test_multiple_games_per_server_concurrent(self) -> None:
        """
        A single JAR server can handle multiple concurrent games correctly.
        This is the Option 1 scenario validated against a process we control.
        """
        with ManagedServer() as srv:
            client = _make_client(srv.base_url)
            n = 6
            errors: list[str] = []

            def worker(seed: int) -> str:
                resp = _new_minimal_game(client, seed=seed)
                gid = resp["gameId"]
                for _ in range(3):
                    r = _end_turn(client, gid)
                    if not r.get("success"):
                        errors.append(f"seed={seed} game={gid}: {r}")
                return gid

            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(worker, i * 11) for i in range(n)]
                game_ids = [f.result() for f in as_completed(futures)]

            self.assertEqual([], errors, f"Concurrent game errors: {errors}")
            self.assertEqual(len(game_ids), len(set(game_ids)))

    def test_sharded_workers_across_two_jar_servers(self) -> None:
        """
        Start 2 JARs and run Option 2 sharding: 8 workers split across both
        servers.  All steps succeed; cross-server game access returns 404.
        """
        with ManagedServer() as srv_a, ManagedServer() as srv_b:
            server_urls = [srv_a.base_url, srv_b.base_url]
            num_workers = 8
            errors: list[str] = []
            results: list[tuple[str, str]] = []  # (url, game_id)
            lock = threading.Lock()

            def sharded_worker(worker_idx: int) -> None:
                url = _worker_index_to_url(worker_idx, server_urls)
                client = _make_client(url)
                resp = _new_minimal_game(client, seed=worker_idx * 17)
                gid = resp["gameId"]
                for _ in range(3):
                    r = _end_turn(client, gid)
                    if not r.get("success"):
                        errors.append(f"w={worker_idx} url={url} game={gid}: {r}")
                with lock:
                    results.append((url, gid))

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(sharded_worker, i) for i in range(num_workers)]
                for f in as_completed(futures):
                    f.result()

            self.assertEqual([], errors, f"Sharded worker errors: {errors}")

            # Cross-server isolation check
            client_a = _make_client(srv_a.base_url)
            client_b = _make_client(srv_b.base_url)
            for creating_url, game_id in results:
                wrong_client = client_b if creating_url == srv_a.base_url else client_a
                with self.assertRaises(requests.exceptions.HTTPError) as ctx:
                    wrong_client.get_state(game_id)
                self.assertEqual(
                    ctx.exception.response.status_code,
                    404,
                    f"Game {game_id} from {creating_url} visible on wrong server",
                )
