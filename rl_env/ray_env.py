"""Ray-native environment factory for Unciv RL.

:func:`make_env_creator` returns an ``env_creator`` callable that is
compatible with ``ray.tune.registry.register_env`` and the RLlib environment-
creation protocol.

**EnvRunner ↔ JVM isolation**

Each RLlib *EnvRunner* (rollout worker) is identified by its ``worker_index``.
All environments inside the **same** EnvRunner share the **same**
``worker_index``, so they are always routed to the same Unciv JVM server.
Different EnvRunners receive different ``worker_index`` values and are therefore
routed to **different** JVM server processes::

    url = server_urls[config.get("worker_index", 0) % len(server_urls)]

This means:

* EnvRunner 0 (``worker_index=0``) → ``server_urls[0]``
* EnvRunner 1 (``worker_index=1``) → ``server_urls[1]``
* EnvRunner 2 (``worker_index=2``) → ``server_urls[0]``   (wraps around)
* … and so on.

Game state is in-memory inside each JVM, so an EnvRunner must always talk to
the same server that created its games.  Because routing is deterministic
(``worker_index % len(server_urls)``), this invariant is maintained throughout
a training run without any additional bookkeeping.

Typical usage – single server, all EnvRunners share one JVM::

    from ray.tune.registry import register_env
    from rl_env.ray_env import make_env_creator

    register_env("unciv_rl", make_env_creator(["http://localhost:8080"]))

Typical usage – one dedicated JVM per EnvRunner (two servers)::

    from ray.tune.registry import register_env
    from rl_env.ray_env import make_env_creator

    register_env(
        "unciv_rl",
        make_env_creator([
            "http://localhost:8080",
            "http://localhost:8081",
        ]),
    )
    # EnvRunner 1 (worker_index=1) → :8081
    # EnvRunner 2 (worker_index=2) → :8080
    # Every Env inside the same EnvRunner shares the same JVM.

See ``docs/Developers/RL-Environment.md`` for full RLlib configuration
examples including the recommended 2-EnvRunner / 2-Env-per-runner setup.
"""

from __future__ import annotations

from typing import Callable, Sequence

from rl_env import UncivEnv

# Type alias – matches the signature expected by ``register_env``.
EnvCreator = Callable[[dict], UncivEnv]


def make_env_creator(
    server_urls: Sequence[str],
    *,
    num_agents: int = 1,
    num_ai: int = 3,
    num_city_states: int = 0,
    no_barbarians: bool = True,
) -> EnvCreator:
    """Return an ``env_creator`` function for use with Ray / RLlib.

    The returned function accepts the config dict that RLlib passes to each
    rollout worker's env factory and returns a fresh
    :class:`~rl_env.UncivEnv` whose server URL is determined by::

        url = server_urls[config.get("worker_index", 0) % len(server_urls)]

    Because all environments inside the same RLlib EnvRunner share the same
    ``worker_index``, they always connect to the **same** Unciv JVM server.
    Environments in **different** EnvRunners connect to different servers,
    giving each EnvRunner its own isolated JVM process.

    Parameters
    ----------
    server_urls:
        One or more Unciv RL server base URLs
        (e.g. ``["http://localhost:8080"]``).  With a single URL every
        EnvRunner connects to the same server.  With N URLs, EnvRunners are
        round-robin sharded so that each gets its own dedicated JVM.
    num_agents:
        Number of RL agents per game (default ``1``).
    num_ai:
        Number of built-in AI players per game (default ``3``).
    num_city_states:
        Number of city-state civilisations per game (default ``0``).
    no_barbarians:
        Disable barbarian spawning (default ``True``).

    Returns
    -------
    env_creator:
        A ``(config: dict) -> UncivEnv`` callable suitable for
        ``ray.tune.registry.register_env`` or
        ``PPOConfig().environment(env_creator=...)``.

    Raises
    ------
    ValueError
        If *server_urls* is empty.
    """
    urls = list(server_urls)
    if not urls:
        raise ValueError("server_urls must contain at least one URL")

    def env_creator(config: dict) -> UncivEnv:
        # worker_index is the EnvRunner index.  All envs within one EnvRunner
        # share the same worker_index and therefore the same JVM server.
        worker_index: int = config.get("worker_index", 0)
        url = urls[worker_index % len(urls)]
        return UncivEnv(
            base_url=url,
            num_agents=num_agents,
            num_ai=num_ai,
            num_city_states=num_city_states,
            no_barbarians=no_barbarians,
        )

    return env_creator
