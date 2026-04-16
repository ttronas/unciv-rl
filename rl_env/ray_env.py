"""Ray-native environment factory for Unciv RL.

:func:`make_env_creator` returns an ``env_creator`` callable that is
compatible with ``ray.tune.registry.register_env`` and the RLlib environment-
creation protocol.  Each Ray rollout worker automatically selects its Unciv RL
server via::

    url = server_urls[config.get("worker_index", 0) % len(server_urls)]

which distributes simulation load across multiple Unciv server JVM processes
without any custom subprocess management or manual thread coordination.

Typical usage – single server, all workers share one JVM process::

    from ray.tune.registry import register_env
    from rl_env.ray_env import make_env_creator

    register_env("unciv_rl", make_env_creator(["http://localhost:8080"]))

Typical usage – multiple servers, workers sharded by ``worker_index``::

    from ray.tune.registry import register_env
    from rl_env.ray_env import make_env_creator

    register_env(
        "unciv_rl",
        make_env_creator([
            "http://localhost:8080",
            "http://localhost:8081",
        ]),
    )
    # Ray automatically routes workers:
    #   worker 0 → :8080   worker 1 → :8081
    #   worker 2 → :8080   worker 3 → :8081 …

See ``docs/Developers/RL-Environment.md`` for full RLlib configuration
examples.
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

    Parameters
    ----------
    server_urls:
        One or more Unciv RL server base URLs
        (e.g. ``["http://localhost:8080"]``).  With a single URL every worker
        connects to the same server.  With multiple URLs workers are
        round-robin sharded by ``worker_index``.
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
