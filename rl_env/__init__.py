"""
rl_env – PettingZoo AEC environment wrapper for the Unciv Kotlin game engine.

Typical usage::

    from rl_env import UncivEnv
    env = UncivEnv(base_url="http://localhost:8080")
    observations, infos = env.reset()
    ...
"""

from rl_env.unciv_env import UncivEnv

__all__ = ["UncivEnv"]
