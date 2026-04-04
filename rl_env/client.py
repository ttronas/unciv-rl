"""
HTTP client for the Unciv RL server's ``/rl/*`` endpoints.

All methods raise :class:`requests.HTTPError` on non-2xx responses and
:class:`requests.ConnectionError` / :class:`requests.Timeout` on network
failures.  A simple exponential-backoff retry policy is applied automatically.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Default retry configuration
# ---------------------------------------------------------------------------

_DEFAULT_RETRIES = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)


def _build_session(retries: Retry | None = None) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries or _DEFAULT_RETRIES)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class UncivRLClient:
    """
    Thin wrapper around the Unciv server's ``/rl/*`` REST API.

    Parameters
    ----------
    base_url:
        Base URL of the running Unciv server, e.g. ``"http://localhost:8080"``.
    timeout:
        Per-request timeout in seconds.
    retries:
        Custom :class:`urllib3.util.retry.Retry` configuration.  Defaults to
        3 retries with exponential back-off.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        retries: Retry | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = _build_session(retries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/rl/{path.lstrip('/')}"

    def _get(self, path: str) -> Any:
        response = self._session.get(self._url(path), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict | None = None) -> Any:
        response = self._session.post(
            self._url(path),
            json=payload or {},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def new_game(
        self,
        num_agents: int = 1,
        num_ai: int = 3,
        num_city_states: int = 6,
        difficulty: str = "Prince",
        include_map_planes: bool = True,
        no_barbarians: bool = True,
        seed: int = 0,
    ) -> dict:
        """
        POST /rl/new_game – start a fresh game.

        Returns a dict with keys ``gameId``, ``agentCivIds``,
        ``currentAgent``, and ``observation``.
        """
        return self._post(
            "new_game",
            {
                "numAgents": num_agents,
                "numAI": num_ai,
                "numCityStates": num_city_states,
                "difficulty": difficulty,
                "includeMapPlanes": include_map_planes,
                "noBarbarians": no_barbarians,
                "seed": seed,
            },
        )

    def get_state(self, game_id: str) -> dict:
        """
        GET /rl/state/{gameId} – retrieve the current observation.

        Returns a full :class:`RLObservationResponse` dict.
        """
        return self._get(f"state/{game_id}")

    def step(self, game_id: str, action: dict) -> dict:
        """
        POST /rl/action/{gameId} – execute *action* for the current agent.

        *action* must contain at least the key ``"macro"``; ``"target"``,
        ``"subaction"``, ``"arg1"``, and ``"arg2"`` default to 0 on the server.

        Returns an :class:`ActionResult` dict with keys ``success``,
        ``message``, and ``reward``.
        """
        payload = {
            "macro": int(action.get("macro", 0)),
            "target": int(action.get("target", 0)),
            "subaction": int(action.get("subaction", 0)),
            "arg1": int(action.get("arg1", 0)),
            "arg2": int(action.get("arg2", 0)),
        }
        return self._post(f"action/{game_id}", payload)

    def get_action_mask(self, game_id: str) -> dict:
        """
        GET /rl/action_mask/{gameId} – compute legal action mask.

        Returns an :class:`RLActionMaskResponse` dict.
        """
        return self._get(f"action_mask/{game_id}")

    def reset(self, game_id: str) -> dict:
        """
        POST /rl/reset/{gameId} – reset the game to a fresh start.

        Returns a :class:`ResetResponse` dict.
        """
        return self._post(f"reset/{game_id}")

    def get_done(self, game_id: str) -> dict:
        """
        GET /rl/done/{gameId} – check termination status.

        Returns a dict with keys ``done``, ``winner``, and ``scores``.
        """
        return self._get(f"done/{game_id}")

    # ------------------------------------------------------------------
    # Convenience: poll until the server is ready
    # ------------------------------------------------------------------

    def wait_until_ready(self, max_wait: float = 60.0, poll_interval: float = 1.0) -> None:
        """
        Block until the server responds to ``GET /isalive``.

        Raises :class:`TimeoutError` if the server does not respond within
        *max_wait* seconds.
        """
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            try:
                resp = self._session.get(
                    f"{self.base_url}/isalive", timeout=5.0
                )
                if resp.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Unciv server at {self.base_url} did not become ready within {max_wait}s"
        )
