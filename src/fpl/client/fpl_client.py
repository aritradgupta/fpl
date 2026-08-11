"""
FPL Official API Async Client.

Handles interaction with live Fantasy Premier League API endpoints:
- bootstrap-static/ (players, teams, gameweeks, settings)
- fixtures/ (full season match schedule & FDR)
- element-summary/{player_id}/ (per-player detailed history)
- entry/{manager_id}/ (user squad import)
"""

import asyncio
from typing import Any

import httpx

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Async client for Fantasy Premier League API."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET and decode an endpoint, retrying transient failures."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            for attempt in range(self.retries + 1):
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    if not retryable or attempt >= self.retries:
                        raise
                    await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError("unreachable")

    async def get_bootstrap_static(self) -> dict[str, Any]:
        """Fetch general FPL data including players (elements), teams, and gameweeks."""
        return await self._get_json("/bootstrap-static/")

    async def get_fixtures(self, event: int | None = None) -> list[dict[str, Any]]:
        """Fetch fixtures. Optionally filter by gameweek event ID."""
        return await self._get_json("/fixtures/", {"event": event} if event is not None else None)

    async def get_player_summary(self, player_id: int) -> dict[str, Any]:
        """Fetch detailed stats and history for a specific player ID."""
        return await self._get_json(f"/element-summary/{player_id}/")

    async def get_user_entry(self, user_id: int) -> dict[str, Any]:
        """Fetch manager entry info by user ID."""
        return await self._get_json(f"/entry/{user_id}/")
