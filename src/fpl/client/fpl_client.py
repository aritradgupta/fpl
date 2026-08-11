"""
FPL Official API Async Client.

Handles interaction with live Fantasy Premier League API endpoints:
- bootstrap-static/ (players, teams, gameweeks, settings)
- fixtures/ (full season match schedule & FDR)
- element-summary/{player_id}/ (per-player detailed history)
- entry/{manager_id}/ (user squad import)
"""

from typing import Any, Dict, List, Optional
import httpx


BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Async client for Fantasy Premier League API."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    async def get_bootstrap_static(self) -> Dict[str, Any]:
        """Fetch general FPL data including players (elements), teams, and gameweeks."""
        url = f"{self.base_url}/bootstrap-static/"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_fixtures(self, event: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch fixtures. Optionally filter by gameweek event ID."""
        url = f"{self.base_url}/fixtures/"
        if event is not None:
            url += f"?event={event}"

        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """Fetch detailed stats and history for a specific player ID."""
        url = f"{self.base_url}/element-summary/{player_id}/"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_user_entry(self, user_id: int) -> Dict[str, Any]:
        """Fetch manager entry info by user ID."""
        url = f"{self.base_url}/entry/{user_id}/"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
