"""Archive a pre-deadline FPL bootstrap snapshot."""

import argparse
import asyncio
from pathlib import Path

from fpl.client import FPLClient
from fpl.modeling.snapshots import archive_bootstrap_snapshot


async def collect_snapshot(args: argparse.Namespace) -> Path:
    """Fetch bootstrap data and write one validated snapshot archive."""
    client = FPLClient(timeout=args.timeout, retries=args.retries)
    bootstrap = await client.get_bootstrap_static()
    return archive_bootstrap_snapshot(
        bootstrap,
        season=args.season,
        gameweek=args.gameweek,
        root=args.root,
        force=args.force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="Season label, for example 2026-27")
    parser.add_argument("--gameweek", required=True, type=int)
    parser.add_argument("--root", default="data/snapshots")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Replace an existing archive")
    args = parser.parse_args()
    path = asyncio.run(collect_snapshot(args))
    print(f"Saved snapshot: {path}")


if __name__ == "__main__":
    main()
