"""
FPL GW1 2026-27 Squad Recommendation Script (Upgraded & Strongly Typed).

Syncs live data from the official FPL API, computes multi-component Expected Points,
and runs the PuLP Integer Program to output the optimal £100.0m 15-player squad,
starting XI, Captain, and Bench order.

Run with: uv run python scripts/recommend_squad.py
"""

import asyncio
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fpl.client import FPLClient
from fpl.data import load_players_df_from_db, sync_bootstrap_to_db
from fpl.optimizer import optimize_squad

console = Console(width=120)


async def generate_recommendation():
    console.print(
        Panel(
            "[bold green]FPL 2026-27 Team Creator — Upgraded Recommendation Engine[/bold green]\n"
            "Fetching live player prices & projections from Fantasy Premier League API...",
            border_style="green",
        )
    )

    client = FPLClient()

    # 1. Sync live data from API
    try:
        bootstrap = await client.get_bootstrap_static()
        count = await sync_bootstrap_to_db(bootstrap)
        console.print(f"  [green]✓ Synced {count} players from official FPL API.[/green]\n")
    except Exception as e:
        console.print(f"  [yellow]⚠ Unable to sync live API ({e}), using cached database...[/yellow]\n")

    # 2. Load players
    players_df = await load_players_df_from_db()

    # 3. Run solver producing strongly typed SquadRecommendation
    squad_rec = optimize_squad(players_df, budget=100.0)

    # 4. Print 15-Man Squad Summary Table
    table = Table(
        title=f"Optimal 15-Man Squad (Budget: £{squad_rec.total_cost:.1f}m / £100.0m | Projected XI xP: {squad_rec.total_expected_points:.2f})",
        border_style="cyan",
    )
    table.add_column("Role", style="magenta", justify="center")
    table.add_column("Pos", style="cyan", justify="center")
    table.add_column("Player", style="bold white")
    table.add_column("Team", style="yellow")
    table.add_column("Cost", justify="right")
    table.add_column("Mins", justify="right")
    table.add_column("Attack xP", justify="right")
    table.add_column("Def/DC xP", justify="right")
    table.add_column("Total xP", justify="right", style="bold green")

    for p in squad_rec.starting_xi:
        proj = p.projection
        table.add_row(
            p.role.value,
            proj.position.value,
            proj.web_name,
            proj.team,
            f"£{proj.cost:.1f}m",
            f"{proj.expected_minutes:.0f}m",
            f"{proj.attack_xp:.2f}",
            f"{proj.clean_sheet_xp + proj.defensive_contribution_xp:.2f}",
            f"{proj.total_xp:.2f}",
        )

    table.add_section()

    for p in squad_rec.bench:
        proj = p.projection
        table.add_row(
            f"BENCH #{p.bench_order}",
            proj.position.value,
            proj.web_name,
            proj.team,
            f"£{proj.cost:.1f}m",
            f"{proj.expected_minutes:.0f}m",
            f"{proj.attack_xp:.2f}",
            f"{proj.clean_sheet_xp + proj.defensive_contribution_xp:.2f}",
            f"{proj.total_xp:.2f}",
        )

    console.print(table)
    console.print(
        f"\n[bold]Captain (2x):[/bold] [green]{squad_rec.captain.web_name}[/green] ({squad_rec.captain.total_xp:.2f} xP) | "
        f"[bold]Vice-Captain:[/bold] [yellow]{squad_rec.vice_captain.web_name}[/yellow] ({squad_rec.vice_captain.total_xp:.2f} xP)\n"
    )


if __name__ == "__main__":
    asyncio.run(generate_recommendation())
