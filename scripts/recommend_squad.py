"""
FPL GW1 2026-27 Squad Recommendation Script (Multi-Solver Suite Support).

Syncs live data from official FPL API and runs your choice of solver model:
- single_period: Fast 0-1 PuLP Integer Linear Program
- multi_period: Multi-Gameweek Horizon ILP Solver
- stochastic: Monte Carlo Scenario Risk Solver
- genetic: Metaheuristic Evolutionary Algorithm

Run: uv run python scripts/recommend_squad.py --solver single_period
"""

import argparse
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
from fpl.models.squad import SolverType
from fpl.optimizer import optimize_squad

console = Console(width=120)


async def generate_recommendation(solver_type: str = "single_period"):
    console.print(
        Panel(
            f"[bold green]FPL 2026-27 Team Creator — Solver Model: [{solver_type.upper()}][/bold green]\n"
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

    # 3. Run selected solver strategy
    squad_rec = optimize_squad(players_df, budget=100.0, solver_type=solver_type)

    # 4. Print 15-Man Squad Summary Table
    table = Table(
        title=f"Optimal Squad [{solver_type.upper()}] (Cost: £{squad_rec.total_cost:.1f}m / £100.0m | XI xP: {squad_rec.total_expected_points:.2f})",
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


def main():
    parser = argparse.ArgumentParser(description="FPL Squad Recommendation CLI")
    parser.add_argument(
        "--solver",
        type=str,
        default="single_period",
        choices=[s.value for s in SolverType],
        help="Solver model strategy: single_period, multi_period, stochastic, genetic",
    )
    args = parser.parse_args()
    asyncio.run(generate_recommendation(solver_type=args.solver))


if __name__ == "__main__":
    main()
