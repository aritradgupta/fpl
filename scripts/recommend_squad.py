"""
FPL GW1 2026-27 Squad Recommendation Script (Multi-Solver Suite Support).

Syncs live data from official FPL API and runs your choice of solver model:
- single_period: Fast 0-1 PuLP Integer Linear Program
- multi_period: Multi-Gameweek Horizon ILP Solver
- stochastic: Monte Carlo Scenario Risk Solver
- genetic: Metaheuristic Evolutionary Algorithm

Run: uv run python scripts/recommend_squad.py --solver genetic --seed 42 --generations 50
"""

import argparse
import asyncio
import sys

import pandas as pd

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


async def generate_recommendation(
    solver_type: str = "single_period",
    seed: int | None = 42,
    generations: int = 50,
    population_size: int = 60,
    risk_aversion: float = 0.15,
    horizon_weeks: int = 3,
    lock_players: list[str] | None = None,
    exclude_players: list[str] | None = None,
    gameweek: int = 1,
):
    console.print(
        Panel(
            f"[bold green]FPL 2026-27 Team Creator — Solver Model: [{solver_type.upper()}][/bold green]\n"
            f"Config: seed={seed}, gen={generations}, pop={population_size}, risk={risk_aversion}, horizon={horizon_weeks}w, locked={lock_players}\n"
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

    try:
        fixtures_df = pd.DataFrame(await client.get_fixtures())
        if fixtures_df.empty:
            fixtures_df = None
    except Exception as exc:
        fixtures_df = None
        console.print(f"  [yellow]⚠ Unable to fetch fixtures ({exc}); using neutral fixture fallback.[/yellow]\n")

    # 3. Run selected solver strategy
    squad_rec = optimize_squad(
        players_df,
        budget=100.0,
        solver_type=solver_type,
        seed=seed,
        generations=generations,
        population_size=population_size,
        risk_aversion=risk_aversion,
        horizon_weeks=horizon_weeks,
        lock_players=lock_players,
        exclude_players=exclude_players,
        fixtures_df=fixtures_df,
        gameweek=gameweek,
    )

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for genetic/stochastic solvers")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations for genetic algorithm")
    parser.add_argument("--population", type=int, default=60, help="Population size for genetic algorithm")
    parser.add_argument(
        "--risk-aversion", type=float, default=0.15, help="Risk aversion parameter for stochastic solver"
    )
    parser.add_argument("--horizon", type=int, default=3, help="Horizon weeks count for multi-period solver")
    parser.add_argument("--gameweek", type=int, default=1, help="First gameweek to project")
    parser.add_argument(
        "--lock", type=str, nargs="*", help="Lock specific player name(s) into squad (e.g. --lock Haaland)"
    )
    parser.add_argument("--exclude", type=str, nargs="*", help="Exclude specific player name(s) from squad")

    args = parser.parse_args()
    asyncio.run(
        generate_recommendation(
            solver_type=args.solver,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population,
            risk_aversion=args.risk_aversion,
            horizon_weeks=args.horizon,
            lock_players=args.lock,
            exclude_players=args.exclude,
            gameweek=args.gameweek,
        )
    )


if __name__ == "__main__":
    main()
