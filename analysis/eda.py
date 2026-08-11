"""
FPL Historical Data — Exploratory Data Analysis

This script loads vaastav's historical FPL data across multiple seasons and
performs a comprehensive analysis to answer key questions that will drive
the design of our expected points model and recommendation engine.

Run: uv run python analysis/eda.py
"""

# pylint: disable=too-many-lines

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors with Rich
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

console = Console(width=120)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data" / "historical" / "data"
RECENT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
ALL_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    *RECENT_SEASONS,
]

# Columns that exist across recent seasons (2020-21 onward) in merged_gw.csv
COMMON_GW_COLS = [
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "selected",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
    "GW",
]

# Additional columns in newer seasons (2022-23+)
EXPECTED_COLS = [
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "starts",
]

# 2025-26 specific
DEFENSIVE_COLS = [
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "recoveries",
    "tackles",
]


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with UTF-8, falling back to latin-1 for older seasons with accented names."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


def load_season_gw(season: str) -> pd.DataFrame | None:
    """Load merged_gw.csv for a given season."""
    path = DATA_ROOT / season / "gws" / "merged_gw.csv"
    if not path.exists():
        console.print(f"[yellow]No merged_gw.csv for {season}[/yellow]")
        return None

    df = _read_csv_safe(path)
    df["season"] = season
    return df


def load_season_fixtures(season: str) -> pd.DataFrame | None:
    """Load fixtures.csv for a given season."""
    path = DATA_ROOT / season / "fixtures.csv"
    if not path.exists():
        return None
    return _read_csv_safe(path)


def load_season_teams(season: str) -> pd.DataFrame | None:
    """Load teams.csv for a given season."""
    path = DATA_ROOT / season / "teams.csv"
    if not path.exists():
        return None
    return _read_csv_safe(path)


def load_all_seasons() -> pd.DataFrame:
    """Load and combine GW data from all recent seasons."""
    frames = []
    for season in RECENT_SEASONS:
        df = load_season_gw(season)
        if df is not None:
            frames.append(df)
            console.print(f"  [green]✓[/green] {season}: {len(df):,} rows, {len(df.columns)} columns")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    console.print(f"\n  [bold green]Combined:[/bold green] {len(combined):,} total rows across {len(frames)} seasons\n")
    return combined


# ──────────────────────────────────────────────
# Section 1: Schema Analysis
# ──────────────────────────────────────────────
def analyze_schemas():
    """Compare column schemas across all seasons."""
    console.rule("[bold cyan]1. Schema Analysis Across Seasons[/bold cyan]")

    season_cols = {}
    for season in ALL_SEASONS:
        df = load_season_gw(season)
        if df is not None:
            season_cols[season] = set(df.columns)

    # Find columns present in ALL seasons
    if not season_cols:
        console.print("[red]No data found![/red]")
        return

    common = set.intersection(*season_cols.values())
    console.print(f"\n[bold]Columns in ALL {len(season_cols)} seasons[/bold] ({len(common)}):")
    console.print(f"  {sorted(common)}\n")

    # Show what was added/removed per season
    table = Table(title="Column Evolution", box=box.SIMPLE_HEAVY)
    table.add_column("Season", style="cyan")
    table.add_column("Total Cols", justify="right")
    table.add_column("Added (vs previous)", style="green")
    table.add_column("Removed (vs previous)", style="red")

    prev_cols = None
    for season in sorted(season_cols.keys()):
        cols = season_cols[season]
        if prev_cols is not None:
            added = cols - prev_cols
            removed = prev_cols - cols
            table.add_row(
                season,
                str(len(cols)),
                ", ".join(sorted(added)) if added else "—",
                ", ".join(sorted(removed)) if removed else "—",
            )
        else:
            table.add_row(season, str(len(cols)), "(baseline)", "—")
        prev_cols = cols

    console.print(table)
    console.print()


# ──────────────────────────────────────────────
# Section 2: Basic Data Profiling
# ──────────────────────────────────────────────
def profile_data(df: pd.DataFrame):
    """Basic data profiling: shape, types, missing values, key distributions."""
    console.rule("[bold cyan]2. Data Profiling[/bold cyan]")

    # Shape
    console.print(f"[bold]Shape:[/bold] {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Position distribution
    if "position" in df.columns:
        pos_dist = df["position"].value_counts()
        table = Table(title="Records by Position", box=box.SIMPLE)
        table.add_column("Position", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Pct", justify="right")
        for pos, count in pos_dist.items():
            table.add_row(pos, f"{count:,}", f"{count / len(df) * 100:.1f}%")
        console.print(table)

    # Season distribution
    if "season" in df.columns:
        season_dist = df["season"].value_counts().sort_index()
        table = Table(title="Records by Season", box=box.SIMPLE)
        table.add_column("Season", style="cyan")
        table.add_column("Player-GWs", justify="right")
        table.add_column("Unique Players", justify="right")
        for season, count in season_dist.items():
            n_players = df[df["season"] == season]["name"].nunique()
            table.add_row(season, f"{count:,}", str(n_players))
        console.print(table)

    # Missing values in key columns
    key_cols = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "bps",
        "creativity",
        "influence",
        "threat",
        "ict_index",
        "value",
        "xP",
        "selected",
    ]
    existing_key_cols = [c for c in key_cols if c in df.columns]
    missing = df[existing_key_cols].isnull().sum()
    if missing.sum() > 0:
        console.print("\n[bold]Missing values in key columns:[/bold]")
        for col, n_missing in missing[missing > 0].items():
            console.print(f"  {col}: {n_missing:,} ({n_missing / len(df) * 100:.1f}%)")
    else:
        console.print("\n[green]✓ No missing values in key columns[/green]")

    # Points distribution
    console.print("\n[bold]Total Points per Player-GW distribution:[/bold]")
    pts = df["total_points"]
    console.print(
        f"  Mean: {pts.mean():.2f} | Median: {pts.median():.0f} | "
        f"Std: {pts.std():.2f} | Min: {pts.min():.0f} | Max: {pts.max():.0f}"
    )
    console.print(f"  Zero-point GWs: {(pts == 0).sum():,} ({(pts == 0).sum() / len(pts) * 100:.1f}%)")
    console.print(f"  Double-digit hauls (≥10): {(pts >= 10).sum():,} ({(pts >= 10).sum() / len(pts) * 100:.1f}%)")
    console.print()


# ──────────────────────────────────────────────
# Section 3: Feature Correlation with Points
# ──────────────────────────────────────────────
def analyze_correlations(df: pd.DataFrame):
    """Which features correlate most with total_points?"""
    console.rule("[bold cyan]3. Feature Correlation with Total Points[/bold cyan]")

    # Only use numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["element", "fixture", "round", "GW", "opponent_team"]
    feature_cols = [c for c in numeric_cols if c not in exclude and c != "total_points"]

    correlations = {}
    for col in feature_cols:
        valid = df[["total_points", col]].dropna()
        if len(valid) > 100:
            correlations[col] = valid["total_points"].corr(valid[col])

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    table = Table(title="Correlation with total_points", box=box.SIMPLE_HEAVY)
    table.add_column("Feature", style="cyan")
    table.add_column("Correlation", justify="right")
    table.add_column("Strength", justify="center")

    for feat, corr in sorted_corr[:25]:
        abs_corr = abs(corr)
        if abs_corr > 0.5:
            strength = "[bold green]Strong[/bold green]"
        elif abs_corr > 0.3:
            strength = "[yellow]Moderate[/yellow]"
        elif abs_corr > 0.1:
            strength = "[dim]Weak[/dim]"
        else:
            strength = "[dim red]Negligible[/dim red]"

        color = "green" if corr > 0 else "red"
        table.add_row(feat, f"[{color}]{corr:+.4f}[/{color}]", strength)

    console.print(table)
    console.print()


# ──────────────────────────────────────────────
# Section 4: Positional Value Analysis
# ──────────────────────────────────────────────
def analyze_positional_value(df: pd.DataFrame):
    """Points-per-million by position, optimal budget allocation."""
    console.rule("[bold cyan]4. Positional Value Analysis[/bold cyan]")

    # Filter to players who played (minutes > 0)
    played = df[df["minutes"] > 0].copy()
    # value is stored in tenths (e.g., 55 = £5.5m)
    played["price_m"] = played["value"] / 10.0

    if "position" not in played.columns:
        console.print("[red]No position column available[/red]")
        return

    # Season-level aggregation per player
    season_agg = (
        played.groupby(["season", "name", "position"])
        .agg(
            total_pts=("total_points", "sum"),
            total_minutes=("minutes", "sum"),
            avg_price=("price_m", "mean"),
            gws_played=("total_points", "count"),
        )
        .reset_index()
    )
    season_agg["pts_per_m"] = season_agg["total_pts"] / season_agg["avg_price"]
    season_agg["pts_per_90"] = (season_agg["total_pts"] / (season_agg["total_minutes"] / 90)).replace(
        [np.inf, -np.inf], np.nan
    )

    # Position-level summary
    table = Table(
        title="Points per £M by Position (Season Totals, Players with ≥10 GWs)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Position", style="cyan")
    table.add_column("Avg Pts/£M", justify="right")
    table.add_column("Median Pts/£M", justify="right")
    table.add_column("Avg Season Pts", justify="right")
    table.add_column("Avg Price", justify="right")
    table.add_column("Player Count", justify="right")

    regulars = season_agg[season_agg["gws_played"] >= 10]

    for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
        pos_data = regulars[regulars["position"] == pos]
        if len(pos_data) == 0:
            continue
        table.add_row(
            pos,
            f"{pos_data['pts_per_m'].mean():.1f}",
            f"{pos_data['pts_per_m'].median():.1f}",
            f"{pos_data['total_pts'].mean():.0f}",
            f"£{pos_data['avg_price'].mean():.1f}m",
            str(len(pos_data)),
        )

    console.print(table)

    # Price band analysis
    console.print("\n[bold]Points per £M by Price Band (regulars, ≥10 GWs):[/bold]")
    price_bands = [
        (3.5, 5.0),
        (5.0, 6.5),
        (6.5, 8.0),
        (8.0, 10.0),
        (10.0, 13.0),
        (13.0, 16.0),
    ]

    table2 = Table(title="Value by Price Band", box=box.SIMPLE)
    table2.add_column("Price Band", style="cyan")
    table2.add_column("Avg Pts/Season", justify="right")
    table2.add_column("Avg Pts/£M", justify="right")
    table2.add_column("Players", justify="right")

    for low, high in price_bands:
        band = regulars[(regulars["avg_price"] >= low) & (regulars["avg_price"] < high)]
        if len(band) > 0:
            table2.add_row(
                f"£{low:.1f}m – £{high:.1f}m",
                f"{band['total_pts'].mean():.0f}",
                f"{band['pts_per_m'].mean():.1f}",
                str(len(band)),
            )

    console.print(table2)
    console.print()


# ──────────────────────────────────────────────
# Section 5: Form Analysis
# ──────────────────────────────────────────────
def analyze_form(df: pd.DataFrame):
    """Does recent form predict next-GW points? What window is optimal?"""
    console.rule("[bold cyan]5. Form Window Analysis[/bold cyan]")

    # Only players who played
    played = df[df["minutes"] > 0].copy()
    played = played.sort_values(["name", "season", "GW"])

    # Calculate rolling averages for different windows
    windows = [3, 5, 7, 10]
    correlations = {}

    for window in windows:
        # Rolling mean of points for each player within a season
        w = window  # capture for lambda
        played[f"form_{window}"] = played.groupby(["name", "season"])["total_points"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).mean().shift(1)
        )
        # Correlation of form_N with actual next GW points
        valid = played[["total_points", f"form_{window}"]].dropna()
        if len(valid) > 100:
            correlations[f"Last {window} GWs"] = valid["total_points"].corr(valid[f"form_{window}"])

    # Also compare season average up to that point
    played["season_avg"] = played.groupby(["name", "season"])["total_points"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    valid = played[["total_points", "season_avg"]].dropna()
    if len(valid) > 100:
        correlations["Season Avg (expanding)"] = valid["total_points"].corr(valid["season_avg"])

    # Also check xP from FPL
    if "xP" in played.columns:
        valid = played[["total_points", "xP"]].dropna()
        valid["xP"] = pd.to_numeric(valid["xP"], errors="coerce")
        valid = valid.dropna()
        if len(valid) > 100:
            correlations["FPL's xP"] = valid["total_points"].corr(valid["xP"])

    table = Table(
        title="Form Predictiveness: Correlation with Next-GW Points",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Predictor", style="cyan")
    table.add_column("Correlation", justify="right")

    for name, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        color = "green" if abs(corr) > 0.15 else "yellow"
        table.add_row(name, f"[{color}]{corr:+.4f}[/{color}]")

    console.print(table)
    console.print(
        "\n[dim]Note: These are all weak-to-moderate correlations, which is expected "
        "in football — high variance is inherent. Even 0.15+ is useful for our model.[/dim]\n"
    )


# ──────────────────────────────────────────────
# Section 6: Home/Away & FDR Analysis
# ──────────────────────────────────────────────
def analyze_home_away_fdr(df: pd.DataFrame):
    """Does home advantage and fixture difficulty actually matter?"""
    console.rule("[bold cyan]6. Home/Away & Fixture Difficulty Analysis[/bold cyan]")

    played = df[df["minutes"] > 0].copy()

    # Normalize was_home to boolean
    played["is_home"] = played["was_home"].apply(lambda x: x if isinstance(x, bool) else str(x).lower() == "true")

    # Home vs Away by position
    table = Table(title="Home vs Away: Avg Points by Position", box=box.SIMPLE_HEAVY)
    table.add_column("Position", style="cyan")
    table.add_column("Home Avg", justify="right")
    table.add_column("Away Avg", justify="right")
    table.add_column("Home Advantage", justify="right")
    table.add_column("N (Home)", justify="right")
    table.add_column("N (Away)", justify="right")

    if "position" in played.columns:
        for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
            pos_data = played[played["position"] == pos]
            if len(pos_data) == 0:
                continue
            home_avg = pos_data[pos_data["is_home"]]["total_points"].mean()
            away_avg = pos_data[~pos_data["is_home"]]["total_points"].mean()
            n_home = pos_data["is_home"].sum()
            n_away = (~pos_data["is_home"]).sum()
            advantage = home_avg - away_avg
            color = "green" if advantage > 0 else "red"
            table.add_row(
                pos,
                f"{home_avg:.2f}",
                f"{away_avg:.2f}",
                f"[{color}]{advantage:+.2f}[/{color}]",
                f"{n_home:,}",
                f"{n_away:,}",
            )

    console.print(table)

    # FDR analysis — load fixtures from recent seasons with FDR data
    console.print("\n[bold]Fixture Difficulty Rating (FDR) Impact:[/bold]")

    fdr_data = []
    for season in RECENT_SEASONS:
        fixtures = load_season_fixtures(season)
        teams = load_season_teams(season)
        if fixtures is None or teams is None:
            continue

        # Check if FDR columns exist
        if "team_h_difficulty" not in fixtures.columns:
            continue

        # Build a mapping: for each fixture, extract team + FDR
        for _, fix in fixtures.iterrows():
            if pd.isna(fix.get("team_h_score")):
                continue  # unplayed fixture
            event = fix.get("event")
            if pd.isna(event):
                continue

            # Home team faces team_h_difficulty
            fdr_data.append(
                {
                    "season": season,
                    "GW": int(event),
                    "team_id": fix["team_h"],
                    "fdr": fix["team_h_difficulty"],
                    "is_home": True,
                }
            )
            # Away team faces team_a_difficulty
            fdr_data.append(
                {
                    "season": season,
                    "GW": int(event),
                    "team_id": fix["team_a"],
                    "fdr": fix["team_a_difficulty"],
                    "is_home": False,
                }
            )

    if fdr_data:
        fdr_df = pd.DataFrame(fdr_data)

        # Merge FDR with player data
        # We need to map player's team number to fixture team_id
        if "opponent_team" in played.columns:
            # For each player-GW, we need their team, not opponent
            # In the GW data, "team" is the player's team name, not ID
            # We'll use a simpler approach: group by FDR levels
            # Join on season + GW + opponent_team (opposing team ID)
            # Actually, let's just look at average points conceded BY FDR level
            console.print("  Analyzing points scored against teams of each FDR level...\n")

            # Merge: player's opponent_team = fixture's team_id, player's GW = fixture's GW
            merged_fdr = played.merge(
                fdr_df,
                left_on=["season", "GW", "opponent_team"],
                right_on=["season", "GW", "team_id"],
                how="inner",
                suffixes=("", "_fdr"),
            )

            if len(merged_fdr) > 0:
                table2 = Table(
                    title="Avg Points by Opponent FDR (higher FDR = harder fixture)",
                    box=box.SIMPLE,
                )
                table2.add_column("FDR", style="cyan", justify="center")
                table2.add_column("Avg Points", justify="right")
                table2.add_column("Avg Goals", justify="right")
                table2.add_column("Avg Assists", justify="right")
                table2.add_column("Matches", justify="right")

                for fdr_val in sorted(merged_fdr["fdr"].unique()):
                    fdr_slice = merged_fdr[merged_fdr["fdr"] == fdr_val]
                    table2.add_row(
                        str(int(fdr_val)),
                        f"{fdr_slice['total_points'].mean():.2f}",
                        f"{fdr_slice['goals_scored'].mean():.3f}",
                        f"{fdr_slice['assists'].mean():.3f}",
                        f"{len(fdr_slice):,}",
                    )

                console.print(table2)
    else:
        console.print("  [yellow]No FDR data available in fixtures[/yellow]")

    console.print()


# ──────────────────────────────────────────────
# Section 7: Player Consistency & Reliability
# ──────────────────────────────────────────────
def analyze_consistency(df: pd.DataFrame):
    """Identify consistent vs boom-or-bust players."""
    console.rule("[bold cyan]7. Player Consistency Analysis[/bold cyan]")

    played = df[df["minutes"] > 0].copy()

    # Per-player-season stats
    player_seasons = (
        played.groupby(["season", "name", "position"])
        .agg(
            total_pts=("total_points", "sum"),
            mean_pts=("total_points", "mean"),
            std_pts=("total_points", "std"),
            gws=("total_points", "count"),
            blanks=("total_points", lambda x: (x <= 2).sum()),
            hauls=("total_points", lambda x: (x >= 10).sum()),
            avg_minutes=("minutes", "mean"),
        )
        .reset_index()
    )
    player_seasons["cv"] = player_seasons["std_pts"] / player_seasons["mean_pts"]
    player_seasons["blank_pct"] = player_seasons["blanks"] / player_seasons["gws"] * 100
    player_seasons["haul_pct"] = player_seasons["hauls"] / player_seasons["gws"] * 100

    # Filter to regulars (≥20 GWs played)
    regulars = player_seasons[player_seasons["gws"] >= 20].copy()

    # Overall consistency stats
    console.print("[bold]Consistency Stats (Players with ≥20 GWs in a season):[/bold]\n")

    table = Table(title="Consistency by Position", box=box.SIMPLE_HEAVY)
    table.add_column("Position", style="cyan")
    table.add_column("Avg Mean Pts/GW", justify="right")
    table.add_column("Avg Std Dev", justify="right")
    table.add_column("Avg CV", justify="right")
    table.add_column("Blank Rate", justify="right")
    table.add_column("Haul Rate", justify="right")

    for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
        pos_data = regulars[regulars["position"] == pos]
        if len(pos_data) == 0:
            continue
        table.add_row(
            pos,
            f"{pos_data['mean_pts'].mean():.2f}",
            f"{pos_data['std_pts'].mean():.2f}",
            f"{pos_data['cv'].mean():.2f}",
            f"{pos_data['blank_pct'].mean():.1f}%",
            f"{pos_data['haul_pct'].mean():.1f}%",
        )

    console.print(table)

    # Top most consistent players (low CV, high total)
    top_consistent = regulars.nlargest(15, "total_pts").sort_values("cv")

    console.print("\n[bold]Most Consistent Among Top Scorers (≥20 GWs, Top 15 by total pts):[/bold]")
    table2 = Table(box=box.SIMPLE)
    table2.add_column("Player", style="cyan")
    table2.add_column("Season")
    table2.add_column("Pos")
    table2.add_column("Total Pts", justify="right")
    table2.add_column("Mean/GW", justify="right")
    table2.add_column("Std Dev", justify="right")
    table2.add_column("CV", justify="right")
    table2.add_column("Blanks%", justify="right")
    table2.add_column("Hauls%", justify="right")

    for _, row in top_consistent.iterrows():
        table2.add_row(
            row["name"],
            row["season"],
            row["position"],
            str(int(row["total_pts"])),
            f"{row['mean_pts']:.1f}",
            f"{row['std_pts']:.1f}",
            f"{row['cv']:.2f}",
            f"{row['blank_pct']:.0f}%",
            f"{row['haul_pct']:.0f}%",
        )

    console.print(table2)
    console.print()


# ──────────────────────────────────────────────
# Section 8: Transfer & Price Patterns
# ──────────────────────────────────────────────
def analyze_transfers(df: pd.DataFrame):
    """How do prices change? Transfer volume patterns."""
    console.rule("[bold cyan]8. Transfer & Pricing Patterns[/bold cyan]")

    played = df.copy()
    played["price_m"] = played["value"] / 10.0

    # Price volatility per season
    console.print("[bold]Price Change Distribution per Season:[/bold]\n")

    for season in RECENT_SEASONS:
        season_data = played[played["season"] == season]
        if len(season_data) == 0:
            continue

        # Get start and end price for each player
        player_prices = (
            season_data.sort_values("GW")
            .groupby("name")
            .agg(
                start_price=("price_m", "first"),
                end_price=("price_m", "last"),
                gws=("GW", "nunique"),
            )
            .reset_index()
        )
        player_prices = player_prices[player_prices["gws"] >= 5]
        player_prices["price_change"] = player_prices["end_price"] - player_prices["start_price"]

        risers = (player_prices["price_change"] > 0).sum()
        fallers = (player_prices["price_change"] < 0).sum()
        stable = (player_prices["price_change"] == 0).sum()
        max_rise = player_prices["price_change"].max()
        max_fall = player_prices["price_change"].min()

        console.print(
            f"  {season}: {risers} risers, {fallers} fallers, {stable} stable | "
            f"Max rise: [green]+£{max_rise:.1f}m[/green], "
            f"Max fall: [red]£{max_fall:.1f}m[/red]"
        )

    console.print()


# ──────────────────────────────────────────────
# Section 9: Captaincy & Ownership Analysis
# ──────────────────────────────────────────────
def analyze_ownership(df: pd.DataFrame):
    """Do highly-owned players deliver? Differential value."""
    console.rule("[bold cyan]9. Ownership vs Performance[/bold cyan]")

    played = df[df["minutes"] > 0].copy()

    # Ownership bands
    if "selected" in played.columns:
        played["selected_num"] = pd.to_numeric(played["selected"], errors="coerce")
        valid = played.dropna(subset=["selected_num"])

        if len(valid) > 0:
            # Normalize selected within each GW (it's absolute count, so varies by season)
            valid["ownership_pct"] = valid.groupby(["season", "GW"])["selected_num"].transform(
                lambda x: x / x.max() * 100 if x.max() > 0 else 0
            )

            ownership_bands = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]

            table = Table(
                title="Average Points by Ownership Band (Normalized)",
                box=box.SIMPLE_HEAVY,
            )
            table.add_column("Ownership %ile", style="cyan")
            table.add_column("Avg Points", justify="right")
            table.add_column("Records", justify="right")

            for low, high in ownership_bands:
                band = valid[(valid["ownership_pct"] >= low) & (valid["ownership_pct"] < high)]
                if len(band) > 0:
                    table.add_row(
                        f"{low}-{high}%",
                        f"{band['total_points'].mean():.2f}",
                        f"{len(band):,}",
                    )

            console.print(table)

    console.print()


# ──────────────────────────────────────────────
# Section 10: Defensive Contribution (2025-26)
# ──────────────────────────────────────────────
def analyze_defensive_contribution(df: pd.DataFrame):
    """New for 2025-26: CBIT/CBIRT scoring impact."""
    console.rule("[bold cyan]10. Defensive Contribution Analysis (2025-26)[/bold cyan]")

    latest = df[df["season"] == "2025-26"].copy()

    if "defensive_contribution" not in latest.columns:
        console.print("[yellow]No defensive_contribution column found[/yellow]")
        return

    played = latest[latest["minutes"] > 0]
    has_dc = played[played["defensive_contribution"] > 0]

    console.print("[bold]Players earning defensive contribution bonus:[/bold]")
    console.print(
        f"  Total GW entries with DC > 0: {len(has_dc):,} / {len(played):,} ({len(has_dc) / len(played) * 100:.1f}%)"
    )

    if "position" in played.columns and len(has_dc) > 0:
        table = Table(
            title="Defensive Contribution by Position",
            box=box.SIMPLE,
        )
        table.add_column("Position", style="cyan")
        table.add_column("DC > 0 Rate", justify="right")
        table.add_column("Avg DC (when > 0)", justify="right")
        table.add_column("Avg Extra Pts", justify="right")

        for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
            pos_played = played[played["position"] == pos]
            pos_dc = has_dc[has_dc["position"] == pos]
            if len(pos_played) == 0:
                continue
            rate = len(pos_dc) / len(pos_played) * 100
            avg_dc = pos_dc["defensive_contribution"].mean() if len(pos_dc) > 0 else 0
            table.add_row(
                pos,
                f"{rate:.1f}%",
                f"{avg_dc:.1f}",
                f"{avg_dc * 2:.1f}",  # 2 pts per DC bonus
            )

        console.print(table)

    # Top DC earners
    dc_leaders = (
        played.groupby(["name", "position"])
        .agg(
            total_dc=("defensive_contribution", "sum"),
            gws=("defensive_contribution", "count"),
            total_pts=("total_points", "sum"),
        )
        .reset_index()
        .nlargest(15, "total_dc")
    )

    if len(dc_leaders) > 0:
        console.print("\n[bold]Top 15 Defensive Contribution Earners (2025-26):[/bold]")
        table2 = Table(box=box.SIMPLE)
        table2.add_column("Player", style="cyan")
        table2.add_column("Position")
        table2.add_column("Total DC", justify="right")
        table2.add_column("DC/GW", justify="right")
        table2.add_column("Total Pts", justify="right")

        for _, row in dc_leaders.iterrows():
            table2.add_row(
                row["name"],
                row["position"],
                str(int(row["total_dc"])),
                f"{row['total_dc'] / row['gws']:.1f}",
                str(int(row["total_pts"])),
            )

        console.print(table2)

    console.print()


# ──────────────────────────────────────────────
# Section 11: Key Findings Summary
# ──────────────────────────────────────────────
def summarize_findings(df: pd.DataFrame):
    """Produce a summary of key findings."""
    console.rule("[bold cyan]11. Key Findings Summary[/bold cyan]")

    played = df[df["minutes"] > 0]

    findings = []

    # 1. Points distribution insight
    pts = played["total_points"]
    zero_pct = (pts == 0).sum() / len(pts) * 100
    findings.append(
        f"• {zero_pct:.0f}% of all player-GW entries score 0 points — "
        f"predicting who gets minutes is as important as predicting who scores."
    )

    # 2. Position value
    if "position" in played.columns:
        pos_means = played.groupby("position")["total_points"].mean()
        if len(pos_means) > 0:
            best_pos = pos_means.idxmax()
            findings.append(
                f"• {best_pos} position has the highest avg points/GW ({pos_means.max():.2f}) — "
                f"but this doesn't account for price."
            )

    # 3. Minutes threshold
    regular_starters = played[played["minutes"] >= 60]
    sub_appearances = played[(played["minutes"] > 0) & (played["minutes"] < 60)]
    findings.append(
        f"• Players with 60+ mins avg {regular_starters['total_points'].mean():.2f} pts/GW "
        f"vs {sub_appearances['total_points'].mean():.2f} for sub appearances — "
        f"identifying nailed starters is critical."
    )

    # 4. Bonus points impact
    if "bonus" in played.columns:
        bonus_corr = played["total_points"].corr(played["bonus"])
        findings.append(
            f"• Bonus points correlation with total_points: {bonus_corr:.3f} — "
            f"BPS-heavy players provide stable returns."
        )

    for finding in findings:
        console.print(finding)

    console.print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():  # noqa: D103
    """Run all EDA analyses on the historical FPL dataset."""
    console.print(
        Panel(
            "[bold]FPL Historical Data — Exploratory Data Analysis[/bold]\n"
            f"Data source: {DATA_ROOT}\n"
            f"Seasons: {', '.join(RECENT_SEASONS)}",
            title="FPL EDA",
            border_style="cyan",
        )
    )

    # 1. Schema analysis
    analyze_schemas()

    # 2. Load combined data
    console.rule("[bold cyan]Loading Data[/bold cyan]")
    df = load_all_seasons()

    # Ensure numeric types for key columns
    numeric_cols = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "bps",
        "creativity",
        "influence",
        "threat",
        "ict_index",
        "value",
        "selected",
        "goals_conceded",
        "saves",
        "own_goals",
        "penalties_missed",
        "penalties_saved",
        "yellow_cards",
        "red_cards",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Run all analyses
    profile_data(df)
    analyze_correlations(df)
    analyze_positional_value(df)
    analyze_form(df)
    analyze_home_away_fdr(df)
    analyze_consistency(df)
    analyze_transfers(df)
    analyze_ownership(df)
    analyze_defensive_contribution(df)
    summarize_findings(df)

    console.print(
        Panel(
            "[bold green]EDA Complete![/bold green]\nReview findings above to inform the expected points model design.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
