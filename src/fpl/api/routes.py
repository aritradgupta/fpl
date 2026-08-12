"""
FastAPI Routes for FPL Team Creator Application.
"""

import asyncio
from contextlib import suppress
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from fpl.api.lab_html import SOLVER_LAB_HTML
from fpl.api.schemas import (
    SolverLabPlayer,
    SolverLabRequest,
    SolverLabResponse,
    SolverLabResult,
    SquadRecommendRequest,
    SquadRecommendResponse,
    SyncResponse,
    TransferRequest,
    TransferResponse,
)
from fpl.client import FPLClient
from fpl.data import load_players_df_from_db, sync_bootstrap_to_db
from fpl.models.squad import ChipType, SolverType
from fpl.optimizer import (
    enrich_df_with_xp,
    optimize_squad,
    optimize_transfers,
    player_stats_from_series,
)

router = APIRouter()
client = FPLClient()


@router.get("/health", summary="Health Check")
async def health_check():
    """Verify application status."""
    return {"status": "ok", "app": "FPL Team Creator Recommendation System"}


@router.get("/lab", response_class=HTMLResponse, include_in_schema=False)
async def solver_lab() -> HTMLResponse:
    """Serve the interactive solver comparison page."""
    return HTMLResponse(SOLVER_LAB_HTML)


@router.get("/lab/players", include_in_schema=False)
async def solver_lab_players() -> list[dict[str, Any]]:
    """Return compact player options for the Solver Lab selectors."""
    df = await load_players_df_from_db()
    if df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        df = await load_players_df_from_db()
    columns = ["id", "web_name", "position", "team", "cost"]
    records = df[columns].sort_values(["position", "web_name"]).to_dict("records")
    return [{str(key): value for key, value in record.items()} for record in records]


@router.post("/sync", response_model=SyncResponse, summary="Sync Live FPL API Data")
async def sync_data():
    """Trigger live data fetch from official FPL API and cache locally in SQLite."""
    try:
        data = await client.get_bootstrap_static()
        count = await sync_bootstrap_to_db(data)
        return SyncResponse(status="success", total_players_synced=count)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync with FPL API: {str(e)}",
        ) from e


@router.get("/players", summary="List & Filter Players")
async def list_players(
    position: str | None = Query(None, description="Filter by position (GKP, DEF, MID, FWD)"),
    team: str | None = Query(None, description="Filter by team name"),
    max_cost: float | None = Query(None, description="Max cost limit in £m"),
):
    """Retrieve player dataset with optional positional and pricing filters."""
    df = await load_players_df_from_db()
    if df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        df = await load_players_df_from_db()

    df = enrich_df_with_xp(df)

    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if team:
        df = df[df["team"].str.lower() == team.lower()]
    if max_cost:
        df = df[df["cost"] <= max_cost]

    return df.to_dict("records")


async def _process_squad_recommendation(
    budget: float,
    club_limit: int,
    chip: ChipType,
    solver_type: SolverType,
    gameweek: int = 1,
    horizon_weeks: int = 3,
):
    """Shared handler for squad optimization."""
    df = await load_players_df_from_db()
    if df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        df = await load_players_df_from_db()

    fixtures_df = None
    with suppress(Exception):
        fixtures_df = pd.DataFrame(await client.get_fixtures())
        if fixtures_df.empty:
            fixtures_df = None

    try:
        rec = optimize_squad(
            df,
            budget=budget,
            club_limit=club_limit,
            chip=chip,
            solver_type=solver_type,
            horizon_weeks=horizon_weeks,
            gameweek=gameweek,
            fixtures_df=fixtures_df,
        )
        return rec.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Optimization failed: {str(e)}",
        ) from e


@router.get(
    "/recommend/squad",
    response_model=SquadRecommendResponse,
    summary="Recommend Optimal 15-Man Squad (GET)",
    operation_id="recommend_squad_get",
)
async def recommend_squad_get(
    budget: float = Query(100.0, ge=50.0, le=120.0, description="Squad budget limit in £m"),
    club_limit: int = Query(3, ge=1, le=5, description="Max players allowed per PL club"),
    chip: ChipType = Query(ChipType.NONE, description="Active chip (none, wildcard, freehit, bboost, 3xc)"),
    solver_type: SolverType = Query(SolverType.SINGLE_PERIOD, description="Solver strategy model"),
    gameweek: int = Query(1, ge=1, le=50, description="First gameweek to project"),
    horizon_weeks: int = Query(3, ge=1, le=10, description="Projection horizon"),
):
    """
    Recommend 15-man squad via GET request using URL query parameters.
    """
    return await _process_squad_recommendation(
        budget=budget,
        club_limit=club_limit,
        chip=chip,
        solver_type=solver_type,
        gameweek=gameweek,
        horizon_weeks=horizon_weeks,
    )


@router.post(
    "/recommend/squad",
    response_model=SquadRecommendResponse,
    summary="Recommend Optimal 15-Man Squad (POST)",
    operation_id="recommend_squad_post",
)
async def recommend_squad_post(req: SquadRecommendRequest):
    """
    Recommend 15-man squad via POST request using JSON body payload.
    """
    return await _process_squad_recommendation(
        budget=req.budget,
        club_limit=req.club_limit,
        chip=req.chip,
        solver_type=req.solver_type,
        gameweek=req.gameweek,
        horizon_weeks=req.horizon_weeks,
    )


@router.post(
    "/recommend/transfers",
    response_model=TransferResponse,
    summary="Recommend Optimal Transfers & Hit Strategy",
    operation_id="recommend_transfers_post",
)
async def recommend_transfers(req: TransferRequest):
    """
    Evaluates transfer decisions for an existing squad considering free transfers,
    hit penalties (-4 pts per hit), bank budget, and chip strategy.
    """
    df = await load_players_df_from_db()
    if df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        df = await load_players_df_from_db()

    curr_squad_df = df[df["id"].isin(req.current_squad_ids)]
    if len(curr_squad_df) != 15:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Found {len(curr_squad_df)} valid players out of 15 requested current squad IDs.",
        )

    current_squad_stats = [player_stats_from_series(row) for _, row in curr_squad_df.iterrows()]
    available_stats = [player_stats_from_series(row) for _, row in df.iterrows()]

    try:
        transfer_rec = optimize_transfers(
            current_squad=current_squad_stats,
            available_players=available_stats,
            free_transfers=req.free_transfers,
            max_transfers=req.max_transfers,
            bank_budget=req.bank_budget,
            chip=req.chip,
            solver_type=req.solver_type,
        )
        return transfer_rec.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transfer optimization failed: {str(e)}",
        ) from e


async def _run_lab_case(
    solver_type: SolverType,
    req: SolverLabRequest,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None,
) -> SolverLabResult:
    """Run one CPU/GPU solver case without blocking the async event loop."""
    started = perf_counter()
    try:
        recommendation = await asyncio.to_thread(
            optimize_squad,
            players_df,
            budget=req.budget,
            club_limit=req.club_limit,
            chip=req.chip,
            solver_type=solver_type,
            horizon_weeks=req.horizon_weeks,
            risk_aversion=req.risk_aversion,
            num_scenarios=req.num_scenarios,
            use_gpu=req.use_gpu,
            generations=req.generations,
            population_size=req.population_size,
            seed=req.seed,
            lock_players=[str(player_id) for player_id in req.lock_player_ids],
            exclude_players=[str(player_id) for player_id in req.exclude_player_ids],
            fixtures_df=fixtures_df,
            gameweek=req.gameweek,
        )
        return SolverLabResult(
            solver=solver_type,
            status="ok",
            runtime_seconds=round(perf_counter() - started, 3),
            total_expected_points=recommendation.total_expected_points,
            total_cost=recommendation.total_cost,
            captain=recommendation.captain.web_name,
            squad_player_ids=sorted(
                player.projection.player_id for player in recommendation.starting_xi + recommendation.bench
            ),
            players=[
                SolverLabPlayer(
                    player_id=selected.projection.player_id,
                    name=selected.projection.web_name,
                    position=selected.projection.position,
                    team=selected.projection.team,
                    cost=selected.projection.cost,
                    role=selected.role.value,
                )
                for selected in recommendation.starting_xi + recommendation.bench
            ],
        )
    except Exception as exc:
        return SolverLabResult(
            solver=solver_type,
            status="failed",
            runtime_seconds=round(perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


@router.post("/lab/run", response_model=SolverLabResponse, summary="Compare Solver Strategies")
async def run_solver_lab(req: SolverLabRequest) -> SolverLabResponse:
    """Run selected solvers over the same cached players and fixture snapshot."""
    players_df = await load_players_df_from_db()
    if players_df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        players_df = await load_players_df_from_db()

    fixtures_df: pd.DataFrame | None = None
    with suppress(Exception):
        fixtures_df = pd.DataFrame(await client.get_fixtures())
        if fixtures_df.empty:
            fixtures_df = None

    results = await asyncio.gather(
        *(_run_lab_case(solver, req, players_df, fixtures_df) for solver in req.solver_types)
    )
    return SolverLabResponse(
        results=list(results),
        fixture_rows=0 if fixtures_df is None else len(fixtures_df),
        player_count=len(players_df),
    )
