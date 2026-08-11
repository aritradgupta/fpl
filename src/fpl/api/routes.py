"""
FastAPI Routes for FPL Team Creator Application.
"""

from fastapi import APIRouter, HTTPException, Query, status

from fpl.api.schemas import (
    SquadRecommendRequest,
    SquadRecommendResponse,
    SyncResponse,
    TransferRequest,
    TransferResponse,
)
from fpl.client import FPLClient
from fpl.data import load_players_df_from_db, sync_bootstrap_to_db
from fpl.models.squad import ChipType
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


async def _process_squad_recommendation(budget: float, club_limit: int, chip: ChipType):
    """Shared handler for squad optimization."""
    df = await load_players_df_from_db()
    if df.empty:
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        df = await load_players_df_from_db()

    try:
        rec = optimize_squad(df, budget=budget, club_limit=club_limit, chip=chip)
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
):
    """
    Recommend 15-man squad via GET request using URL query parameters.
    """
    return await _process_squad_recommendation(budget=budget, club_limit=club_limit, chip=chip)


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
    return await _process_squad_recommendation(budget=req.budget, club_limit=req.club_limit, chip=req.chip)


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
        )
        return transfer_rec.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transfer optimization failed: {str(e)}",
        ) from e
