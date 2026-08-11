"""
Pydantic API Request & Response Schemas for FPL Team Creator.
"""

from pydantic import BaseModel, Field

from fpl.models.player import Position
from fpl.models.squad import ChipType, SquadRole


class PlayerProjectionOut(BaseModel):
    """Output projection schema for individual player."""

    player_id: int
    web_name: str
    position: Position
    team: str
    cost: float
    expected_minutes: float
    appearance_xp: float
    attack_xp: float
    clean_sheet_xp: float
    defensive_contribution_xp: float
    bonus_xp: float
    fixture_multiplier: float
    total_xp: float


class SelectedPlayerOut(BaseModel):
    """Output schema for a selected squad player with role."""

    projection: PlayerProjectionOut
    role: SquadRole
    bench_order: int | None = None


class SquadRecommendRequest(BaseModel):
    """Input request schema for 15-man squad recommendation."""

    budget: float = Field(default=100.0, ge=50.0, le=120.0, description="Squad budget limit in £m")
    club_limit: int = Field(default=3, ge=1, le=5, description="Max players allowed per PL club")
    chip: ChipType = Field(default=ChipType.NONE, description="Active chip (none, wildcard, freehit, bboost, 3xc)")


class SquadRecommendResponse(BaseModel):
    """Output response schema for 15-man squad recommendation."""

    total_expected_points: float
    total_cost: float
    squad_size: int
    chip_active: ChipType
    captain: PlayerProjectionOut
    vice_captain: PlayerProjectionOut
    starting_xi: list[SelectedPlayerOut]
    bench: list[SelectedPlayerOut]


class TransferRequest(BaseModel):
    """Input request schema for transfer evaluation."""

    current_squad_ids: list[int] = Field(..., min_length=15, max_length=15, description="15 current squad player IDs")
    free_transfers: int = Field(default=1, ge=1, le=5, description="Banked free transfers")
    max_transfers: int = Field(default=2, ge=1, le=3, description="Max transfers to evaluate")
    bank_budget: float = Field(default=0.0, ge=0.0, le=20.0, description="Bank balance in £m")
    chip: ChipType = Field(default=ChipType.NONE, description="Active chip strategy")


class SingleTransferOut(BaseModel):
    """Output schema for a single transfer swap."""

    player_out: PlayerProjectionOut
    player_in: PlayerProjectionOut
    cost_difference: float
    net_xp_gain: float


class TransferResponse(BaseModel):
    """Output response schema for transfer recommendations."""

    transfers: list[SingleTransferOut]
    transfers_count: int
    free_transfers_used: int
    hits_cost: int
    gross_xp_gain: float
    net_xp_gain: float
    recommended_squad: SquadRecommendResponse


class SyncResponse(BaseModel):
    """Output response schema for FPL API data sync."""

    status: str
    total_players_synced: int
