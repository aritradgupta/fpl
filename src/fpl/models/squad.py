"""
Strongly Typed Squad & Transfer Optimization Models.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from fpl.models.player import PlayerProjection


class SquadRole(StrEnum):
    """Role assigned to a player in a recommended squad."""

    STARTER = "STARTER"
    CAPTAIN = "CAPTAIN"
    VICE_CAPTAIN = "VICE_CAPTAIN"
    BENCH = "BENCH"


class ChipType(StrEnum):
    """FPL Chip strategy choices."""

    NONE = "none"
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


class SolverType(StrEnum):
    """Available squad recommendation solver models."""

    SINGLE_PERIOD = "single_period"
    MULTI_PERIOD = "multi_period"
    STOCHASTIC = "stochastic"
    GENETIC = "genetic"


class SelectedPlayer(BaseModel):
    """A player included in a recommended squad with assigned role and projection."""

    model_config = ConfigDict(frozen=True)

    projection: PlayerProjection
    role: SquadRole
    bench_order: int | None = Field(default=None, description="1, 2, 3 or 4 if on bench")


class SquadRecommendation(BaseModel):
    """Complete 15-man squad recommendation output."""

    model_config = ConfigDict(frozen=True)

    total_expected_points: float = Field(..., description="Projected total XI points (including captain 2x/3x)")
    total_cost: float = Field(..., description="Total squad purchase cost in £m")
    squad_size: int = Field(default=15, description="Number of players in squad")
    chip_active: ChipType = Field(default=ChipType.NONE, description="Chip activated for this Gameweek")
    captain: PlayerProjection
    vice_captain: PlayerProjection
    starting_xi: list[SelectedPlayer]
    bench: list[SelectedPlayer]


class SingleTransfer(BaseModel):
    """A single transfer action (one player out, one player in)."""

    model_config = ConfigDict(frozen=True)

    player_out: PlayerProjection
    player_in: PlayerProjection
    cost_difference: float = Field(..., description="Financial impact (£m)")
    net_xp_gain: float = Field(..., description="Expected points gain for this swap")


class TransferRecommendation(BaseModel):
    """Optimal transfer decision recommendation across a gameweek horizon."""

    model_config = ConfigDict(frozen=True)

    transfers: list[SingleTransfer]
    transfers_count: int
    free_transfers_used: int
    hits_cost: int = Field(default=0, description="Point penalty (-4 per hit)")
    gross_xp_gain: float = Field(..., description="Gross expected points gain before hits")
    net_xp_gain: float = Field(..., description="Net expected points gain after hit penalty")
    recommended_squad: SquadRecommendation


class HorizonGameweekRecommendation(BaseModel):
    """Machine-readable plan for one gameweek in a multi-week strategy."""

    model_config = ConfigDict(frozen=True)

    gameweek: int
    squad_player_ids: list[int]
    starting_player_ids: list[int]
    captain_id: int
    vice_captain_id: int
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers_used: int
    hits: int
    expected_points: float
    chip: ChipType = ChipType.NONE


class HorizonRecommendation(BaseModel):
    """Complete transfer-aware multi-gameweek recommendation."""

    model_config = ConfigDict(frozen=True)

    total_expected_points: float
    total_hits: int
    initial_bank: float
    final_bank: float
    gameweeks: list[HorizonGameweekRecommendation]
