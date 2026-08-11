"""
Strongly Typed Player & Fixture Models for FPL Recommendation Engine.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Position(StrEnum):
    """Player position enumeration."""

    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

    @classmethod
    def from_element_type(cls, element_type: int) -> "Position":
        """Convert numerical FPL element type to Position Enum."""
        mapping = {1: cls.GKP, 2: cls.DEF, 3: cls.MID, 4: cls.FWD}
        return mapping.get(element_type, cls.MID)


class FixtureContext(BaseModel):
    """Upcoming fixture context for a player."""

    model_config = ConfigDict(frozen=True)

    event_id: int = Field(..., description="Gameweek Event ID")
    opponent_team_id: int = Field(..., description="Opponent Team ID")
    opponent_short_name: str = Field(default="OPP", description="Opponent short name (e.g. ARS)")
    fdr: int = Field(default=3, ge=1, le=5, description="Fixture Difficulty Rating 1-5")
    is_home: bool = Field(default=True, description="True if home fixture")


class PlayerStats(BaseModel):
    """Strongly typed player metrics container."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="Unique FPL Player ID")
    web_name: str = Field(..., description="Display name on FPL website")
    first_name: str = Field(default="", description="Player first name")
    second_name: str = Field(default="", description="Player last name")
    position: Position = Field(..., description="Playing position (GKP, DEF, MID, FWD)")
    team: str = Field(..., description="Club name (e.g. Arsenal)")
    team_code: int = Field(default=0, description="Club numerical code")
    cost: float = Field(..., ge=3.5, le=16.0, description="Current price in £m (e.g. 5.5)")
    ep_next: float = Field(default=0.0, ge=0.0, description="FPL official next GW expected points")
    form: float = Field(default=0.0, ge=0.0, description="Recent form rating")
    points_per_game: float = Field(default=0.0, ge=0.0, description="Season points per game average")
    total_points: int = Field(default=0, ge=0, description="Total season points accumulated")
    minutes: int = Field(default=0, ge=0, description="Total minutes played this season")
    goals_scored: int = Field(default=0, ge=0, description="Goals scored this season")
    assists: int = Field(default=0, ge=0, description="Assists made this season")
    clean_sheets: int = Field(default=0, ge=0, description="Clean sheets earned this season")
    selected_by_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Ownership percentage")
    defensive_contribution_per_90: float = Field(
        default=0.0, ge=0.0, description="Average CBIT/CBIRT contributions per 90 mins"
    )
    expected_goals_per_90: float = Field(default=0.0, ge=0.0, description="Underlying xG per 90 mins")
    expected_assists_per_90: float = Field(default=0.0, ge=0.0, description="Underlying xA per 90 mins")
    ict_index: float = Field(default=0.0, ge=0.0, description="ICT Index score")


class PlayerProjection(BaseModel):
    """Detailed mathematical breakdown of calculated Expected Points (xP)."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    web_name: str
    position: Position
    team: str
    cost: float
    expected_minutes: float = Field(..., ge=0.0, le=90.0, description="Projected minutes to play")
    appearance_xp: float = Field(..., ge=0.0, description="Base points for playing (1pt <60m, 2pts >=60m)")
    attack_xp: float = Field(..., ge=0.0, description="xP from expected goals and assists")
    clean_sheet_xp: float = Field(..., ge=0.0, description="xP from clean sheets")
    defensive_contribution_xp: float = Field(..., ge=0.0, description="xP from 2025-26 CBIT/CBIRT threshold")
    bonus_xp: float = Field(..., ge=0.0, description="Expected bonus points")
    fixture_multiplier: float = Field(..., ge=0.5, le=1.5, description="FDR & Home/Away composite multiplier")
    total_xp: float = Field(..., ge=0.0, description="Final combined Expected Points projection")
