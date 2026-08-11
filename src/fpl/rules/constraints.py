"""
FPL Rules & Constraints Engine.

Encodes all official Fantasy Premier League (2026-27) rules:
- Squad budget & composition limits
- Starting XI formation constraints
- Transfer costs and chip rules
"""

# Official 2026-27 Budget & Squad Limits
TOTAL_BUDGET: float = 100.0  # £100.0m
SQUAD_SIZE: int = 15
MAX_PER_TEAM: int = 3

# Positional Requirements for 15-man Squad
POSITION_LIMITS: dict[str, int] = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

# Element type IDs in official API mapping
ELEMENT_TYPE_MAP: dict[int, str] = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

POSITION_NAME_TO_TYPE: dict[str, int] = {v: k for k, v in ELEMENT_TYPE_MAP.items()}

# Formation Constraints for 11-man Starting XI
STARTING_XI_SIZE: int = 11
STARTING_XI_CONSTRAINTS: dict[str, tuple[int, int]] = {
    "GKP": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

# Transfer & Scoring Rules
FREE_TRANSFERS_PER_GW: int = 1
MAX_BANKED_TRANSFERS: int = 5
HIT_COST_PER_EXTRA_TRANSFER: int = 4

# Scoring Matrix (points earned per action)
SCORING_RULES: dict[str, dict[str, int]] = {
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assists": {"GKP": 3, "DEF": 3, "MID": 3, "FWD": 3},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},  # per 2 goals conceded
    "penalties_saved": {"GKP": 5, "DEF": 5, "MID": 5, "FWD": 5},
    "penalties_missed": {"GKP": -2, "DEF": -2, "MID": -2, "FWD": -2},
    "yellow_cards": {"GKP": -1, "DEF": -1, "MID": -1, "FWD": -1},
    "red_cards": {"GKP": -3, "DEF": -3, "MID": -3, "FWD": -3},
    "own_goals": {"GKP": -2, "DEF": -2, "MID": -2, "FWD": -2},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},  # CBIT/CBIRT threshold
}


def validate_squad_composition(squad: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate if a 15-man squad satisfies all official FPL rules.

    Args:
        squad: List of dicts representing players with keys 'position', 'team', 'cost'.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    if len(squad) != SQUAD_SIZE:
        errors.append(f"Squad must contain exactly {SQUAD_SIZE} players (got {len(squad)}).")

    # Check total cost
    total_cost = sum(p.get("cost", 0.0) for p in squad)
    if total_cost > TOTAL_BUDGET:
        errors.append(f"Total squad cost (£{total_cost:.1f}m) exceeds budget (£{TOTAL_BUDGET:.1f}m).")

    # Check position counts
    pos_counts: dict[str, int] = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in squad:
        pos = p.get("position", "")
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            errors.append(f"Unknown position '{pos}' for player {p.get('name')}.")

    for pos, required in POSITION_LIMITS.items():
        actual = pos_counts.get(pos, 0)
        if actual != required:
            errors.append(f"Position '{pos}' requires exactly {required} players (got {actual}).")

    # Check team limits (max 3 per PL club)
    team_counts: dict[str, int] = {}
    for p in squad:
        team = str(p.get("team", ""))
        team_counts[team] = team_counts.get(team, 0) + 1

    for team, count in team_counts.items():
        if count > MAX_PER_TEAM:
            errors.append(f"Team '{team}' has {count} players selected (max allowed is {MAX_PER_TEAM}).")

    return len(errors) == 0, errors


def validate_starting_xi(xi: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate if an 11-man starting lineup is a valid FPL formation.

    Args:
        xi: List of 11 player dicts with 'position'.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    if len(xi) != STARTING_XI_SIZE:
        errors.append(f"Starting XI must contain exactly {STARTING_XI_SIZE} players (got {len(xi)}).")

    pos_counts: dict[str, int] = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in xi:
        pos = p.get("position", "")
        if pos in pos_counts:
            pos_counts[pos] += 1

    for pos, (min_req, max_req) in STARTING_XI_CONSTRAINTS.items():
        actual = pos_counts.get(pos, 0)
        if actual < min_req or actual > max_req:
            errors.append(f"Starting XI '{pos}' count ({actual}) is out of valid range [{min_req}, {max_req}].")

    return len(errors) == 0, errors
