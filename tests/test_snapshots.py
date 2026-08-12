import pandas as pd
import pytest

from fpl.modeling.snapshot_join import join_snapshot_outcomes
from fpl.modeling.snapshots import snapshot_from_bootstrap, validate_snapshot


def test_snapshot_from_bootstrap_normalizes_player_pool():
    snapshot = snapshot_from_bootstrap(
        {
            "teams": [{"id": 1, "name": "Arsenal"}],
            "elements": [{
                "id": 10,
                "web_name": "Player",
                "element_type": 3,
                "team": 1,
                "now_cost": 75,
                "status": "a",
                "chance_of_playing_next_round": None,
                "can_select": True,
                "can_transact": True,
            }],
        },
        season="2026-27",
        gameweek=1,
        snapshot_time="2026-08-12T10:00:00Z",
    )
    assert snapshot.loc[0, "team"] == "Arsenal"
    assert snapshot.loc[0, "cost"] == 7.5
    assert snapshot.loc[0, "chance_of_playing"] == 100.0


def test_snapshot_validation_rejects_duplicates_and_invalid_costs():
    frame = pd.DataFrame([{
        "season": "s", "gameweek": 1, "snapshot_time": "t", "player_id": 1,
        "web_name": "p", "position": 3, "team_id": 1, "team": "a", "cost": 2.0,
        "status": "a", "chance_of_playing": 100, "can_select": True, "can_transact": True,
    }])
    with pytest.raises(ValueError, match="cost"):
        validate_snapshot(frame)


def test_snapshot_join_keeps_players_without_outcome_rows():
    snapshot = snapshot_from_bootstrap(
        {"teams": [{"id": 1, "name": "Arsenal"}], "elements": [
            {"id": 1, "web_name": "A", "element_type": 3, "team": 1, "now_cost": 75,
             "status": "a", "can_select": True, "can_transact": True},
            {"id": 2, "web_name": "B", "element_type": 3, "team": 1, "now_cost": 75,
             "status": "a", "can_select": True, "can_transact": True},
        ]},
        season="s", gameweek=1, snapshot_time="t",
    )
    outcomes = pd.DataFrame([{"season": "s", "gameweek": 1, "player_id": 1, "total_points": 6, "minutes": 90}])
    joined = join_snapshot_outcomes(snapshot, outcomes)
    assert joined["player_id"].tolist() == [1, 2]
    assert joined.loc[joined["player_id"] == 2, "target_points"].item() == 0.0
