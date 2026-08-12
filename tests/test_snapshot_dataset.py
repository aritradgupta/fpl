import pandas as pd

from fpl.modeling.snapshot_dataset import build_snapshot_training_dataset
from fpl.modeling.snapshots import snapshot_from_bootstrap


def test_snapshot_training_dataset_uses_full_pool_and_prior_history():
    snapshot_rows = []
    for gameweek in (1, 2):
        snapshot = snapshot_from_bootstrap(
            {"teams": [{"id": 1, "name": "Arsenal"}], "elements": [
                {"id": 1, "web_name": "A", "element_type": 3, "team": 1, "now_cost": 75,
                 "status": "a", "can_select": True, "can_transact": True},
                {"id": 2, "web_name": "B", "element_type": 3, "team": 1, "now_cost": 75,
                 "status": "a", "can_select": True, "can_transact": True},
            ]},
            season="s", gameweek=gameweek, snapshot_time="t",
        )
        snapshot_rows.append(snapshot)
    snapshots = pd.concat(snapshot_rows, ignore_index=True)
    outcomes = pd.DataFrame([
        {"season": "s", "gameweek": 1, "player_id": 1, "total_points": 8, "minutes": 90},
        {"season": "s", "gameweek": 2, "player_id": 1, "total_points": 2, "minutes": 30},
    ])
    dataset = build_snapshot_training_dataset(snapshots, outcomes, history_windows=(2,))
    player_two_gw2 = dataset[(dataset.element == 2) & (dataset.GW == 2)].iloc[0]
    player_one_gw2 = dataset[(dataset.element == 1) & (dataset.GW == 2)].iloc[0]
    assert len(dataset) == 4
    assert player_two_gw2.target_points == 0
    assert player_two_gw2.total_points_lag1 == 0
    assert player_one_gw2.total_points_lag1 == 8
