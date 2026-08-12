import pandas as pd

from fpl.optimizer.replay import replay_projection_policies


def _pool() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pid = 1
    for position, count, cost in [("GKP", 3, 4.5), ("DEF", 6, 5.0), ("MID", 6, 6.5), ("FWD", 5, 7.5)]:
        for index in range(count):
            rows.append({
                "id": pid,
                "web_name": f"{position}_{index}",
                "position": position,
                "team": f"Team_{index % 5}",
                "cost": cost,
                "minutes": 900,
                "total_points": 50,
                "points_per_game": 4.0,
                "heuristic_xp": float(index + 1),
                "blended_xp": float(count - index),
                "actual_points": float(index % 3),
            })
            pid += 1
    return pd.DataFrame(rows)


def test_replay_compares_two_projection_policies():
    result = replay_projection_policies([_pool(), _pool()])

    assert result.heuristic.gameweeks == 2
    assert result.blended.gameweeks == 2
    assert result.heuristic.realized_points >= 0.0
    assert result.blended.average_captain_points >= 0.0


def test_replay_requires_projection_and_actual_columns():
    frame = _pool().drop(columns="blended_xp")
    try:
        replay_projection_policies([frame])
    except ValueError as error:
        assert "blended_xp" in str(error)
    else:
        raise AssertionError("Expected missing replay column to fail")
