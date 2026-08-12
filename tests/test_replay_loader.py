from pathlib import Path

from fpl.optimizer.replay import build_replay_frames_from_gw_history


def test_replay_loader_uses_point_in_time_metadata():
    season = Path("data/historical/data/2024-25")
    frames = build_replay_frames_from_gw_history(season)

    assert frames
    frame = frames[0]
    assert {"id", "position", "team", "cost", "actual_points", "heuristic_xp", "blended_xp"}.issubset(frame)
    assert frame["id"].is_unique
    assert frame["team"].notna().all()
    assert (frame["cost"] > 0).all()
