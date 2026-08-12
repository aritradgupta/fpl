"""Run chronological multi-season solver replay."""

import pandas as pd

from fpl.modeling.features import build_next_gameweek_dataset
from fpl.modeling.models import BoostedTreePredictor
from fpl.optimizer.replay import (
    build_replay_frames_from_gw_history,
    replay_projection_policies,
)

SEASONS = ["2016-17", "2017-18", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def main() -> None:
    raw = pd.read_csv("data/historical/data/cleaned_merged_seasons.csv", low_memory=False)
    available = [season for season in SEASONS if season in set(raw["season_x"].astype(str))]
    for index, target_season in enumerate(available[2:], start=2):
        train_seasons = available[:index]
        train = raw[raw["season_x"].isin(train_seasons)]
        predictor = BoostedTreePredictor().fit(build_next_gameweek_dataset(train, history_windows=(3, 5)))
        frames = build_replay_frames_from_gw_history(
            f"data/historical/data/{target_season}",
            predictor=predictor,
        )
        result = replay_projection_policies(frames)
        heuristic = result.heuristic
        blended = result.blended
        print(
            f"{target_season}: "
            f"heuristic={heuristic.realized_points:.0f} ({heuristic.average_points:.2f}/GW), "
            f"blended={blended.realized_points:.0f} ({blended.average_points:.2f}/GW), "
            f"delta={blended.realized_points - heuristic.realized_points:+.0f}"
        )


if __name__ == "__main__":
    main()
