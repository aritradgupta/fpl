"""Run a season-level heuristic versus learned projection replay."""

import pandas as pd

from fpl.modeling.features import build_next_gameweek_dataset
from fpl.modeling.models import BoostedTreePredictor
from fpl.optimizer.replay import build_replay_frames_from_gw_history, replay_projection_policies


def main() -> None:
    raw = pd.read_csv("data/historical/data/cleaned_merged_seasons.csv", low_memory=False)
    train_raw = raw[raw["season_x"].isin([
        "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24",
    ])]
    model = BoostedTreePredictor().fit(build_next_gameweek_dataset(train_raw, history_windows=(3, 5)))
    frames = build_replay_frames_from_gw_history("data/historical/data/2024-25", predictor=model)
    print(replay_projection_policies(frames))


if __name__ == "__main__":
    main()
