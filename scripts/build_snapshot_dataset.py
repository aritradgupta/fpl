"""Build a causal training dataset from snapshot and outcome CSV archives."""

import argparse

import pandas as pd

from fpl.modeling import build_snapshot_training_dataset, load_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Pre-deadline snapshot CSV")
    parser.add_argument("--outcomes", required=True, help="Outcome CSV with season/gameweek/player_id")
    parser.add_argument("--output", required=True, help="Output training CSV")
    args = parser.parse_args()

    snapshot = load_snapshots(args.snapshots)
    outcomes = pd.read_csv(args.outcomes, low_memory=False)
    dataset = build_snapshot_training_dataset(snapshot, outcomes)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {len(dataset):,} training rows to {args.output}")


if __name__ == "__main__":
    main()
