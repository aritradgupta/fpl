"""Train and persist the component-level FPL predictor."""

import argparse
from datetime import UTC, datetime

import pandas as pd

from fpl.modeling.models import BoostedTreePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Causal training CSV")
    parser.add_argument("--output", required=True, help="Model artifact path")
    parser.add_argument("--holdout-gameweeks", type=int, default=6)
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    model = BoostedTreePredictor().fit(data)
    model.save(args.output, metadata={
        "trained_at": datetime.now(UTC).isoformat(),
        "training_rows": len(data),
        "holdout_gameweeks": args.holdout_gameweeks,
    })
    print(f"Saved predictor artifact to {args.output}")


if __name__ == "__main__":
    main()
