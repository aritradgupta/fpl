# EPL Fantasy Premier League Team Creator & Recommendation Engine (2026-27)

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed_by-uv-000000.svg)](https://github.com/astral-sh/uv)
[![Linter](https://img.shields.io/badge/linter-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Typing](https://img.shields.io/badge/mypy-checked-blue.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)](https://docs.pytest.org/)

A data-driven **Fantasy Premier League (FPL) Recommendation System** and **Integer Linear Programming (ILP) Squad Optimizer** built for the **2026-27 season**.

The application combines a multi-component Expected Points (xP) projection model, live integration with official FPL API endpoints, SQLite local caching, an ILP solver powered by `PuLP` / COIN-OR CBC, and a FastAPI web application.

---

## Key Features

- **Multi-Component Expected Points (xP) Engine**:
  - **Expected Minutes (xM)**: Rotation risk & start probability model (60+ mins threshold).
  - **Attack xP**: Position-weighted expected goals ($xG$) and assists ($xA$) scoring matrix.
  - **Defense xP**: Clean Sheet probability modeling based on defensive strength.
  - **Defensive Contribution xP (CBIT/CBIRT)**: Evaluates 2025-26 defensive contribution rules (+2 pts threshold floor for defensive midfielders and ball-winning center-backs).
  - **BPS & Bonus Magnet Weighting**: Derived from ICT Index and historical Bonus Points System metrics.
  - **Composite Fixture Scaling**: Fixture Difficulty Rating (FDR 1–5) and Home/Away advantage multipliers.

- **PuLP Integer Linear Programming (ILP) Solver**:
  - **15-Man Squad Optimization**: Solves binary decision variables $x_i \in \{0, 1\}$ to select an optimal £100.0m squad under position limits (2 GKP, 5 DEF, 5 MID, 3 FWD) and club constraints (max 3 players per Premier League team).
  - **Starting XI & Captaincy Optimization**: Solves formation constraints (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), assigns Captain (2x/3x multiplier) and Vice-Captain, and orders the bench (reserve GKP on Bench #1, outfielders ordered by xP).
  - **Transfer & Hit Strategy Evaluator**: Compares current squad against transfer options, accounting for banked free transfers (1 to 5), hit penalties (-4 pts per hit), bank budget, and chip strategy execution (`wildcard`, `freehit`, `bboost`, `3xc`).

- **FastAPI Web Service & REST API**:
  - Full REST API with Swagger UI (`/docs`).
  - Supports both GET query parameters and POST JSON payloads for squad and transfer recommendations.
  - Auto-syncs live player data from the official FPL API on startup into an async SQLite database (`aiosqlite`).

- **Rich Terminal CLI Tool**:
  - Live terminal recommendation runner ([`scripts/recommend_squad.py`](file:///d:/repos/fpl/scripts/recommend_squad.py)) rendering formatted color tables with minute projections and xP breakdowns.

---

## Project Architecture

```C
d:\repos\fpl\
├── pyproject.toml               # Package configuration & tool settings (Pylint, Mypy, Pytest)
├── .gitignore                   # Version control ignore rules
├── README.md                    # Project documentation
├── analysis/
│   └── eda.py                   # 5-Season Exploratory Data Analysis Script
├── scripts/
│   └── recommend_squad.py       # Terminal CLI tool for live GW recommendations
├── tests/
│   ├── test_expected_points.py  # Unit tests for Expected Points engine
│   └── test_optimizer.py        # Unit tests for PuLP ILP Solver & Transfer optimizer
└── src/fpl/
    ├── __init__.py              # FastAPI Application & Lifespan Server Entrypoint
    ├── api/                     # REST API Layer
    │   ├── routes.py            # FastAPI Route Handlers
    │   └── schemas.py           # Pydantic Request & Response Schemas
    ├── client/                  # FPL API Integration
    │   └── fpl_client.py        # Async HTTP client for Fantasy Premier League API
    ├── data/                    # Storage & Cache Layer
    │   └── database.py          # Async SQLite database persistence (fpl_cache.db)
    ├── models/                  # Strongly Typed Domain Models
    │   ├── player.py            # PlayerStats, FixtureContext, PlayerProjection, Position
    │   └── squad.py             # SquadRecommendation, SelectedPlayer, TransferRecommendation, ChipType
    ├── optimizer/               # Mathematical Recommendation Engine
    │   ├── expected_points.py   # Multi-component Expected Points (xP) Engine
    │   └── solver.py            # PuLP Integer Linear Programming (ILP) Solver
    └── rules/                   # FPL Rules Engine
        └── constraints.py       # Squad, Budget, Formation & Transfer Rules
```

---

## Quickstart & Installation

### Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package installer)

### Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/aritradgupta/fpl.git
cd fpl
uv sync
```

---

## Usage

### 1. Run Live Squad Recommendation CLI

Fetch live player prices & projections from the official FPL API and generate recommendations:

```bash
# Single-Period PuLP solver (default)
uv run python scripts/recommend_squad.py --solver single_period

# Lock Erling Haaland into the squad (--lock Haaland)
uv run python scripts/recommend_squad.py --lock Haaland

# Multi-Period Horizon solver (5-week block)
uv run python scripts/recommend_squad.py --solver multi_period --gameweek 1 --horizon 5

# Stochastic Risk-Adjusted Scenario solver (CUDA GPU accelerated)
uv run python scripts/recommend_squad.py --solver stochastic --risk-aversion 0.25

# PyTorch CUDA GPU Vectorized Genetic Algorithm (custom seed, generations & population)
uv run python scripts/recommend_squad.py --solver genetic --seed 42 --generations 80 --population 100
```

#### CLI Command-Line Arguments

| Argument | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--solver` | `str` | `single_period` | `single_period`, `multi_period`, `stochastic`, `genetic` | Selects solver strategy model |
| `--lock` | `str list` | `None` | Any player name | Locks specific player(s) into squad (e.g. `--lock Haaland`) |
| `--exclude` | `str list` | `None` | Any player name | Excludes specific player(s) from squad (e.g. `--exclude Palmer`) |
| `--seed` | `int` | `42` | Any integer | Random seed (`42` for fixed/reproducible, or custom int) |
| `--generations` | `int` | `50` | $\ge 1$ | Number of evolutionary generations (`genetic` solver) |
| `--population` | `int` | `60` | $\ge 10$ | Chromosome population size (`genetic` solver) |
| `--risk-aversion` | `float` | `0.15` | $0.0 - 2.0$ | Risk aversion parameter $\lambda$ (`stochastic` solver) |
| `--horizon` | `int` | `3` | $1 - 10$ | Number of upcoming gameweeks in multi-period block (`multi_period` solver) |
| `--gameweek` | `int` | `1` | $1 - 50$ | First gameweek used for fixture difficulty and home/away context |

The CLI and squad recommendation API fetch official fixtures and pass them to every solver. If fixtures cannot be reached, the system falls back to the neutral projection model so cached/offline analysis remains usable.

### 2. Start FastAPI Web Server

Launch the local web server:

```bash
uv run fpl
```

Open your browser to:

- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **List Players**: `http://127.0.0.1:8000/players`
- **Squad Recommendation**: `http://127.0.0.1:8000/recommend/squad`
- **Interactive Solver Lab**: `http://127.0.0.1:8000/lab`

The Solver Lab runs the selected single-period, multi-period, stochastic, and genetic solvers over the same cached player pool and official fixture snapshot. It reports runtime, status, projected points, cost, captain, and selected player IDs side by side. A failed solver is reported independently so one experimental strategy does not hide the others' results.

---

## REST API Endpoints

| Method       | Endpoint               | Description                                              |
| :----------- | :--------------------- | :------------------------------------------------------- |
| `GET`        | `/health`              | Application status health check                          |
| `POST`       | `/sync`                | Triggers live FPL API data fetch and SQLite cache update |
| `GET`        | `/players`             | List and filter players by position, team, or max cost   |
| `GET / POST` | `/recommend/squad`     | Solves ILP for optimal 15-man squad & starting XI        |
| `POST`       | `/recommend/transfers` | Evaluates optimal transfers & hit strategy (-4 pts/hit)  |

---

## Testing & Quality Verification

### Run Pytest Suite

```bash
uv run pytest tests/
```

### Historical Solver Replay

The replay harness in [`src/fpl/optimizer/replay.py`](src/fpl/optimizer/replay.py)
compares heuristic and blended projections using the same constrained ILP. Each
gameweek frame must contain point-in-time `id`, `position`, `team`, `cost`, and
`actual_points` columns, plus the two forecast columns. Do not substitute
end-of-season team or price metadata: that leaks future information into the
historical decision.

### Run Static Type Checker (`mypy`)

```bash
uv run mypy src/fpl
```

### Run Fast Linter & Formatter (`ruff`)

```bash
# Check code for linting issues
uv run ruff check .

# Format all code files
uv run ruff format .
```

---

## GPU Acceleration (NVIDIA CUDA)

To enable hardware tensor acceleration for Monte Carlo simulations on NVIDIA GPUs (e.g., RTX 4060):

```bash
uv add torch
```

`pyproject.toml` is configured with the official PyTorch CUDA index for Windows (`pytorch-cu124`). Once installed, [`src/fpl/optimizer/gpu.py`](file:///d:/repos/fpl/src/fpl/optimizer/gpu.py) automatically utilizes your GPU!

---

## License

MIT License. Developed for EPL Fantasy Premier League 2026-27 team management.
