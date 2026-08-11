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

Fetch live player prices & projections from the official FPL API and generate the optimal 15-man squad:

```bash
uv run python scripts/recommend_squad.py
```

### 2. Start FastAPI Web Server

Launch the local web server:

```bash
uv run fpl
```

Open your browser to:

- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **List Players**: `http://127.0.0.1:8000/players`
- **Squad Recommendation**: `http://127.0.0.1:8000/recommend/squad`

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

## License

MIT License. Developed for EPL Fantasy Premier League 2026-27 team management.

## FPL analysis and squad optimizer

An async Python application for combining the official Fantasy Premier League API with local historical data analysis.

## Quick start

   uv sync
   uv run python scripts/recommend_squad.py
   uv run pytest

The API starts with uv run fpl and exposes interactive documentation at /docs. Live bootstrap data is cached in data/fpl_cache.db; the app can continue to read the cache when the official API is unavailable.

## Project layout

- src/fpl/client: resilient async client for official endpoints.
- src/fpl/data: SQLite cache and bootstrap synchronization.
- src/fpl/optimizer: expected-points model and PuLP squad/transfer optimization.
- src/fpl/rules: centralized squad, formation, transfer, and scoring rules.
- analysis/eda.py: exploratory analysis of vaastav historical data.

The optimizer uses current API prices in £m, requires the official 2 GKP / 5 DEF / 5 MID / 3 FWD squad shape, enforces a three-player-per-club limit by default, and checks solver status before returning a recommendation.

## Data caveats

Historical vaastav files and the official API do not always use identical column names or definitions. Treat projections as a baseline for analysis rather than a guarantee. Fixture context should be supplied explicitly when using the projection functions for a specific gameweek.
