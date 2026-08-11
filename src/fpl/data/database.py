"""
SQLite Database Layer for FPL Data Persistence & Sync.

Manages local caching of live FPL API data:
- Players table
- Teams table
- Sync timestamp logging
"""

from pathlib import Path
from typing import Dict
import aiosqlite
import pandas as pd


DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "fpl_cache.db"


async def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite database tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                web_name TEXT,
                first_name TEXT,
                second_name TEXT,
                position TEXT,
                team TEXT,
                team_code INTEGER,
                cost REAL,
                ep_next REAL,
                form REAL,
                points_per_game REAL,
                total_points INTEGER,
                minutes INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                selected_by_percent REAL,
                raw_json TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT,
                short_name TEXT,
                strength INTEGER,
                strength_overall_home INTEGER,
                strength_overall_away INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def sync_bootstrap_to_db(bootstrap_data: Dict, db_path: Path = DB_PATH) -> int:
    """Sync live bootstrap-static JSON data to SQLite database."""
    await init_db(db_path)

    teams_list = bootstrap_data.get("teams", [])
    team_id_to_name = {t["id"]: t["name"] for t in teams_list}

    element_types = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    elements = bootstrap_data.get("elements", [])

    async with aiosqlite.connect(db_path) as db:
        for t in teams_list:
            await db.execute(
                """
                INSERT OR REPLACE INTO teams (
                    id, name, short_name, strength,
                    strength_overall_home, strength_overall_away
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    t["id"],
                    t["name"],
                    t["short_name"],
                    t.get("strength", 3),
                    t.get("strength_overall_home", 1000),
                    t.get("strength_overall_away", 1000),
                ),
            )

        for p in elements:
            pos = element_types.get(p.get("element_type", 3), "MID")
            team_name = team_id_to_name.get(p.get("team"), "Unknown")
            cost = p.get("now_cost", 0) / 10.0

            await db.execute(
                """
                INSERT OR REPLACE INTO players (
                    id, web_name, first_name, second_name, position, team, team_code,
                    cost, ep_next, form, points_per_game, total_points, minutes,
                    goals_scored, assists, clean_sheets, selected_by_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p["id"],
                    p.get("web_name", ""),
                    p.get("first_name", ""),
                    p.get("second_name", ""),
                    pos,
                    team_name,
                    p.get("team_code", 0),
                    cost,
                    float(p.get("ep_next") or 0.0),
                    float(p.get("form") or 0.0),
                    float(p.get("points_per_game") or 0.0),
                    p.get("total_points", 0),
                    p.get("minutes", 0),
                    p.get("goals_scored", 0),
                    p.get("assists", 0),
                    p.get("clean_sheets", 0),
                    float(p.get("selected_by_percent") or 0.0),
                ),
            )

        await db.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES ('last_sync', datetime('now'))"
        )
        await db.commit()

    return len(elements)


async def load_players_df_from_db(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load player dataset from SQLite into pandas DataFrame."""
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT * FROM players") as cursor:
            rows = await cursor.fetchall()
            cols = [col[0] for col in cursor.description]
            return pd.DataFrame(rows, columns=cols)
