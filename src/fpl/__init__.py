"""
FPL Team Creator Application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import httpx

from fpl.api import router as api_router
from fpl.client import FPLClient
from fpl.data import init_db, sync_bootstrap_to_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize database and auto-sync FPL API data on startup."""
    await init_db()
    try:
        client = FPLClient()
        data = await client.get_bootstrap_static()
        await sync_bootstrap_to_db(data)
        print("✓ Live FPL API data auto-synced successfully on startup.")
    except (httpx.HTTPError, OSError) as e:
        print(f"⚠ Auto-sync on startup skipped (offline mode): {e}")
    yield


app = FastAPI(
    title="FPL Team Creator Recommendation System",
    description="Optimal squad, starting XI, and transfer recommendations for Fantasy Premier League 2026-27.",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routes under both /api and root level for convenience
app.include_router(api_router, prefix="/api")
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive API documentation."""
    return RedirectResponse(url="/docs")


def main() -> None:
    """Entry point for starting the FastAPI web server."""
    uvicorn.run("fpl:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
