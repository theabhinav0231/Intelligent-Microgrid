"""
P2P Energy Marketplace — FastAPI Application
Now refactored with OOP Service Layer and Thin Controllers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .database import init_db
from .routers import router as market_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes high-level resources and ensures tables exist."""
    # Note: in production with PostgreSQL, use Alembic for migrations.
    # For SQLite prototype, init_db() ensures tables exist.
    init_db()
    yield
    # Cleanup logic (if any) goes here


app = FastAPI(
    title="⚡ Microgrid P2P Energy Marketplace",
    description="Double auction marketplace for 75 microgrid nodes. Now with OOP & Settlement.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to dashboard domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routers ──
app.include_router(market_router)


@app.get("/health", summary="Health check", tags=["Market"])
def health_check():
    """Simple health check for monitoring."""
    return {
        "status": "healthy", 
        "service": "microgrid-marketplace", 
        "version": "2.0.0",
        "features": ["cda_matching", "auth_v1", "settlement_v1"]
    }

# Run: uvicorn marketplace.main:app --host 0.0.0.0 --port 8000 --reload
