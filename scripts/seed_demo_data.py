#!/usr/bin/env python3
"""
DEMO DATA SEEDER — Pench Eye

All rows created here are SIMULATED and flagged is_demo=True.
Works against whatever DATABASE_URL is configured (SQLite by default,
PostgreSQL + pgvector under docker compose).

Usage:
    python scripts/seed_demo_data.py            # seed if empty
    python scripts/seed_demo_data.py --reset    # wipe demo rows and re-seed
"""
import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))


async def main(reset: bool) -> None:
    from app.core.database import AsyncSessionLocal, create_tables
    from app.core.seed import seed_all

    await create_tables()
    async with AsyncSessionLocal() as session:
        summary = await seed_all(session, reset=reset)

    print("Pench Eye demo data seeded (all rows labelled DEMO):")
    for key, value in summary.items():
        print(f"  {key:<14} {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Pench Eye demo data.")
    parser.add_argument("--reset", action="store_true", help="Delete existing demo rows first.")
    args = parser.parse_args()
    asyncio.run(main(args.reset))
