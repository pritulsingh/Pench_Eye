from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from app.core.config import settings

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    Path(settings.DATABASE_URL.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
else:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def create_tables():
    import app.models  # noqa: F401  — populate metadata before create_all

    async with engine.begin() as conn:
        if not IS_SQLITE:
            from sqlalchemy import text

            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception:
                pass
        await conn.run_sync(Base.metadata.create_all)
