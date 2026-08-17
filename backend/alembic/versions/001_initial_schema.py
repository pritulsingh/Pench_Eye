"""initial schema

Creates extensions and relies on SQLAlchemy create_all for table creation.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite is the local fallback and does not support PostgreSQL extensions.
    if op.get_bind().dialect.name == "postgresql":
        op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        op.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
        op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
    # Tables are managed by SQLAlchemy create_all() in app startup (dev mode)
    # In production, generate full migration with: alembic revision --autogenerate


def downgrade() -> None:
    pass