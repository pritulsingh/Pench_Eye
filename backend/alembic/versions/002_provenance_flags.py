"""Add provenance flags needed to separate synthetic and live records."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002_provenance_flags"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table in ("camera_stations", "embeddings"):
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "is_demo" not in columns:
            op.add_column(
                table,
                sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.false()),
            )

    # Existing seed cameras are identifiable by the seed routine's explicit description.
    op.execute(
        sa.text(
            "UPDATE camera_stations SET is_demo = true "
            "WHERE description = 'Simulated camera station for the Pench Eye demo dataset.'"
        )
    )
    op.execute(
        sa.text(
            "UPDATE embeddings SET is_demo = true WHERE observation_id IN "
            "(SELECT id FROM observations WHERE is_demo = true)"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table in ("embeddings", "camera_stations"):
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "is_demo" in columns:
            op.drop_column(table, "is_demo")