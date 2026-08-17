"""
Portable SQLAlchemy column types.

The system targets PostgreSQL (+pgvector) in Docker, but must also run on
SQLite so the prototype can start with zero infrastructure. These decorators
pick the native PostgreSQL type when available and fall back to portable
representations elsewhere.
"""
import json
import uuid
from typing import Any, Optional

from sqlalchemy import JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.types import TypeDecorator


class GUID(TypeDecorator):
    """UUID column: native uuid on PostgreSQL, 36-char string elsewhere."""

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value: Any, dialect) -> Optional[Any]:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value: Any, dialect) -> Optional[uuid.UUID]:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class JSONType(TypeDecorator):
    """JSON column: JSONB on PostgreSQL, JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class VectorType(TypeDecorator):
    """
    Embedding column: pgvector `vector(dim)` on PostgreSQL when the extension
    bindings are installed, otherwise a JSON-encoded float list.

    Similarity search uses the pgvector operator when available and an
    in-Python cosine fallback otherwise (see ReIDService.search_similar).
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 512):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector

                return dialect.type_descriptor(Vector(self.dim))
            except ImportError:
                pass
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            try:
                import pgvector.sqlalchemy  # noqa: F401

                return list(value)
            except ImportError:
                pass
        return json.dumps([float(v) for v in value])

    def process_result_value(self, value: Any, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            return json.loads(value)
        return list(value)
