"""Run the Alembic migrations from Python without needing alembic.ini."""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    """Build an in-memory Alembic config for db/migrations, optionally with a database URL."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("path_separator", "os")
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    """Upgrade the database to a revision (default: head)."""
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    """Downgrade the database to a revision (default: base)."""
    command.downgrade(alembic_config(url), revision)


def main(argv: list[str]) -> int:
    """Upgrade to the given revision, or downgrade for 'base' or an option-like argument; return 0.
    """
    target = argv[0] if argv else "head"
    if target == "base" or target.startswith("-"):
        downgrade(revision=target)
    else:
        upgrade(revision=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
