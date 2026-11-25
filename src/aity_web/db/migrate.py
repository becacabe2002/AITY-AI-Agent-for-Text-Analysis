"""Run the Alembic migrations from Python (tests, the W9 deploy, `make bootstrap`).

    python -m aity_web.db.migrate            # upgrade head, URL from AITY_DATABASE_URL
    python -m aity_web.db.migrate base       # downgrade to an empty database

Equivalent to `alembic upgrade head` from the repo root, but works from any
directory because it does not need alembic.ini.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    """An in-memory Alembic config pointing at db/migrations. ``url`` (any
    form database_url() accepts) overrides AITY_DATABASE_URL."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("path_separator", "os")
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


def main(argv: list[str]) -> int:
    target = argv[0] if argv else "head"
    if target == "base" or target.startswith("-"):
        downgrade(revision=target)
    else:
        upgrade(revision=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
